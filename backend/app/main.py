"""Local single-user FastAPI app. Use ONE uvicorn worker in this development release."""
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from . import db
from .config import prepare_storage, settings
from .logging_setup import get_logger
from .schemas import ChatRequest, ConversationCreate, ConversationRename, ProfileUpdate, MemoryCreate, SearchRequest
from .knowledge.ingest import process_pdf
from .knowledge.tables import load_tables
from .knowledge.retrieval import search, get_passage, DenseUnavailable
from .agent.model import ModelClient
from .agent.workflow import AgentRunner

logger=get_logger('api')


@asynccontextmanager
async def lifespan(app:FastAPI):
    prepare_storage()
    db.initialize()
    logger.info('backend ready data_dir=%s model=%s provider=%s', settings.data_dir, settings.model_name, settings.model_provider)
    yield


app=FastAPI(title='Adaptive Agentic RAG (local development)',version='0.1.15',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=settings.cors_origins,allow_methods=['GET','POST','PUT','PATCH','DELETE'],
                   allow_headers=['Content-Type','Last-Event-ID'],allow_credentials=False)


def owned_paper(paper_id,required=True):
    p=db.one('SELECT * FROM papers WHERE id=? AND owner_id=?',(paper_id,settings.owner_id))
    if required and not p:raise HTTPException(404,'Paper not found')
    return p


def owned_conversation(conversation_id):
    c=db.one('SELECT * FROM conversations WHERE id=? AND owner_id=?',(conversation_id,settings.owner_id))
    if not c:raise HTTPException(404,'Conversation not found')
    return c


def owned_run(run_id):
    r=db.one('''SELECT r.* FROM runs r JOIN conversations c ON c.id=r.conversation_id
                WHERE r.id=? AND c.owner_id=?''',(run_id,settings.owner_id))
    if not r:raise HTTPException(404,'Run not found')
    for key in ('paper_ids','citations','metrics'):
        r[key]=db.decode(r[key], [] if key!='metrics' else {})
    return r


@app.get('/api/health')
async def health():
    model=ModelClient()
    return {'status':'ok','model_base_url':settings.model_base_url,'model':settings.model_name,
            'model_provider':model.provider,'model_available':await model.is_available(), 'mode':'single-user-local'}


@app.post('/api/papers',status_code=202)
async def create_paper(background:BackgroundTasks,file:UploadFile=File(...)):
    if not (file.filename or '').lower().endswith('.pdf'):
        raise HTTPException(400,'Only PDF files are accepted')
    content=await file.read(settings.max_upload_mb*1024*1024+1)
    if len(content)>settings.max_upload_mb*1024*1024:
        raise HTTPException(413,'PDF too large')
    if not content.startswith(b'%PDF-'):
        raise HTTPException(400,'Invalid PDF signature')
    paper_id=uuid4().hex
    pdf_path=settings.data_dir/'pdfs'/f'{paper_id}.pdf'
    pdf_path.write_bytes(content)
    title=Path(file.filename).name[:180]
    db.execute('''INSERT INTO papers(id,owner_id,title,source_type,document_path,status,created_at)
               VALUES(?,?,?,?,?,?,?)''',(paper_id,settings.owner_id,title,'pdf',f'pdfs/{paper_id}.pdf','processing',db.now()))
    logger.info('paper=%s upload queued filename=%s bytes=%d',paper_id,title,len(content))
    background.add_task(process_pdf,paper_id,pdf_path)
    return {'id':paper_id,'title':title,'status':'processing'}


@app.get('/api/papers')
async def list_papers():
    return db.rows('''SELECT id,title,source_type,status,parser,error,created_at FROM papers
                      WHERE owner_id=? ORDER BY created_at DESC''',(settings.owner_id,))


@app.get('/api/papers/{paper_id}')
async def get_paper(paper_id:str):
    p=owned_paper(paper_id)
    p.pop('document_path',None)
    return p


def resolve_pdf_path(paper):
    """Prefer portable paths; recover old absolute paths after a directory move."""
    original=Path(paper['document_path'])
    path=original if original.is_absolute() else settings.data_dir/original
    if path.is_file():
        return path
    # v0.1.4 wrote absolute paths. An update may move the project and retain the DB.
    # Only use the expected file for this paper ID, never an arbitrary neighboring PDF.
    candidate=settings.data_dir/'pdfs'/f"{paper['id']}.pdf"
    if candidate.is_file():
        db.execute('UPDATE papers SET document_path=? WHERE id=?',
                   (f"pdfs/{paper['id']}.pdf",paper['id']))
        logger.info('paper=%s restored portable PDF path',paper['id'])
        return candidate
    logger.warning('paper=%s PDF file missing stored_path=%s fallback=%s',
                   paper['id'],paper['document_path'],candidate)
    return None


@app.get('/api/papers/{paper_id}/tables')
async def get_paper_tables(paper_id:str, number:int | None = None):
    # Diagnostic endpoint: exposes parsed cells and their PDF page, without
    # exposing files or QASPER gold answers.
    owned_paper(paper_id)
    tables=await asyncio.to_thread(load_tables,[paper_id],number)
    return {'paper_id':paper_id,'tables':tables}


@app.get('/api/papers/{paper_id}/document')
async def pdf_document(paper_id:str):
    p=owned_paper(paper_id)
    if p['source_type']!='pdf' or not p['document_path']:
        raise HTTPException(404,'This QASPER record has no local PDF')
    path=resolve_pdf_path(p)
    if path is None:
        raise HTTPException(404,'PDF file missing. If the project directory was moved, restore data/pdfs or re-upload the PDF.')
    return FileResponse(path,media_type='application/pdf',headers={'Content-Disposition':'inline'})


@app.delete('/api/papers/{paper_id}')
async def delete_paper(paper_id:str):
    p=owned_paper(paper_id)
    # Delete cached vector arrays that contain this paper, including multi-paper indexes.
    cached=db.rows('SELECT DISTINCT cache_file FROM vector_cache WHERE paper_id=?',(paper_id,))
    db.execute('DELETE FROM papers WHERE id=?',(paper_id,))
    for item in cached:
        (settings.data_dir/'indexes'/item['cache_file']).unlink(missing_ok=True)
        db.execute('DELETE FROM vector_cache WHERE cache_file=?',(item['cache_file'],))
    if p['document_path']:
        path=resolve_pdf_path(p)
        if path:path.unlink(missing_ok=True)
    return {'deleted':paper_id}


@app.post('/api/search')
async def search_papers(req:SearchRequest):
    for paper_id in req.paper_ids:
        p=owned_paper(paper_id)
        if p['status']!='ready':raise HTTPException(409,f'Paper {paper_id} is not ready')
    try:return await asyncio.to_thread(search,req.query,req.paper_ids,req.strategy,req.top_k,req.rerank)
    except DenseUnavailable as e:raise HTTPException(503,str(e)) from e


@app.get('/api/passage/{chunk_id:path}')
async def passage(chunk_id:str,paper_id:str):
    owned_paper(paper_id)
    evidence=get_passage(chunk_id,[paper_id])
    if not evidence:raise HTTPException(404,'Passage not found')
    return evidence


@app.post('/api/conversations',status_code=201)
async def create_conversation(req:ConversationCreate):
    c={'id':uuid4().hex,'title':req.title,'created_at':db.now()}
    db.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
               (c['id'],settings.owner_id,c['title'],c['created_at']))
    return c


@app.get('/api/conversations')
async def list_conversations():
    # Include EVERY conversation. Last message time keeps the sidebar order useful.
    return db.rows('''SELECT c.id,c.title,c.created_at,
                      COALESCE(m.last_message_at,c.created_at) AS updated_at,
                      COALESCE(m.message_count,0) AS message_count
                      FROM conversations c
                      LEFT JOIN (
                        SELECT conversation_id,COUNT(*) AS message_count,
                               MAX(created_at) AS last_message_at
                        FROM messages GROUP BY conversation_id
                      ) m ON m.conversation_id=c.id
                      WHERE c.owner_id=?
                      ORDER BY updated_at DESC,c.created_at DESC,c.id DESC''',(settings.owner_id,))


@app.patch('/api/conversations/{conversation_id}')
async def rename_conversation(conversation_id:str,req:ConversationRename):
    owned_conversation(conversation_id)
    title=req.title.strip()
    if not title:
        raise HTTPException(400,'Conversation title cannot be empty')
    db.execute('UPDATE conversations SET title=? WHERE id=? AND owner_id=?',
               (title,conversation_id,settings.owner_id))
    logger.info('conversation=%s renamed',conversation_id)
    return {'id':conversation_id,'title':title}


@app.delete('/api/conversations/{conversation_id}')
async def delete_conversation(conversation_id:str):
    owned_conversation(conversation_id)
    active=db.one('''SELECT id FROM runs WHERE conversation_id=? AND status IN ('pending','running') LIMIT 1''',
                  (conversation_id,))
    if active:
        raise HTTPException(409,'This conversation has an active run. Cancel or finish it first.')
    # SQLite ON DELETE CASCADE removes messages, runs and persisted SSE events.
    db.execute('DELETE FROM conversations WHERE id=? AND owner_id=?',(conversation_id,settings.owner_id))
    logger.info('conversation=%s deleted',conversation_id)
    return {'deleted':conversation_id}


@app.get('/api/conversations/{conversation_id}')
async def get_conversation(conversation_id:str):
    c=owned_conversation(conversation_id)
    messages=db.rows('SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at,id',(conversation_id,))
    for m in messages:m['citations']=db.decode(m['citations'],[])
    return dict(c,messages=messages)


@app.post('/api/agent/runs',status_code=202)
async def create_run(req:ChatRequest,background:BackgroundTasks):
    owned_conversation(req.conversation_id)
    for pid in req.paper_ids:
        p=owned_paper(pid)
        if p['status']!='ready':raise HTTPException(409,f'Paper {pid} is not ready')
    run_id=uuid4().hex
    with db.connect() as c:
        c.execute('''INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at)
                     VALUES(?,?,?,?,?,?,?)''',(run_id,req.conversation_id,'pending',req.mode,req.question,db.encode(req.paper_ids),db.now()))
        c.execute('''INSERT INTO messages(id,conversation_id,role,content,citations,run_id,created_at)
                     VALUES(?,?,?,?,?,?,?)''',(uuid4().hex,req.conversation_id,'user',req.question,'[]',run_id,db.now()))
        c.execute('UPDATE conversations SET title=? WHERE id=? AND title=?',
                  (req.question[:85],req.conversation_id,'New conversation'))
    logger.info('run=%s accepted conversation=%s mode=%s papers=%s',run_id,req.conversation_id,req.mode,req.paper_ids)
    background.add_task(AgentRunner().run,run_id,req)
    return {'id':run_id,'status':'pending','mode':req.mode,'conversation_id':req.conversation_id}


@app.get('/api/agent/runs/{run_id}')
async def get_run(run_id:str):
    return owned_run(run_id)


@app.get('/api/debug/runs')
async def debug_recent_runs():
    """Convenient single-user demo diagnostics. No separate log platform."""
    return db.rows('''SELECT r.id,r.status,r.mode,r.question,r.error,r.created_at,r.completed_at
                      FROM runs r JOIN conversations c ON c.id=r.conversation_id
                      WHERE c.owner_id=? ORDER BY r.created_at DESC LIMIT 15''',(settings.owner_id,))


@app.get('/api/agent/runs/{run_id}/debug')
async def debug_run(run_id:str):
    run=owned_run(run_id)
    events=db.rows('''SELECT id,event_type,payload,created_at FROM run_events
                      WHERE run_id=? ORDER BY id''',(run_id,))
    for event in events:
        event['payload']=db.decode(event['payload'],{})
    return {'run':run,'events':events,
            'log_file':str(settings.data_dir/'logs'/'app.log')}


@app.post('/api/agent/runs/{run_id}/cancel')
async def cancel_run(run_id:str):
    r=owned_run(run_id)
    if r['status'] in ('pending','running'):
        db.execute('UPDATE runs SET cancel_requested=1 WHERE id=?',(run_id,))
        return {'id':run_id,'cancel_requested':True}
    return {'id':run_id,'status':r['status'],'cancel_requested':False}


@app.get('/api/agent/runs/{run_id}/events')
async def run_events(run_id:str,request:Request,after:int=0):
    owned_run(run_id)
    header=request.headers.get('last-event-id')
    if header and header.isdigit():after=max(after,int(header))
    async def generate():
        last=max(0,after)
        while True:
            if await request.is_disconnected():break
            events=db.rows('SELECT * FROM run_events WHERE run_id=? AND id>? ORDER BY id LIMIT 150',(run_id,last))
            for item in events:
                last=item['id']
                yield f"id: {last}\nevent: {item['event_type']}\ndata: {item['payload']}\n\n"
            run=owned_run(run_id)
            if run['status'] in ('failed','succeeded','partial','cancelled') and not events:
                yield f"event: stream_end\ndata: {json.dumps({'status':run['status']})}\n\n"
                break
            if not events: yield ': heartbeat\n\n'
            await asyncio.sleep(0.4)
    return StreamingResponse(generate(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})


@app.get('/api/learning/profile')
async def get_profile():
    p=db.one('SELECT * FROM profile WHERE owner_id=?',(settings.owner_id,))
    return {'language':p['language'],'explanation_level':p['explanation_level'],
            'domains':db.decode(p['domains'],[]),'goals':db.decode(p['goals'],[]),
            'memory_enabled':bool(p['memory_enabled'])}


@app.put('/api/learning/profile')
async def update_profile(req:ProfileUpdate):
    db.execute('''UPDATE profile SET language=?,explanation_level=?,domains=?,goals=?,memory_enabled=? WHERE owner_id=?''',
               (req.language,req.explanation_level,db.encode(req.domains),db.encode(req.goals),int(req.memory_enabled),settings.owner_id))
    return await get_profile()


@app.get('/api/learning/memories')
async def list_memories():
    return db.rows('SELECT * FROM memories WHERE owner_id=? ORDER BY updated_at DESC',(settings.owner_id,))


@app.post('/api/learning/memories/confirm',status_code=201)
async def add_confirmed_memory(req:MemoryCreate):
    # Explicit user confirmation: never turn questions into automatic claims of mastery.
    mid=uuid4().hex
    db.execute('''INSERT INTO memories(id,owner_id,domain,topic,memory_type,content,mastery,created_at,updated_at)
                  VALUES(?,?,?,?,?,?,?,?,?)''',
               (mid,settings.owner_id,req.domain,req.topic,req.memory_type,req.content,req.mastery,db.now(),db.now()))
    return db.one('SELECT * FROM memories WHERE id=?',(mid,))


@app.patch('/api/learning/memories/{memory_id}')
async def change_memory(memory_id:str,req:MemoryCreate):
    updated=db.execute('''UPDATE memories SET domain=?,topic=?,memory_type=?,content=?,mastery=?,updated_at=?
                          WHERE id=? AND owner_id=?''',
                       (req.domain,req.topic,req.memory_type,req.content,req.mastery,db.now(),memory_id,settings.owner_id))
    if not updated:raise HTTPException(404,'Memory not found')
    return db.one('SELECT * FROM memories WHERE id=?',(memory_id,))


@app.delete('/api/learning/memories/{memory_id}')
async def remove_memory(memory_id:str):
    deleted=db.execute('DELETE FROM memories WHERE id=? AND owner_id=?',(memory_id,settings.owner_id))
    if not deleted:raise HTTPException(404,'Memory not found')
    return {'deleted':memory_id}

@app.get('/api/papers/{paper_id}/chunks')
async def list_paper_chunks(paper_id:str):
    owned_paper(paper_id)
    out=db.rows('SELECT id,seq,section,content,paragraph_refs,pages FROM chunks WHERE paper_id=? ORDER BY seq',(paper_id,))
    for c in out:
        c['paragraph_refs']=db.decode(c['paragraph_refs'],[])
        c['pages']=db.decode(c['pages'],[])
    return out

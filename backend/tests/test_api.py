from pathlib import Path
import fitz
from fastapi.testclient import TestClient
from app.main import app
from app import db


CLIENT=TestClient(app)


def test_pdf_upload_parse_read_delete(tmp_path):
    doc=fitz.open()
    page=doc.new_page()
    page.insert_text((60,90),'The algorithm uses a transformer to retrieve scientific paragraphs.')
    raw=doc.tobytes()
    doc.close()
    with CLIENT:
        r=CLIENT.post('/api/papers',files={'file':('example.pdf',raw,'application/pdf')})
        assert r.status_code==202,r.text
        pid=r.json()['id']
        p=CLIENT.get('/api/papers/'+pid).json()
        assert p['status']=='ready',p
        assert p['parser'] in ('docling','pymupdf')
        assert CLIENT.get('/api/papers/'+pid+'/document').status_code==200
        s=CLIENT.post('/api/search',json={'query':'transformer','paper_ids':[pid],'strategy':'keyword'})
        assert s.status_code==200,s.text
        assert s.json()['evidences']
        assert CLIENT.delete('/api/papers/'+pid).status_code==200
        assert CLIENT.get('/api/papers/'+pid).status_code==404


def test_profile_memory_and_conversation():
    with CLIENT:
        c=CLIENT.post('/api/conversations',json={'title':'Paper study'}).json()
        assert CLIENT.get('/api/conversations/'+c['id']).status_code==200
        x=CLIENT.put('/api/learning/profile',json={'language':'en','explanation_level':'beginner','domains':['NLP'],'goals':[], 'memory_enabled':True})
        assert x.status_code==200
        assert x.json()['domains']==['NLP']
        m=CLIENT.post('/api/learning/memories/confirm',json={'domain':'NLP','topic':'attention','memory_type':'progress','content':'I studied attention','mastery':'studied'})
        assert m.status_code==201
        assert len(CLIENT.get('/api/learning/memories').json())==1
        assert CLIENT.delete('/api/learning/memories/'+m.json()['id']).status_code==200
        assert CLIENT.get('/api/learning/memories').json()==[]


def test_upload_signature_rejected():
    with CLIENT:
        response=CLIENT.post('/api/papers',files={'file':('bad.pdf',b'this is not a PDF','application/pdf')})
        assert response.status_code==400


def test_http_run_and_sse_replay(monkeypatch):
    import numpy as np
    import app.main as entry
    from app.agent.workflow import AgentRunner
    from app.knowledge import retrieval
    from app.knowledge.ingest import import_qasper_file
    class FakeLocalModel:
        async def chat(self,messages,**kwargs):
            return {'content':'{"sufficient":true,"missing":""}','usage':{},'latency_ms':1}
        async def stream(self,messages,**kwargs):
            yield 'Combined dense and BM25 [1]'
    monkeypatch.setattr(entry,'AgentRunner',lambda:AgentRunner(FakeLocalModel()))
    monkeypatch.setattr(retrieval,'_dense_scores',lambda q,chunks:np.array([float('BM25' in c['content']) for c in chunks],dtype='float32'))
    demo=Path(__file__).resolve().parents[2]/'examples'/'qasper-mini.json'
    with CLIENT:
        import_qasper_file(demo)
        conv=CLIENT.post('/api/conversations',json={'title':'Test'}).json()['id']
        response=CLIENT.post('/api/agent/runs',json={
            'conversation_id':conv,'question':'What is BM25?', 'paper_ids':['qasper-demo-paper-001'],
            'mode':'basic','teaching':False,'use_memory':False,'allow_web':False})
        assert response.status_code==202,response.text
        run=response.json()['id']
        result=CLIENT.get('/api/agent/runs/'+run).json()
        assert result['status']=='succeeded',result
        assert result['citations']
        # Late SSE client receives all persisted events and stream_end.
        stream=CLIENT.get('/api/agent/runs/'+run+'/events')
        assert stream.status_code==200
        assert 'event: answer_delta' in stream.text
        assert 'event: run_completed' in stream.text
        assert 'event: stream_end' in stream.text


def test_pdf_document_recovers_legacy_absolute_path():
    """Old database paths can be recovered when the same PDF exists in current data/pdfs."""
    from pathlib import Path
    doc=fitz.open()
    doc.new_page().insert_text((60,90),'Portable PDF path test')
    raw=doc.tobytes()
    doc.close()
    with CLIENT:
        r=CLIENT.post('/api/papers',files={'file':('portable.pdf',raw,'application/pdf')})
        pid=r.json()['id']
        saved=db.one('SELECT document_path FROM papers WHERE id=?',(pid,))['document_path']
        assert saved==f'pdfs/{pid}.pdf'
        # Simulate an existing DB copied from a different Windows project path.
        db.execute('UPDATE papers SET document_path=? WHERE id=?',
                   (f'E:/old-location/backend/data/pdfs/{pid}.pdf',pid))
        response=CLIENT.get(f'/api/papers/{pid}/document')
        assert response.status_code==200
        assert response.content==raw
        assert db.one('SELECT document_path FROM papers WHERE id=?',(pid,))['document_path']==saved
        assert CLIENT.delete(f'/api/papers/{pid}').status_code==200


def test_pdf_document_missing_source_reports_actionable_404():
    doc=fitz.open()
    doc.new_page().insert_text((60,90),'Missing PDF test')
    raw=doc.tobytes()
    doc.close()
    with CLIENT:
        pid=CLIENT.post('/api/papers',files={'file':('missing.pdf',raw,'application/pdf')}).json()['id']
        (db.settings.data_dir/'pdfs'/f'{pid}.pdf').unlink()
        response=CLIENT.get(f'/api/papers/{pid}/document')
        assert response.status_code==404
        assert 're-upload' in response.json()['detail'].lower()
        assert db.one('SELECT status FROM papers WHERE id=?',(pid,))['status']=='ready'

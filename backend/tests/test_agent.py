import asyncio
from uuid import uuid4
from pathlib import Path
from app import db
from app.agent.workflow import AgentRunner
from app.knowledge.ingest import import_qasper_file
from app.schemas import ChatRequest


class FakeLocalModel:
    model='local-test-double'
    async def chat(self,messages,**kwargs):
        return {'content':'{"sufficient":true,"missing":""}','usage':{},'latency_ms':1}
    async def stream(self,messages,**kwargs):
        yield 'Combined dense and BM25 [1]'


def test_run_persists_result_and_events():
    demo=Path(__file__).resolve().parents[2]/'examples'/'qasper-mini.json'
    import_qasper_file(demo)
    conv=uuid4().hex
    run=uuid4().hex
    with db.connect() as c:
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
            (conv,db.settings.owner_id,'Agent test',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
            (run,conv,'pending','basic','BM25 dense method',db.encode(['qasper-demo-paper-001']),db.now()))
    req=ChatRequest(conversation_id=conv,question='BM25 dense method',paper_ids=['qasper-demo-paper-001'],mode='basic')
    # No LLM API is contacted, but real retrieval and database persistence run.
    asyncio.run(AgentRunner(model=FakeLocalModel()).run(run,req))
    r=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert r['status']=='failed' or r['status']=='succeeded'
    # If embedding weights are cached, the full success path is exercised.
    if r['status']=='succeeded':
        assert '[1]' in r['answer']
        assert db.decode(r['citations'],[])
    else:
        assert 'Dense retrieval requires' in r['error'] or 'model' in r['error'].lower()
    assert db.rows('SELECT * FROM run_events WHERE run_id=?',(run,))


def test_explicit_cancel():
    demo=Path(__file__).resolve().parents[2]/'examples'/'qasper-mini.json'
    import_qasper_file(demo)
    conv=uuid4().hex
    run=uuid4().hex
    with db.connect() as c:
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
            (conv,db.settings.owner_id,'Cancel test',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at,cancel_requested) VALUES(?,?,?,?,?,?,?,?)',
            (run,conv,'pending','basic','test',db.encode(['qasper-demo-paper-001']),db.now(),1))
    req=ChatRequest(conversation_id=conv,question='test',paper_ids=['qasper-demo-paper-001'],mode='basic')
    asyncio.run(AgentRunner(FakeLocalModel()).run(run,req))
    assert db.one('SELECT status FROM runs WHERE id=?',(run,))['status']=='cancelled'


def test_baseline_success_with_deterministic_vector_stub(monkeypatch):
    # The stub only replaces unavailable offline embedding weights; all RAG plumbing is real.
    import numpy as np
    from app.knowledge import retrieval
    monkeypatch.setattr(retrieval,'_dense_scores',lambda q,chunks:np.array([float('BM25' in c['content']) for c in chunks],dtype='float32'))
    demo=Path(__file__).resolve().parents[2]/'examples'/'qasper-mini.json'
    import_qasper_file(demo)
    for mode in ('basic','hybrid'):
        conv=uuid4().hex;run=uuid4().hex
        with db.connect() as c:
            c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                      (conv,db.settings.owner_id,'Success test',db.now()))
            c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                      (run,conv,'pending',mode,'BM25 method',db.encode(['qasper-demo-paper-001']),db.now()))
        req=ChatRequest(conversation_id=conv,question='BM25 method',paper_ids=['qasper-demo-paper-001'],mode=mode)
        asyncio.run(AgentRunner(FakeLocalModel()).run(run,req))
        r=db.one('SELECT * FROM runs WHERE id=?',(run,))
        assert r['status']=='succeeded',r['error']
        assert db.decode(r['citations'],[])
        assert db.decode(r['metrics'],{})['retrieved_evidence']
        ev=db.rows('SELECT event_type FROM run_events WHERE run_id=?',(run,))
        assert any(x['event_type']=='answer_delta' for x in ev)
        assert any(x['event_type']=='run_completed' for x in ev)

"""Day-1 end-to-end HTTP/SQLite acceptance on the included SYNTHETIC paper.

The fake LLM/dense encoder make these tests deterministic/offline; do not
interpret them as quality scores or real Ollama/GPU integration evidence.
"""
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app import db
from app.agent.workflow import AgentRunner
from app.knowledge import retrieval
from app.knowledge.ingest import import_qasper_file
from app.main import app
from test_multi_question import _fake_langgraph


DEMO = Path(__file__).resolve().parents[2] / 'examples' / 'qasper-mini.json'


class Day1Model:
    def __init__(self):
        self.prompts=[]
        self.last_stream_meta={}

    async def chat(self,messages,**kwargs):
        self.prompts.append((kwargs.get('json_mode'), messages))
        if messages[0]['content'].startswith('Judge if the cited passages'):
            return {'content':'{"sufficient":true,"missing":""}','latency_ms':1}
        return {'content':json.dumps({
            'answer':'The paper combines dense similarity and BM25 using reciprocal rank fusion (RRF).',
            'citations':[1],
        }), 'latency_ms':1}

    async def stream(self,messages,**kwargs):
        self.prompts.append((False,messages))
        self.last_stream_meta={'finish_reason':'stop'}
        yield 'The paper combines dense similarity and BM25 using reciprocal rank fusion (RRF). [1]'


def _install(monkeypatch):
    import app.main as entry
    _fake_langgraph(monkeypatch)
    model=Day1Model()
    monkeypatch.setattr(entry,'AgentRunner',lambda:AgentRunner(model))
    monkeypatch.setattr(retrieval,'_dense_scores',
        lambda q, chunks:np.array([1.0 if 'BM25' in c['content'] else 0.1 for c in chunks],dtype='float32'))
    return model


def _post_run(client,mode,question,pid,use_memory=False):
    conversation=client.post('/api/conversations',json={'title':'Day1 acceptance'}).json()['id']
    response=client.post('/api/agent/runs',json={
        'conversation_id':conversation,'question':question,'paper_ids':[pid],
        'mode':mode,'teaching':False,'use_memory':use_memory,'allow_web':False,
    })
    assert response.status_code==202, response.text
    rid=response.json()['id']
    state=client.get(f'/api/agent/runs/{rid}').json()
    assert state['status'] in ('succeeded','partial'),(mode,state.get('error'),state)
    assert state['answer'] and 'fusion' in state['answer'].lower()
    assert state['citations'],(mode,state['answer'])
    assert all(c['paper_id']==pid and c['chunk_id'].startswith(pid+':') for c in state['citations'])
    assert any(c['paragraph_refs'] for c in state['citations'])
    history=client.get(f'/api/conversations/{conversation}').json()
    assert [m['role'] for m in history['messages']]==['user','assistant']
    assert history['messages'][1]['content']==state['answer']
    sse=client.get(f'/api/agent/runs/{rid}/events')
    assert sse.status_code==200
    assert 'event: answer_delta' in sse.text
    assert 'event: stream_end' in sse.text
    details=client.get(f'/api/agent/runs/{rid}/debug').json()
    assert details['events']
    return state


def test_three_modes_and_trace_using_real_http_and_qasper_rows(monkeypatch):
    _install(monkeypatch)
    with TestClient(app) as client:
        import_qasper_file(DEMO,split='dev')
        pid='qasper-demo-paper-001'
        assert client.get('/api/papers').status_code==200
        q='Explain how the paper combines dense similarity and BM25 with RRF.'
        states={mode:_post_run(client,mode,q,pid) for mode in ('basic','hybrid','adaptive')}
        assert all(state['mode']==mode for mode,state in states.items())
        assert states['basic']['metrics']['retrieval_rounds']==1
        assert states['hybrid']['metrics']['retrieval_rounds']==1
        assert states['adaptive']['metrics']['retrieval_rounds']>=1
        passage=states['adaptive']['citations'][0]
        evidence=client.get('/api/passage/'+passage['chunk_id'],params={'paper_id':pid})
        assert evidence.status_code==200 and evidence.json()['paragraph_refs']
        # QASPER has paragraph references, not PDF page numbers.
        assert not passage['page_numbers']


def test_confirmed_memory_is_used_only_with_learner_opt_in(monkeypatch):
    model=_install(monkeypatch)
    with TestClient(app) as client:
        import_qasper_file(DEMO,split='dev')
        client.put('/api/learning/profile',json={
            'language':'en','explanation_level':'beginner','domains':['NLP'],
            'goals':['Understand retrieval'],'memory_enabled':True})
        entry=client.post('/api/learning/memories/confirm',json={
            'domain':'NLP','topic':'BM25','memory_type':'progress',
            'content':'I have studied keyword matching but not rank fusion.',
            'mastery':'studied'})
        assert entry.status_code==201
        memory_id=entry.json()['id']
        q='Explain how BM25 and dense retrieval are combined in this study.'
        used=_post_run(client,'adaptive',q,'qasper-demo-paper-001',use_memory=True)
        assert used['status']=='succeeded'
        # The answer-generation prompt, not the evidence-sufficiency judge,
        # should contain the user-confirmed educational context.
        answer_prompts=[str(messages) for _,messages in model.prompts
                        if not messages[0]['content'].startswith('Judge if the cited passages')]
        assert any('I have studied keyword matching' in x for x in answer_prompts)
        model.prompts.clear()
        _post_run(client,'adaptive',q,'qasper-demo-paper-001',use_memory=False)
        answer_prompts=[str(m) for _,m in model.prompts]
        assert all('I have studied keyword matching' not in x for x in answer_prompts)
        # A disabled profile switch overrides request.use_memory=True.
        client.put('/api/learning/profile',json={
            'language':'en','explanation_level':'beginner','domains':['NLP'],
            'goals':['Understand retrieval'],'memory_enabled':False})
        model.prompts.clear()
        _post_run(client,'adaptive',q,'qasper-demo-paper-001',use_memory=True)
        assert all('I have studied keyword matching' not in str(messages)
                   for _,messages in model.prompts)
        assert client.patch(f'/api/learning/memories/{memory_id}',json={
            'domain':'NLP','topic':'BM25','memory_type':'progress',
            'content':'I now understand reciprocal rank fusion.','mastery':'confirmed'}).status_code==200
        assert client.delete(f'/api/learning/memories/{memory_id}').status_code==200
        assert client.get('/api/learning/memories').json()==[]


def test_invalid_pdf_and_qasper_document_do_not_generate_fabricated_source(monkeypatch):
    _install(monkeypatch)
    with TestClient(app) as client:
        import_qasper_file(DEMO,split='dev')
        r=client.get('/api/papers/demo-paper-001/document')
        assert r.status_code==404
        assert client.post('/api/papers',files={
            'file':('fake.pdf',b'%PDF-not-a-valid-document','application/pdf')}).status_code==202
        assert client.post('/api/search',json={
            'query':'BM25','paper_ids':['not-a-paper'],'strategy':'keyword'}).status_code==404

import asyncio
import json
import httpx
import pytest
from fastapi.testclient import TestClient
from app.agent.model import ModelClient, ModelUnavailable
from app.main import app
from app import db


def _mock_transport(monkeypatch, handler):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr('app.agent.model.httpx.AsyncClient',
                        lambda **kwargs: original(transport=transport, **kwargs))


def test_ollama_native_thinking_off_and_stream(monkeypatch):
    sent=[]
    def responder(req):
        if req.url.path == '/api/tags':
            return httpx.Response(200, json={'models':[{'name':'qwen3:4b'}]})
        payload=json.loads(req.content)
        sent.append(payload)
        assert req.url.path == '/api/chat'
        assert payload['think'] is False
        if payload['stream']:
            lines=[{'message':{'content':'CC ','thinking':''},'done':False},
                   {'message':{'content':'means Chunk Coverage.'},'done':False},
                   {'done':True,'done_reason':'stop','eval_count':5,'prompt_eval_count':8}]
            return httpx.Response(200,text='\n'.join(json.dumps(x) for x in lines)+'\n')
        return httpx.Response(200,json={'message':{'content':'CC means Chunk Coverage.'},
                                        'done_reason':'stop','eval_count':9})
    _mock_transport(monkeypatch,responder)
    client=ModelClient(base_url='http://127.0.0.1:11434/v1',model='qwen3:4b')
    assert client.provider == 'ollama'
    assert asyncio.run(client.is_available())
    answer=asyncio.run(client.chat([{'role':'user','content':'What is CC?'}]))
    assert 'Chunk Coverage' in answer['content']
    async def collect():
        return ''.join([part async for part in client.stream([{'role':'user','content':'CC?'}])])
    assert 'Chunk Coverage' in asyncio.run(collect())
    assert len(sent)==2


def test_openai_stream_reasoning_only_explains_error(monkeypatch):
    def responder(req):
        assert req.url.path=='/v1/chat/completions'
        chunks=[{'choices':[{'delta':{'reasoning_content':'I am thinking'},'finish_reason':None}]},
                {'choices':[{'delta':{},'finish_reason':'length'}]}]
        return httpx.Response(200,text='\n'.join('data: '+json.dumps(x)+'\n' for x in chunks)+'\ndata: [DONE]\n')
    _mock_transport(monkeypatch,responder)
    client=ModelClient(base_url='http://127.0.0.1:8001/v1',model='qwen',provider='openai')
    async def collect():
        return [part async for part in client.stream([{'role':'user','content':'test'}])]
    with pytest.raises(ModelUnavailable,match='reasoning_chars=13'):
        asyncio.run(collect())


def test_debug_runs_and_log_file():
    from app.logging_setup import get_logger
    log=get_logger('test')
    log.info('diagnostic_test_marker')
    with TestClient(app) as client:
        response=client.get('/api/debug/runs')
        assert response.status_code==200
        assert isinstance(response.json(),list)
    assert (db.settings.data_dir/'logs'/'app.log').exists()
    assert 'diagnostic_test_marker' in (db.settings.data_dir/'logs'/'app.log').read_text(encoding='utf-8')


def test_failed_answer_is_diagnosable(monkeypatch):
    from pathlib import Path
    from app.agent.workflow import AgentRunner
    from app.knowledge import retrieval
    from app.knowledge.ingest import import_qasper_file
    import app.main as entry
    import numpy as np

    class EmptyModel:
        async def chat(self, messages, **kwargs):
            return {'content':'{"sufficient":true}', 'usage':{}, 'latency_ms':1}
        async def stream(self, messages, **kwargs):
            if False:
                yield ''

    monkeypatch.setattr(entry,'AgentRunner',lambda:AgentRunner(EmptyModel()))
    monkeypatch.setattr(retrieval,'_dense_scores',lambda q,chunks:np.ones(len(chunks),dtype='float32'))
    demo=Path(__file__).resolve().parents[2]/'examples'/'qasper-mini.json'
    with TestClient(app) as client:
        import_qasper_file(demo)
        conv=client.post('/api/conversations',json={'title':'Error debug test'}).json()['id']
        response=client.post('/api/agent/runs',json={'conversation_id':conv,'question':'CC?',
            'paper_ids':['qasper-demo-paper-001'],'mode':'basic'})
        run_id=response.json()['id']
        run=client.get(f'/api/agent/runs/{run_id}').json()
        assert run['status']=='failed'
        assert 'empty response' in run['error']
        debug=client.get(f'/api/agent/runs/{run_id}/debug').json()
        assert debug['run']['id']==run_id
        assert any(event['event_type']=='run_failed' for event in debug['events'])
        assert run_id in (db.settings.data_dir/'logs'/'app.log').read_text(encoding='utf-8')


def test_ollama_json_mode_for_evidence_assessment(monkeypatch):
    def responder(req):
        payload=json.loads(req.content)
        assert payload['think'] is False
        assert payload['format']=='json'
        assert payload['options']['num_predict']==220
        return httpx.Response(200,json={'message':{'content':'{"sufficient":true,"missing":""}'},
                                        'done_reason':'stop'})
    _mock_transport(monkeypatch,responder)
    client=ModelClient(base_url='http://127.0.0.1:11434/v1',model='qwen3:4b')
    result=asyncio.run(client.chat([{'role':'user','content':'Assess'}],max_tokens=220,json_mode=True))
    assert json.loads(result['content'])['sufficient'] is True


def test_openai_json_mode_is_optional():
    client=ModelClient(base_url='http://127.0.0.1:8001/v1',model='qwen',provider='openai')
    _,payload,_=client._request([{'role':'user','content':'Hi'}],0,200,False,json_mode=True)
    assert payload['response_format']=={'type':'json_object'}
    _,payload,_=client._request([{'role':'user','content':'Hi'}],0,200,True)
    assert 'response_format' not in payload

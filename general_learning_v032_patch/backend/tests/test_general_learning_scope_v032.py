"""General Learning: isolated sources, API contract and fail-closed behavior."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import db
from app.main import app
from app.schemas import ChatRequest
from app.agent import background_learning as bg
from app.agent.workflow import AgentRunner


def _make_run(question='请解释 Q-Learning'):
    cid='general-test-conv'
    db.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
               (cid,'local-user','test',db.now()))
    rid='general-test-run'
    db.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) '
               'VALUES(?,?,?,?,?,?,?)',(rid,cid,'pending','adaptive',question,'[]',db.now()))
    return rid,ChatRequest(conversation_id=cid,scope='general',paper_ids=[],question=question,
                           mode='adaptive',teaching=True,allow_web=True)


class SourceModel:
    async def chat(self,messages,**kwargs):
        return {'content':'{"answer":"Q-Learning 是一种通过奖励更新动作价值的方法。假设机器人找到出口获得奖励，它会逐渐偏向更好的动作。实际学习质量依赖奖励设计。","citations":[1]}',
                'usage':{},'latency_ms':2}


SOURCES=[{'title':'Q-learning','source_url':'https://en.wikipedia.org/wiki/Q-learning',
          'content':'Q-learning is a model-free reinforcement learning algorithm to learn a policy.',
          'source_type':'external','provider':'Wikipedia'}]


def test_scope_contract_preserves_paper_qa():
    with pytest.raises(ValidationError):
        ChatRequest(conversation_id='x',question='q',paper_ids=[])
    valid=ChatRequest(conversation_id='x',question='q',scope='general',paper_ids=[])
    assert valid.mode=='adaptive'
    with pytest.raises(ValidationError):
        ChatRequest(conversation_id='x',question='q',scope='general',paper_ids=['some-paper'])
    with pytest.raises(ValidationError):
        ChatRequest(conversation_id='x',question='q',scope='general',paper_ids=[],mode='basic')


def test_topic_and_wikipedia_source_parser():
    assert bg.search_topic('我不是很了解QLearning，详细且生动地讲解一下相关的基础知识').lower()=='qlearning'
    assert bg.search_topic('请解释 Self-Attention 的原理').lower()=='self-attention'
    parsed=bg._wiki_sources({'query':{'pages':[
        {'pageid':2,'title':'Q-learning','extract':'A basic learning method','index':1},
        {'pageid':3,'title':'Bad','extract':'','index':2}]}},'en')
    assert len(parsed)==1
    assert parsed[0]['source_url']=='https://en.wikipedia.org/wiki/Q-learning'


def test_general_does_not_touch_paper_retriever(monkeypatch):
    import app.agent.workflow as workflow
    async def fake_search(question,**kwargs):return SOURCES
    def no_paper(*args,**kwargs):raise AssertionError('Paper retrieval must not execute')
    monkeypatch.setattr(workflow,'search_background',fake_search)
    monkeypatch.setattr(workflow.retrieval,'search',no_paper)
    rid,req=_make_run()
    asyncio.run(AgentRunner(SourceModel()).run(rid,req))
    run=db.one('SELECT * FROM runs WHERE id=?',(rid,))
    assert run['status']=='succeeded'
    citations=db.decode(run['citations'],[])
    assert len(citations)==1
    assert citations[0]['source_url']=='https://en.wikipedia.org/wiki/Q-learning'
    assert citations[0]['paper_id'] is None
    assert citations[0]['source_type']=='external'
    assert 'Q-Learning' in run['answer']
    assert db.decode(run['metrics'],{})['retrieval_rounds']==0
    assert db.decode(run['metrics'],{})['knowledge_scope']=='general'


def test_empty_search_abstains_without_paper_citation(monkeypatch):
    import app.agent.workflow as workflow
    async def empty(question,**kwargs):return []
    monkeypatch.setattr(workflow,'search_background',empty)
    rid,req=_make_run()
    asyncio.run(AgentRunner(SourceModel()).run(rid,req))
    run=db.one('SELECT * FROM runs WHERE id=?',(rid,))
    assert run['status']=='partial'
    assert db.decode(run['citations'],[])==[]


def test_general_without_web_never_fetches(monkeypatch):
    import app.agent.workflow as workflow
    async def no_web(*args,**kwargs):raise AssertionError('No networking allowed')
    monkeypatch.setattr(workflow,'search_background',no_web)
    rid,req=_make_run();req.allow_web=False
    asyncio.run(AgentRunner(SourceModel()).run(rid,req))
    assert db.one('SELECT status FROM runs WHERE id=?',(rid,))['status']=='partial'


def test_bad_model_citation_does_not_fabricate_reference(monkeypatch):
    import app.agent.workflow as workflow
    async def fake_search(*args,**kwargs):return SOURCES
    class BadModel:
        async def chat(self,*args,**kwargs):
            return {'content':'{"answer":"My explanation","citations":[999]}','latency_ms':1}
    monkeypatch.setattr(workflow,'search_background',fake_search)
    rid,req=_make_run()
    asyncio.run(AgentRunner(BadModel()).run(rid,req))
    run=db.one('SELECT * FROM runs WHERE id=?',(rid,))
    assert run['status']=='partial'
    assert db.decode(run['citations'],[])==[]


def test_api_accepts_general_without_paper(monkeypatch):
    import app.main as entry
    import app.agent.workflow as workflow
    async def fake_search(*args,**kwargs):return SOURCES
    monkeypatch.setattr(workflow,'search_background',fake_search)
    monkeypatch.setattr(entry,'AgentRunner',lambda:AgentRunner(SourceModel()))
    with TestClient(app) as client:
        cid=client.post('/api/conversations',json={'title':'study'}).json()['id']
        r=client.post('/api/agent/runs',json={'conversation_id':cid,'question':'Q-Learning',
             'paper_ids':[],'scope':'general','mode':'adaptive','allow_web':True})
        assert r.status_code==202,r.text
        record=client.get('/api/agent/runs/'+r.json()['id']).json()
        assert record['status']=='succeeded'
        assert len(record['citations'])==1
        assert record['citations'][0]['source_type']=='external'
        rejected=client.post('/api/agent/runs',json={'conversation_id':cid,'question':'q',
             'paper_ids':[],'mode':'basic'})
        assert rejected.status_code==422

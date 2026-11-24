"""v0.3.17 regressions for the user-observed T03/T04/T06/T07 failures.

No live Wikipedia, Ollama, or PDF vectors are required by these tests.
The intentionally unrelated journal page must never be a definition source.
"""
import asyncio
import json
from uuid import uuid4

import httpx
import pytest

from app import db
from app.config import settings
from app.agent import background_learning as bg
from app.agent import workflow
from app.agent.memory_management import is_memory_listing_request
from app.agent.workflow import AgentRunner
from app.schemas import ChatRequest


CASES=[
    ('Explain the query, key, and value vectors in self-attention.',
     'Attention (machine learning)', 'https://en.wikipedia.org/wiki/Attention_(machine_learning)',
     'Self-attention uses queries, keys and values to compute attention weights.'),
    ('What is a B-tree index in SQL?',
     'B-tree', 'https://en.wikipedia.org/wiki/B-tree',
     'B-trees are used in databases for efficient index searches.'),
    ('Explain the difference between set union and set intersection.',
     'Union (set theory)', 'https://en.wikipedia.org/wiki/Union_(set_theory)',
     'Union combines all elements of both sets; intersection keeps only shared elements.'),
]


def test_extracted_subjects_are_short_and_support_multiple_concepts():
    assert bg.search_topic(CASES[0][0]) == 'self-attention'
    assert bg.search_topics(CASES[1][0]) == ['B-tree index','B-tree']
    assert bg.search_topics(CASES[2][0]) == [
        'set union','Union (set theory)','set intersection','Intersection (set theory)']


@pytest.mark.parametrize('question,title,url,content',CASES)
def test_source_filter_uses_named_topic_not_instruction(question,title,url,content):
    sources=[{'title':title,'source_url':url,'content':content},
             {'title':'Journal of Machine Learning Research',
              'source_url':'https://en.wikipedia.org/wiki/Journal_of_Machine_Learning_Research',
              'content':'A journal for research publications.'}]
    diagnostics={}
    result=bg.verified_background_sources(sources,question,diagnostics=diagnostics)
    assert [r['title'] for r in result] == [title]
    assert diagnostics['topics'] == bg.search_topics(question)
    assert any(x['status']=='title_mismatch' for x in diagnostics['filter_reasons'])


def test_wikipedia_query_rewriting_recovers_technical_subjects(monkeypatch):
    requested=[]
    def handler(req):
        if req.url.host=='api.openalex.org':
            raise AssertionError('Direct Wikipedia title evidence is enough')
        params=req.url.params
        requested.append(params.get('gsrsearch') or 'suggestion')
        topic=params.get('gsrsearch')
        if topic in ('Attention (machine learning)','B-tree','Union (set theory)',
                     'Intersection (set theory)'):
            title=topic
            result={'query':{'pages':[{'pageid':100,'index':1,'title':title,
                                      'extract':f'{title} describes this technical concept.'}]}}
        elif params.get('list')=='search':
            result={'query':{'searchinfo':{}}}
        else:
            result={'query':{'pages':[]}}
        return httpx.Response(200,json=result)
    client=httpx.AsyncClient
    monkeypatch.setattr(bg.httpx,'AsyncClient',lambda *a,**kw:client(
        *a,**{**kw,'transport':httpx.MockTransport(handler)}))
    for question,_,_,_ in CASES:
        diagnostics={}
        results=asyncio.run(bg.search_background(question, diagnostics=diagnostics))
        assert bg.verified_background_sources(results,question)
        assert diagnostics['query_candidates']==bg.search_topics(question)
    assert 'Attention (machine learning)' in requested
    assert 'B-tree' in requested
    assert 'Union (set theory)' in requested
    assert 'Intersection (set theory)' in requested


def test_similar_but_different_field_or_journal_is_still_rejected():
    sources=[{'title':'Quantum machine learning',
              'source_url':'https://en.wikipedia.org/wiki/Quantum_machine_learning',
              'content':'Quantum machine learning uses quantum algorithms.'},
             {'title':'Journal of Machine Learning Research',
              'source_url':'https://en.wikipedia.org/wiki/Journal_of_Machine_Learning_Research',
              'content':'A scientific journal.'}]
    assert bg.verified_background_sources(sources,'What is machine learning?')==[]


def make_request(question, *, scope='general', paper_ids=None):
    cid=uuid4().hex; rid=uuid4().hex
    db.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
               (cid,settings.owner_id,'study',db.now()))
    req=ChatRequest(conversation_id=cid,question=question,scope=scope,paper_ids=paper_ids or [],
                    mode='adaptive',allow_web=True,teaching=True,use_memory=True)
    db.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) '
               'VALUES(?,?,?,?,?,?,?)',
               (rid,cid,'pending','adaptive',question,db.encode(req.paper_ids),db.now()))
    return rid,req


class ForbiddenModel:
    async def chat(self,*a,**kw):
        raise AssertionError('No model call is needed to list saved notes')


def save_note(topic):
    db.execute('INSERT INTO memories(id,owner_id,domain,topic,memory_type,content,mastery,created_at,updated_at) '
               'VALUES(?,?,?,?,?,?,?,?,?)',
               (uuid4().hex,settings.owner_id,'ML',topic,'progress','I have studied '+topic,
                'studied',db.now(),db.now()))


@pytest.mark.parametrize('question,scope',[
    ('List my saved learning notes.','paper'),
    ('Show all my memories.','general'),
    ('列出我的学习记录','paper'),
])
def test_list_memories_uses_local_db_and_never_searches(monkeypatch, question,scope):
    for topic in ('Chunk Coverage','Self-Attention','Clustering','Code Coverage'):
        save_note(topic)
    def no_retrieval(*a,**kw):
        raise AssertionError('Listing notes must not retrieve any papers')
    async def no_background(*a,**kw):
        raise AssertionError('Listing notes must not perform any web search')
    monkeypatch.setattr(workflow.retrieval,'search',no_retrieval)
    monkeypatch.setattr(workflow,'search_background',no_background)
    rid,req=make_request(question,scope=scope)
    asyncio.run(AgentRunner(ForbiddenModel()).run(rid,req))
    row=db.one('SELECT * FROM runs WHERE id=?',(rid,))
    metrics=db.decode(row['metrics'])
    assert row['status']=='succeeded'
    assert metrics['memory_management_action']=='list'
    assert metrics['memory_records_total']==4
    assert metrics['memory_records_returned']==4
    assert metrics['llm_calls']==0
    assert metrics['retrieval_rounds']==0
    assert metrics['learning_memory_applied'] is False
    assert db.decode(row['citations'])==[]
    assert all(topic in row['answer'] for topic in
               ('Chunk Coverage','Self-Attention','Clustering','Code Coverage'))
    assert ('You have 4' if metrics['response_language']=='en' else '4 条') in row['answer']


def test_recap_intent_does_not_capture_personalized_question():
    assert is_memory_listing_request('List my saved learning notes.')
    assert not is_memory_listing_request('Based on my saved learning note, explain CC.')
    assert not is_memory_listing_request('What is machine learning?')
    assert not is_memory_listing_request('Explain the difference between two notes in the paper.')


class ExternalModel:
    async def chat(self,messages,**kw):
        return {'content':json.dumps({'answer':'A B-tree is a balanced search tree used for indexing in databases.',
                                       'citations':[1]}),'latency_ms':1}


def test_general_learning_uses_verified_btree_source_and_records_search_metadata(monkeypatch):
    async def source_search(*args,**kwargs):
        diag=kwargs.get('diagnostics')
        if diag is not None: diag['queries_attempted']=['B-tree index','B-tree']
        return [{'title':'B-tree','source_url':'https://en.wikipedia.org/wiki/B-tree',
                 'content':'B-tree is a balanced tree used for database indexing.',
                 'source_type':'external','provider':'Wikipedia'}]
    monkeypatch.setattr(workflow,'search_background',source_search)
    rid,req=make_request('What is a B-tree index in SQL?')
    asyncio.run(AgentRunner(ExternalModel()).run(rid,req))
    row=db.one('SELECT * FROM runs WHERE id=?',(rid,))
    metrics=db.decode(row['metrics'])
    assert row['status']=='succeeded',metrics
    assert metrics['external_search_topics']==['B-tree index','B-tree']
    assert metrics['external_sources_found']==1
    assert metrics['llm_calls']==1
    assert db.decode(row['citations'])[0]['source_url']=='https://en.wikipedia.org/wiki/B-tree'

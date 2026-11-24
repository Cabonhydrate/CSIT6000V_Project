"""Release-candidate guards: generalizable source quality and English failures.

All tests are offline. Exact Wikipedia page titles alone do not make an
encyclopedia disambiguation page sufficient technical evidence.
"""
import asyncio
from uuid import uuid4

import httpx

from app import db
from app.agent import background_learning as bg
from app.agent import workflow
from app.agent.workflow import AgentRunner
from app.schemas import ChatRequest

QKV='Explain the query, key, and value vectors in self-attention.'


def source(title,content,**extra):
    return {'title':title,'content':content,
            'source_url':'https://en.wikipedia.org/wiki/'+title.replace(' ','_'),**extra}


def test_disambiguation_rejected_even_if_title_is_exact():
    navigation=source('Self-attention','Self-attention can mean: Attention (machine learning), '
                      'or self-attention in psychology.')
    diag={}
    assert bg.verified_background_sources([navigation],QKV,diagnostics=diag)==[]
    assert diag['filter_reasons'][0]['status']=='disambiguation'
    assert bg.verified_background_sources([source('Self-attention','A topic index',
                                                      pageprops={'disambiguation':''})],QKV)==[]
    assert bg.verified_background_sources([source('Self-attention (disambiguation)',
                                                      'A list of topics')],QKV)==[]


def test_required_components_are_question_driven_not_hardcoded_definitions():
    assert bg.requested_aspects(QKV)==('query','key','value')
    assert bg.requested_aspects('Explain the red, green, and blue channels in an image.')==(
        'red','green','blue')
    assert bg.requested_aspects('What is a B-tree index in SQL?')==()
    assert bg.requested_aspects('Compare set union and intersection.')==()
    assert bg._covers_aspects('Queries, keys, and values are projections.',('query','key','value'))


def test_a_relevant_title_without_requested_details_is_not_sufficient_evidence():
    entry=source('Attention (machine learning)',
                 'Attention assigns importance to tokens in a sequence.')
    diag={}
    assert bg.verified_background_sources([entry],QKV,diagnostics=diag)==[]
    assert diag['aspect_coverage']=='insufficient_source_detail'
    entry['content']='The query is matched against the keys and combines associated values.'
    assert bg.verified_background_sources([entry],QKV)


def test_network_wikipedia_disambiguation_falls_through_to_real_article(monkeypatch):
    original=httpx.AsyncClient
    requests=[]
    def respond(req):
        p=req.url.params
        q=p.get('gsrsearch')
        requests.append(q)
        if q=='self-attention':
            pages=[{'pageid':1,'title':'Self-attention','index':1,
                    'extract':'Self-attention can mean: Attention (machine learning), or cognition.',
                    'pageprops':{'disambiguation':''}}]
        elif q=='Attention (machine learning)':
            pages=[{'pageid':2,'title':'Attention (machine learning)','index':1,
                    'extract':'In self-attention, the queries are matched against keys to retrieve values.'}]
        else:
            pages=[]
        if p.get('list')=='search':
            return httpx.Response(200,json={'query':{'searchinfo':{}}})
        return httpx.Response(200,json={'query':{'pages':pages}})
    monkeypatch.setattr(bg.httpx,'AsyncClient',lambda *a,**kw:original(
        *a,**{**kw,'transport':httpx.MockTransport(respond)}))
    found=asyncio.run(bg.search_background(QKV))
    assert [x['title'] for x in found]==['Attention (machine learning)']
    assert bg.verified_background_sources(found,QKV)
    assert 'self-attention' in requests and 'Attention (machine learning)' in requests


def test_long_article_expanded_only_for_missing_named_components(monkeypatch):
    original=httpx.AsyncClient
    requested=[]
    def respond(req):
        p=req.url.params
        requested.append(dict(p))
        if p.get('pageids'):
            pages=[{'pageid':20, 'title':'Attention (machine learning)',
                'extract':('An introduction to attention.\n\n'
                           'The query is compared with keys to form attention weights. '
                           'These weights combine the corresponding values.\n\nOther details.')}]
        elif p.get('gsrsearch')=='Attention (machine learning)':
            pages=[{'pageid':20,'title':'Attention (machine learning)','index':1,
                    'extract':'Attention assigns different importance weights in a neural network.'}]
        else:
            pages=[]
        if p.get('list')=='search':
            return httpx.Response(200,json={'query':{'searchinfo':{}}})
        return httpx.Response(200,json={'query':{'pages':pages}})
    monkeypatch.setattr(bg.httpx,'AsyncClient',lambda *a,**kw:original(
        *a,**{**kw,'transport':httpx.MockTransport(respond)}))
    found=asyncio.run(bg.search_background(QKV))
    assert bg.verified_background_sources(found,QKV)
    assert any(p.get('pageids')=='20' for p in requested)
    assert 'query is compared with keys' in found[0]['content']
    assert all('_pageid' not in x for x in found)


def test_detail_expansion_does_not_invent_missing_information(monkeypatch):
    original=httpx.AsyncClient
    def respond(req):
        p=req.url.params
        if p.get('list')=='search':
            return httpx.Response(200,json={'query':{'searchinfo':{}}})
        pages=[{'pageid':20,'title':'Attention (machine learning)','index':1,
                'extract':'Attention assigns importance weights to tokens.'}]
        return httpx.Response(200,json={'query':{'pages':pages}})
    monkeypatch.setattr(bg.httpx,'AsyncClient',lambda *a,**kw:original(
        *a,**{**kw,'transport':httpx.MockTransport(respond)}))
    found=asyncio.run(bg.search_background(QKV))
    assert bg.verified_background_sources(found,QKV)==[]


class BadTeachingModel:
    async def chat(self,messages,**kwargs):
        return {'content':'{}','latency_ms':1}


def test_english_teaching_failure_is_not_a_chinese_fallback(monkeypatch):
    from test_english_preferences_and_memory import create_run
    from test_day1_learning_memory import fake_langgraph
    fake_langgraph(monkeypatch)
    rid, req=create_run(monkeypatch,teaching=True,
        question='Please explain the main idea of Chunk Coverage for beginners.')
    asyncio.run(AgentRunner(BadTeachingModel()).run(rid,req))
    rec=db.one('SELECT * FROM runs WHERE id=?',(rid,))
    assert rec['status']=='partial'
    assert 'Paper evidence was retrieved' in rec['answer']
    assert not any('\u4e00'<=ch<='\u9fff' for ch in rec['answer'])


class SequencedExternalModel:
    def __init__(self, texts):
        self.texts=iter(texts)
    async def chat(self,messages,**kwargs):
        return {'content':next(self.texts),'latency_ms':1}


def _general_run(monkeypatch, model, question=QKV):
    import json
    from test_external_memory_routing_v0317 import make_request
    from app.config import settings
    db.execute('UPDATE profile SET language=? WHERE owner_id=?',('en',settings.owner_id))
    async def search(*args,**kwargs):
        return [source('Attention (machine learning)',
                       'The query is compared with keys to compute weights over values.')]
    monkeypatch.setattr(workflow,'search_background',search)
    rid,req=make_request(question)
    asyncio.run(AgentRunner(model).run(rid,req))
    rec=db.one('SELECT * FROM runs WHERE id=?',(rid,))
    return rec,db.decode(rec['metrics'])


def test_named_components_missing_from_generated_answer_trigger_retry(monkeypatch):
    incomplete='{"answer":"Attention weights indicate relevant tokens.","citations":[1]}'
    complete='{"answer":"A query matches keys and then retrieves a weighted sum of values.","citations":[1]}'
    rec,metrics=_general_run(monkeypatch,SequencedExternalModel([incomplete,complete]))
    assert rec['status']=='succeeded',metrics
    assert metrics['general_validation_attempts'][0]['reason']=='model_answer_missing_named_components'
    assert metrics['general_validation_attempts'][1]['passed'] is True
    assert 'query matches keys' in rec['answer']


def test_wrong_language_uses_one_bounded_retry(monkeypatch):
    chinese='{"answer":"查询向量是目标，键向量和数值向量被检索。","citations":[1]}'
    english='{"answer":"The query matches keys to weight values.","citations":[1]}'
    rec,metrics=_general_run(monkeypatch,SequencedExternalModel([chinese,english]))
    assert rec['status']=='succeeded',metrics
    assert metrics['general_validation_attempts'][0]['reason']=='wrong_response_language'
    assert metrics['general_validation_attempts'][1]['passed'] is True
    assert 'query matches keys' in rec['answer']


def test_unsupported_generated_components_do_not_get_success(monkeypatch):
    incomplete='{"answer":"This is a form of machine learning attention.","citations":[1]}'
    rec,metrics=_general_run(monkeypatch,SequencedExternalModel([incomplete,incomplete]))
    assert rec['status']=='partial'
    assert metrics['general_validation_issue']=='model_answer_missing_named_components'
    assert db.decode(rec['citations'])==[]
    assert 'could not be verified' in rec['answer']

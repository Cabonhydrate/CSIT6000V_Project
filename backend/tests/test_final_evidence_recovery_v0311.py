"""Source-grounded recovery regressions for W02 and G01.

Test with a fictional paper metric so the application never needs a CC Gold
answer. Network calls and LLM output are mocked; real Ollama is tested by user.
"""
import asyncio
import httpx

from app import db
from app.agent import background_learning as bg
from app.agent import workflow
from app.agent.workflow import AgentRunner
from app.skills.mixed_comparison import sourced_shared_coverage
from test_adaptive_web_comparison_v038 import (
    setup, make_run, EXTERNAL, REPLY,
)
from test_final_output_recovery_v040 import ComparisonReplies


PAPER = ('In this study, Record Coverage (RC) measures the fraction of '
         'unique corpus records retrieved at least once across a test suite.')
EXT = ('Code coverage is a percentage measure of the degree to which '
       'the source code of a program is executed when a test suite runs.')


def test_specialized_article_is_not_general_concept_definition():
    assert bg.source_title_relevance({'title':'Quantum machine learning'},
                                     'Maching Learning') < .76
    assert bg.source_title_relevance({'title':'Machine learning algorithms'},
                                     'Maching Learning') < .76
    assert bg.source_title_relevance({'title':'Journal of Machine Learning Research'},
                                     'Maching Learning') < .76
    assert bg.source_title_relevance({'title':'Machine learning'},
                                     'Maching Learning') >= .76


def test_explicit_alias_can_be_used_without_accepting_related_subfields():
    alias={'title':'Artificial neural network',
           'content':'Artificial neural network is also known as neural network.'}
    assert bg.source_title_relevance(alias,'neural network') >= .76
    assert bg.source_title_relevance({**alias,'content':'Used in neural network studies.'},
                                     'neural network') < .76


def test_wikipedia_spelling_suggestion_fetches_exact_concept(monkeypatch):
    calls=[]
    def handler(request):
        if request.url.host=='api.openalex.org':
            raise AssertionError('Corrected Wikipedia evidence is enough')
        params=request.url.params
        calls.append(str(params))
        if params.get('list')=='search':
            return httpx.Response(200,json={
                'query':{'searchinfo':{'suggestion':'Machine learning'}}})
        if params.get('gsrsearch')=='Machine learning':
            pages=[{'pageid':3,'index':1,'title':'Machine learning',
                    'extract':'Machine learning is the study of algorithms that learn from data.'}]
        else:
            pages=[{'pageid':1,'index':1,'title':'Quantum machine learning',
                    'extract':'Quantum machine learning studies quantum algorithms.'},
                   {'pageid':2,'index':2,'title':'Journal of Machine Learning Research',
                    'extract':'A journal about machine learning.'}]
        return httpx.Response(200,json={'query':{'pages':pages}})
    real=httpx.AsyncClient
    monkeypatch.setattr(bg.httpx,'AsyncClient',lambda *a,**kwargs: real(
        *a,**{**kwargs,'transport':httpx.MockTransport(handler)}))
    sources=asyncio.run(bg.search_background('Maching Learning是什么？'))
    assert [s['title'] for s in sources]==['Machine learning']
    assert len(calls)==3


def test_no_topic_exact_match_fails_closed_even_when_subfield_exists(monkeypatch):
    def handler(request):
        if request.url.host=='api.openalex.org':
            return httpx.Response(200,json={'results':[]})
        if request.url.params.get('list')=='search':
            return httpx.Response(200,json={'query':{'searchinfo':{}}})
        return httpx.Response(200,json={'query':{'pages':[
            {'pageid':4,'index':1,'title':'Quantum machine learning',
             'extract':'This describes quantum algorithms for ML.'}]}})
    real=httpx.AsyncClient
    monkeypatch.setattr(bg.httpx,'AsyncClient',lambda *a,**kwargs: real(
        *a,**{**kwargs,'transport':httpx.MockTransport(handler)}))
    assert asyncio.run(bg.search_background('Maching Learning是什么？'))==[]


def test_source_derived_similarity_requires_both_metrics_to_define_test_fractions():
    shared=sourced_shared_coverage(PAPER,EXT,'Record Coverage','Code coverage')
    assert '覆盖' in shared and '文档检索等同于程序执行' in shared
    assert sourced_shared_coverage(PAPER,'The program uses source code.',
                                   'Record Coverage','Code coverage')==''
    assert sourced_shared_coverage('A neural technique learns embeddings.',EXT,
                                   'Neural Technique','Code coverage')==''


def test_persistent_model_comparison_error_is_safely_replaced_from_evidence(monkeypatch):
    explicit_percent={**EXTERNAL,'content':EXT}
    setup(monkeypatch,found=(explicit_percent,))
    _,rid,req=make_run()
    bad={**REPLY,'similarity':(
        '两者均通过测试套件运行时的代码执行情况来评估程序执行的覆盖程度。')}
    model=ComparisonReplies([bad,bad])
    asyncio.run(AgentRunner(model).run(rid,req))
    row=db.one('SELECT * FROM runs WHERE id=?',(rid,))
    metrics=db.decode(row['metrics'])
    citations=db.decode(row['citations'])
    assert row['status']=='succeeded'
    assert [v['passed'] for v in metrics['mixed_comparison_validation']]==[False,False,True]
    assert metrics['mixed_comparison_validation'][2]['attempt']=='source_fallback'
    assert metrics['mixed_comparison_source_derived_similarity'] is True
    assert [c['source_type'] for c in citations]==['paper','external']
    assert '两者都利用覆盖比例' in row['answer']
    assert '两者均通过测试套件运行时的代码执行' not in row['answer']


def test_invalid_paper_claim_cannot_be_laundered_by_source_similarity(monkeypatch):
    setup(monkeypatch,found=({**EXTERNAL,'content':EXT},))
    _,rid,req=make_run()
    bad={**REPLY,
         'paper':'NCI 衡量测试套件未曾检索到的记录数量。',
         'similarity':'两者均通过测试套件运行时的代码执行情况来评估程序执行的覆盖程度。'}
    asyncio.run(AgentRunner(ComparisonReplies([bad,bad])).run(rid,req))
    row=db.one('SELECT * FROM runs WHERE id=?',(rid,))
    assert row['status']=='partial'
    assert db.decode(row['citations'])==[]
    assert db.decode(row['metrics']).get('mixed_comparison_source_derived_similarity') is None

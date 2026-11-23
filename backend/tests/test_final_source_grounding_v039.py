"""v0.3.9 last-mile source consistency across Paper QA and General Learning.

The workflow is exercised with mock model/network services. Test fixtures use a
fictional paper; real CC is only used as an *additional* regression fixture.
"""
import asyncio
import json

import httpx

from app import db
from app.agent import background_learning as bg
from app.agent import workflow
from app.agent.workflow import AgentRunner
from app.skills.mixed_comparison import (
    source_acronym_expansion, validate_comparison_grounding,
    external_source_matches,
)
from test_adaptive_web_comparison_v038 import (
    make_run, setup, EXTERNAL, REPLY, Model,
)
from test_general_learning_scope_v032 import _make_run, SourceModel


PAPER_TEXT = ('In this paper, we introduce Novel Coverage Index (NCI). '
              'NCI measures the fraction of unique corpus records retrieved '
              'at least once across a test suite.')
BAD_NAME = {**REPLY, 'paper': 'NCI（Novel Corpus Index）衡量测试检索过的记录占比。'}
BAD_SIMILARITY = {**REPLY,
                  'similarity': '两者都关注程序源代码在测试中被执行到的覆盖区域。'}


def test_title_ranking_prefers_corrected_concept_over_journal_name():
    journal = {'title': 'Journal of Machine Learning Research',
               'content': 'A peer-reviewed journal about machine learning.'}
    concept = {'title': 'Machine learning',
               'content': 'Machine learning is a subfield of artificial intelligence.'}
    assert bg.source_title_relevance(concept, 'Maching Learning') >= .76
    assert bg.source_title_relevance(journal, 'Maching Learning') < .76
    assert bg.relevant_sources([journal, concept], 'Maching Learning') == [concept]


def test_title_relevance_supports_disambiguation_without_keyword_laundering():
    assert bg.source_title_relevance({'title': 'Transformer (deep learning architecture)'},
                                     'Transformer') >= .76
    assert bg.source_title_relevance({'title': 'Artificial neural network',
                                      'content': 'Artificial neural network, also known as neural network, models neurons.'},
                                     'neural network') >= .76
    assert bg.source_title_relevance({'title': 'Journal of Machine Learning Research'},
                                     'Machine learning') < .76
    assert bg.source_title_relevance({'title': '时序差分学习'}, '时序差分学习') == 1


def test_wikipedia_reranking_does_not_use_search_order_as_ground_truth(monkeypatch):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if 'wikipedia.org' in request.url.host:
            return httpx.Response(200, json={'query': {'pages': [
                {'pageid': 101, 'index': 1, 'title': 'Journal of Machine Learning Research',
                 'extract': 'The Journal of Machine Learning Research is a scientific journal.'},
                {'pageid': 102, 'index': 2, 'title': 'Machine learning',
                 'extract': 'Machine learning is a field of AI that learns from data.'},
            ]}})
        raise AssertionError('A relevant Wikipedia definition was found; no OpenAlex fallback')
    real = httpx.AsyncClient
    monkeypatch.setattr(bg.httpx, 'AsyncClient', lambda *a, **kw: real(
        *a, **{**kw, 'transport': httpx.MockTransport(handler)}))
    result = asyncio.run(bg.search_background('Maching Learning是什么？'))
    assert [x['title'] for x in result] == ['Machine learning']
    assert len(calls) == 1


def test_acronym_expansion_from_actual_paper_words_not_guessed():
    assert source_acronym_expansion('NCI', [{'content': PAPER_TEXT}]) == 'Novel Coverage Index'
    assert source_acronym_expansion('CC', [{'content':
        'We introduce Chunk Coverage (CC) as a criterion. CC measures retrieval.'}]) == 'Chunk Coverage'
    assert source_acronym_expansion('NCI', [{'content': 'No long-form name appears.'}]) == ''


def test_wrong_abbreviation_expansion_rejected_even_fullwidth_parentheses():
    msg = validate_comparison_grounding(
        BAD_NAME, paper_term='NCI', paper_sources=[{'content': PAPER_TEXT}],
        paper_excerpt='NCI measures the fraction of unique corpus records retrieved.',
        external_excerpt=EXTERNAL['content'])
    assert 'Novel Coverage Index' in msg
    assert 'Novel Corpus Index' in msg
    good = {**REPLY, 'paper': 'NCI（Novel Coverage Index）衡量测试检索过的记录比例。'}
    assert not validate_comparison_grounding(
        good, paper_term='NCI', paper_sources=[{'content': PAPER_TEXT}],
        paper_excerpt=PAPER_TEXT, external_excerpt=EXTERNAL['content'])


def test_comparison_similarity_cannot_attribue_code_executions_to_retrieval():
    error = validate_comparison_grounding(
        BAD_SIMILARITY, paper_term='NCI', paper_sources=[{'content': PAPER_TEXT}],
        paper_excerpt=PAPER_TEXT, external_excerpt=EXTERNAL['content'])
    assert '相似点' in error
    assert not validate_comparison_grounding(
        REPLY, paper_term='NCI', paper_sources=[{'content': PAPER_TEXT}],
        paper_excerpt=PAPER_TEXT, external_excerpt=EXTERNAL['content'])


def test_external_comparison_must_be_about_named_topic_not_page_mention():
    assert external_source_matches(EXTERNAL, 'code coverage')
    journal = {'title': 'Journal of Machine Learning Research',
               'content': 'Machine learning is a topic covered by this journal.',
               'source_url': 'https://en.wikipedia.org/wiki/Journal_of_Machine_Learning_Research'}
    assert not external_source_matches(journal, 'machine learning')


def test_general_learning_refuses_offtopic_journal_citation(monkeypatch):
    bad_source = {'title': 'Journal of Machine Learning Research',
                  'content': 'A peer-reviewed journal on machine learning.',
                  'source_url': 'https://en.wikipedia.org/wiki/Journal_of_Machine_Learning_Research',
                  'source_type': 'external', 'provider': 'Wikipedia'}
    async def fake_search(*a, **kw):
        return [bad_source]
    monkeypatch.setattr(workflow, 'search_background', fake_search)
    run, req = _make_run('Maching Learning是什么？')
    asyncio.run(AgentRunner(SourceModel()).run(run, req))
    row = db.one('SELECT * FROM runs WHERE id=?', (run,))
    assert row['status'] == 'partial'
    assert db.decode(row['citations']) == []
    metrics = db.decode(row['metrics'])
    assert metrics['external_sources_found_raw'] == 1
    assert metrics['external_sources_found'] == 0


def test_general_learning_uses_direct_topic_url_and_citation(monkeypatch):
    good_source = {'title': 'Machine learning',
                   'content': 'Machine learning is a field of AI concerned with learning from data.',
                   'source_url': 'https://en.wikipedia.org/wiki/Machine_learning',
                   'source_type': 'external', 'provider': 'Wikipedia'}
    class ConceptModel:
        async def chat(self, *_args, **_kwargs):
            return {'content': json.dumps({'answer': '机器学习从数据中学习模式。',
                                           'citations': [1]}, ensure_ascii=False),
                    'latency_ms': 1}
    async def fake_search(*a, **kw):
        return [good_source]
    monkeypatch.setattr(workflow, 'search_background', fake_search)
    run, req = _make_run('Maching Learning是什么？')
    asyncio.run(AgentRunner(ConceptModel()).run(run, req))
    row = db.one('SELECT * FROM runs WHERE id=?', (run,))
    assert row['status'] == 'succeeded'
    assert db.decode(row['citations'])[0]['source_url'].endswith('/Machine_learning')


class ScriptedComparisonModel(Model):
    def __init__(self, answers):
        super().__init__()
        self.answers = answers
        self.answer_calls = 0
    async def chat(self, messages, **kwargs):
        if 'Judge if the cited passages' in messages[0]['content']:
            return {'content': '{"sufficient":true,"missing":""}', 'latency_ms': 1}
        answer = self.answers[min(self.answer_calls, len(self.answers) - 1)]
        self.answer_calls += 1
        return {'content': json.dumps(answer, ensure_ascii=False), 'latency_ms': 1}


def test_mixed_comparison_wrong_long_name_repairs_before_success(monkeypatch):
    setup(monkeypatch)
    _, run, req = make_run()
    model = ScriptedComparisonModel([BAD_NAME, REPLY])
    asyncio.run(AgentRunner(model).run(run, req))
    row = db.one('SELECT * FROM runs WHERE id=?', (run,))
    assert row['status'] == 'succeeded'
    validations = db.decode(row['metrics'])['mixed_comparison_validation']
    assert [x['passed'] for x in validations] == [False, True]
    assert 'Novel Coverage Index' in validations[0]['reason']
    assert [x['source_type'] for x in db.decode(row['citations'])] == ['paper', 'external']


def test_mixed_comparison_does_not_launder_bad_similarity(monkeypatch):
    setup(monkeypatch)
    _, run, req = make_run()
    model = ScriptedComparisonModel([BAD_SIMILARITY, BAD_SIMILARITY])
    asyncio.run(AgentRunner(model).run(run, req))
    row = db.one('SELECT * FROM runs WHERE id=?', (run,))
    assert row['status'] == 'partial'
    assert db.decode(row['citations']) == []
    assert '相似点' in db.decode(row['metrics'])['mixed_comparison_issue']

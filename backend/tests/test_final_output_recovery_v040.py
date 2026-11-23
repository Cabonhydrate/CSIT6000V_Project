"""Last-mile regressions from real W02/G01 traces, with independent fake sources.

No network or actual model is needed; these tests exercise the production
comparison grounding check and the General Learning run lifecycle.
"""
import asyncio
import json

from app import db
from app.agent import workflow
from app.agent.workflow import AgentRunner
from app.skills.mixed_comparison import validate_comparison_grounding
from test_general_learning_scope_v032 import _make_run

PAPER = ('In this paper, we introduce Novel Coverage Index (NCI). '
         'NCI measures the fraction of unique corpus records retrieved at least once '
         'across a test suite.')
EXTERNAL = ('Code coverage measures the source code of a program executed '
            'when a particular test suite is run.')
BASE = {'paper': 'NCI（Novel Coverage Index）衡量测试检索过的语料记录比例。',
        'external': 'Code coverage 衡量测试执行过的源代码。',
        'difference': '不同的是前者衡量检索记录，后者衡量执行的程序代码。'}


def issue(similarity):
    return validate_comparison_grounding(
        {**BASE, 'similarity': similarity},
        paper_term='NCI', paper_sources=[{'content': PAPER}],
        paper_excerpt=PAPER, external_excerpt=EXTERNAL)


def test_real_w02_claim_about_code_execution_is_rejected():
    assert '相似点' in issue(
        '两者均通过测试套件运行时的代码执行情况来评估程序执行的覆盖程度，'
        '且都用于指导测试优化（如优先选择能扩展未测试区域的查询）。')


def test_other_shared_execution_phrasings_are_rejected():
    for claim in ('双方都依赖程序代码在测试中被执行的覆盖范围。',
                  '二者都统计被执行的代码行。',
                  '两者共同衡量代码运行的比例。'):
        assert issue(claim), claim


def test_real_shared_coverage_and_comparative_clauses_are_preserved():
    for claim in ('两者都衡量测试覆盖目标空间的程度。',
                  '二者都能帮助定位尚未触及的区域。',
                  '两者都是覆盖率指标，前者关注文档检索而后者关注代码执行。',
                  '两者都反映测试覆盖范围，不过代码覆盖率衡量程序代码执行。'):
        assert issue(claim) == '', claim


def test_negated_false_commonality_is_not_treated_as_assertion():
    assert issue('不能说两者都以源代码执行情况衡量覆盖程度。') == ''


GOOD = {'title': 'Machine learning', 'source_url': 'https://en.wikipedia.org/wiki/Machine_learning',
        'content': ('Machine learning is a branch of artificial intelligence that develops '
                    'methods allowing systems to learn from data.'),
        'source_type': 'external', 'provider': 'Wikipedia'}


class Responses:
    def __init__(self, *messages):
        self.messages = list(messages)
        self.calls = 0

    async def chat(self, _msgs, **_kwargs):
        index = min(self.calls, len(self.messages) - 1)
        self.calls += 1
        return {'content': self.messages[index], 'latency_ms': 1}


def _run(monkeypatch, output):
    async def source_search(*_args, **_kwargs):
        return [GOOD]
    monkeypatch.setattr(workflow, 'search_background', source_search)
    rid, req = _make_run('Maching Learning是什么？')
    model = Responses(*output)
    asyncio.run(AgentRunner(model).run(rid, req))
    row = db.one('SELECT * FROM runs WHERE id=?', (rid,))
    return row, db.decode(row['metrics']), db.decode(row['citations']), model


VALID = json.dumps({
    'answer': '机器学习是人工智能的一个领域，通过从数据中学习规律完成任务。',
    'citations': [1]}, ensure_ascii=False)


def test_general_bad_json_retries_once_then_returns_verified_source(monkeypatch):
    row, metrics, cites, model = _run(monkeypatch, ('这不是JSON结构', VALID))
    assert row['status'] == 'succeeded'
    assert model.calls == 2
    assert [v['passed'] for v in metrics['general_validation_attempts']] == [False, True]
    assert metrics['general_validation_attempts'][0]['reason'] == 'model_answer_missing_or_invalid_json'
    assert cites[0]['source_url'] == GOOD['source_url']
    assert cites[0]['paper_id'] is None


def test_general_missing_citation_retries_but_never_fabricates(monkeypatch):
    no_citation = json.dumps({'answer': '机器学习是从数据学习的方法。', 'citations': []}, ensure_ascii=False)
    row, metrics, cites, model = _run(monkeypatch, (no_citation, VALID))
    assert row['status'] == 'succeeded'
    assert [v['passed'] for v in metrics['general_validation_attempts']] == [False, True]
    assert metrics['general_validation_attempts'][0]['reason'] == 'model_citations_missing_or_invalid'
    assert cites[0]['source_type'] == 'external'


def test_general_internal_reasoning_is_rejected_then_repaired(monkeypatch):
    internal = json.dumps({'answer':'首先，用户的问题是 Machine Learning 是什么。我需要根据来源回答。',
                           'citations':[1]}, ensure_ascii=False)
    row, metrics, cites, model = _run(monkeypatch, (internal, VALID))
    assert row['status'] == 'succeeded'
    assert metrics['general_validation_attempts'][0]['reason'] == 'model_answer_contains_internal_analysis'
    assert '首先' not in row['answer']


def test_general_two_invalid_answers_remain_partial_with_no_citations(monkeypatch):
    row, metrics, cites, model = _run(monkeypatch, ('not-json', '{"answer":"abc","citations":[999]}'))
    assert row['status'] == 'partial'
    assert model.calls == 2
    assert metrics['general_validation_issue'] == 'model_citations_missing_or_invalid'
    assert cites == []
    assert '外部资料已检索到' in row['answer']


def test_general_retry_cannot_invent_pdf_citations(monkeypatch):
    no_citation = json.dumps({'answer':'解释', 'citations':[99]}, ensure_ascii=False)
    row, metrics, cites, _ = _run(monkeypatch, (no_citation, no_citation))
    assert row['status'] == 'partial'
    assert cites == []


def test_no_external_source_does_not_call_llm(monkeypatch):
    async def no_source(*_args, **_kwargs):
        return []
    monkeypatch.setattr(workflow, 'search_background', no_source)
    rid, req = _make_run('Maching Learning是什么？')
    model = Responses(VALID)
    asyncio.run(AgentRunner(model).run(rid, req))
    row = db.one('SELECT * FROM runs WHERE id=?',(rid,))
    assert row['status'] == 'partial'
    assert model.calls == 0
    assert db.decode(row['citations']) == []

from test_adaptive_web_comparison_v038 import make_run as make_comparison_run, setup as setup_comparison, REPLY


class ComparisonReplies:
    def __init__(self, replies):
        self.replies = replies
        self.number = 0

    async def chat(self, messages, **_kwargs):
        if 'Judge if the cited passages' in messages[0]['content']:
            return {'content': '{"sufficient": true, "missing": ""}', 'latency_ms': 1}
        reply = self.replies[min(self.number, len(self.replies) - 1)]
        self.number += 1
        return {'content': json.dumps(reply, ensure_ascii=False), 'latency_ms': 1}


def test_w02_full_workflow_rejects_code_execution_commonality_and_repairs(monkeypatch):
    setup_comparison(monkeypatch)
    _, run, request = make_comparison_run()
    bad = {**REPLY, 'similarity': (
        '两者均通过测试套件运行时的代码执行情况来评估程序执行的覆盖程度，'
        '且都用于指导测试优化。')}
    model = ComparisonReplies([bad, REPLY])
    asyncio.run(AgentRunner(model).run(run, request))
    row = db.one('SELECT * FROM runs WHERE id=?', (run,))
    metrics = db.decode(row['metrics'])
    refs = db.decode(row['citations'])
    assert row['status'] == 'succeeded'
    assert [a['passed'] for a in metrics['mixed_comparison_validation']] == [False, True]
    assert '代码执行' in metrics['mixed_comparison_validation'][0]['reason']
    assert [r['source_type'] for r in refs] == ['paper', 'external']
    assert '检索' in row['answer'] and '代码' in row['answer']


def test_w02_full_workflow_never_launders_persistently_bad_comparison(monkeypatch):
    setup_comparison(monkeypatch)
    _, run, request = make_comparison_run()
    bad = {**REPLY, 'similarity': (
        '两者均通过测试套件运行时的代码执行情况来评估程序执行的覆盖程度。')}
    asyncio.run(AgentRunner(ComparisonReplies([bad, bad])).run(run, request))
    row = db.one('SELECT * FROM runs WHERE id=?', (run,))
    metrics = db.decode(row['metrics'])
    assert row['status'] == 'partial'
    assert [v['passed'] for v in metrics['mixed_comparison_validation']] == [False, False]
    assert db.decode(row['citations']) == []

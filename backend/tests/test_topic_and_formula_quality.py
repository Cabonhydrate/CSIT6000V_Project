"""Regressions for seven real academic assistant demo questions.

These tests use local DB + mocked model; no network, Ollama, TACC or gold answers.
"""
import asyncio
from app import db
from app.agent.workflow import _paper_mentions_focus, _analytical_answer_valid, AgentRunner
from app.schemas import ChatRequest
from test_routing_analytic import _seed_run


def test_formula_must_explain_variables_not_just_state_ratio():
    q='论文中 CC 的数学定义是什么？请解释公式中的每个变量。'
    english_ratio=('CC is the proportion of distinct chunks retrieved by all test queries '
                   'divided by the total number of chunks.')
    assert not _analytical_answer_valid(english_ratio,'formula',q)
    chinese_ratio='CC = 不同检索块数 / 总块数，其中分子是检索到的块数，分母是所有块的数量。'
    assert not _analytical_answer_valid(chinese_ratio,'formula',q)
    grounded=('CC(T)=|∪ R_k(q)| / |C|，其中 T 表示测试查询集合，'
              'C 表示语料库中的全部 Chunk，R_k(q) 表示查询 q 检索到的前 k 个 Chunk，'
              'k 表示每次查询返回的 Chunk 数，使用并集去重。')
    assert _analytical_answer_valid(grounded,'formula',q)


def test_named_topic_scan_not_confused_by_generic_algorithms():
    paper, _, _ = _seed_run('这篇论文提出了哪些量子计算算法？')
    assert _paper_mentions_focus('quantum computing',[paper]) is False
    assert _paper_mentions_focus('Chunk Coverage',[paper]) is True
    assert _paper_mentions_focus('algorithms',[paper]) is None


class FocusModel:
    async def chat(self, messages, **kwargs):
        assert kwargs['json_mode'] is True
        return {'content':'{"focus_term":"quantum computing"}', 'latency_ms':3}
    async def stream(self,*args,**kwargs):
        raise AssertionError('A source-unsupported topic must not reach answer generation')
        yield ''


def test_unrelated_question_abstains_without_citations_or_search():
    question='这篇论文提出了哪些量子计算算法？'
    paper,conversation,run=_seed_run(question)
    asyncio.run(AgentRunner(FocusModel()).run(run,ChatRequest(
        conversation_id=conversation, question=question, paper_ids=[paper],
        mode='adaptive', teaching=True)))
    record=db.one('SELECT status,answer,metrics,citations FROM runs WHERE id=?',(run,))
    metrics=db.decode(record['metrics'],{})
    assert record['status']=='partial'
    assert '量子计算' not in record['answer'] or '未找到' in record['answer']
    assert 'quantum computing' in record['answer']
    assert metrics['named_topic_in_paper'] is False
    assert metrics['answer_complete'] is False
    assert metrics['llm_calls']==1
    assert metrics['tool_calls']==0
    assert db.decode(record['citations'],[])==[]


def test_formula_english_ratio_is_marked_partial(monkeypatch):
    from test_routing_analytic import _run
    from test_multi_question import _fake_langgraph
    _fake_langgraph(monkeypatch)
    response='{"answer":"CC is the ratio of retrieved chunks / total chunks.","citations":[1]}'
    record, _model, _queries = _run(monkeypatch,
        '论文中 CC 的数学定义是什么？请解释公式中的每个变量。', response)
    assert record['status']=='partial'
    assert db.decode(record['metrics'])['answer_complete'] is False


def test_analytic_answer_prefers_few_citations(monkeypatch):
    from test_routing_analytic import _run
    response='{"answer":"传统指标关注单问题答案，而未覆盖整体检索区域；本文提出覆盖率指标补充测试充分性。","citations":[1,2,3,4,5]}'
    record, _model, _queries = _run(monkeypatch,
        '为什么作者认为传统 RAG 评价指标不足？',response)
    assert record['status']=='succeeded'
    assert len(db.decode(record['citations']))<=3

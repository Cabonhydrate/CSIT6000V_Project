"""General, non-paper-specific regression checks for the delivery candidate.

No model calls, web search or Gold Answer injection; tests check generic
quality boundaries and consent/benchmark isolation.
"""
import asyncio
from uuid import uuid4

from app import db
from app.agent.workflow import AgentRunner, _answer_is_bad
from app.schemas import ChatRequest
from app.skills.teaching import validate_and_render_teaching
from test_day1_learning_memory import CaptureModel, prepare


EVIDENCE = [{
    'content': ('Chunk Coverage (CC) measures the fraction of distinct document '
                'chunks retrieved across test queries, relative to all chunks '
                'in the corpus.'),
    'evidence_id': 'p:c1',
}]
QUESTION = '比较 Chunk Coverage 和传统测试的代码覆盖率。'


def _teaching(definition: str, takeaway: str):
    return validate_and_render_teaching({
        'definition': definition,
        'example': '假设语料有5个片段，查询检索到3个不同片段，因此CC=3/5。',
        'takeaway': takeaway,
        'citations': [1],
    }, EVIDENCE, QUESTION)


def test_code_coverage_comparison_is_not_document_code_confusion():
    r = _teaching(
        'Chunk Coverage 衡量不同文档片段被检索到的比例。',
        '与代码覆盖率不同，前者统计被检索的文档片段，后者统计代码的执行范围。')
    assert not r.issue, r.issue
    assert '代码覆盖率' in r.text


def test_explicit_confusion_about_code_chunks_is_still_rejected():
    r = _teaching('Chunk Coverage 表示测试执行的代码块比例。',
                  '统计测试执行过的代码片段。')
    assert '代码片段' in r.issue


def test_negated_code_coverage_claim_is_not_rejected():
    r = _teaching('Chunk Coverage 不是代码覆盖率，而是文档片段检索覆盖比例。',
                  '不能把检索覆盖和代码执行覆盖混为一谈。')
    assert not r.issue, r.issue


def test_internal_planning_is_not_allowed_as_final_answer():
    bad = ('首先，问题是：“What architecture does the decoder have?” 这是英文，'
           '但是用户要求中文直接回答。让我仔细阅读每个摘录，'
           '从摘录中提取架构信息。')
    assert _answer_is_bad(bad)
    assert _answer_is_bad('我需要分析来自论文的证据，然后提取结论。')
    assert not _answer_is_bad('论文中的模型采用 LSTM 解码器，利用上下文进行序列预测。[1]')
    assert not _answer_is_bad('首先，问题是数据稀疏，作者设计了检索增强的解决方案。[1]')


def test_adaptive_benchmark_has_no_profile_when_both_controls_are_disabled(monkeypatch):
    paper = prepare(monkeypatch)
    model = CaptureModel()
    conv, run = uuid4().hex, uuid4().hex
    question = '这篇论文如何定义 Chunk Coverage？'
    with db.connect() as c:
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                  (conv,db.settings.owner_id,'Ablation run',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) '
                  'VALUES(?,?,?,?,?,?,?)',
                  (run,conv,'pending','adaptive',question,db.encode([paper]),db.now()))
    asyncio.run(AgentRunner(model).run(run,ChatRequest(
        conversation_id=conv,question=question,paper_ids=[paper],
        mode='adaptive',teaching=False,use_memory=False,allow_web=False)))
    metrics = db.decode(db.one('SELECT metrics FROM runs WHERE id=?',(run,))['metrics'])
    assert metrics['learning_memory_candidates'] == 0
    assert metrics['learning_context_applied'] is False
    assert all('仅用于个性化讲解' not in str(prompt)
               for prompt in model.chat_inputs + model.stream_inputs)

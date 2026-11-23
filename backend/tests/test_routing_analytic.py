"""Regression tests for the three real-world academic question failures in v0.1.9.

No network/API calls: exercise routing, evidence prompts, citation validation and DB states.
"""
import asyncio
from uuid import uuid4

from app import db
from app.agent.workflow import (
    AgentRunner, _question_intent, _simple_definition, _split_question_parts,
    _explicit_acronym_answer, _retrieval_query, _select_strategy,
)
from app.schemas import ChatRequest
from test_multi_question import _fake_langgraph


FORMULA = '论文中 CC 的数学定义是什么？请解释公式中的每个变量。'
WHY = '为什么作者认为传统 RAG 评价指标不足？'
COMPARE = 'CC-guided testing 与 Random Selection 有什么区别？'


def test_intents_do_not_match_what_inside_why_or_comparison():
    assert _question_intent(FORMULA) == 'formula'
    assert _question_intent(WHY) == 'explanation'
    assert _question_intent(COMPARE) == 'comparison'
    for question in (FORMULA, WHY, COMPARE):
        assert not _simple_definition(question)
    assert _simple_definition('论文中的 CC 是什么意思？')
    assert _simple_definition('CC-guided testing 是什么？')
    assert _select_strategy(FORMULA, 0, '') == 'hybrid'
    assert _select_strategy(COMPARE, 0, '') == 'hybrid'
    assert _select_strategy(WHY, 0, '') == 'dense'


def test_followup_inherits_previous_topic():
    assert _split_question_parts(FORMULA) == ['论文中 CC 的数学定义是什么；请解释公式中的每个变量']
    assert _split_question_parts('文中的 CC 是什么？CC-guided test 是什么？') == [
        '文中的 CC 是什么', 'CC-guided test 是什么'
    ]
    assert _split_question_parts('方法是什么？它有哪些局限？') == ['方法是什么；它有哪些局限']
    assert 'Chunk Coverage' in _retrieval_query(FORMULA,'formula',['Chunk Coverage'])


def _seed_run(question):
    paper, conversation, run = (uuid4().hex for _ in range(3))
    with db.connect() as conn:
        conn.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                     (paper, db.settings.owner_id,'Example study','pdf','ready',db.now()))
        conn.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                     (f'{paper}:c0',paper,0,'Definitions',
                      'Chunk Coverage (CC) measures distinct document chunks returned by test queries. '
                      'CC(T)=|Union of retrieved chunks| / |All corpus chunks|. '
                      'T is the set of test queries, R(q) the returned chunks for query q, and D the corpus chunks.',
                      db.encode(['paragraph-1']),db.encode([7])))
        conn.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                     (conversation,db.settings.owner_id,'Routing test',db.now()))
        conn.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                     (run,conversation,'pending','adaptive',question,db.encode([paper]),db.now()))
    return paper,conversation,run


class MockModel:
    def __init__(self,answer):
        self.answer=answer
        self.prompts=[]
    async def chat(self,messages,**kwargs):
        self.prompts.append(messages)
        assert kwargs.get('json_mode') is True
        if 'Judge if the cited passages' in messages[0]['content']:
            return {'content':'{"sufficient":true,"missing":""}', 'latency_ms':1}
        return {'content':self.answer, 'latency_ms':1}
    async def stream(self,*args,**kwargs):
        raise AssertionError('An analytical question must never use freeform streaming')
        yield ''


def _run(monkeypatch,question,reply):
    _fake_langgraph(monkeypatch)
    paper,conversation,run=_seed_run(question)
    from app.knowledge import retrieval
    strategies=[]
    async_evidence={
        'evidence_id':f'{paper}:c0','source_type':'paper', 'paper_id':paper,
        'chunk_id':f'{paper}:c0','section':'Definitions','page_numbers':[7],
        'paragraph_refs':['paragraph-1'], 'content':
        'Chunk Coverage (CC) is the ratio of distinct retrieved chunks to all corpus chunks. '
        'CC(T)=|Union R(q)| / |D|, where T denotes test questions, R(q) retrieved chunks and D all chunks. '
        'The authors argue that relevance-only metrics cannot detect unexercised retrieval regions. '
        'The CC-guided method selects questions covering new chunks, while random selection chooses uniformly.'
    }
    def fake_search(query,paper_ids,strategy,top_k,rerank=False,run_id=None):
        strategies.append((query,strategy))
        return {'evidences':[async_evidence],'latency_ms':2}
    monkeypatch.setattr(retrieval,'search',fake_search)
    model=MockModel(reply)
    asyncio.run(AgentRunner(model).run(run,ChatRequest(
        conversation_id=conversation,question=question,paper_ids=[paper],
        mode='adaptive',teaching=True)))
    stored=db.one('SELECT * FROM runs WHERE id=?',(run,))
    return stored,model,strategies


def test_math_formula_uses_topic_and_is_complete(monkeypatch):
    reply='{"answer":"CC(T)=|∪R(q)| / |D|，其中 T 是测试问题集合，R(q) 是问题 q 检索到的 Chunk，D 是文档库中的全部 Chunk；并集用于避免重复计数。","citations":[1]}'
    rec,model,queries=_run(monkeypatch,FORMULA,reply)
    assert rec['status']=='succeeded',rec['error']
    assert db.decode(rec['metrics'])['question_intent']=='formula'
    assert db.decode(rec['metrics'])['answer_complete'] is True
    assert 'R(q)' in rec['answer'] and 'D 是' in rec['answer']
    assert all(x[1]=='hybrid' for x in queries)
    assert 'Chunk Coverage' in queries[0][0]
    assert '公式中的每个变量' in model.prompts[-1][-1]['content']
    assert db.decode(rec['citations'])[0]['page_numbers']==[7]


def test_math_formula_missing_variables_is_partial_not_success(monkeypatch):
    reply='{"answer":"CC 就是 Chunk Coverage。","citations":[1]}'
    rec,model,_=_run(monkeypatch,FORMULA,reply)
    assert rec['status']=='partial',rec['error']
    assert db.decode(rec['metrics'])['answer_complete'] is False
    assert len(model.prompts)==3  # 1 evidence check, 2 attempts to answer
    assert '部分回答' in rec['answer']
    assert 'CC 就是 Chunk Coverage' in rec['answer']
    assert len(db.decode(rec['citations'])) == 1  # Cited partial information is preserved


def test_why_question_uses_explanation_not_definition(monkeypatch):
    reply='{"answer":"传统指标通常评估单次检索相关性或下游答案，而不能显示哪些文档区域尚未被测试查询触及，因此作者希望通过覆盖率补足这一缺口。","citations":[1]}'
    rec,model,queries=_run(monkeypatch,WHY,reply)
    assert rec['status']=='succeeded',rec['error']
    assert queries[0][1]=='dense'
    assert '不足' in model.prompts[-1][-1]['content']
    assert not any('Verified expansion:' in m['content'] for prompt in model.prompts for m in prompt)


def test_comparison_question_uses_analytic_comparison(monkeypatch):
    reply='{"answer":"CC-guided testing 优先选择能覆盖新 Chunk 的测试问题；Random Selection 则随机选择。两者的选择依据不同，不能在没有具体实验数据时声称效果差异。","citations":[1]}'
    rec,model,queries=_run(monkeypatch,COMPARE,reply)
    assert rec['status']=='succeeded',rec['error']
    assert queries[0][1]=='hybrid'
    assert '比较双方' in model.prompts[-1][0]['content']
    assert 'Random Selection' in rec['answer']


def test_compound_acronym_is_not_the_verified_abbreviation_answer():
    passage=[{'content':'Chunk Coverage (CC) is a metric.'}]
    assert _explicit_acronym_answer('CC-guided testing 是什么？',passage) is None

"""Synthetic regression tests for safe answer recovery, not benchmark gold."""
import asyncio
import json
from uuid import uuid4

from app import db
from app.agent.workflow import AgentRunner, _source_quote
from app.knowledge import retrieval
from app.schemas import ChatRequest
from app.skills.math_support import _literal_equation, safe_math_fallback
from test_multi_question import _fake_langgraph


PAPER = ('We define Corpus Reachability (CR) as CR(T) = '
         '|∪_{q∈T} R_k(q)| / |D|. Here D is the whole corpus and T is the query suite.')
EVIDENCE = [{'evidence_id':'p:3','paper_id':'p','chunk_id':'p:3',
             'page_numbers':[3], 'section':'Definitions','content':PAPER}]


def test_mathematical_fallback_uses_literal_paper_equation():
    q='有人说 T = ∪_{i=1}^k R_k(q) 是 CR 的定义。请判断并写出正确公式。'
    assert _literal_equation(q,EVIDENCE)==(1,'CR(T) = |∪_{q∈T} R_k(q)| / |D|')
    result=safe_math_fallback(q,EVIDENCE)
    assert result and result.kind=='literal_source_equation'
    assert 'CR(T)' in result.text and '∪_{q∈T}' in result.text
    assert result.citation==1
    assert not safe_math_fallback(q,[dict(EVIDENCE[0],content='CR is a useful metric.')])
    assert not safe_math_fallback('为什么作者选用这种方法？',EVIDENCE)
    bad=[dict(EVIDENCE[0],content='The incorrect formula is CR(T) = |∪R_k(q)| / |D|.')]
    assert not safe_math_fallback(q,bad)


def test_citation_excerpt_prefers_literal_definition_over_earlier_operator():
    long_content='Other methods use ∪ in another context. '+'irrelevant context. '*100
    long_content+='\n'+PAPER
    excerpt=_source_quote(dict(EVIDENCE[0],content=long_content),
                          '请写出 CR 的正式公式。',chars=700)
    assert 'CR(T) =' in excerpt
    assert '∪_{q∈T}' in excerpt


def test_computed_union_witness_is_mathematically_valid_and_source_gated():
    q='有人说 ∑_{q∈T}|R_k(q)|/|D| 是否总等于 CR 的论文定义？请举反例。'
    result=safe_math_fallback(q,EVIDENCE)
    assert result and result.kind=='computed_union_counterexample'
    assert 'A={a}' in result.text and 'B={a}' in result.text
    assert '2/2=1' in result.text and '1/2=0.5' in result.text
    assert '自构造' in result.text and result.citation==1
    assert not safe_math_fallback(q,[dict(EVIDENCE[0],content='CR results are reported for multiple datasets.')])


def _seed(monkeypatch, question, reply, fail=False):
    _fake_langgraph(monkeypatch)
    paper, convo, run = (uuid4().hex for _ in range(3))
    with db.connect() as conn:
        conn.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                     (paper, db.settings.owner_id, 'Synthetic reachability paper','pdf','ready',db.now()))
        conn.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                     (convo, db.settings.owner_id, 'Math recovery test',db.now()))
        conn.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                     (run,convo,'pending','adaptive',question,db.encode([paper]),db.now()))
        conn.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                     (f'{paper}:3',paper,0,'Definitions',PAPER,db.encode([]),db.encode([3])))
    evidence=[dict(EVIDENCE[0],evidence_id=f'{paper}:3',chunk_id=f'{paper}:3',paper_id=paper)]
    monkeypatch.setattr(retrieval,'search',lambda *a,**kw:{'evidences':evidence,'latency_ms':1})
    class Model:
        async def chat(self, messages, **kwargs):
            if messages[0]['content'].startswith('Judge if the cited passages'):
                return {'content':'{"sufficient":true,"missing":""}', 'latency_ms':1}
            if fail:
                raise RuntimeError('model backend unavailable')
            return {'content':reply,'latency_ms':1}
    asyncio.run(AgentRunner(Model()).run(run,ChatRequest(
        conversation_id=convo,question=question,paper_ids=[paper],mode='adaptive')))
    return db.one('SELECT * FROM runs WHERE id=?',(run,))


def test_full_workflow_keeps_sourced_equation_if_both_model_attempts_invalid(monkeypatch):
    q='有人声称 T = ∪_{i=1}^k R_k(q) 是 CR 的定义。如不正确，请写出正确公式。'
    row=_seed(monkeypatch,q,'{}')
    assert row['status']=='partial'
    assert 'CR(T)' in row['answer'] and '∪_{q∈T}' in row['answer']
    assert len(db.decode(row['citations']))==1
    metrics=db.decode(row['metrics'])
    assert metrics['safe_math_fallback']=='literal_source_equation'
    assert [a['passed'] for a in metrics['validation_attempts']]==[False,False]


def test_full_workflow_computes_witness_without_model_fabrication(monkeypatch):
    q='有人把 CR 写成 ∑_{q∈T}|R_k(q)|/|D|。这是否总等于论文定义？请构造一个反例。'
    row=_seed(monkeypatch,q,'{}')
    assert row['status']=='partial'
    assert '2/2=1' in row['answer'] and '1/2=0.5' in row['answer']
    assert db.decode(row['metrics'])['math_witness_precomputed'] is True
    assert db.decode(row['metrics'])['safe_math_fallback']=='computed_union_counterexample'
    assert db.decode(row['citations'])[0]['page_numbers']==[3]


def test_failed_run_has_visible_error_and_stage(monkeypatch):
    row=_seed(monkeypatch,'请写出 CR 的数学定义。','',fail=True)
    assert row['status']=='failed'
    assert '内部错误' in row['answer']
    assert 'model backend unavailable' in row['error']
    assert db.decode(row['metrics'])['failure_type']=='RuntimeError'
    assert db.decode(row['metrics'])['failure_stage']

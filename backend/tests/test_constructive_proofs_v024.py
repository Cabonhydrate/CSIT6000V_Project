"""General-purpose constructive math and exact-formula evidence regression tests.

All examples here use a tiny synthetic index or synthetic expressions; no
Gold answer, dataset-specific value, or external evaluator belongs in runtime.
"""
import json

from app import db
from app.agent.workflow import (
    _question_intent, _ensure_formula_definition_evidence, _source_quote,
)
from app.skills.registry import select_answer_skills
from app.skills.quality import requests_counterexample, counterexample_issue
from test_routing_analytic import _run, MockModel


def test_universal_equation_example_routes_to_verdict_and_constructive_skill():
    q = ('有人将 Coverage Index 写为 ∑_{q∈T}|R(q)|/|D|。'
         '这是否总等于原文定义？请构造两个检索命中相同元素的小例子说明。')
    assert requests_counterexample(q)
    assert _question_intent(q) == 'claim_check'
    assert select_answer_skills('claim_check', teaching=False, question=q).names == (
        'verifyClaims', 'constructCounterexample')
    assert not requests_counterexample('请举例解释什么是集合并集。')
    assert 'constructCounterexample' not in select_answer_skills('formula',teaching=False,
        question='举一个例子解释覆盖率公式').names


def test_constructed_example_is_required_and_corpus_is_not_a_second_result_set():
    q='这是否总等于数学定义？请构造两个返回相同元素的例子说明∑与∪的差别。'
    mere_analogy='两个不同研究领域都能提到相同词，因此两种公式不相同。'
    assert '两个实际集合' in counterexample_issue(q, mere_analogy)
    # If only R(q1) and R(q2) are disjoint, they must not be treated as
    # overlapping merely because both occur in the corpus universe C.
    disjoint='C={a,b}，R(q1)={a}，R(q2)={b}。求和2/2=1，并集2/2=1。'
    assert '没有共同元素' in counterexample_issue(q,disjoint)
    overlap='假设 C={a,b}，R(q1)={a}，R(q2)={a}；求和2/2=1，而并集1/2=0.5。'
    assert counterexample_issue(q, overlap)==''


def test_constructive_skills_validate_latex_fractions():
    q='这两个集合公式是否总相等？构造一个反例，说明∑与∪的差别。'
    answer=(r'假设 R(q_1)=\{a\}，R(q_2)=\{a\}。'
            r'逐个求和是 \frac{2}{2}=1，集合并集得到 \frac{1}{2}=0.5。')
    assert counterexample_issue(q,answer)==''


def test_formula_definition_from_actual_index_preferred_to_indirect_evidence(monkeypatch):
    q='请写出论文里 CR(T) 的正式数学公式，并解释符号。'
    indirect=[{'evidence_id':'first','content':'CR is related to dataset coverage.',
               'paper_id':'p','page_numbers':[2],'section':'Overview'}]
    chunks=[{'id':'p:c2','paper_id':'p','seq':4,'section':'Definitions','pages':db.encode([5]),
             'paragraph_refs':db.encode(['p5-s3']),
             'content':'Corpus Reachability (CR) is CR(T)=|∪_{q∈T} R(q)|/|D|.'}]
    monkeypatch.setattr(db,'rows',lambda *args,**kwargs:chunks)
    chosen=_ensure_formula_definition_evidence(indirect,q,['p'])
    assert chosen[0]['evidence_id']=='p:c2'
    assert chosen[0]['page_numbers']==[5]
    assert 'CR(T)=' in _source_quote(chosen[0],q)
    assert chosen[1]['evidence_id']=='first'
    assert _ensure_formula_definition_evidence(indirect,'什么是 CR？',['p'])==indirect


def test_full_workflow_repairs_missing_constructed_example(monkeypatch):
    q=('有人将检索覆盖度写为 ∑_{q∈T}|R(q)|/|D|。这是否总等于论文定义？'
       '请构造两个查询检索到相同元素的小例子说明。')
    first={'claims':[{'claim':'求和式是否总等于并集定义','verdict':'refuted',
           'explanation':'不同领域的查询可能检索到相同资料。','evidence_ids':[1]}]}
    repaired={'claims':[{'claim':'求和式是否总等于并集定义','verdict':'refuted',
           'explanation':('假设同一语料 D={c1,c2}，R(q1)={c1}，R(q2)={c1}。'
                          '逐次求和2/2=1，而并集去重为1/2=0.5，故不等价。'),
           'evidence_ids':[1]}]}
    original=MockModel.chat
    seen=[]
    async def fake_chat(self,messages,**kwargs):
        if messages[0]['content'].startswith('Judge if the cited passages'):
            return await original(self,messages,**kwargs)
        seen.append(messages)
        reply=first if len(seen)==1 else repaired
        return {'content':json.dumps(reply,ensure_ascii=False),'latency_ms':1}
    monkeypatch.setattr(MockModel,'chat',fake_chat)
    rec,_,_= _run(monkeypatch,q,'unused')
    assert len(seen)==2
    assert rec['status']=='succeeded',rec['answer']
    assert '2/2=1' in rec['answer'] and '1/2=0.5' in rec['answer']
    assert '两个实际集合' in seen[1][0]['content'] or '数学反例' in seen[1][1]['content']
    stats=db.decode(rec['metrics'])
    assert stats['question_intent']=='claim_check'
    assert stats['answer_skills'][:2]==['verifyClaims','constructCounterexample']
    assert [x['passed'] for x in stats['validation_attempts']]==[False,True]


def test_full_workflow_requires_formal_not_prose_when_explicitly_requested(monkeypatch):
    q=('有人声称同一个元素三次检索应该计三次。请判断正误，'
       '并写出正式的 CC(T) 公式，解释并集为何去重。')
    first={'claims':[{'claim':'重复检索应当计数三次','verdict':'refuted',
            'explanation':'覆盖率是不同元素数量/全部元素数量，因此不会重复计数。',
            'evidence_ids':[1]}]}
    repaired={'claims':[{'claim':'重复检索应当计数三次','verdict':'refuted',
            'explanation':'同一个元素多次检索在并集中只出现一次。','evidence_ids':[1]}],
            'formula':'CC(T)=|∪_{q∈T} R(q)| / |D|',
            'variables':[{'symbol':'T','meaning':'测试查询集合'},
                         {'symbol':'D','meaning':'语料元素集合'}],
            'formula_citations':[1]}
    original=MockModel.chat
    seen=[]
    async def fake_chat(self,messages,**kwargs):
        if messages[0]['content'].startswith('Judge if the cited passages'):
            return await original(self,messages,**kwargs)
        seen.append(messages)
        return {'content':json.dumps(first if len(seen)==1 else repaired,ensure_ascii=False),'latency_ms':1}
    monkeypatch.setattr(MockModel,'chat',fake_chat)
    rec,_,_=_run(monkeypatch,q,'unused')
    assert rec['status']=='succeeded',rec['answer']
    assert len(seen)==2
    assert 'CC(T)' in rec['answer'] and '∪' in rec['answer']
    assert [v['passed'] for v in db.decode(rec['metrics'])['validation_attempts']]==[False,True]

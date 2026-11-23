"""Regression for v0.1.21 review: prevent answerable-but-abstained regressions.

Uses synthetic papers, tables and model responses. No paper-specific gold answers
in production code, no Ollama/Internet dependency, and no external eval suite.
"""
import json

from app import db
from app.skills.registry import select_answer_skills, skill_instructions
from app.agent.workflow import _focused_repair_messages, _source_quote
from test_routing_analytic import _run, MockModel
from test_multi_table_claims_v016 import execute_case, ClaimModel


def test_counterexample_skill_is_composable_and_loads_on_demand():
    q='给出集合并集与集合大小求和不相等的具体反例。'
    plan=select_answer_skills('formula',teaching=True,question=q)
    assert plan.names==('explainFormula','constructCounterexample','explainForBeginner')
    instructions=skill_instructions(plan)
    assert 'both calculated results' in instructions
    assert 'hypothetical' in instructions
    assert 'constructCounterexample' not in select_answer_skills('formula',teaching=False,
                                                                 question='解释集合并集的公式').names


def test_focused_repair_keeps_original_citation_numbers_and_source_facts():
    ev=[{'section':'Page 4','page_numbers':[4],
         'content':'Table 2. Measurements. Some numbers.'},
        {'section':'Page 3','page_numbers':[3],
         'content':'To assess whether this was caused by the generator, we repeated '
                   'the experiment using a stronger generator (Another-LM).'}]
    q='作者用什么实验解释 Table 2 中的例外？'
    messages=_focused_repair_messages(q,'explanation',ev,'没有补充实验说明',
                                      select_answer_skills('explanation',teaching=False,question=q))
    assert len(messages)==2
    assert '没有补充实验说明' in messages[1]['content']
    assert '[2]' in messages[1]['content']
    assert 'Another-LM' in messages[1]['content']
    assert 'Return JSON' in messages[0]['content']


def test_formula_reference_excerpt_prefers_equation_over_long_page_header():
    q='请写出 Corpus Reachability 的数学定义公式'
    content=('General background. '*250+'CR(T)=|∪_{q∈T}R(q)|/|C|. Definitions follow.')
    snippet=_source_quote({'content':content},q,chars=450)
    assert 'CR(T)=' in snippet
    assert len(snippet)<=450


def test_invalid_first_formula_is_fixed_using_compact_source_prompt(monkeypatch):
    incorrect={'formula':'T','variables':[{'symbol':'T','meaning':'查询集合'},
             {'symbol':'C','meaning':'全集'}], 'explanation':'计算覆盖率。','citations':[1]}
    corrected={'formula':r'CC(T)=\frac{|\bigcup_{q\in T}R(q)|}{|D|}',
               'variables':[{'symbol':'T','meaning':'查询集合'},
                            {'symbol':'C','meaning':'全部文档块'}],
               'explanation':'R(q) 是每个查询返回的集合，先并集去重。',
               'citations':[1]}
    original=MockModel.chat
    seen=[]
    async def fake_chat(self,messages,**kw):
        if messages[0]['content'].startswith('Judge if the cited passages'):
            return await original(self,messages,**kw)
        seen.append(messages)
        answer=incorrect if len(seen)==1 else corrected
        return {'content':json.dumps(answer,ensure_ascii=False),'latency_ms':1}
    monkeypatch.setattr(MockModel,'chat',fake_chat)
    q='请写出论文中 CC 的数学定义公式，并解释集合与变量。'
    rec,_,_=_run(monkeypatch,q,'{}')
    assert rec['status']=='succeeded',rec['answer']
    assert len(seen)==2
    assert len(seen[0])==2
    assert len(seen[1])==2  # a fresh prompt, not an extended previous conversation
    assert '上一版校验失败' in seen[1][1]['content']
    assert 'CC(T)' in rec['answer']
    traces=db.decode(rec['metrics'])['validation_attempts']
    assert [x['passed'] for x in traces]==[False,True]
    assert [x['attempt'] for x in traces]==[1,2]
    assert len(db.decode(rec['citations']))==1
    assert 'semantic_validation_issue' not in db.decode(rec['metrics'])


def test_claim_check_recovers_with_short_prompt_in_existing_two_calls(monkeypatch):
    original=ClaimModel.chat
    attempts=[]
    wrong='Table 3 的 Atlas 加速 2 倍，但所有 APFD 指标 CC 都更好。'
    fixed=('1. Table 3 的 Average 平均为 1.6×，此项正确。\n'
           '2. Atlas 的实际值是 Table 3 的 Random 20 和 Coverage-guided 10，20/10=2.0×，不能套用 1.6×。\n'
           '3. 并非所有 APFD 都更好；Table 9 的 Response Groundedness 中 '
           'R=0.670，而 CC=0.610，这是反例。')
    async def fake_chat(self,messages,**kw):
        if messages[0]['content'].startswith('Judge if the cited passages'):
            return await original(self,messages,**kw)
        attempts.append(messages)
        reply=wrong if len(attempts)==1 else fixed
        return {'content':json.dumps({'answer':reply,'citations':[]},ensure_ascii=False),'latency_ms':1}
    monkeypatch.setattr(ClaimModel,'chat',fake_chat)
    rec,_=execute_case(monkeypatch,'ignored')
    assert rec['status']=='succeeded',rec['answer']
    assert len(attempts)==2
    assert 'STRUCTURED TABLE CELLS' in attempts[1][1]['content']
    assert 'Response Groundedness' in attempts[1][1]['content']
    assert 'CC=0.610' in rec['answer']
    assert {x['page_numbers'][0] for x in db.decode(rec['citations'])}=={2,4}
    assert db.decode(rec['metrics'])['answer_complete'] is True


def test_bad_values_still_rejected_after_focused_repair(monkeypatch):
    wrong=('Table 3 Atlas 为 20/10=2.0×。Table 9 Atlas Response Groundedness：'
           'Random=0.610，CC-guided=0.670。')
    rec,_=execute_case(monkeypatch,wrong)
    assert rec['status']=='partial',rec['answer']
    metrics=db.decode(rec['metrics'])
    assert len(metrics['validation_attempts'])==2
    assert all(not x['passed'] for x in metrics['validation_attempts'])
    assert 'Random' in metrics['numeric_fact_conflict']
    assert 'Random=0.610' not in rec['answer']  # unsafe candidate must not leak


def test_math_counterexample_requires_two_object_results_not_analogy(monkeypatch):
    q='请构造一个集合反例说明两次检索结果基数求和不等于并集基数。'
    bad='举两个数据集中的不同问题就可以说明不相等。'
    rec,_,_=_run(monkeypatch,q,json.dumps({'answer':bad,'citations':[1]},ensure_ascii=False))
    assert rec['status']=='partial'
    assert '数学反例' in db.decode(rec['metrics'])['semantic_validation_issue']
    answer=('设 C={c1,c2}，R(q1)={c1}，R(q2)={c1}；求和得 2/2=1，'
            '并集去重仅 1/2=0.5。因此两者不等价。')
    rec,_,_=_run(monkeypatch,q,json.dumps({'answer':answer,'citations':[1]},ensure_ascii=False))
    assert rec['status']=='succeeded',rec['answer']


def test_counterexample_validator_checks_arithmetic_not_only_set_appearance():
    from app.skills.quality import counterexample_issue
    q='请用具体的集合反例解释并集与检索结果数量求和的不同'
    wrong='R(q1)={c1}, R(q2)={c1}，求和 2/2=0.5，并集 1/2=1。'
    assert '分数计算不成立' in counterexample_issue(q,wrong)
    right='R(q1)={c1}, R(q2)={c1}，求和 2/2=1，并集 1/2=0.5。'
    assert counterexample_issue(q,right)==''


def test_explicit_formal_math_request_is_not_satisfied_by_prose_ratio():
    from app.skills.quality import formal_equation_issue
    q='请写出该指标的正确公式，并解释每个符号。'
    source=[{'content':'This measure is M(T)=|∪_{q∈T} R(q)| / |C|.'}]
    bad='M 表示所有已检索元素占总元素数量的比例。'
    assert '正式数学等式' in formal_equation_issue(q,bad,source)
    formal=r'M(T)=\frac{|\bigcup_{q\in T} R(q)|}{|C|}'
    assert formal_equation_issue(q,'',source,formal)==''


def test_validation_attempt_diagnostics_do_not_mark_a_valid_run_as_abstained(monkeypatch):
    answer=('Table 3 的 Average 行平均为1.6×；但 Atlas 行是 Random 20，'
            'Coverage-guided 10，所以20/10=2.0×，而非1.6×。'
            'Table 9 Atlas Response Groundedness：R=0.670，CC=0.610，'
            '所以并非所有 APFD 更好。')
    rec,model=execute_case(monkeypatch,answer,citations=[])
    assert rec['status']=='succeeded',rec['answer']
    metrics=db.decode(rec['metrics'])
    assert len(metrics['validation_attempts'])==1
    assert metrics['validation_attempts'][0]['passed'] is True
    assert metrics['validation_attempts'][0]['candidate_preview']
    assert metrics.get('abstained_after_validation') is None
    assert metrics.get('validation_repair_succeeded') is None


def test_experimental_citation_excerpt_contains_changed_condition():
    q='论文又用什么补充实验解释 Table 2 的例外？'
    content=('Background details. '*150+'To assess whether the surprising effect depends on the '
             'generation model, we repeated this experiment with a stronger model '
             '(Another-LM) and obtained different measurements.')
    preview=_source_quote({'content':content},q,chars=520)
    assert 'Another-LM' in preview
    assert len(preview)<=520

"""Regression for independent evaluation findings: symbolic definition, universal
claims and follow-up experiment. Production logic contains no named-paper answers.
"""
import json

from app.skills.quality import (
    _source_math_lhs, formula_grounding_issue, formal_equation_issue,
    counterexample_issue, universal_table_issue, experiment_followup_issue)
from app.agent.workflow import _adjacent_explanation_evidence, _analytical_context
from app import db
from app.config import settings
from uuid import uuid4


SOURCE=[{'content': ('We introduce Coverage Index (CI). The fraction of distinct '
                     'items across all test queries is: CI(T) = |∪_{q∈T} R(q)| / |C|. '
                     'Each query contributes a set; E ← E ∪ R.')}]
Q='判断号称 Coverage Index 定义的公式是否正确：T = ∪_{i=1}^k {R(q)}。请写出正确公式并解释每个符号。'


def test_source_defined_formula_subject_rejects_a_plausible_but_wrong_lhs():
    assert _source_math_lhs(Q,SOURCE)=='CI(T)'
    assert '左侧' in formula_grounding_issue(Q, r'T=\bigcup_{i=1}^{k}R(q)',SOURCE)
    assert '原文' in formal_equation_issue(Q,'',SOURCE,r'T=\bigcup_{i=1}^{k}R(q)')
    correct=r'CI(T)=\frac{|\bigcup_{q\in T}R(q)|}{|C|}'
    assert formula_grounding_issue(Q,correct,SOURCE)==''
    assert formal_equation_issue(Q,'',SOURCE,correct)==''


def test_formal_indexed_definition_does_not_accept_a_word_ratio_or_missing_index():
    assert '索引' in formal_equation_issue(Q,'',SOURCE,r'CI(T)=\frac{distinct}{all}')
    assert '索引' in formal_equation_issue(Q,'',SOURCE,r'CI(T)=\bigcup R(q)/|C|')


def test_math_counterexample_requires_objects_and_numerical_results():
    q='请构造反例说明各查询检索数量求和与集合并集的结果不一定相同'
    assert '集合' in counterexample_issue(q,'两个数据集可能出现重叠，因此不相等。')
    correct=r'R(q1)={c1}, R(q2)={c1}, |C|=2；求和为 2/2=1，而并集只有 1/2=0.5。'
    assert counterexample_issue(q,correct)==''


TABLE={'table_number':6,'headers':['Dataset','Metric','R','CC'],
       'rows':[{'Dataset':'SampleA','Metric':'MetricX','R':'0.501','CC':'0.502'},
               {'Dataset':'SampleA','Metric':'MetricY','R':'0.700','CC':'0.600'}]}
UNIVERSAL='有人说 SampleA 的所有 APFD 指标中 CC 都优于 R，请逐项验证。'


def test_universal_claim_with_counterexample_must_be_refuted():
    bad={'claims':[{'claim':'所有 APFD 指标 CC 都优于 R', 'verdict':'supported',
                     'explanation':'MetricY 中 CC=0.600，R=0.700。','evidence_ids':[1]}]}
    rendered='全部指标都优于 R。MetricY 的 CC=0.600，小于 R=0.700。'
    assert 'supported' in universal_table_issue(UNIVERSAL,bad,rendered,[TABLE])
    good={'claims':[dict(bad['claims'][0], verdict='refuted')]}
    assert universal_table_issue(UNIVERSAL,good,rendered,[TABLE])==''
    assert '反例' in universal_table_issue(UNIVERSAL,good,'所有指标都好。',[TABLE])
    assert universal_table_issue(UNIVERSAL,{'answer':'x'},
        '这个全称说法错误：MetricY 中 CC=0.600，小于 R=0.700。',[TABLE])==''


def test_experiment_explanation_requires_actual_changed_experimental_condition():
    q='论文又用什么实验解释这一例外？'
    context=[{'content':('To assess whether this effect is specific to the choice of '
                         'generation model, we repeated the experiment using a stronger '
                         'generator (Nova-3-mini). With this generator, the prior '
                         'exception disappeared.')}]
    vague='作者比较了不同指标的 APFD 值，发现有一个例外。'
    assert '补充/重复' in experiment_followup_issue(q,vague,context)
    generic='作者使用相同的数据集重新做实验，结果不同。'
    assert '模型/配置' in experiment_followup_issue(q,generic,context)
    precise='作者使用更强的生成模型 Nova-3-mini 重新进行实验，这次策略更好。'
    assert experiment_followup_issue(q,precise,context)==''


def test_experiment_neighbor_is_citable_and_excerpt_is_centered_on_followup():
    paper=uuid4().hex
    with db.connect() as conn:
        conn.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                     (paper,settings.owner_id,'Study','pdf','ready',db.now()))
        for seq,page,text in [
                (0,3,'The study finds a surprising exception. Table 2. First results.'),
                (1,2,'Background notes. '+('Some unrelated methodology. '*120)+
                 'To assess whether this result was caused by the generator, we '
                 'repeated the study using a stronger generator (Nova-3-mini). '
                 'This supplementary experiment changed the result.'),
                (2,4,'General conclusion.')]:
            conn.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                         (f'{paper}:c{seq}',paper,seq,f'Page {page}',text,
                          db.encode([f'p{page}']),db.encode([page])))
    selected=[dict(evidence_id=f'{paper}:c0',paper_id=paper,chunk_id=f'{paper}:c0',
                   section='Page 3',page_numbers=[3],content='Table 2. Results and exception.',
                   paragraph_refs=['p3'])]
    q='根据 Table 2，论文又用什么实验解释这一例外？'
    expanded=_adjacent_explanation_evidence(selected,q,[paper])
    assert expanded[0]['page_numbers']==[3]
    assert expanded[1]['page_numbers']==[2]
    context=_analytical_context(expanded,q,'explanation')
    assert 'Nova-3-mini' in context
    assert '[2]' in context


def test_full_agent_rejects_wrong_definition_even_when_json_is_well_formed(monkeypatch):
    from test_routing_analytic import _run
    from app.skills import select_answer_skills
    q='判断下面这个 Chunk Coverage 的公式是否正确：T = ∪_{i=1}^{k} R_k(q)。如果错误请写出正确公式并解释每个符号。'
    assert 'explainFormula' in select_answer_skills('claim_check', teaching=True, question=q).names
    bad={'claims':[{'claim':'当前公式是正确的','verdict':'supported',
                    'explanation':'该公式表达了覆盖率。','evidence_ids':[1]}],
         'formula':r'T=\bigcup_{i=1}^k R_k(q)',
         'variables':[{'symbol':'T','meaning':'测试查询集合'}]}
    stored,_,_=_run(monkeypatch,q,json.dumps(bad,ensure_ascii=False))
    assert stored['status']=='partial',stored['answer']
    metrics=db.decode(stored['metrics'])
    assert '公式左侧' in metrics['semantic_validation_issue'] or '原文' in metrics['semantic_validation_issue']
    assert metrics['answer_complete'] is False


def test_experiment_question_with_citable_neighbor_can_repair_bad_response(monkeypatch):
    from app.agent.workflow import AgentRunner
    from app.knowledge import retrieval
    from app.schemas import ChatRequest
    from test_multi_question import _fake_langgraph
    import asyncio
    _fake_langgraph(monkeypatch)
    paper,conversation,run=(uuid4().hex for _ in range(3))
    q='根据 Table 5 中的结果，论文又用什么实验解释这一例外？'
    table='Table 5. Outcome summary with an unexpected exception in Groundedness.'
    note=('Methodological context. '+('Other detail. '*180)+
          'To assess whether this effect is specific to the choice of generation model, '
          'we repeated the experiment using a stronger generator (Nova-3-mini). '
          'With that generator the exception was no longer present.')
    with db.connect() as conn:
        conn.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                     (paper,settings.owner_id,'Study','pdf','ready',db.now()))
        conn.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                     (conversation,settings.owner_id,'Study',db.now()))
        conn.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                     (run,conversation,'pending','adaptive',q,db.encode([paper]),db.now()))
        for i,pg,txt in [(0,17,table),(1,16,note)]:
            conn.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                         (f'{paper}:c{i}',paper,i,f'Page {pg}',txt,db.encode([f'p{pg}']),db.encode([pg])))
    ev=dict(evidence_id=f'{paper}:c0',paper_id=paper,chunk_id=f'{paper}:c0',
            section='Page 17',page_numbers=[17],paragraph_refs=['p17'],content=table)
    monkeypatch.setattr(retrieval,'search',lambda *args,**kw: {'evidences':[ev],'latency_ms':2})

    class StepModel:
        def __init__(self):self.prompts=[]
        async def chat(self,messages,**kwargs):
            self.prompts.append(messages)
            if 'Judge if the cited passages' in messages[0]['content']:
                return {'content':'{"sufficient":true,"missing":""}', 'latency_ms':1}
            data={'answer': '作者比较了几个指标的表格结果，没有进行额外验证。',
                  'citations':[1]}
            if len(self.prompts)>=2:
                data={'answer':'作者使用更强的 Nova-3-mini 重新进行实验，使原本的例外消失。',
                      'citations':[2]}
            return {'content':json.dumps(data,ensure_ascii=False),'latency_ms':1}
        async def stream(self,*args,**kwargs):
            raise AssertionError('Analytical questions do not freeform stream')
            yield ''
    model=StepModel()
    asyncio.run(AgentRunner(model).run(run,ChatRequest(
        conversation_id=conversation,question=q,paper_ids=[paper],mode='adaptive')))
    record=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert record['status']=='succeeded',record['answer']
    assert len(model.prompts)>=2
    assert any('Nova-3-mini' in m['content'] for prompt in model.prompts for m in prompt)
    assert 'Nova-3-mini' in record['answer']
    assert db.decode(record['citations'])[0]['page_numbers']==[16]


def test_full_agent_rejects_supported_universal_claim_despite_correct_numbers(monkeypatch):
    from test_multi_table_claims_v016 import execute_case
    bad=('Table 3 Average 为 1.6×；Atlas 为 20/10=2.0×。'
         'Table 9 的所有 APFD 指标 CC 均优于 Random，这一判断正确。'
         'Response Groundedness 中 CC=0.610，Random=0.670。')
    rec,_=execute_case(monkeypatch,bad)
    assert rec['status']=='partial'
    metrics=db.decode(rec['metrics'])
    assert metrics['answer_complete'] is False
    assert '没有明确否定' in metrics['semantic_validation_issue']


def test_disjoint_sets_are_not_valid_duplicate_counterexample():
    q='构造两个检索集合的反例，说明并集与求和不能直接替换。'
    wrong='R(q1)={c1}，R(q2)={c2}，求和得到2/2=1，并集得到2/2=1。'
    assert '没有共同元素' in counterexample_issue(q,wrong)

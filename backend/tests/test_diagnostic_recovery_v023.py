"""Source-backed regression scenarios for misplaced formula, missing experiment
citation and a false aggregate verdict. Uses synthetic tables, no test gold in
runtime modules, no external evaluator, no Ollama calls.
"""
import asyncio
import json
from uuid import uuid4

from app import db
from app.config import settings
from app.agent.workflow import AgentRunner
from app.schemas import ChatRequest
from app.knowledge import retrieval
from app.skills.quality import (
    recover_embedded_formula, matching_experiment_citation,
    reconcile_source_aggregate_claims, aggregate_table_verdict_issue,
)
from test_multi_question import _fake_langgraph
from test_multi_table_claims_v016 import TABLE_A, TABLE_B, execute_case, ClaimModel
from test_routing_analytic import _run, MockModel


def test_formula_written_in_claim_explanation_is_not_lost():
    question = ('请判断 Coverage Index 的错误定义 T = ∪_{i=1}^k R_k(q) 是否正确，'
                '如不正确请写出正确公式。')
    source = [{'content': 'Coverage Index (CI) is CI(T) = |∪_{q∈T} R_k(q)| / |C|.'}]
    claim = {'claims': [{'claim': 'T = ∪_{i=1}^k R_k(q)', 'verdict': 'refuted',
                         'explanation': 'The correct definition is CI(T) = |∪_{q∈T} R_k(q)| / |C|, where T is a set.'}]}
    assert recover_embedded_formula(question, claim, source) == 'CI(T) = |∪_{q∈T} R_k(q)| / |C|'
    assert recover_embedded_formula(question, {'claims':[dict(claim['claims'][0], explanation='T is incorrect.')]},source) == ''


def test_aggregate_verdict_corrected_only_when_source_is_unambiguous():
    data = {'claims': [
        {'claim': 'Coverage-guided 对 Random 的平均加速为 1.6×', 'verdict': 'refuted',
         'explanation': '错误理解。', 'evidence_ids': [1]},
        {'claim': '平均加速为 1.6×，因此 Atlas 也应当是 1.6×', 'verdict': 'refuted',
         'explanation': '这是错误的类推。', 'evidence_ids':[1]},
    ]}
    corrected, count = reconcile_source_aggregate_claims(data,[TABLE_A,TABLE_B])
    assert count == 1
    assert corrected['claims'][0]['verdict'] == 'supported'
    assert 'Average' in corrected['claims'][0]['explanation']
    assert corrected['claims'][1] == data['claims'][1]
    assert data['claims'][0]['verdict'] == 'refuted'  # caller's original untouched
    assert aggregate_table_verdict_issue('', corrected,[TABLE_A,TABLE_B]) == ''
    mismatched = {'claims':[{'claim':'平均加速为 1.7×', 'verdict':'supported', 'explanation':'wrong'}]}
    fixed, n = reconcile_source_aggregate_claims(mismatched,[TABLE_A])
    assert n == 1 and fixed['claims'][0]['verdict'] == 'refuted'
    assert 'Table 3' in fixed['claims'][0]['explanation']


def test_experiment_citation_requires_shared_named_configuration():
    src=[{'content':'Table 8 describes the initial results.'},
         {'content':'To investigate the exception, we repeated the experiment using '
                    'a stronger query generator (Model-X9).'}]
    good='作者使用更强的 Model-X9 重新实验，观察了不同的结果。'
    assert matching_experiment_citation(good,src)==2
    assert matching_experiment_citation('作者重复了实验。',src) is None
    assert matching_experiment_citation('作者使用 Model-W3 重新实验。',src) is None


def test_full_workflow_recovers_formula_embedded_in_verdict(monkeypatch):
    q=('判断下面这个号称是 Chunk Coverage 定义的公式是否正确：'
       'T = ∪_{i=1}^k {R_k(q)}。如不正确，请写出正确公式并解释每个符号。')
    payload={'claims':[{'claim':'T = ∪_{i=1}^k {R_k(q)}', 'verdict':'refuted',
             'explanation': ('这个公式把测试查询集合当成了检索集合。正确的 CC 数学定义是 '
                             'CC(T) = |∪_{q∈T} R_k(q)| / |C|，其中 T 是测试集，'
                             'C 是全集，R_k(q) 是检索结果集合，k 是单次结果条数。'),
             'evidence_ids':[1]}]}
    rec,_,_= _run(monkeypatch,q,json.dumps(payload,ensure_ascii=False))
    assert rec['status']=='succeeded',rec['answer']
    assert 'CC(T)' in rec['answer'] and '∪' in rec['answer']
    assert db.decode(rec['metrics']).get('recovered_embedded_formula') is True
    assert len(db.decode(rec['citations'])) >= 1


def test_full_workflow_corrects_simple_aggregate_without_losing_other_claims(monkeypatch):
    original=ClaimModel.chat
    calls=[]
    payload={'claims':[
        {'claim':'Coverage-guided 对 Random 的平均加速为 1.6×', 'verdict':'refuted',
         'explanation':'我认为平均是另外的数值。','evidence_ids':[1]},
        {'claim':'Atlas 也加速 1.6×','verdict':'refuted',
         'explanation':'Atlas 在 Table 3 为 20/10=2.0×，不同于 1.6×。','evidence_ids':[1]},
        {'claim':'所有 APFD 指标 CC 都优于 R','verdict':'refuted',
         'explanation':'Table 9 Atlas Response Groundedness 为 R=0.670，CC=0.610，故存在反例。',
         'evidence_ids':[2]},
    ]}
    async def fake(self,messages,**kw):
        if messages[0]['content'].startswith('Judge if the cited passages'):
            return await original(self,messages,**kw)
        calls.append(messages)
        return {'content':json.dumps(payload,ensure_ascii=False),'latency_ms':1}
    monkeypatch.setattr(ClaimModel,'chat',fake)
    rec,_ = execute_case(monkeypatch,'unused')
    assert rec['status']=='succeeded',rec['answer']
    assert rec['answer'].startswith('1. **正确**')
    assert 'Table 3' in rec['answer'] and '0.610' in rec['answer']
    metrics=db.decode(rec['metrics'])
    assert metrics['source_aggregate_verdicts_corrected']==1
    assert metrics['validation_attempts'][0]['passed'] is True
    assert len(calls)==1


def test_full_workflow_repairs_missing_followup_experiment_citation(monkeypatch):
    _fake_langgraph(monkeypatch)
    paper,conversation,run = (uuid4().hex for _ in range(3))
    q=('根据 Table 8，Atlas 的 Response Groundedness 指标下，'
       'Random 与 Coverage-guided 的 APFD 分别是多少？作者用什么补充实验解释这一例外？')
    table={
        'headers':['Dataset','RAG Eval. Metric','R','CC'],
        'rows':[{'Dataset':'Atlas','RAG Eval. Metric':'Response Groundedness',
                 'R':'0.650','CC':'0.620'}],
        'caption':'Table 8. APFD comparison.', 'table_number':8,'page':4,
    }
    chunks = [
        {'evidence_id':f'{paper}:c0','paper_id':paper,'chunk_id':f'{paper}:c0',
         'section':'Page 4','page_numbers':[4],'paragraph_refs':['page4-p0'],
         'content':'Table 8. APFD comparison. Atlas Response Groundedness: R=0.650, CC=0.620.'},
        {'evidence_id':f'{paper}:c1','paper_id':paper,'chunk_id':f'{paper}:c1',
         'section':'Page 3','page_numbers':[3],'paragraph_refs':['page3-p0'],
         'content': ('To investigate the anomaly, we repeated the experiment using '
                     'a stronger query generator (Model-X9). After the change '
                     'the Coverage-guided strategy performed better on all metrics.')},
    ]
    with db.connect() as c:
        c.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                  (paper, settings.owner_id,'Synthetic study','pdf','ready',db.now()))
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                  (conversation, settings.owner_id,'Synthetic case',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                  (run,conversation,'pending','adaptive',q,db.encode([paper]),db.now()))
        c.execute('INSERT INTO paper_tables(paper_id,table_number,page,headers,rows_json,caption) VALUES(?,?,?,?,?,?)',
                  (paper,8,4,db.encode(table['headers']),db.encode(table['rows']),table['caption']))
        for ix, ev in enumerate(chunks):
            c.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                      (ev['chunk_id'],paper,ix,ev['section'],ev['content'],db.encode(ev['paragraph_refs']),db.encode(ev['page_numbers'])))
    monkeypatch.setattr(retrieval,'search',lambda *a,**kw:{'evidences':chunks,'latency_ms':1})
    first={'answer':'Random=0.650，CC=0.620。', 'citations':[1]}
    second={'answer':('Random=0.650，CC=0.620。作者用更强的 Model-X9 作为查询生成器'
                      '重复实验后，Coverage-guided 在所有指标上优于 Random。'), 'citations':[1]}
    model=MockModel('unused')
    seen=[]
    async def fake_chat(messages,**kw):
        if messages[0]['content'].startswith('Judge if the cited passages'):
            return {'content':'{"sufficient":true,"missing":""}', 'latency_ms':1}
        seen.append(messages)
        return {'content':json.dumps(first if len(seen)==1 else second,ensure_ascii=False),'latency_ms':1}
    monkeypatch.setattr(model,'chat',fake_chat)
    asyncio.run(AgentRunner(model).run(run,ChatRequest(
        conversation_id=conversation,question=q,paper_ids=[paper],mode='adaptive')))
    rec=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert rec['status']=='succeeded',rec['answer']
    assert 'Model-X9' in rec['answer']
    cites=db.decode(rec['citations'])
    assert {c['page_numbers'][0] for c in cites}=={3,4}
    assert db.decode(rec['metrics'])['experiment_citation_repaired'] is True
    assert db.decode(rec['metrics'])['validation_attempts'][-1]['passed'] is True

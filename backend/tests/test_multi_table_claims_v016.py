"""Cross-table claim checks: real source cells, no paper-specific answers.

Regression for a valid metric in table B incorrectly rejected against a
single selected row in table A. Runs the whole Adaptive task using a mock LLM.
"""
import asyncio
import json
from uuid import uuid4
from app import db
from app.config import settings
from app.agent.facts import numeric_claim_conflict, claim_table_sources
from app.agent.workflow import AgentRunner
from app.knowledge import retrieval
from app.schemas import ChatRequest
from test_multi_question import _fake_langgraph


TABLE_A={
    'table_number':3,'page':2,'caption':'Table 3. Number of test executions to target coverage.',
    'headers':['Dataset','Coverage-guided','Random','Speedup (R)'],
    'rows':[
        {'Dataset':'Atlas','Coverage-guided':'10','Random':'20','Speedup (R)':'2.0×'},
        {'Dataset':'Average','Coverage-guided':'–','Random':'–','Speedup (R)':'1.6×'},
    ]
}
TABLE_B={
    'table_number':9,'page':4,'caption':'Table 9. Fault detection APFD metrics.',
    'headers':['Dataset','RAG Eval. Metric','R','CC'],
    'rows':[
        {'Dataset':'Atlas','RAG Eval. Metric':'Faithfulness','R':'0.500','CC':'0.700'},
        {'Dataset':'Atlas','RAG Eval. Metric':'Response Groundedness','R':'0.670','CC':'0.610'},
    ]
}
QUESTION=('有人说 Table 3 的平均 1.6 倍意味着 Atlas 也是 1.6 倍，'
          '并且 Table 9 中 Atlas 的所有 APFD 指标下 CC-guided 都比 Random 好。请逐条核验。')


def test_multi_table_numeric_verification_scopes_values_by_source():
    a=dict(TABLE_A,paper_id='synthetic'); b=dict(TABLE_B,paper_id='synthetic')
    assert numeric_claim_conflict(
        'Table 3 平均为 1.6 倍，Atlas 为 20/10=2 倍。\n'
        'Table 9 中 Atlas Response Groundedness：Random=0.670，CC-guided=0.610。',
        [a,b]) == ''
    wrong=numeric_claim_conflict(
        'Table 9 中 Atlas Response Groundedness：Random=0.610，CC-guided=0.670。',[a,b])
    assert wrong and 'Random' in wrong
    missing=numeric_claim_conflict('Table 9 的 Response Groundedness：CC=0.999。',[a,b])
    assert missing and '0.999' in missing
    misattributed=numeric_claim_conflict('Table 3 记录 CC=0.610。',[a,b])
    assert misattributed and 'Table 3' in misattributed


def prepare_case():
    paper,conversation,run=(uuid4().hex for _ in range(3))
    sources=[]
    with db.connect() as c:
        c.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                  (paper,settings.owner_id,'Two tables','pdf','ready',db.now()))
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                  (conversation,settings.owner_id,'multi-table test',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                  (run,conversation,'pending','adaptive',QUESTION,db.encode([paper]),db.now()))
        for i,table in enumerate((TABLE_A,TABLE_B)):
            c.execute('INSERT INTO paper_tables(paper_id,table_number,page,headers,rows_json,caption) VALUES(?,?,?,?,?,?)',
                      (paper,table['table_number'],table['page'],db.encode(table['headers']),
                       db.encode(table['rows']),table['caption']))
            content=(f"Table {table['table_number']}. {table['caption']}\n" + '\n'.join(
                ' | '.join(f'{h}: {r.get(h, "")}' for h in table['headers']) for r in table['rows']))
            key=f'{paper}:c{i}'
            c.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                      (key,paper,i,f'Page {table["page"]}',content,db.encode([f'p{i}']),
                       db.encode([table['page']])))
            sources.append({'evidence_id':key,'paper_id':paper,'chunk_id':key,
                            'section':f'Page {table["page"]}','content':content,
                            'page_numbers':[table['page']],'paragraph_refs':[f'p{i}']})
    return paper,conversation,run,sources


class ClaimModel:
    def __init__(self, reply, citations=None):
        self.reply=reply
        self.citations=citations if citations is not None else [1,2]
        self.prompts=[]
    async def chat(self,messages,**kwargs):
        self.prompts.append(messages)
        if messages[0]['content'].startswith('Judge if the cited passages'):
            return {'content':'{"sufficient":true,"missing":""}','latency_ms':1}
        return {'content':json.dumps({'answer':self.reply,'citations':self.citations},ensure_ascii=False),
                'latency_ms':1}
    async def stream(self,*args,**kwargs):
        raise AssertionError('Claim check must be structured')
        yield ''


def execute_case(monkeypatch,reply,citations=None):
    _fake_langgraph(monkeypatch)
    paper,conversation,run,sources=prepare_case()
    def fake_search(*args,**kwargs):
        return {'evidences':sources,'latency_ms':2}
    monkeypatch.setattr(retrieval,'search',fake_search)
    model=ClaimModel(reply,citations)
    asyncio.run(AgentRunner(model).run(run,ChatRequest(
        conversation_id=conversation,question=QUESTION,paper_ids=[paper],mode='adaptive')))
    return db.one('SELECT * FROM runs WHERE id=?',(run,)),model


def test_agent_accepts_correct_claims_from_two_tables(monkeypatch):
    response=('1. 平均提升 1.6× 正确，Table 3 的 Average 行是 1.6×。[1]\n'
              '2. Atlas 不等于平均值。Table 3 的 Atlas 行是 20/10=2.0×。[1]\n'
              '3. 并非所有指标都更好。Table 9 的 Atlas Response Groundedness：'
              'Random=0.670，CC-guided=0.610，后者更低。[2]')
    rec,model=execute_case(monkeypatch,response)
    assert rec['status']=='succeeded',rec['answer']
    metrics=db.decode(rec['metrics'])
    assert metrics.get('answer_complete') is True
    assert not metrics.get('numeric_fact_conflict')
    assert [x['page_numbers'] for x in db.decode(rec['citations'])]==[[2],[4]]
    assert 'Table 9' in db.decode(rec['citations'])[1]['quote']
    assert any('1.6×' in m['content'] and '0.610' in m['content']
               for prompt in model.prompts for m in prompt)


def test_agent_rejects_swapped_values_within_metric_row(monkeypatch):
    reply=('Table 3 Atlas 是 20/10=2.0×。Table 9 Atlas Response Groundedness：'
           'Random=0.610，CC-guided=0.670。')
    rec,_=execute_case(monkeypatch,reply)
    assert rec['status']=='partial'
    metrics=db.decode(rec['metrics'])
    assert 'Random' in metrics['numeric_fact_conflict']
    assert 'Table 3 对应数据行' not in metrics['numeric_fact_conflict']


def test_claim_source_selection_keeps_both_table_sections():
    paper,_,_,_=prepare_case()
    selected=claim_table_sources(QUESTION,[paper])
    assert {s['table_number'] for s in selected}=={3,9}
    table3=next(x for x in selected if x['table_number']==3)
    assert {'Atlas','Average'}=={r['Dataset'] for r in table3['rows']}
    table9=next(x for x in selected if x['table_number']==9)
    assert any(r['RAG Eval. Metric']=='Response Groundedness' for r in table9['rows'])


def test_incomplete_multi_table_citations_do_not_count_as_success(monkeypatch):
    partial='Table 3 的 Atlas 需要 10 次，Random 需要 20 次；加速比为 20/10=2.0×。'
    rec,_=execute_case(monkeypatch,partial,citations=[1])
    assert rec['status']=='partial'
    metrics=db.decode(rec['metrics'])
    assert 'Table 9' in metrics['answer_validation_issue']
    assert 'Table 3' in rec['answer'] and 'Table 9' in rec['answer']
    assert [c['page_numbers'] for c in db.decode(rec['citations'])]==[[2],[4]]
    assert '解释部分未通过核验' in rec['answer']


def test_derived_ratio_is_not_treated_as_source_measurement():
    a=dict(TABLE_A,paper_id='synthetic');b=dict(TABLE_B,paper_id='synthetic')
    assert numeric_claim_conflict('Table 3 的比值 10/20=0.500，Atlas 2.0×。',[a,b])==''
    bad=numeric_claim_conflict('Table 3 的比值 10/20=0.999。',[a,b])
    assert '计算式' in bad

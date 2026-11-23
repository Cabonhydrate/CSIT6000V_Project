"""General source-anchor and answer-repair regression tests for v0.1.13.

These fake passages stand in for any paper's labeled tables/algorithms. They
intentionally do not depend on the original Chunk Coverage PDF.
"""
import asyncio
import json
from app import db
from app.agent.workflow import (
    AgentRunner, _source_anchors, _anchored_evidence, _analytical_context,
    _analytical_issue,
)
from app.schemas import ChatRequest
from app.knowledge import retrieval
from test_routing_analytic import _seed_run
from test_multi_question import _fake_langgraph
from test_contextual_tasks_v012 import ALGORITHM, SPEEDUP


def _item(paper, seq, content, page):
    return {'evidence_id':f'{paper}:c{seq}', 'source_type':'paper',
            'paper_id':paper, 'chunk_id':f'{paper}:c{seq}',
            'section':f'Page {page}', 'page_numbers':[page],
            'paragraph_refs':[f'page{page}-p0'], 'content':content}


ALGORITHM_TEXT='''Algorithm 1 Coverage-guided test generation.
Require: Chunk corpus C, retriever R, initial tests T0, budget B, target tau (τ).
Initialize T = T0 and E = empty set. For each query in T, execute top-k retrieval,
subtract one from budget B and update E = E union retrieved chunks.
Compute CC(T) = |E| / |C|. Stop if coverage reaches τ. Stop the loop when B runs out.
Set U = C minus E and generate new questions from uncovered chunks in U.
Add new test queries to T, then repeat the loop until termination.'''

TABLE_TEXT='''Table 4. Tests required to reach 50% of attainable coverage.
Dataset  Coverage-guided  Overlap-biased  Random  Speedup (OB)  Speedup (R)
MIMIC-IV 248 1,301 530 5.2x 2.1x
ConvFinQA 871 2,259 1,226 2.6x 1.4x'''


class TwoDrafts:
    def __init__(self, replies):
        self.replies=replies
        self.prompts=[]
    async def chat(self, messages, **kwargs):
        self.prompts.append(messages)
        assert kwargs['json_mode'] is True
        return {'content':json.dumps(self.replies[min(len(self.prompts)-1,len(self.replies)-1)],ensure_ascii=False),
                'latency_ms':2}
    async def stream(self,*args,**kwargs):
        raise AssertionError('Structured analysis should not use free stream')
        yield ''


def _run(monkeypatch, question, chunks, replies):
    _fake_langgraph(monkeypatch)
    paper,conv,run=_seed_run(question)
    rows=[_item(paper,seq,content,page) for seq,content,page in chunks]
    calls=[]
    def fake_search(query,paper_ids,strategy,top_k,rerank=False,run_id=None):
        calls.append((query,strategy))
        return {'evidences':rows,'latency_ms':1}
    monkeypatch.setattr(retrieval,'search',fake_search)
    model=TwoDrafts(replies)
    asyncio.run(AgentRunner(model).run(run,ChatRequest(
        conversation_id=conv,question=question,paper_ids=[paper],mode='adaptive',teaching=True)))
    return db.one('SELECT * FROM runs WHERE id=?',(run,)),model,calls,run


def test_source_anchors_are_general():
    assert _source_anchors('根据 Table 4 与 Algorithm 1') == [('table','4'),('algorithm','1')]
    assert _source_anchors('请比较表 2 和算法 3') == [('table','2'),('algorithm','3')]
    assert _source_anchors('不要引用量子计算') == []
    ev=[{'content':'Unrelated chapter'}, {'content':ALGORITHM_TEXT}, {'content':'Table 4 data'}]
    assert _anchored_evidence(ev,ALGORITHM)[0]['content'].startswith('Algorithm 1')
    assert 'budget B' in _analytical_context(_anchored_evidence(ev,ALGORITHM),ALGORITHM,'procedure')


def test_algorithm_answer_repairs_missing_conditions_instead_of_retrieving_again(monkeypatch):
    draft={'answer':'初始化 E，逐条检索查询并更新已覆盖 Chunk，B 表示预算。','citations':[1]}
    fixed={'answer':'先初始化测试集 T 和已覆盖集合 E；依次检索测试问题并更新 E。'
          '其中 B 是预算，τ 是目标覆盖率，U = C - E 是尚未覆盖的 Chunk 集合。'
          '针对 U 生成新查询并加入 T。B 耗尽或 CC 达到 τ 时停止。',
          'citations':[1]}
    rec,model,calls,run=_run(monkeypatch,ALGORITHM,
                            [(9,'An unrelated paragraph about statistics.',12),
                             (6,ALGORITHM_TEXT,7)], [draft,fixed])
    assert rec['status']=='succeeded',rec['answer']
    assert len(calls)==1  # Exact Algorithm 1 source was already retrieved
    assert len(model.prompts)==2  # draft + targeted repair, no judge/retrieval loop
    assert '没有解释题目要求的符号' in model.prompts[1][-1]['content']
    assert '预算' in rec['answer'] and 'τ' in rec['answer']
    assert db.decode(rec['citations'])[0]['page_numbers']==[7]
    assert 'Algorithm 1' in db.decode(rec['citations'])[0]['quote']
    events=db.rows('SELECT * FROM run_events WHERE run_id=?',(run,))
    assert any(x['event_type']=='answer_validation' for x in events)


def test_table_calculation_repairs_missing_arithmetic_without_retrieval_loop(monkeypatch):
    first={'answer':'Table 4 shows 248 coverage-guided tests, 530 random, 1301 overlap-biased.',
           'citations':[1]}
    second={'answer':'达到 50% 的目标：Coverage-guided 为 248 次，Random 为 530 次，'
            'Overlap-biased 为 1301 次；相对 Random 的加速比是 530/248≈2.14 倍。',
            'citations':[1]}
    rec,model,calls,_=_run(monkeypatch,SPEEDUP,
                           [(14,TABLE_TEXT,15),(2,'Random material unrelated to the specified table.',3)],
                           [first,second])
    assert rec['status']=='succeeded',rec['answer']
    assert '530/248≈2.14' in rec['answer']
    assert len(calls)==1 and len(model.prompts)==2
    assert '计算式' in model.prompts[1][-1]['content']
    assert 'MIMIC-IV' in db.decode(rec['citations'])[0]['quote']


def test_unfixed_answer_is_partial_but_keeps_cited_useful_information(monkeypatch):
    first={'answer':'表格显示目标方案 248 次，随机方案 530 次，但尚未计算比值。',
           'citations':[1]}
    rec,model,calls,_=_run(monkeypatch,SPEEDUP,[(14,TABLE_TEXT,15)],[first,first])
    assert rec['status']=='partial'
    assert '248' in rec['answer'] and '部分回答' in rec['answer']
    assert db.decode(rec['citations'])[0]['page_numbers']==[15]
    assert 'answer_validation_issue' in db.decode(rec['metrics'])
    assert len(calls)==1


def test_missing_anchor_does_not_pretend_we_located_requested_table():
    evidence=[{'content':'Table 3 discusses unrelated distributions'}]
    assert _anchored_evidence(evidence,'What is in Table 4?')==evidence
    assert _analytical_issue('248 and 530, here is the speedup.', 'calculation', SPEEDUP)

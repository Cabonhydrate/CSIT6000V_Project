"""Regression tests derived from T05/T14/T17 failed real paper prompts.

The original PDF supplied the answer in every case. These tests verify the
planning and completion contract independently of Ollama and network access.
"""
import asyncio
from app import db
from app.agent.workflow import _split_question_parts, _question_intent, _analytical_answer_valid, AgentRunner
from app.schemas import ChatRequest
from test_routing_analytic import _seed_run
from test_multi_question import _fake_langgraph

ALGORITHM = '按照 Algorithm 1，逐步说明 CC-guided test generation 如何工作。B、τ、E、U 分别表示什么？何时停止？'
SPEEDUP = '根据 Table 4，MIMIC-IV 达到最大可达覆盖率的 50% 时，CC-guided、Random、Overlap-biased 分别需要多少次测试？计算相对 Random 的加速比。'
EXCEPTION = '根据 Table 5，MIMIC-IV 的 Response Groundedness 指标下，Random 与 CC-guided 的 APFD 分别是多少？论文又用什么实验解释这一例外？'


def test_dependent_followup_uses_antecedent():
    for question in (ALGORITHM,SPEEDUP,EXCEPTION):
        assert len(_split_question_parts(question))==1
    assert _question_intent(ALGORITHM)=='procedure'
    assert _question_intent(SPEEDUP)=='calculation'
    assert _question_intent(EXCEPTION)=='explanation'
    assert len(_split_question_parts('CC是什么？CC-guided test是什么？'))==2


def test_calculation_and_procedure_have_meaningful_completion_checks():
    assert not _analytical_answer_valid('Table 4 中是 248、530、1301，计算相对 Random 的加速比。','calculation',SPEEDUP)
    assert _analytical_answer_valid('分别是248、530、1301；530/248≈2.14倍。','calculation',SPEEDUP)
    assert not _analytical_answer_valid('分别是248、530、1301；530/248≈2.5倍。','calculation',SPEEDUP)
    assert not _analytical_answer_valid('B 是预算，E 为已访问块，U 为未访问块。','procedure',ALGORITHM)
    assert not _analytical_answer_valid('遍历并检索，B 是预算，τ 是目标阈值，耗尽或达到阈值时停止。','procedure',ALGORITHM)
    assert _analytical_answer_valid('首先初始化 E，再遍历每个查询并更新 E、U。预算 B 耗尽或覆盖率达到 τ 时停止。','procedure',ALGORITHM)


class PaperModel:
    def __init__(self, answer):
        self.answer=answer
        self.calls=[]
    async def chat(self,messages,**kwargs):
        self.calls.append(messages)
        if 'Judge if the cited passages' in messages[0]['content']:
            return {'content':'{"sufficient":true,"missing":""}','latency_ms':1}
        return {'content':self.answer,'latency_ms':2}
    async def stream(self,*args,**kwargs):
        raise AssertionError('Analytical answers should use JSON mode, not free stream')
        yield ''


def _try(monkeypatch,question,answer):
    _fake_langgraph(monkeypatch)
    paper,conversation,run=_seed_run(question)
    from app.knowledge import retrieval
    evidence={
        'evidence_id':f'{paper}:c2','paper_id':paper,'chunk_id':f'{paper}:c2','source_type':'paper',
        'section':'Section evidence', 'page_numbers':[16], 'paragraph_refs':['p0'],
        'content':('Algorithm 1: initialize T and E. Execute the retriever, update E, '
                   'stop when budget B expires or CC reaches threshold τ. '
                   'Table 4: MIMIC-IV CC-guided 248, random 530, overlap-biased 1301. '
                   'Table 5: MIMIC-IV Response Groundedness random 0.615 vs CC 0.597. '
                   'The authors repeated the experiment with stronger GPT-5-mini query generator; '
                   'CC-guided then exceeded random on every measure.')}
    queries=[]
    def fake_search(query,paper_ids,strategy,top_k,rerank=False,run_id=None):
        queries.append((query,strategy))
        return {'evidences':[evidence],'latency_ms':3}
    monkeypatch.setattr(retrieval,'search',fake_search)
    model=PaperModel(answer)
    asyncio.run(AgentRunner(model).run(run,ChatRequest(
        conversation_id=conversation,question=question,paper_ids=[paper],mode='adaptive',teaching=False)))
    rec=db.one('SELECT * FROM runs WHERE id=?',(run,))
    return rec,model,queries


def test_algorithm_question_can_answer_stopping_conditions(monkeypatch):
    rec,model,queries=_try(monkeypatch,ALGORITHM,
        '{"answer":"初始化 T、E，遍历查询并把检索结果并入 E，U 是 C 与 E 的差集。B 是查询预算，τ 是目标覆盖率；当 B 耗尽或 CC 达到 τ 时停止。","citations":[1]}')
    assert rec['status']=='succeeded',rec['error']
    assert db.decode(rec['metrics'])['question_intent']=='procedure'
    assert '停止' in rec['answer'] and '何时停止' not in rec['answer']
    assert queries[0][1]=='hybrid'


def test_speedup_question_must_show_calculation(monkeypatch):
    rec,model,queries=_try(monkeypatch,SPEEDUP,
        '{"answer":"CC-guided需要248次、Random需要530次、Overlap-biased需要1301次；相对Random的加速比为530/248≈2.14倍。","citations":[1]}')
    assert rec['status']=='succeeded',rec['error']
    assert '530/248' in rec['answer']
    assert db.decode(rec['metrics'])['question_intent']=='calculation'


def test_bad_speedup_not_reported_as_succeeded(monkeypatch):
    rec,model,_=_try(monkeypatch,SPEEDUP,
        '{"answer":"CC-guided是248次，Random是530次，Overlap-biased是1301次。下面计算加速比。","citations":[1]}')
    assert rec['status']=='partial',rec['error']
    assert db.decode(rec['metrics'])['answer_complete'] is False


def test_exception_question_retains_both_table_and_experiment(monkeypatch):
    rec,model,queries=_try(monkeypatch,EXCEPTION,
        '{"answer":"Random为0.615，CC-guided为0.597。作者使用更强的GPT-5-mini重新生成临床测试查询，重复实验后CC-guided在所有指标上均超过Random。","citations":[1]}')
    assert rec['status']=='succeeded',rec['error']
    assert 'GPT-5-mini' in rec['answer']
    assert db.decode(rec['metrics'])['question_intent']=='explanation'
    assert '这一例外' in model.calls[-1][-1]['content']

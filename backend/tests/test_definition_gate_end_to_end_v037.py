"""P0-B v0.3.7: verify the same final-answer gate across modes and retries.

No production rules keyed on CC; NCI and EPI are fictional metrics. Source
contexts simulate mixed PDF retrieval with a direct definition at rank 6.
"""
import asyncio
import json
from uuid import uuid4

import pytest

from app import db
from app.agent.workflow import AgentRunner
from app.schemas import ChatRequest
from app.knowledge import retrieval
from app.skills.definition_grounding import (definition_source,
    prioritize_definition_evidence, definition_issue)

Q='什么是 NCI？'
DIRECT=('Novel Coverage Index (NCI) measures the fraction of distinct source '
        'records retrieved at least once across a test suite, out of all records '
        'in the indexed corpus.')
UNRELATED=[
    'Prior work discusses coverage and efficiency.',
    'Our context evaluation uses Exact Sentence Ratio = |Sext(q)| / |S(c(q))|.',
    'Embedding is a common information retrieval component.',
    'NCI is investigated in prior experiments, but this paragraph does not define it.',
    'Retrieval ranking and diversity are measured in separate sections.',
]


def evs(above=True):
    texts=[*UNRELATED, DIRECT, 'We report experiments.', 'Other metrics were also measured.']
    if not above:
        texts=[DIRECT, *UNRELATED]
    return [{'evidence_id':f'p:c{i}', 'paper_id':'p', 'chunk_id':f'p:c{i}',
             'section':f'Page {i+1}', 'page_numbers':[i+1], 'content':text,
             'paragraph_refs':[f'page{i+1}-p0']} for i,text in enumerate(texts)]


def test_false_anchor_from_other_metric_is_ignored():
    assert definition_source(Q,evs()[:5]) is None
    a=definition_source(Q,evs())
    assert a and a.index==6 and a.ratio and a.retrieved


def test_promote_hidden_definition_without_fabricating_content():
    original=evs()
    ordered,source=prioritize_definition_evidence(Q,original)
    assert source and source.index==1
    assert ordered[0] is original[5]
    assert sorted(x['evidence_id'] for x in ordered)==sorted(x['evidence_id'] for x in original)
    assert definition_source(Q,[ordered[0]]).source_excerpt in ordered[0]['content']


def test_source_already_in_first_five_keeps_numbering():
    original=evs(False)
    arranged,source=prioritize_definition_evidence(Q,original)
    assert source.index==1 and [id(i) for i in arranged]==[id(i) for i in original]


@pytest.mark.parametrize('answer', [
    'NCI = 3 [1]',
    'NCI 是未被检索的记录数量。[1]',
    'NCI 计算公式为 |Sext(q)| / |S(c(q))|，评估被检索的记录比例。[1]',
    'NCI 是用于衡量生成回答相关性的指标。[1]',
])
def test_real_definition_source_rejects_wrong_definition(answer):
    promoted,anchor=prioritize_definition_evidence(Q,evs())
    assert definition_issue(Q,answer,promoted,[1],anchor)


def test_not_found_subject_is_exposed():
    assert definition_source(Q,evs()[:5]) is None


class Model:
    def __init__(self,stream,chat_results=(),finish='stop'):
        self.text=stream
        self.chat_results=list(chat_results)
        self.last_stream_meta={'finish_reason':finish}
        self.calls=[]
    async def stream(self,messages,**kwargs):
        self.calls.append(('stream',messages))
        yield self.text
    async def chat(self,messages,**kwargs):
        self.calls.append(('chat',messages))
        assert kwargs.get('json_mode')
        result=self.chat_results.pop(0)
        return {'content':json.dumps({'answer':result},ensure_ascii=False),
                'latency_ms':1, 'usage':{}}


def run(monkeypatch,mode,model,evidence=None):
    paper,conv,run_id=[uuid4().hex for _ in range(3)]
    with db.connect() as con:
        con.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                    (paper,db.settings.owner_id,'Fictional index metric','qasper','ready',db.now()))
        con.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                    (conv,db.settings.owner_id,'test',db.now()))
        con.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) '
                    'VALUES(?,?,?,?,?,?,?)',(run_id,conv,'pending',mode,Q,db.encode([paper]),db.now()))
    e=[{**item,'paper_id':paper,'evidence_id':f'{paper}:c{i}',
        'chunk_id':f'{paper}:c{i}'} for i,item in enumerate(evidence or evs())]
    def fake_search(question,paper_ids,strategy,top_k,rerank,run_id):
        assert strategy==('dense' if mode=='basic' else 'hybrid')
        return {'evidences':e,'latency_ms':0.4}
    monkeypatch.setattr(retrieval,'search',fake_search)
    # In isolated mocks, local acronym fallback should not be invoked if the
    # evidence already has a direct definition; other tests exercise that path.
    asyncio.run(AgentRunner(model).run(run_id,ChatRequest(
        conversation_id=conv,question=Q,paper_ids=[paper],mode=mode,
        teaching=False,use_memory=False)))
    row=db.one('SELECT * FROM runs WHERE id=?',(run_id,))
    return row,db.decode(row['metrics']),db.decode(row['citations']),model


@pytest.mark.parametrize('mode',['basic','hybrid'])
def test_one_repair_after_promoted_source(monkeypatch,mode):
    model=Model('NCI = 3 [1]', ['NCI 衡量至少被检索一次的记录占总记录数的比例。[1]'])
    row,metrics,citations,model=run(monkeypatch,mode,model)
    assert row['status']=='succeeded',metrics
    assert metrics['definition_gate_status']=='source_found'
    assert metrics['definition_source_id'].endswith(':c5')
    assert metrics['definition_repair_succeeded']
    assert metrics['definition_final_check']=='passed'
    assert len(citations)==1 and citations[0]['page_numbers']==[6]
    assert 'NCI = 3' not in row['answer']
    assert '(NCI) measures the fraction' in citations[0]['quote']
    assert len(model.calls)==2


def test_length_retry_then_definition_repair_still_validated(monkeypatch):
    model=Model('首先，用户的问题是...我需要先检查论文。',[
        'NCI 是生成回答的准确率。[1]',
        'NCI 衡量至少检索一次的记录占全部记录的比例。[1]'], finish='length')
    row,metrics,citations,model=run(monkeypatch,'basic',model)
    assert row['status']=='succeeded',metrics
    assert metrics['answer_presentation_repaired']
    assert metrics['definition_repair_succeeded']
    assert [i['passed'] for i in metrics['definition_validation_attempts']]==[False,True]
    assert len(citations)==1
    assert len(model.calls)==3


def test_bad_twice_must_remain_partial(monkeypatch):
    model=Model('NCI 是未被检索到的部分比例。[1]',[
        'NCI 是生成模型回答的准确率。[1]'])
    row,metrics,citations,_=run(monkeypatch,'hybrid',model)
    assert row['status']=='partial'
    assert metrics['definition_abstained']
    assert metrics['answer_complete'] is False
    assert not citations
    assert '未被检索' not in row['answer']


def test_sound_uncited_repair_quotes_only_real_source(monkeypatch):
    model=Model('NCI = 3 [1]',[
        'NCI 衡量至少被检索一次的记录数占全语料记录总数的比例。'])
    row,metrics,citations,_=run(monkeypatch,'basic',model)
    assert row['status']=='succeeded',metrics
    assert metrics.get('definition_source_fallback')
    assert len(citations)==1
    assert DIRECT in ' '.join(citations[0]['quote'].split())
    assert 'retrieved at least once' in row['answer']


def test_definition_source_without_subject_definition_fails_closed(monkeypatch):
    evidence=evs()[:5]
    model=Model('NCI 是程序的准确率。[1]')
    monkeypatch.setattr(retrieval,'find_definition_evidence',lambda *args,**kwargs: [])
    row,metrics,citations,_=run(monkeypatch,'basic',model,evidence)
    assert row['status']=='partial'
    assert metrics['definition_gate_status']=='source_not_found'
    assert metrics['definition_abstained']
    assert not citations


def test_explicit_reference_to_unrelated_context_metric_is_not_accepted(monkeypatch):
    model=Model('NCI 是检索记录占整个语料的比例。公式为 Sext(q)/S(c(q))。[1]',[
        'NCI 是检索记录占整个语料的比例。[1]'])
    row,metrics,citations,_=run(monkeypatch,'hybrid',model)
    assert row['status']=='succeeded',metrics
    assert metrics['definition_validation_attempts'][0]['passed'] is False
    assert metrics['definition_validation_attempts'][1]['passed'] is True


def test_other_named_metric_is_generic_not_gold_key():
    e=[{'content':'Evaluation Passage Index (EPI) measures the fraction of '
                  'retrieved passages out of all passages in the corpus.'}]
    assert not definition_issue('什么是 EPI？',
                                'EPI 衡量已检索片段占全部片段的比例。[1]',e,[1])


def test_replayed_cc_failure_samples_are_rejected_without_production_gold():
    # Literal source sentence and actual examples from the browser acceptance
    # review; this data is test-only, never imported into the agent.
    actual=[
        {'content':'Background material: Context Relevance(q) = |Sext(q)| / |S(c(q))|.'},
        {'content':'CC makes uncovered regions visible; the rest of this paragraph discusses test plans.'},
        {'content':'CC measures the fraction of corpus chunks that are retrieved '
                   'at least once across a test suite, providing a structural view '
                   'of which parts of the retrieval space have been exercised.'},
    ]
    source=definition_source('什么是 CC？',actual)
    assert source and source.index==3 and source.ratio and source.retrieved
    bad_formula=('CC 用于评估检索上下文对生成响应的必要性，计算公式为 '
                 '|Sext(q)| / |S(c(q))|，指已检索的文档片段比例。[3]')
    wrong_direction='CC 评估未被测试覆盖的文档块占比。[3]'
    correct='CC 衡量测试套件中至少检索到一次的不同文档块占总语料库块数的比例。[3]'
    assert definition_issue('什么是 CC？',bad_formula,actual,[3],source)
    assert definition_issue('什么是 CC？',wrong_direction,actual,[3],source)
    assert not definition_issue('什么是 CC？',correct,actual,[3],source)


def test_irrelevant_fraction_not_confused_with_retrieval_coverage():
    ev=[{'content': 'Retrieval Coverage Index (RCI) measures the fraction of corpus '
                    'documents that were retrieved at least once in a test suite.'}]
    assert definition_issue(
        '什么是 RCI？',
        'RCI 是检索到的无关文档占比。[1]',ev,[1])
    assert not definition_issue(
        '什么是 RCI？',
        'RCI 是测试时检索到的文档占总文档数量的比例。[1]',ev,[1])

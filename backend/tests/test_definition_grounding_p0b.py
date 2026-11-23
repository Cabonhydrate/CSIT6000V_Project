"""Source-driven P0-B definition and citation tests (no actual paper Gold)."""
import asyncio
from uuid import uuid4

import pytest
from app import db
from app.agent.workflow import AgentRunner
from app.schemas import ChatRequest
from app.knowledge import retrieval
from app.skills.definition_grounding import definition_issue, definition_source


Q = '什么是 NCI？'
SOURCE = ('Novel Coverage Index (NCI) measures the fraction of distinct '
          'source records retrieved at least once by tests, out of all records '
          'in the indexed corpus.')
E = [
    {'evidence_id': 'p:c0', 'content': 'Previous work studies retrieval systems and vector indices.', 'section':'Intro', 'page_numbers':[1]},
    {'evidence_id': 'p:c1', 'content': SOURCE, 'section':'Definition', 'page_numbers':[2]},
]


def test_source_anchor_is_extracted_from_current_paper_not_answer_key():
    match = definition_source(Q,E)
    assert match.index == 2
    assert match.subject == 'NCI'
    assert match.ratio and match.retrieved
    assert SOURCE in match.source_excerpt


@pytest.mark.parametrize('text,fragment', [
    ('NCI=3 [2]', '比例/分数'),
    ('NCI 指未被检索到的文档片段比例。[2]', '未检索'),
    ('NCI 是未被实际使用的文档片段占比。[2]', '未检索'),
    ('NCI 衡量被检索的文档比例。[1]', '引用没有包含'),
])
def test_reject_inverse_dimensions_and_unrelated_citation(text,fragment):
    citation = [int(x) for x in __import__('re').findall(r'\[(\d+)\]',text)]
    assert fragment in definition_issue(Q,text,E,citation)


@pytest.mark.parametrize('answer', [
    'NCI 表示被至少一次检索到的记录占全部记录的比例。[2]',
    'NCI = 3/10，表示命中的记录数除以总记录数。[2]',
    'NCI 是已检索到的记录占比，并不表示那些未被检索的记录比例。[2]',
])
def test_accept_true_definition_and_valid_fraction(answer):
    assert not definition_issue(Q,answer,E,[2])


def test_no_source_definition_does_not_invent_one():
    unrelated = [{'content':'Some experimental results are shown in Table 2.'}]
    assert definition_source(Q,unrelated) is None
    assert definition_issue(Q,'NCI=3 [1]',unrelated,[1]) == ''


def test_other_metric_from_another_paper_is_checked_without_special_rules():
    q = '什么是 RS？'
    ev = [{'content': ('Recall Share (RS) measures the proportion of unique '
                        'retrieved passages over the full passage inventory.')}]
    assert definition_issue(q,'RS=7 [1]',ev,[1])
    assert not definition_issue(q,'RS 是已检索片段占语料库全部片段的比例。[1]',ev,[1])


def test_named_non_ratio_algorithm_is_not_forced_into_percentage():
    q = '什么是 BM？'
    ev = [{'content': 'Branch Method (BM) is the algorithm used for search.'}]
    assert not definition_issue(q,'BM 是一种搜索算法。[1]',ev,[1])


class WrongThenCorrect:
    def __init__(self,wrong,repaired):
        self.wrong = wrong
        self.repaired = repaired
        self.chat_calls = 0
        self.last_stream_meta = {'finish_reason':'stop'}
    async def stream(self,messages,**kwargs):
        yield self.wrong
    async def chat(self,messages,**kwargs):
        self.chat_calls+=1
        assert kwargs.get('json_mode') is True
        assert '可靠的论文原文摘录' in messages[-1]['content']
        assert SOURCE in messages[-1]['content']
        return {'content':__import__('json').dumps({'answer':self.repaired},ensure_ascii=False),
                'latency_ms':1,'usage':{}}


def run_case(monkeypatch,mode,wrong,repaired):
    paper,conv,run = [uuid4().hex for _ in range(3)]
    with db.connect() as c:
        c.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                  (paper,db.settings.owner_id,'Independent research','qasper','ready',db.now()))
        for i,content in enumerate([E[0]['content'],SOURCE]):
            c.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                      (f'{paper}:c{i}',paper,i,f'Section {i+1}',content,
                       db.encode([f's{i}-p0']),db.encode([i+1])))
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                  (conv,db.settings.owner_id,'test',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) '
                  'VALUES(?,?,?,?,?,?,?)',(run,conv,'pending',mode,Q,db.encode([paper]),db.now()))

    async def fake(q, paper_ids, strategy, top_k, rerank, run_id):
        assert strategy == ('dense' if mode=='basic' else 'hybrid')
        return {'evidences':[{
            'evidence_id':f'{paper}:c{i}', 'chunk_id':f'{paper}:c{i}',
            'paper_id':paper,'content':content,'section':f'Section {i+1}',
            'page_numbers':[i+1], 'paragraph_refs':[f's{i}-p0'],
        } for i,content in enumerate([E[0]['content'],SOURCE])], 'latency_ms':1}
    monkeypatch.setattr(retrieval,'search', lambda *args: asyncio.run(fake(*args)))
    # Basic/Hybrid: the stream uses the model wrong answer; only invalid
    # definitions trigger one JSON repair call.
    model=WrongThenCorrect(wrong,repaired)
    asyncio.run(AgentRunner(model).run(run,ChatRequest(conversation_id=conv,
        question=Q,paper_ids=[paper],mode=mode,teaching=False,use_memory=False)))
    row=db.one('SELECT * FROM runs WHERE id=?',(run,))
    return row,model,db.decode(row['metrics']),db.decode(row['citations'])


@pytest.mark.parametrize('mode', ['basic','hybrid'])
def test_real_workflow_repairs_bare_number_and_rebinds_real_citation(monkeypatch,mode):
    row,model,metrics,citations=run_case(monkeypatch,mode,
        'NCI=3。[1]', 'NCI 衡量至少一次检索到的不同记录占全语料记录的比例。[2]')
    assert row['status']=='succeeded',row.get('error')
    assert model.chat_calls == 1
    assert metrics['definition_repair_succeeded'] is True
    assert [s['passed'] for s in metrics['definition_validation_attempts']] == [False,True]
    assert len(citations)==1 and citations[0]['page_numbers']==[2]
    assert '比例' in row['answer'] and 'NCI=3' not in row['answer']


def test_real_workflow_rejects_twice_wrong_without_false_success(monkeypatch):
    row,model,metrics,citations=run_case(monkeypatch,'basic',
        'NCI=3。[2]', 'NCI 是未被使用的记录比例。[2]')
    assert row['status']=='partial',row.get('error')
    assert metrics['definition_abstained'] is True
    assert citations==[]
    assert 'NCI=3' not in row['answer'] and '未被使用' not in row['answer']


def test_real_workflow_accepts_valid_without_model_repair(monkeypatch):
    row,model,metrics,citations=run_case(monkeypatch,'basic',
        'NCI 衡量被检索的不同记录占全部记录的比例。[2]', '')
    assert row['status']=='succeeded',row.get('error')
    assert model.chat_calls==0
    assert metrics['definition_source_verified'] is True
    assert len(citations)==1


def test_real_cc_paper_excerpt_detects_the_two_previous_bad_answers():
    # A short literal excerpt from the paper, used only in regression data.
    # The production checker never contains CC or this answer.
    excerpt = ({'content': 'We introduce Chunk Coverage (CC), a structural '
               'adequacy criterion for RAG systems. Given a test suite T, '
               'CC measures the proportion of chunks in the corpus that are '
               'retrieved at least once across all test executions.'})
    refs = [excerpt]
    assert '未检索' in definition_issue(
        '什么是CC？','CC是上下文片段中未被实际使用的部分。[1]',refs,[1])
    assert '比例' in definition_issue('什么是CC？','CC=3。[1]',refs,[1])
    assert definition_issue(
        '什么是CC？','CC衡量至少被检索一次的不同Chunk占总Chunk数量的比例。[1]',refs,[1]) == ''

"""Regression tests: multiple explicit questions must not collapse to acronym-only path."""
import asyncio
import sys
import types
from uuid import uuid4
from app import db
from app.agent.workflow import (
    AgentRunner,_explicit_acronym_answer,_simple_definition,
    _split_question_parts,_render_multi_answers,
)
from app.schemas import ChatRequest


def _fake_langgraph(monkeypatch):
    graph=types.ModuleType('langgraph.graph')
    graph.START='START'; graph.END='END'
    class Graph:
        def __init__(self,*args): self.nodes={}; self.router=None
        def add_node(self,name,fn): self.nodes[name]=fn
        def add_edge(self,*args): pass
        def add_conditional_edges(self,source,router,paths): self.router=router
        def compile(self,checkpointer=None): return self
        async def ainvoke(self,state,config=None):
            for _ in range(4):
                for node in ('plan','retrieve','assess'):
                    state.update(await self.nodes[node](state))
                if self.router(state)=='end': break
            return state
    graph.StateGraph=Graph
    monkeypatch.setitem(sys.modules,'langgraph',types.ModuleType('langgraph'))
    monkeypatch.setitem(sys.modules,'langgraph.graph',graph)
    monkeypatch.setitem(sys.modules,'langgraph.checkpoint',types.ModuleType('langgraph.checkpoint'))
    saver=types.ModuleType('langgraph.checkpoint.memory'); saver.InMemorySaver=lambda: object()
    monkeypatch.setitem(sys.modules,'langgraph.checkpoint.memory',saver)


def _seed(question):
    paper_id=uuid4().hex; conv=uuid4().hex; run=uuid4().hex
    with db.connect() as conn:
        conn.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                  (paper_id,db.settings.owner_id,'Chunk Coverage study','pdf','ready',db.now()))
        rows=[
            ('Abstract','We introduce Chunk Coverage (CC), the fraction of distinct corpus chunks retrieved at least once by a test suite.',[1]),
            ('Method','CC-guided test selection prioritizes candidate test queries that retrieve previously uncovered chunks, aiming to expand coverage.',[7]),
            ('Discussion','The paper compares CC-guided tests against random test selection for retrieval test effectiveness.',[14]),
        ]
        for seq,(section,content,pages) in enumerate(rows):
            conn.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                         (paper_id+f':c{seq}',paper_id,seq,section,content,db.encode([f'p{seq}']),db.encode(pages)))
        conn.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                     (conv,db.settings.owner_id,'Demo',db.now()))
        conn.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                     (run,conv,'pending','adaptive',question,db.encode([paper_id]),db.now()))
    return paper_id,conv,run


class Model:
    def __init__(self,answer_second=True):
        self.calls=0
        self.answer_second=answer_second
        self.prompts=[]
    async def chat(self,messages,**kwargs):
        self.calls+=1
        assert kwargs.get('json_mode') is True
        prompt=messages[-1]['content']
        self.prompts.append(prompt)
        assert 'CC-guided test是什么' in prompt
        assert 'CC-guided test selection prioritizes candidate test queries' in prompt
        if self.answer_second:
            result='{"answer":"CC-guided test 通过优先选择能检索到此前未覆盖 Chunk 的问题来扩大覆盖范围。","citations":[2]}'
        else:
            result='{"answer":"","citations":[]}'
        return {'content':result,'latency_ms':3,'usage':{}}
    async def stream(self,*args,**kwargs):
        raise AssertionError('Structured multi-question answers must not free-stream')
        yield ''


def test_compound_definition_not_shortcut():
    full='文中的CC是什么意思？CC-guided test是什么？'
    assert _split_question_parts(full)==['文中的CC是什么意思','CC-guided test是什么']
    assert not _simple_definition(full)
    passage=[{'content':'Chunk Coverage (CC) measures coverage of retrieval tests.'}]
    assert _explicit_acronym_answer(full,passage) is None
    assert _explicit_acronym_answer('CC-guided test是什么？',passage) is None
    assert _explicit_acronym_answer('CC是什么？',passage) is not None


def test_compound_questions_get_two_cited_answers(monkeypatch):
    _fake_langgraph(monkeypatch)
    question='文中的CC是什么意思？CC-guided test是什么？'
    paper,conv,run=_seed(question)
    model=Model()
    asyncio.run(AgentRunner(model).run(run,ChatRequest(conversation_id=conv,question=question,
                                             paper_ids=[paper],mode='adaptive',teaching=False)))
    rec=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert rec['status']=='succeeded',rec['error']
    # First question is a source-verifiable abbreviation; second needs one model call.
    assert model.calls==1
    assert 'CC-guided test' in rec['answer'] and 'Chunk Coverage' in rec['answer']
    assert '1. 文中的CC是什么意思' in rec['answer']
    assert '2. CC-guided test是什么' in rec['answer']
    metrics=db.decode(rec['metrics'])
    assert metrics['subquestions_total']==2 and metrics['subquestions_answered']==2
    assert metrics['answer_complete'] is True
    assert metrics['retrieval_rounds']==2
    citations=db.decode(rec['citations'])
    assert len(citations)==2
    assert citations[0]['chunk_id'].endswith(':c0')
    assert citations[1]['chunk_id'].endswith(':c1')


def test_missing_second_answer_is_partial(monkeypatch):
    _fake_langgraph(monkeypatch)
    question='文中的CC是什么意思？CC-guided test是什么？'
    paper,conv,run=_seed(question)
    model=Model(answer_second=False)
    asyncio.run(AgentRunner(model).run(run,ChatRequest(conversation_id=conv,question=question,
                                             paper_ids=[paper],mode='adaptive',teaching=False)))
    rec=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert rec['status']=='partial'
    assert model.calls==2
    assert rec['answer'].count('当前检索证据不足')==1
    metrics=db.decode(rec['metrics'])
    assert metrics['answer_complete'] is False and metrics['subquestions_answered']==1
    events=db.rows('SELECT event_type,payload FROM run_events WHERE run_id=? ORDER BY id',(run,))
    assert any(e['event_type']=='run_completed' and db.decode(e['payload'])['status']=='partial'
               for e in events)


def test_scoped_answers_keep_later_question_citations():
    from app.agent.workflow import _render_scoped_answers
    # This reproduces the production bug: passage c2 appears SIXTH in the global
    # merged evidence pool, so old evidence[:5] removed it before generation.
    e1=[{'evidence_id':f'c{i}','content':f'Question one passage {i}', 'chunk_id':f'c{i}'} for i in range(4)]
    e2=[{'evidence_id':'c14','content':'Test comparison','chunk_id':'c14'},
        {'evidence_id':'c2','content':'Coverage-guided test selection retrieves new chunks','chunk_id':'c2'}]
    reply,cites,answered=_render_scoped_answers(
        ['What is CC?', 'What are CC-guided tests?'],
        [{'answer':'CC means Chunk Coverage', 'citations':[1]},
         {'answer':'CC-guided tests prioritize questions reaching uncovered chunks.', 'citations':[2]}],
        [e1,e2],language='en')
    assert answered==2
    assert '[2]' in reply
    assert [c['evidence_id'] for c in cites]==['c0','c2']


def test_repeated_answer_does_not_count_as_two_completions():
    evidence=[{'evidence_id':'1','content':'A'},{'evidence_id':'2','content':'B'}]
    text,citations,complete=_render_multi_answers(['CC 是什么','CC-guided test是什么'],
        {'answers':[{'id':1,'answer':'CC 是 Chunk Coverage','citations':[1]},
                    {'id':2,'answer':'CC 是 Chunk Coverage','citations':[2]}]},evidence,'zh')
    assert complete==1 and len(citations)==1
    assert '现有论文证据不足' in text


def test_long_chunk_excerpt_finds_relevant_sentence():
    from app.agent.workflow import _answer_evidence
    long='Unrelated introductory discussion. '*100
    long+=' CC-guided tests select queries that reach previously uncovered chunks.'
    source=[{'section':'Method','content':long,'page_numbers':[4]}]
    excerpt=_answer_evidence(source,chars=400,focus_query='CC-guided tests是什么')
    assert 'CC-guided tests select queries' in excerpt
    assert '第4页' in excerpt


def test_beginner_multi_question_uses_explanation_without_losing_second_part(monkeypatch):
    _fake_langgraph(monkeypatch)
    question='文中的CC是什么意思？CC-guided test是什么？'
    paper,conv,run=_seed(question)
    class TeachingModel(Model):
        async def chat(self,messages,**kwargs):
            if 'Verified expansion' in messages[-1]['content']:
                self.calls+=1
                return {'content':'{"explanation":"表示被测试问题检索到的不同文本块比例。","paper_role":"衡量检索测试覆盖范围。","example":"100个块中检索到60个，覆盖率是60%。"}',
                        'latency_ms':3}
            return await super().chat(messages,**kwargs)
    model=TeachingModel()
    asyncio.run(AgentRunner(model).run(run,ChatRequest(conversation_id=conv,question=question,
                                             paper_ids=[paper],mode='adaptive',teaching=True)))
    rec=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert rec['status']=='succeeded'
    assert model.calls==2
    assert '举个例子' in rec['answer']
    assert 'CC-guided test 通过优先选择' in rec['answer']

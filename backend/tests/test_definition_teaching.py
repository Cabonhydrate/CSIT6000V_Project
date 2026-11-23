"""The verified acronym is factual; teaching details are optional and source-constrained."""
import asyncio
import sys
import types
from uuid import uuid4
from app import db
from app.agent.workflow import (
    AgentRunner, _definition_teaching_messages, _render_definition_teaching,
)
from app.schemas import ChatRequest


def test_teaching_render_keeps_source_verified_definition():
    base='论文中的 CC 是 Chunk Coverage 的缩写。[1]'
    filled, ok=_render_definition_teaching(base,1,{
        'explanation':'它表示测试问题覆盖到多少个不同的文档块。',
        'paper_role':'用于检查检索测试是否触及足够多的文档区域。',
        'example':'如果有 100 个块，其中 60 个至少被检索一次，覆盖率就是 60%。',
    })
    assert ok and filled.startswith(base)
    assert filled.count('[1]')==3
    assert '通俗理解' in filled
    assert '举个例子（假设）' in filled
    assert '在这篇论文中' in filled
    assert '60%' in filled


def test_teaching_prompt_contains_verified_source_and_language():
    excerpt={'content':'Chunk Coverage (CC) measures the fraction of corpus chunks retrieved at least once.',
             'section':'Introduction','page_numbers':[1]}
    messages=_definition_teaching_messages('CC 是什么意思？','CC','Chunk Coverage',[excerpt],1,'zh','beginner')
    assert len(messages)==2
    assert 'JSON' in messages[0]['content']
    assert 'Chunk Coverage (CC)' in messages[1]['content']
    assert 'Source passage' in messages[1]['content']
    assert '举' in messages[0]['content'] or '例' in messages[0]['content']


def test_instruction_echo_falls_back_to_verified_definition():
    base='论文中的 CC 是 Chunk Coverage 的缩写。[1]'
    output, enriched=_render_definition_teaching(base,1,{
        'explanation':'首先，用户的问题是 CC 是什么意思？回顾用户指令：回答中文。',
        'paper_role':'', 'example':''})
    assert not enriched
    assert output==base


def _fake_langgraph(monkeypatch):
    # Simulate the public graph shape solely to test the adaptive orchestration
    # branch without downloading langgraph into an offline test environment.
    graph=types.ModuleType('langgraph.graph')
    graph.START='START';graph.END='END'
    class StateGraph:
        def __init__(self,*args):self.nodes={};self.route=None
        def add_node(self,name,fn):self.nodes[name]=fn
        def add_edge(self,*args):pass
        def add_conditional_edges(self,source,route,paths):self.route=route
        def compile(self,checkpointer=None):return self
        async def ainvoke(self,state,config=None):
            for _ in range(4):
                state.update(await self.nodes['plan'](state))
                state.update(await self.nodes['retrieve'](state))
                state.update(await self.nodes['assess'](state))
                if self.route(state)=='end':break
            return state
    graph.StateGraph=StateGraph
    checkpoint=types.ModuleType('langgraph.checkpoint.memory')
    checkpoint.InMemorySaver=lambda:object()
    monkeypatch.setitem(sys.modules,'langgraph',types.ModuleType('langgraph'))
    monkeypatch.setitem(sys.modules,'langgraph.graph',graph)
    monkeypatch.setitem(sys.modules,'langgraph.checkpoint',types.ModuleType('langgraph.checkpoint'))
    monkeypatch.setitem(sys.modules,'langgraph.checkpoint.memory',checkpoint)


class TeachingModel:
    def __init__(self,content):self.content=content;self.calls=[]
    async def chat(self,messages,**kwargs):
        self.calls.append((messages,kwargs))
        return {'content':self.content,'latency_ms':2,'usage':{}}
    async def stream(self,*args,**kwargs):
        raise AssertionError('Definition path must not invoke free-form streaming output')
        yield ''


def _run_definition(model,teaching):
    db.execute("UPDATE profile SET language='zh', explanation_level='beginner' WHERE owner_id=?",
               (db.settings.owner_id,))
    paper_id=uuid4().hex; conv=uuid4().hex;run=uuid4().hex
    with db.connect() as c:
        c.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                  (paper_id,db.settings.owner_id,'Demo','pdf','ready',db.now()))
        c.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                  (paper_id+':c0',paper_id,0,'Abstract',
                   'Chunk Coverage (CC) is a metric. CC measures the fraction of corpus chunks '
                   'retrieved at least once across a test suite.',db.encode(['abstract-0']),db.encode([1])))
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                  (conv,db.settings.owner_id,'Demo',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                  (run,conv,'pending','adaptive','文中的 CC 是什么意思？',db.encode([paper_id]),db.now()))
    asyncio.run(AgentRunner(model).run(run,ChatRequest(conversation_id=conv,
        question='文中的 CC 是什么意思？',paper_ids=[paper_id],mode='adaptive',teaching=teaching)))
    return db.one('SELECT * FROM runs WHERE id=?',(run,))


def test_adaptive_beginner_teaching_uses_one_structured_call(monkeypatch):
    _fake_langgraph(monkeypatch)
    m=TeachingModel('{"explanation":"在这组测试问题中，至少检索过一次的文档块所占的比例。",'
                    '"paper_role":"衡量 RAG 检索测试的覆盖程度。",'
                    '"example":"100 个块中有 60 个被至少检索到一次，CC 就是 60%。"}')
    run=_run_definition(m,True)
    assert run['status']=='succeeded',run['error']
    assert len(m.calls)==1
    assert m.calls[0][1]['json_mode'] is True
    assert '举个例子' in run['answer']
    assert run['answer'].startswith('论文中的 CC 是 Chunk Coverage 的缩写。[1]')
    assert db.decode(run['metrics'])['definition_enriched'] is True
    assert len(db.decode(run['citations']))==1


def test_adaptive_without_beginner_skill_is_brief_and_zero_llm(monkeypatch):
    _fake_langgraph(monkeypatch)
    m=TeachingModel('{}')
    run=_run_definition(m,False)
    assert run['status']=='succeeded',run['error']
    assert run['answer']=='论文中的 CC 是 Chunk Coverage 的缩写。[1]'
    assert not m.calls
    assert db.decode(run['metrics'])['llm_calls']==0


def test_bad_teaching_output_still_returns_grounded_definition(monkeypatch):
    _fake_langgraph(monkeypatch)
    m=TeachingModel('{"explanation":"回顾用户指令：不得重复问题", "paper_role":"", "example":""}')
    run=_run_definition(m,True)
    assert run['status']=='succeeded',run['error']
    assert '回顾用户' not in run['answer']
    assert db.decode(run['metrics'])['definition_enriched'] is False

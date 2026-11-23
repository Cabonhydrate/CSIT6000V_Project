"""Day 1: personalization reaches real answer prompts, not just database CRUD."""
import asyncio
import sys
import types
from uuid import uuid4

from app import db
from app.agent.learning_context import choose_learning_memories, make_learning_context
from app.agent.workflow import AgentRunner
from app.schemas import ChatRequest
from app.knowledge import retrieval


QUESTION = ('请根据我的学习背景，解释论文中的 Chunk Coverage。'
            '重点帮助我理解尚不熟悉的部分，并给一个简单的例子。')
NOTE = '我知道 RAG 基本流程，但不熟悉 Chunk Coverage 和集合并集，请先举一个文档块的例子。'


def fake_langgraph(monkeypatch):
    graph=types.ModuleType('langgraph.graph')
    graph.START='START'; graph.END='END'
    class StateGraph:
        def __init__(self,*args):self.nodes={};self.route=None
        def add_node(self,name,fn):self.nodes[name]=fn
        def add_edge(self,*args):pass
        def add_conditional_edges(self,source,route,paths):self.route=route
        def compile(self,checkpointer=None):return self
        async def ainvoke(self,state,config=None):
            for _ in range(3):
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


class CaptureModel:
    def __init__(self):self.chat_inputs=[];self.stream_inputs=[];self.last_stream_meta={}
    async def chat(self,messages,**kwargs):
        self.chat_inputs.append(messages)
        system=messages[0]['content']
        if 'distinctive' in system:return {'content':'{"focus_term":"Chunk Coverage"}','latency_ms':1}
        if 'Judge if the cited passages' in system:
            return {'content':'{"sufficient":true,"missing":""}','latency_ms':1}
        if 'definition' in system and 'example' in system:
            return {'content': (
                '{"definition":"Chunk Coverage 是测试查询覆盖到的不同文档片段数量占全部文档片段数量的比例。",'
                '"example":"某篇论文被划分成5个文档块，检索多次共命中3个不同块，覆盖率是3/5=60%。",'
                '"takeaway":"重复命中同一个文档块只计一次，它衡量检索覆盖而不是答案准确率。",'
                '"citations":[1]}'), 'latency_ms':1}
        return {'content':'{"answer":"Chunk Coverage 计算被检索到的不同文档片段所占比例。[1]","citations":[1]}','latency_ms':1}
    async def stream(self,messages,**kwargs):
        self.stream_inputs.append(messages)
        yield 'Chunk Coverage 是所有查询检索到的不同文档片段占文档块总数的比例。举例：5 个块中检索到 3 个，CC=60%。[1]'


def prepare(monkeypatch):
    fake_langgraph(monkeypatch)
    paper_id=uuid4().hex
    db.execute('UPDATE profile SET language=?,explanation_level=?,domains=?,goals=?,memory_enabled=1 WHERE owner_id=?',
               ('zh','beginner',db.encode(['RAG']),db.encode(['理解研究论文']),db.settings.owner_id))
    with db.connect() as c:
        c.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                  (paper_id,db.settings.owner_id,'Paper','pdf','ready',db.now()))
        c.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                  (paper_id+':c0',paper_id,0,'Introduction',
                   'Chunk Coverage (CC) measures the fraction of distinct document chunks retrieved at least once by a suite of queries.',
                   db.encode(['p6-0']),db.encode([6])))
        c.execute('INSERT INTO memories(id,owner_id,domain,topic,memory_type,content,mastery,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',
                  (uuid4().hex,db.settings.owner_id,'RAG','Chunk Coverage','progress',NOTE,'studied',db.now(),db.now()))
    async def search(q,ids,strategy,top_k,rerank,run_id):
        return {'evidences':[{'evidence_id':paper_id+':c0','chunk_id':paper_id+':c0',
                'paper_id':paper_id,'content':'Chunk Coverage (CC) measures the fraction of distinct document chunks retrieved at least once by a suite of queries.',
                'section':'Introduction','page_numbers':[6],'paragraph_refs':['p6-0'],'source_type':'paper'}],
                'latency_ms':2}
    def search_sync(*args):return asyncio.run(search(*args))
    monkeypatch.setattr(retrieval,'search',search_sync)
    return paper_id


def do_run(paper_id,model,*,use_memory=True,mode='adaptive'):
    run_id=uuid4().hex
    conv=uuid4().hex
    with db.connect() as c:
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                  (conv,db.settings.owner_id,'New learning session',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                  (run_id,conv,'pending',mode,QUESTION,db.encode([paper_id]),db.now()))
    request=ChatRequest(conversation_id=conv,question=QUESTION,paper_ids=[paper_id],mode=mode,
                        teaching=True,use_memory=use_memory)
    asyncio.run(AgentRunner(model).run(run_id,request))
    rec=db.one('SELECT * FROM runs WHERE id=?',(run_id,))
    assert rec['status']=='succeeded',rec.get('error')
    return db.decode(rec['metrics'])


def test_learning_context_uses_mastery_without_claiming_studied_is_mastered():
    ctx=make_learning_context({'language':'zh','explanation_level':'beginner','domains':['RAG']},
                              [{'topic':'Chunk Coverage','content':NOTE,'mastery':'studied'}],include_memories=True)
    assert NOTE in ctx
    assert '学过（不代表掌握）' in ctx
    assert '不是论文证据' in ctx
    assert '不要把学习记录当作论文事实' in ctx


def test_topic_selection_prefers_relevant_and_background_question_can_recall():
    records=[{'topic':'Neural Networks','domain':'AI','content':'I studied backpropagation'},
             {'topic':'Chunk Coverage','domain':'RAG','content':NOTE}]
    assert choose_learning_memories('解释 Chunk Coverage',records,limit=1)[0]['topic']=='Chunk Coverage'
    assert len(choose_learning_memories('根据我的学习背景说明知识点',records))==2
    assert choose_learning_memories('什么是完全无关的商业报价',records)==[]


def test_memory_reaches_answer_generation_across_conversations(monkeypatch):
    paper=prepare(monkeypatch)
    for _ in range(2):
        model=CaptureModel()
        metrics=do_run(paper,model)
        generation_calls = [call for call in model.chat_inputs
                            if 'definition' in call[0]['content'] and 'example' in call[0]['content']]
        assert generation_calls
        final_prompt=generation_calls[0][-1]['content']
        assert NOTE in final_prompt
        assert '学过（不代表掌握）' in final_prompt
        assert '论文证据' in final_prompt
        assert 'distinct document chunks' in final_prompt
        assert metrics['learning_memory_candidates']==1
        assert metrics['learning_context_applied'] is True
        assert all(NOTE not in str(call) for call in model.chat_inputs
                   if 'definition' not in call[0]['content']), 'do not feed memory into retrieval/planning'


def test_memory_off_does_not_leak_into_answer_prompt(monkeypatch):
    paper=prepare(monkeypatch)
    m=CaptureModel()
    metrics=do_run(paper,m,use_memory=False)
    assert NOTE not in str(m.chat_inputs)
    assert NOTE not in str(m.stream_inputs)
    assert metrics['learning_memory_candidates']==0
    db.execute('UPDATE profile SET memory_enabled=0 WHERE owner_id=?',(db.settings.owner_id,))
    m=CaptureModel()
    metrics=do_run(paper,m,use_memory=True)
    assert NOTE not in str(m.chat_inputs)
    assert NOTE not in str(m.stream_inputs)
    assert metrics['learning_memory_candidates']==0


def test_baselines_never_receive_adaptive_learning_context(monkeypatch):
    paper=prepare(monkeypatch)
    for mode in ('basic','hybrid'):
        m=CaptureModel()
        metrics=do_run(paper,m,mode=mode)
        assert NOTE not in str(m.stream_inputs)
        assert metrics['learning_context_applied'] is False


def test_beginner_answer_reflection_is_repaired_not_displayed(monkeypatch):
    """A cited internal plan must not be accepted as a beginner explanation."""
    from app.agent.workflow import _answer_is_bad

    reflected = ('首先，用户是中文学习者，解释层次是beginner。'
                 '学习背景：已经了解RAG基本流程，但还不熟悉集合并集。'
                 '从摘录中提取Chunk Coverage的定义和关键点，先罗列所有来源。[1]')
    assert _answer_is_bad(reflected)
    assert not _answer_is_bad(
        'Chunk Coverage 计算不同检索文档块的占比。假设有5个块，'
        '多次查询合计命中3个不同块，CC=3/5=60%。重复命中不重复计数。[1]')

    paper = prepare(monkeypatch)

    class ReflectionThenAnswer(CaptureModel):
        def __init__(self):
            super().__init__()
            self.answers = 0

        async def stream(self, messages, **kwargs):
            self.stream_inputs.append(messages)
            yield reflected

        async def chat(self, messages, **kwargs):
            system = messages[0]['content']
            self.chat_inputs.append(messages)
            if 'distinctive' in system:
                return {'content': '{"focus_term":"Chunk Coverage"}', 'latency_ms': 1}
            if 'Judge if the cited passages' in system:
                return {'content': '{"sufficient":true,"missing":""}', 'latency_ms': 1}
            self.answers += 1
            if self.answers == 1:
                return {'content': '{"answer":"首先，用户是中文学习者，先罗列证据"}', 'latency_ms': 1}
            return {'content': (
                '{"definition":"Chunk Coverage 是多次查询检索到的不同文档块占全部文档块的比例。",'
                '"example":"假设一篇论文有5个块，两次查询合计命中3个不同块，CC=3/5=60%。",'
                '"takeaway":"同一块命中多次也只计数一次。","citations":[1]}'), 'latency_ms': 1}

    model = ReflectionThenAnswer()
    metrics = do_run(paper, model)
    assert model.answers == 2, 'one bounded structured repair should be enough'
    assert metrics.get('teaching_answer_repaired') is True
    assert '仅输出一个 JSON 对象' in str(model.chat_inputs) or 'JSON 对象' in str(model.chat_inputs)

"""v0.3.8: implicit similarity must compare both concepts with separate sources.

All sources/model outputs are simulated; production contains no CC Gold answer.
The paper definition deliberately sits at rank 6, behind irrelevant passages.
"""
import asyncio
import json
import sys
import types
from uuid import uuid4

import pytest

from app import db
from app.agent import workflow
from app.agent.workflow import AgentRunner, _question_intent
from app.schemas import ChatRequest
from app.skills.mixed_comparison import (implicit_background_comparison,
                                        external_source_matches, validate_comparison_fields)

QUESTION = 'NCI好像和测试中的code coverage有些类似'
DIRECT = ('Novel Coverage Index (NCI) measures the fraction of unique corpus records '
          'retrieved at least once across a test suite, out of all indexed records.')
EXTERNAL = {'title':'Code coverage','source_url':'https://en.wikipedia.org/wiki/Code_coverage',
            'content':('Code coverage is a measure of the degree to which the source '
                       'code of a program is executed when a particular test suite runs.'),
            'source_type':'external','provider':'Wikipedia'}
REPLY = {'paper':'NCI 衡量测试套件至少检索过一次的语料库记录占全部记录的比例。',
         'external':'代码覆盖率衡量测试执行过的程序源代码所占比例。',
         'similarity':'两者都通过覆盖比例反映测试触及了多少目标对象。',
         'difference':'不同的是前者统计被检索到的语料记录，后者统计被执行的程序代码。'}


def fake_graph(monkeypatch):
    graph=types.ModuleType('langgraph.graph')
    graph.START='START';graph.END='END'
    class Graph:
        def __init__(self,*args):self.nodes={};self.router=None
        def add_node(self,name,fn):self.nodes[name]=fn
        def add_edge(self,*args):pass
        def add_conditional_edges(self,src,router,paths):self.router=router
        def compile(self,checkpointer=None):return self
        async def ainvoke(self,state,config=None):
            for _ in range(5):
                for node in ('plan','retrieve','assess'):
                    state.update(await self.nodes[node](state))
                step=self.router(state)
                if step=='web':
                    state.update(await self.nodes['web'](state));return state
                if step=='end':return state
            raise AssertionError('Graph did not terminate')
    graph.StateGraph=Graph
    monkeypatch.setitem(sys.modules,'langgraph',types.ModuleType('langgraph'))
    monkeypatch.setitem(sys.modules,'langgraph.graph',graph)
    monkeypatch.setitem(sys.modules,'langgraph.checkpoint',types.ModuleType('langgraph.checkpoint'))
    saver=types.ModuleType('langgraph.checkpoint.memory');saver.InMemorySaver=lambda:object()
    monkeypatch.setitem(sys.modules,'langgraph.checkpoint.memory',saver)


def make_run(question=QUESTION,web=True):
    paper,conv,run=(uuid4().hex for _ in range(3))
    with db.connect() as conn:
        conn.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                     (paper,db.settings.owner_id,'Fictional metrics study','pdf','ready',db.now()))
        chunks=['The authors compared a small metric with standard test adequacy.',
                'Context precision is a separate quality property.',
                'The authors studied failure detection.',
                'An unrelated paragraph about vector indexes.',
                'A table describes response quality metrics.', DIRECT,
                'A conclusion describes testing workflows.',
                'An appendix discusses implementation.']
        for i,c in enumerate(chunks):
            conn.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) '
                         'VALUES(?,?,?,?,?,?,?)',
                         (paper+':c'+str(i),paper,i,'Page '+str(i+1),c,
                          db.encode(['page'+str(i+1)+'-p0']),db.encode([i+1])))
        conn.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                     (conv,db.settings.owner_id,'Mixed sources',db.now()))
        conn.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) '
                     'VALUES(?,?,?,?,?,?,?)',
                     (run,conv,'pending','adaptive',question,db.encode([paper]),db.now()))
    return paper,run,ChatRequest(conversation_id=conv,paper_ids=[paper],question=question,
                                 mode='adaptive',teaching=True,allow_web=web,use_memory=False)


class Model:
    def __init__(self,answer=None):
        self.reply=answer or REPLY
        self.prompts=[]
    async def chat(self,messages,**kwargs):
        self.prompts.append(messages)
        if 'Judge if the cited passages' in messages[0]['content']:
            return {'content':'{"sufficient":true,"missing":""}', 'latency_ms':1}
        return {'content':json.dumps(self.reply,ensure_ascii=False),'latency_ms':1}
    async def stream(self,*args,**kwargs):
        raise AssertionError('Implicit comparison must never use freeform streaming')
        yield ''


def setup(monkeypatch,found=(EXTERNAL,)):
    fake_graph(monkeypatch)
    # This historical regression intentionally checks the Chinese UI labels.
    db.execute('UPDATE profile SET language=? WHERE owner_id=?', ('zh',db.settings.owner_id))
    async def background(q,**kw):
        assert q=='code coverage'
        return list(found)
    async def no_academic(q):
        raise AssertionError('Do not use paper metadata search for background definitions')
    monkeypatch.setattr(workflow,'search_background',background)
    monkeypatch.setattr(workflow,'academic_search',no_academic)
    def paper_search(q,ids,strategy,k,rerank=False,run_id=None):
        rows=db.rows('SELECT * FROM chunks WHERE paper_id=? ORDER BY seq',(ids[0],))
        return {'latency_ms':1,'evidences':[
            {'evidence_id':r['id'],'paper_id':r['paper_id'],
             'chunk_id':r['id'],'source_type':'paper','section':r['section'],
             'page_numbers':db.decode(r['pages']),'paragraph_refs':db.decode(r['paragraph_refs']),
             'content':r['content']} for r in rows]}
    monkeypatch.setattr(workflow.retrieval,'search',paper_search)


def test_routing_implicit_similarity_not_definition():
    assert implicit_background_comparison(QUESTION)==('NCI','code coverage')
    assert _question_intent(QUESTION)=='comparison'
    assert _question_intent('NCI-guided testing 与 Random selection 有什么区别？')=='comparison'
    assert implicit_background_comparison('NCI-guided testing 与 Random selection 有什么区别？') is None
    assert implicit_background_comparison('请解释 NCI 并给一个类似例子') is None
    assert _question_intent('什么是 NCI？')=='definition'


def test_chinese_background_concept_and_source():
    assert implicit_background_comparison('NCI和代码覆盖率有点像') == ('NCI','代码覆盖率')
    assert implicit_background_comparison('NCI和代码覆盖率一样吗') == ('NCI','代码覆盖率')
    source = {'title':'代码覆盖率','source_url':'https://zh.wikipedia.org/wiki/代码覆盖率',
              'content':'代码覆盖率衡量测试中执行到的程序代码的比例。'}
    assert external_source_matches(source,'代码覆盖率')


def test_source_filter_rejects_unrelated_or_untrusted_web_result():
    assert external_source_matches(EXTERNAL,'code coverage')
    assert not external_source_matches({**EXTERNAL,'title':'Other','content':'A different field'},'code coverage')
    assert not external_source_matches({**EXTERNAL,'source_url':'https://evil.com/Code_coverage'},'code coverage')
    assert not external_source_matches({**EXTERNAL,'source_url':'javascript:alert(1)'},'code coverage')


def test_empty_comparison_fields_fails_closed():
    assert validate_comparison_fields({**REPLY,'difference':''})[1]
    assert validate_comparison_fields({**REPLY,'difference':'这两个一样相同'})[1]
    assert validate_comparison_fields(REPLY)[1]==''


def test_implicit_comparison_uses_two_distinct_real_sources(monkeypatch):
    setup(monkeypatch)
    paper,run,req=make_run()
    model=Model()
    asyncio.run(AgentRunner(model).run(run,req))
    row=db.one('SELECT * FROM runs WHERE id=?',(run,))
    metrics=db.decode(row['metrics'])
    cites=db.decode(row['citations'])
    assert row['status']=='succeeded',row['error']
    assert metrics['question_intent']=='comparison'
    assert metrics['retrieval_rounds']==1
    assert metrics['external_comparison_results']==1
    assert metrics['mixed_comparison_sources']==['paper','external']
    assert metrics['mixed_comparison_validation'][0]['passed'] is True
    assert '**相似点**' in row['answer'] and '**不同点**' in row['answer']
    assert '代码覆盖率' in row['answer'] and 'NCI' in row['answer']
    assert [c['source_type'] for c in cites]==['paper','external']
    assert cites[0]['paper_id']==paper and cites[0]['source_url'] is None
    assert cites[0]['page_numbers']==[6]
    assert 'NCI) measures the fraction' in cites[0]['quote']
    assert cites[1]['paper_id'] is None and 'wikipedia.org' in cites[1]['source_url']
    assert all('source code' not in c['quote'] for c in cites[:1])
    assert '论文术语：NCI' in model.prompts[-1][-1]['content']


def test_web_failure_does_not_misattribute_background_to_pdf(monkeypatch):
    setup(monkeypatch,found=())
    _,run,req=make_run()
    asyncio.run(AgentRunner(Model()).run(run,req))
    row=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert row['status']=='partial'
    assert db.decode(row['citations'])==[]
    assert '不能可靠判断' in row['answer']
    assert db.decode(row['metrics'])['mixed_comparison_issue']=='external_definition_not_found'


def test_missing_second_side_cannot_be_marked_succeeded(monkeypatch):
    setup(monkeypatch)
    bad={**REPLY,'difference':'','similarity':'本次只解释第一个术语。'}
    _,run,req=make_run()
    model=Model(answer=bad)
    asyncio.run(AgentRunner(model).run(run,req))
    row=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert row['status']=='partial'
    assert db.decode(row['citations'])==[]
    assert len(db.decode(row['metrics'])['mixed_comparison_validation'])==2


def test_reversed_paper_definition_is_not_laundered_by_valid_sources(monkeypatch):
    setup(monkeypatch)
    bad={**REPLY,'paper':'NCI 衡量测试套件未曾检索到的语料库记录占全部记录的比例。'}
    _,run,req=make_run()
    asyncio.run(AgentRunner(Model(answer=bad)).run(run,req))
    row=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert row['status']=='partial'
    assert db.decode(row['citations'])==[]


def test_general_learning_remains_independent(monkeypatch):
    assert implicit_background_comparison('我不是很了解 Q-learning，详细讲解') is None

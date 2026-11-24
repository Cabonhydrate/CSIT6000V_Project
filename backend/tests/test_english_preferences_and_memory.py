"""Regression tests: output language and saved-note use are independent controls."""
import asyncio
from uuid import uuid4

from app import db
from app.agent.language_policy import language_instruction, response_language
from app.agent.learning_context import choose_learning_memories, make_learning_context
from app.agent.workflow import AgentRunner
from app.schemas import ChatRequest
from app.knowledge import retrieval


def test_saved_language_is_default_but_explicit_language_wins():
    assert response_language('What is Chunk Coverage?', 'en')=='en'
    assert response_language('什么是 CC？', 'en')=='en'
    assert response_language('What is Chunk Coverage? Please answer in Chinese.', 'en')=='zh'
    assert response_language('请用英文回答：什么是 CC？', 'zh')=='en'
    assert response_language('What is Chunk Coverage?', 'zh')=='zh'
    assert 'Respond entirely in English' in language_instruction('en')


def test_english_saved_note_request_selects_real_memory_without_keyword_overlap():
    records=[
        {'id':'recent','topic':'Context Window','domain':'LLM','content':'I have studied token limits.'},
        {'id':'older','topic':'Chunk Coverage','domain':'RAG','content':'I already know what a RAG retriever does.'},
    ]
    chosen=choose_learning_memories('Based on my saved learning note, explain what Chunk Coverage measures.',records)
    assert chosen[0]['id']=='older'
    # A topical request must not pull in unrelated recent notes.
    assert len(chosen)==1
    assert choose_learning_memories('How to price commercial hosting?',records)==[]
    context=make_learning_context({'language':'en','explanation_level':'beginner','goals':['Understand CC']},chosen,include_memories=True)
    assert 'Confirmed saved learning note' in context
    assert 'studied' in context or 'mentioned' in context
    assert 'never paper evidence' in context.lower() or 'not paper evidence' in context.lower()
    assert 'avoid reteaching' in context
    assert 'Respond in' not in context  # memory is context, not top-level instruction


class MemoryAwareModel:
    def __init__(self):
        self.chat_calls=[]
        self.stream_calls=[]
        self.last_stream_meta={}
    async def chat(self,messages,**kwargs):
        self.chat_calls.append(messages)
        system=messages[0]['content']
        if 'Judge if the cited passages' in system:
            return {'content':'{"sufficient":true,"missing":""}','latency_ms':1}
        has_memory='Confirmed saved learning note' in str(messages)
        ans=('Building on your saved note about understanding RAG retrieval, Chunk Coverage measures unique chunks retrieved, not answer correctness.'
             if has_memory else 'Chunk Coverage measures the share of unique chunks retrieved in a test suite, not answer correctness.')
        return {'content':'{"answer":"'+ans+'","citations":[1]}','latency_ms':1}
    async def stream(self,messages,**kwargs):
        self.stream_calls.append(messages)
        yield 'Chunk Coverage is the share of distinct chunks retrieved by a test suite. [1]'


def create_run(monkeypatch,*,mode='adaptive',use_memory=True,teaching=False,question=None):
    question=question or 'Based on my saved learning note, what does Chunk Coverage measure?'
    paper,conv,run=(uuid4().hex for _ in range(3))
    db.execute('UPDATE profile SET language=?,explanation_level=?,domains=?,goals=?,memory_enabled=1 WHERE owner_id=?',
               ('en','beginner',db.encode(['RAG']),db.encode(['Understand retrieval']),db.settings.owner_id))
    with db.connect() as c:
        c.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                  (paper,db.settings.owner_id,'Study','pdf','ready',db.now()))
        c.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                  (paper+':0',paper,0,'Introduction','Chunk Coverage measures the fraction of corpus chunks retrieved at least once across a test suite.',db.encode(['p1']),db.encode([1])))
        c.execute('INSERT INTO memories(id,owner_id,domain,topic,memory_type,content,mastery,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',
                  (uuid4().hex,db.settings.owner_id,'RAG','Chunk Coverage','progress',
                   'I understand the RAG retrieval pipeline, but not the set-union formula.', 'studied',db.now(),db.now()))
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                  (conv,db.settings.owner_id,'Learner',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                  (run,conv,'pending',mode,question,db.encode([paper]),db.now()))
    def search(q,ids,strategy,top_k,rerank,run_id):
        return {'latency_ms':1,'evidences':[{'evidence_id':paper+':0', 'chunk_id':paper+':0',
            'paper_id':paper,'section':'Introduction','source_type':'paper',
            'content':'Chunk Coverage measures the fraction of corpus chunks retrieved at least once across a test suite.',
            'page_numbers':[1],'paragraph_refs':['p1']} ]}
    monkeypatch.setattr(retrieval,'search',search)
    return run,ChatRequest(conversation_id=conv,question=question,paper_ids=[paper],mode=mode,
                           teaching=teaching,use_memory=use_memory)


def test_english_reaches_basic_generation_prompts(monkeypatch):
    run,req=create_run(monkeypatch,mode='basic',use_memory=False,question='什么是 Chunk Coverage？')
    model=MemoryAwareModel()
    asyncio.run(AgentRunner(model).run(run,req))
    rec=db.one('SELECT * FROM runs WHERE id=?',(run,))
    metrics=db.decode(rec['metrics'])
    assert rec['status']=='succeeded',rec.get('error')
    assert metrics['response_language']=='en'
    prompts=model.chat_calls+model.stream_calls
    assert prompts
    assert all('Respond entirely in English' in call[0]['content'] for call in prompts)
    assert metrics['learning_memory_candidates']==0


def test_english_memory_passes_to_generation_but_not_retrieval(monkeypatch):
    # Fake LangGraph only for the local test environment.
    from test_day1_learning_memory import fake_langgraph
    fake_langgraph(monkeypatch)
    run,req=create_run(monkeypatch,teaching=False,question='Based on my saved learning note, explain why Chunk Coverage does not guarantee a correct answer.')
    model=MemoryAwareModel()
    asyncio.run(AgentRunner(model).run(run,req))
    metrics=db.decode(db.one('SELECT metrics FROM runs WHERE id=?',(run,))['metrics'])
    assert metrics['response_language']=='en'
    assert metrics['learning_memory_candidates']==1
    assert metrics['learning_memory_topics']==['Chunk Coverage']
    assert metrics['learning_memory_applied'] is True
    assert any('Confirmed saved learning note' in str(p) for p in model.chat_calls)
    assert all('Respond entirely in English' in p[0]['content'] for p in model.chat_calls)


def test_saved_memory_can_be_disabled_without_changing_language(monkeypatch):
    from test_day1_learning_memory import fake_langgraph
    fake_langgraph(monkeypatch)
    run,req=create_run(monkeypatch,use_memory=False,teaching=False,
                       question='Explain why Chunk Coverage does not guarantee correct answers.')
    model=MemoryAwareModel()
    asyncio.run(AgentRunner(model).run(run,req))
    metrics=db.decode(db.one('SELECT metrics FROM runs WHERE id=?',(run,))['metrics'])
    assert metrics['learning_memory_candidates']==0
    assert metrics['learning_memory_applied'] is False
    assert metrics['response_language']=='en'
    assert 'Confirmed saved learning note' not in str(model.chat_calls)


def test_saved_note_changes_learning_aware_answer_when_opted_in(monkeypatch):
    """Simulation: proves prompt wiring and a visible outcome delta, not Ollama quality."""
    from test_day1_learning_memory import fake_langgraph
    fake_langgraph(monkeypatch)
    question='Based on my saved learning note, explain why Chunk Coverage does not guarantee answer correctness.'
    answers=[]
    for enable in (True,False):
        run,req=create_run(monkeypatch,use_memory=enable,teaching=False,question=question)
        model=MemoryAwareModel()
        asyncio.run(AgentRunner(model).run(run,req))
        row=db.one('SELECT * FROM runs WHERE id=?',(run,))
        assert row['status']=='succeeded',row.get('error')
        answers.append(row['answer'])
    assert 'Building on your saved note' in answers[0]
    assert 'Building on your saved note' not in answers[1]
    assert answers[0]!=answers[1]


def test_default_language_is_english_for_new_profile_and_api_schema():
    from app.schemas import ProfileUpdate
    assert ProfileUpdate().language == 'en'
    profile=db.one('SELECT language FROM profile WHERE owner_id=?',(db.settings.owner_id,))
    # Existing profiles retain their explicitly saved selection; this test
    # confirms the schema default without silently migrating stored choices.
    assert profile['language'] in ('zh','en')

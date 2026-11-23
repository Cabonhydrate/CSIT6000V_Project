"""Regression tests for Nov-20 instruction-echo defect in Qwen3:4b answers."""
import asyncio
from uuid import uuid4
import numpy as np
from app import db
from app.agent.workflow import AgentRunner, _answer_is_bad, _simple_definition, _explicit_acronym_answer
from app.knowledge import retrieval
from app.schemas import ChatRequest


def test_explicit_acronym_expansion_is_source_grounded():
    assert retrieval.acronym_in_question('文中的 CC 是什么意思？') == ['CC']
    assert retrieval.explicit_expansion('We introduce Chunk Coverage (CC) as our metric.', 'CC') == 'Chunk Coverage'
    assert retrieval.explicit_expansion('Mismatching CC value in this sentence', 'CC') is None
    assert _simple_definition('文中的 CC 是什么意思？')


def test_definition_lookup_uses_paper_chunk_not_gold_labels():
    paper_id=uuid4().hex
    with db.connect() as c:
        c.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                  (paper_id,db.settings.owner_id,'Demo','qasper','ready',db.now()))
        c.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                  (paper_id+':c0',paper_id,0,'Introduction',
                   'The authors propose Chunk Coverage (CC), measuring tested retrieval coverage.',
                   db.encode(['intro-0']),db.encode([])))
    evidence=retrieval.find_definition_evidence('CC 是什么意思？',[paper_id])
    assert len(evidence)==1
    assert evidence[0]['chunk_id']==paper_id+':c0'
    assert evidence[0]['paragraph_refs']==['intro-0']


def test_prompt_echo_is_not_emitted_and_is_repaired(monkeypatch):
    class EchoingModel:
        last_stream_meta={'finish_reason':'length'}
        def __init__(self):
            self.chat_calls=0
        async def stream(self,messages,**kwargs):
            yield '首先，用户的问题是：CC 是什么意思？回顾用户指令：列出提示词。'
        async def chat(self,messages,**kwargs):
            self.chat_calls+=1
            assert kwargs['json_mode'] is True
            return {'content':'{"answer":"CC 是 Chunk Coverage（块覆盖率）的缩写。[1]"}',
                    'usage':{},'latency_ms':1}
    monkeypatch.setattr(retrieval,'_dense_scores',lambda query,chunks:np.ones(len(chunks),dtype='float32'))
    paper_id=uuid4().hex
    conv=uuid4().hex
    run=uuid4().hex
    with db.connect() as c:
        c.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                  (paper_id,db.settings.owner_id,'Demo','qasper','ready',db.now()))
        c.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) VALUES(?,?,?,?,?,?,?)',
                  (paper_id+':c0',paper_id,0,'Intro','Chunk Coverage (CC) measures retrieval test coverage.',
                   db.encode(['intro-0']),db.encode([])))
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                  (conv,db.settings.owner_id,'Demo',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                  (run,conv,'pending','basic','CC 是什么意思？',db.encode([paper_id]),db.now()))
    model=EchoingModel()
    asyncio.run(AgentRunner(model).run(run,ChatRequest(
        conversation_id=conv,question='CC 是什么意思？',paper_ids=[paper_id],mode='basic')))
    output=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert output['status']=='succeeded',output['error']
    assert model.chat_calls==1
    assert '用户的问题' not in output['answer']
    assert 'Chunk Coverage' in output['answer']
    assert len(db.decode(output['citations']))==1
    emitted=''.join(db.decode(e['payload'])['text'] for e in db.rows(
        'SELECT payload FROM run_events WHERE run_id=? AND event_type=? ORDER BY id',(run,'answer_delta')))
    assert emitted==output['answer']
    assert '用户的问题' not in emitted


def test_quality_gate_rejects_instruction_echo():
    assert _answer_is_bad('首先，用户的问题是 CC 是什么意思？回顾用户指令：...',short=True)
    assert not _answer_is_bad('CC 是 Chunk Coverage 的缩写。[1]',short=True)


def test_direct_answer_is_explicitly_grounded_not_hardcoded():
    passage=[{'content':'We propose Chunk Coverage (CC) to evaluate retrieval tests.'}]
    assert _explicit_acronym_answer('文中的 CC 是什么意思？',passage)==('论文中的 CC 是 Chunk Coverage 的缩写。[1]',1)
    assert _explicit_acronym_answer('文中的 CC 是什么意思？',[{'content':'CC score is shown in Table 1.'}]) is None
    assert _explicit_acronym_answer('文章里 MMR 是什么意思？',[{'content':'Maximal Marginal Relevance (MMR) is used for diversity.'}])==('论文中的 MMR 是 Maximal Marginal Relevance 的缩写。[1]',1)
    # Detailed formula questions are not shortcut into abbreviation answers.
    assert _explicit_acronym_answer('CC 的定义公式是什么？',passage) is None

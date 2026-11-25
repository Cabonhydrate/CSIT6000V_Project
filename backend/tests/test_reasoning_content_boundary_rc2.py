"""RC2 safety/reproducibility regression: no raw reasoning reaches answers or F1.

Synthetic HTTP responses only; no QASPER questions, gold answers or live model.
"""
import asyncio
import json

import httpx
import pytest

from app.agent.model import ModelClient, ModelUnavailable, final_answer_only


def _mock(monkeypatch, payload_handler, *, provider='ollama'):
    original = httpx.AsyncClient
    monkeypatch.setattr('app.agent.model.httpx.AsyncClient',
                        lambda **kwargs: original(transport=httpx.MockTransport(payload_handler), **kwargs))
    return ModelClient(base_url=('http://127.0.0.1:11434/v1' if provider=='ollama'
                                 else 'http://127.0.0.1:8001/v1'),
                       model='qwen3:4b', provider=provider)


def _as_stream(chunks):
    return httpx.Response(200, text='\n'.join(json.dumps(c) for c in chunks)+'\n')


def test_orphan_closing_tag_discards_all_preamble():
    full = ('Hmm, the user is asking me a question. I should analyze the excerpts.\n'
            'Some internal working...\n</think>\n\nFinal grounded answer [1].')
    clean, removed = final_answer_only(full)
    assert clean == 'Final grounded answer [1].'
    assert removed > 60


def test_explicit_and_repeated_reasoning_blocks():
    text = '<think>first analysis</think>\n<analysis>second analysis</analysis>\nAnswer [1].'
    assert final_answer_only(text)[0] == 'Answer [1].'


@pytest.mark.parametrize('bad', [
    '<think>internal reasoning without end',
    '<think>internal reasoning</think>',
    'internal reasoning\n</think>',
    'Hmm, the user is asking about a paper, and I should respond.',
    '<analysis>first</analysis> <think>unfinished second',
])
def test_incomplete_or_only_reasoning_fails_closed(bad):
    with pytest.raises(ModelUnavailable, match='reasoning|planning'):
        final_answer_only(bad)


def test_legitimate_text_is_preserved():
    for text in ('The method uses a graph. [1]',
                 '{"answer":"The method uses a graph [1]"}',
                 'The user study involved 171 participants.'):
        assert final_answer_only(text)[0] == text


def test_ollama_chat_reasoning_and_json_mode(monkeypatch):
    def handler(req):
        body=json.loads(req.content)
        assert body['think'] is False
        if body.get('format')=='json':
            raw='<think>do not display</think>{"answer":"Final answer [1]"}'
        else:
            raw='Let me analyze the evidence.\n</think>\nFinal answer [1]'
        return httpx.Response(200, json={'message':{'content':raw},'done_reason':'stop'})
    model=_mock(monkeypatch,handler)
    async def main():
        x=await model.chat([{'role':'user','content':'question'}])
        y=await model.chat([{'role':'user','content':'question'}],json_mode=True)
        return x,y
    plain, structured=asyncio.run(main())
    assert plain['content']=='Final answer [1]'
    assert json.loads(structured['content'])['answer']=='Final answer [1]'


def test_ollama_stream_close_tag_split_across_frames(monkeypatch):
    def handler(req):
        chunks=[{'message':{'content':'Hmm, the user is asking about the evidence.\n</th'},'done':False},
                {'message':{'content':'ink>\nFinal'},'done':False},
                {'message':{'content':' answer [1].'},'done':False},
                {'done':True,'done_reason':'stop','eval_count':20}]
        return _as_stream(chunks)
    model=_mock(monkeypatch,handler)
    async def main():
        return [v async for v in model.stream([{'role':'user','content':'question'}])]
    parts=asyncio.run(main())
    assert parts==['Final answer [1].']
    assert model.last_stream_meta['finish_reason']=='stop'


def test_ollama_stream_does_not_emit_on_open_tag_or_failed_transport(monkeypatch):
    def handler(req):
        return _as_stream([{'message':{'content':'<think>unclosed internal reasoning'},'done':True}])
    model=_mock(monkeypatch,handler)
    observed=[]
    async def main():
        async for value in model.stream([{'role':'user','content':'question'}]):
            observed.append(value)
    with pytest.raises(ModelUnavailable,match='reasoning'):
        asyncio.run(main())
    assert observed == []


def test_ollama_stream_no_done_frame_emits_nothing(monkeypatch):
    model=_mock(monkeypatch,lambda req:_as_stream([{'message':{'content':'Answer.'},'done':False}]))
    observed=[]
    async def main():
        async for value in model.stream([{'role':'user','content':'question'}]):
            observed.append(value)
    with pytest.raises(ModelUnavailable,match='done=true'):
        asyncio.run(main())
    assert observed == []


def test_openai_compatible_chat_and_stream_also_filtered(monkeypatch):
    def handler(req):
        body=json.loads(req.content)
        if body['stream']:
            return httpx.Response(200, text='\n'.join(
                'data: '+json.dumps(part) for part in [
                    {'choices':[{'delta':{'content':'<think>private '},'finish_reason':None}]},
                    {'choices':[{'delta':{'content':'thought</think> Answer [1]'},'finish_reason':'stop'}]},
                ])+'\ndata: [DONE]\n')
        return httpx.Response(200,json={'choices':[{'message':{'content':'<think>secret</think> Answer [1]'},'finish_reason':'stop'}]})
    model=_mock(monkeypatch,handler,provider='openai')
    async def main():
        one=await model.chat([{'role':'user','content':'question'}])
        two=[part async for part in model.stream([{'role':'user','content':'question'}])]
        return one,two
    one,two=asyncio.run(main())
    assert one['content']=='Answer [1]'
    assert two==['Answer [1]']

@pytest.mark.parametrize('mode', ['basic', 'hybrid'])
def test_paper_rag_persists_and_scores_only_final_answer(monkeypatch,mode):
    """Exercise the real AgentRunner + ModelClient + SQLite + F1, with mocked HTTP."""
    from pathlib import Path
    from uuid import uuid4
    import numpy as np
    from app import db
    from app.agent.workflow import AgentRunner
    from app.evaluation import answer_f1
    from app.knowledge import retrieval
    from app.knowledge.ingest import import_qasper_file
    from app.schemas import ChatRequest

    monkeypatch.setattr(retrieval, '_dense_scores', lambda q, chunks:
                        np.array([float('BM25' in c['content']) for c in chunks], dtype='float32'))
    fixture=Path(__file__).resolve().parents[2]/'examples'/'qasper-mini.json'
    import_qasper_file(fixture)
    def handler(req):
        body=json.loads(req.content)
        assert body['think'] is False
        assert body['stream'] is True
        chunks=[{'message':{'content':'Hmm, the user is asking about the method.\n'},'done':False},
                {'message':{'content':'I should check the evidence.\n</thi'},'done':False},
                {'message':{'content':'nk> The method uses BM25 and dense retrieval [1].'},'done':False},
                {'done':True,'done_reason':'stop'}]
        return _as_stream(chunks)
    model=_mock(monkeypatch,handler)
    with db.connect() as con:
        con.execute("UPDATE profile SET language='en', memory_enabled=0")
    conversation_id,run_id=uuid4().hex,uuid4().hex
    with db.connect() as con:
        con.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                    (conversation_id,db.settings.owner_id,'RC2 output test',db.now()))
        con.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                    (run_id,conversation_id,'pending',mode,'BM25 method',
                     db.encode(['qasper-demo-paper-001']),db.now()))
    req=ChatRequest(conversation_id=conversation_id,question='BM25 method',
                    paper_ids=['qasper-demo-paper-001'],mode=mode,
                    use_memory=False,teaching=False,allow_web=False)
    asyncio.run(AgentRunner(model=model).run(run_id,req))
    row=db.one('SELECT status, answer, error FROM runs WHERE id=?',(run_id,))
    assert row['status']=='succeeded',row['error']
    assert row['answer']=='The method uses BM25 and dense retrieval [1].'
    events=db.rows('SELECT event_type,payload FROM run_events WHERE run_id=?',(run_id,))
    streamed=' '.join(str(event['payload']) for event in events)
    assert '</think>' not in streamed and 'I should check the evidence' not in streamed
    assert answer_f1(row['answer'],['The method uses BM25 and dense retrieval']) > 0.8

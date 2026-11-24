"""v0.3.16: only relevant user notes may affect an answer.

All sources/model output in these tests are local fixtures, not live Ollama.
"""
import asyncio
from uuid import uuid4

from app import db
from app.agent.learning_context import choose_learning_memories, learning_focus_gaps, learning_focus_visible
from app.agent.workflow import AgentRunner
from test_day1_learning_memory import fake_langgraph
from test_english_preferences_and_memory import create_run, MemoryAwareModel


def record(key, topic, content, domain='RAG'):
    return {'id': key, 'topic':topic,'domain':domain,'content':content,'mastery':'studied'}


def test_unrelated_question_retrieves_nothing_even_with_same_domain():
    notes=[record('c','Chunk Coverage','I know the RAG retrieval pipeline but not set union.'),
           record('f','FAISS','I have studied nearest-neighbor indexing.')]
    assert choose_learning_memories('Explain transformer positional encoding.',notes)==[]
    assert choose_learning_memories('Explain the RAG retrieval process.',notes)==[]
    assert choose_learning_memories('Tell me about my favorite restaurants.',notes)==[]


def test_explicit_saved_note_phrase_cannot_force_unrelated_retrieval():
    notes=[record('c','Chunk Coverage','I have studied set union.'),
           record('n','Neural Networks','I have studied backpropagation.', 'AI')]
    assert choose_learning_memories('Based on my saved learning note, explain SQL indexing.',notes)==[]
    assert [m['id'] for m in choose_learning_memories(
        'Based on my saved learning note, explain Chunk Coverage.',notes)] == ['c']


def test_multi_subject_question_selects_both_in_relevance_order():
    notes=[record('other','LSTM','I studied recurrent neural networks.', 'NLP'),
           record('cc','Chunk Coverage','I struggle with set union.'),
           record('code','Code Coverage','I studied line and branch coverage.','Testing')]
    selected=choose_learning_memories('Compare Chunk Coverage and code coverage.', notes)
    assert {m['id'] for m in selected} == {'cc','code'}


def test_general_recap_is_explicit_and_bounded():
    notes=[record(str(i),f'Topic {i}',f'Learning note about concept {i}') for i in range(10)]
    assert [m['id'] for m in choose_learning_memories('Please list my saved learning notes.',notes)]==['0','1','2']
    assert choose_learning_memories('Explain a topic based on my saved learning note.',notes)==[]
    assert len(choose_learning_memories('根据我的学习背景说明知识点',notes))==3


def test_same_topic_is_bounded_and_latest_note_wins():
    notes=[record('new','Chunk Coverage','I now understand set union.'),
           record('older','Chunk Coverage','I did not understand set union.'),
           record('oldest','Chunk Coverage','I read about retrieval.')]
    result=choose_learning_memories('Explain Chunk Coverage.',notes)
    assert [n['id'] for n in result]==['new','older']


def test_note_body_requires_distinctive_technical_overlap():
    notes=[record('gap','Math Basics','I struggle with set union.'),
           record('unrelated','Neural Networks','I studied model training.')]
    assert [m['id'] for m in choose_learning_memories('What is set union?',notes)]==['gap']
    assert choose_learning_memories('What is machine learning?',notes)==[]


def test_question_with_hyphenated_topic_matches():
    notes=[record('q','Q-Learning','I learned Q-values.','Reinforcement Learning')]
    assert [m['id'] for m in choose_learning_memories('How does Q Learning work?',notes)]==['q']
    assert [m['id'] for m in choose_learning_memories('Explain Q-Learning.',notes)]==['q']


def test_gap_requires_actual_named_explanation_not_generic_coverage_example():
    gaps=learning_focus_gaps([record('c','Chunk Coverage',
        'I know RAG, but not familiar with set union and how Chunk Coverage is calculated.')])
    assert gaps
    assert not learning_focus_visible('Chunk Coverage is the fraction retrieved. For example, coverage might be 90%.',gaps)
    assert learning_focus_visible('Set union combines {A, B} and {B, C} into {A, B, C}.',gaps)


def test_runner_filters_irrelevant_notes_and_exposes_metrics(monkeypatch):
    fake_langgraph(monkeypatch)
    run,request=create_run(monkeypatch,teaching=False, question='Explain Chunk Coverage.')
    # One relevant note is already inserted; a newer irrelevant note must not be selected.
    with db.connect() as c:
        c.execute('INSERT INTO memories(id,owner_id,domain,topic,memory_type,content,mastery,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',
                  (uuid4().hex,db.settings.owner_id,'NLP','Transformer','progress',
                   'I am not familiar with attention mechanisms.','studied',db.now(), '9999-12-31T23:59:59Z'))
    model=MemoryAwareModel()
    asyncio.run(AgentRunner(model).run(run,request))
    stats=db.decode(db.one('SELECT metrics FROM runs WHERE id=?',(run,))['metrics'])
    assert stats['learning_memory_available']==2
    assert stats['learning_memory_candidates']==1
    assert stats['learning_memory_topics']==['Chunk Coverage']
    assert stats['learning_memory_filtered']==1
    assert stats['learning_memory_applied'] is True
    prompts=str(model.chat_calls + model.stream_calls)
    assert 'Confirmed saved learning note: topic=Chunk Coverage' in prompts
    assert 'not familiar with attention mechanisms' not in prompts


def test_runner_scans_older_note_beyond_previous_30_note_window(monkeypatch):
    fake_langgraph(monkeypatch)
    run,request=create_run(monkeypatch,teaching=False,question='Explain Chunk Coverage.')
    with db.connect() as c:
        for i in range(40):
            c.execute('INSERT INTO memories(id,owner_id,domain,topic,memory_type,content,mastery,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',
                      (uuid4().hex,db.settings.owner_id,'NLP',f'Irrelevant subject {i}',
                       'progress','I read an unrelated manual.','mentioned',db.now(),f'9999-12-31T23:{i:02}:00Z'))
    model=MemoryAwareModel()
    asyncio.run(AgentRunner(model).run(run,request))
    stats=db.decode(db.one('SELECT metrics FROM runs WHERE id=?',(run,))['metrics'])
    assert stats['learning_memory_available']==41
    assert stats['learning_memory_candidates']==1
    assert stats['learning_memory_topics']==['Chunk Coverage']

"""v0.3.15 regression: learner's *named gap* must be explained, not any keyword.

All model output in these tests is simulated. Academic facts/citations remain
subject to the existing validation pipeline, and memories are not evidence.
"""
import asyncio
import json

from app import db
from app.agent import workflow
from app.agent.learning_context import learning_focus_gaps, learning_focus_visible
from test_day1_learning_memory import fake_langgraph
from test_english_preferences_and_memory import create_run
from test_memory_numeric_recovery_v0314 import SequenceTeachingModel


QUESTION = ('Explain what Chunk Coverage measures, why repeatedly retrieving '
            'the same chunk does not increase coverage, and why high coverage '
            'does not guarantee correct answers.')
NOTE = ('I already understand the RAG retrieval pipeline, but I am not '
        'familiar with set union and how Chunk Coverage is calculated.')
NORMAL = ('Chunk Coverage measures the fraction of distinct corpus chunks '
          'retrieved in a test suite. Repeated retrieval only counts each '
          'chunk once, and retrieval coverage cannot guarantee answer quality.')
GENERIC_EXAMPLE = (NORMAL + ' For example, a test suite might cover 90% '
                   'of chunks but still produce inaccurate answers.')
GAP_EXAMPLE = (NORMAL + ' Set union combines the chunks from different queries. '
               'For example, query one retrieves {A, B} and query two retrieves '
               '{B, C}; the union is {A, B, C}, with B counted only once.')


def test_focus_requires_named_concept_not_an_unrelated_percentage_example():
    gaps = learning_focus_gaps([{'content': NOTE}])
    assert gaps and 'set union' in gaps[0]
    assert not learning_focus_visible(GENERIC_EXAMPLE, gaps)
    assert learning_focus_visible(GAP_EXAMPLE, gaps)
    assert not learning_focus_visible('Set union is mentioned. The metric is important.', gaps)
    assert learning_focus_visible('For example, consider {A, B} and {B, C}. '
                                  'Their set union is {A, B, C}.', gaps)


def test_unrelated_examples_do_not_satisfy_named_gap():
    gaps = learning_focus_gaps([{'content': 'Not familiar with attention mechanism.'}])
    assert gaps
    assert not learning_focus_visible('Attention mechanism needs weighting. ' +
                                      'x' * 500 + ' For example, image classification.', gaps)
    assert learning_focus_visible('For example, consider a query and a key. '
                                  'The attention mechanism weights them.', gaps)


def test_real_analytical_flow_attempts_repair_when_generic_example_fools_old_check(monkeypatch):
    fake_langgraph(monkeypatch)
    run_id, request = create_run(monkeypatch, teaching=True, use_memory=True, question=QUESTION)
    db.execute('UPDATE memories SET content=? WHERE owner_id=?', (NOTE, db.settings.owner_id))
    model = SequenceTeachingModel([
        json.dumps({'answer': GENERIC_EXAMPLE, 'citations': [1]}),
        json.dumps({'answer': GAP_EXAMPLE, 'citations': [1]}),
    ])
    asyncio.run(workflow.AgentRunner(model).run(run_id, request))
    row = db.one('SELECT * FROM runs WHERE id=?', (run_id,))
    stats = db.decode(row['metrics'])
    assert row['status'] == 'succeeded', (row['answer'], stats)
    assert stats['learning_memory_applied'] is True
    assert stats['learning_focus_requested'] is True
    assert stats['learning_focus_rewrite_attempted'] is True
    assert stats['learning_focus_covered'] is True
    assert len(model.generation_calls) == 2
    assert 'set union' in row['answer'].lower()
    assert '90%' not in row['answer']


def test_empty_repair_is_reported_as_incomplete_personalization_not_fake_success(monkeypatch):
    fake_langgraph(monkeypatch)
    run_id, request = create_run(monkeypatch, teaching=True, use_memory=True, question=QUESTION)
    db.execute('UPDATE memories SET content=? WHERE owner_id=?', (NOTE, db.settings.owner_id))
    model = SequenceTeachingModel([
        json.dumps({'answer': GENERIC_EXAMPLE, 'citations': [1]}),
        '{"answer":"", "citations":[]}',
    ])
    asyncio.run(workflow.AgentRunner(model).run(run_id, request))
    row = db.one('SELECT * FROM runs WHERE id=?', (run_id,))
    stats = db.decode(row['metrics'])
    assert row['status'] == 'succeeded', (row['answer'], stats)
    assert '90%' in row['answer']   # valid *hypothetical* example retained
    assert stats['learning_focus_requested'] is True
    assert stats['learning_focus_rewrite_attempted'] is True
    assert stats['learning_focus_rewrite_failed'] is True
    assert stats['learning_focus_covered'] is False


def test_memory_off_does_not_inject_or_rewrite_saved_gap(monkeypatch):
    fake_langgraph(monkeypatch)
    run_id, request = create_run(monkeypatch, teaching=True, use_memory=False, question=QUESTION)
    db.execute('UPDATE memories SET content=? WHERE owner_id=?', (NOTE, db.settings.owner_id))
    model = SequenceTeachingModel([json.dumps({'answer': GENERIC_EXAMPLE, 'citations':[1]})])
    asyncio.run(workflow.AgentRunner(model).run(run_id, request))
    row = db.one('SELECT * FROM runs WHERE id=?', (run_id,))
    stats = db.decode(row['metrics'])
    assert row['status'] == 'succeeded', (row['answer'], stats)
    assert stats['learning_memory_applied'] is False
    assert stats['learning_focus_requested'] is False
    assert stats.get('learning_focus_rewrite_attempted') is not True
    assert len(model.generation_calls) == 1

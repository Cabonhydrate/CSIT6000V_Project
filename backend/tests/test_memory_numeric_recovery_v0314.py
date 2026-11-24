"""v0.3.14: factual safety and observable learner-oriented recovery.

No live network or Ollama is required. Never use an actual paper answer as a
hard-coded model response or evaluation oracle.
"""
import asyncio
import json
from app import db
from app.agent import workflow
from app.agent.learning_context import learning_focus_gaps, learning_focus_visible
from app.skills.numeric_provenance import audit_numeric_provenance
from test_routing_analytic import _run
from test_english_preferences_and_memory import create_run
from test_day1_learning_memory import fake_langgraph

SOURCES=[{'content': 'Retrieval coverage measures the fraction of corpus passages reached by a test suite.'}]


def test_modal_hypothesis_in_english_is_not_reported_study_result():
    records, issue = audit_numeric_provenance('Explain retrieval coverage',
        'For example, a test suite might cover 90% of chunks, while returning weak evidence.', SOURCES)
    assert issue == ''
    assert records == [{'value': '90%', 'kind': 'hypothetical', 'verified': True}]


def test_hypothesis_does_not_waive_next_english_reported_claim():
    records, issue = audit_numeric_provenance('Explain retrieval coverage',
        'For example, a test suite might cover 90% of chunks. The study reports 97.3% accuracy.', SOURCES)
    assert len(records) == 2 and records[0]['verified']
    assert '97.3%' in issue


def test_unmarked_or_paper_modal_percentage_is_not_synthetic():
    assert '90%' in audit_numeric_provenance('Explain the result',
        'The experiment might have achieved 90% coverage.', SOURCES)[1]
    assert '90%' in audit_numeric_provenance('Explain the result',
        'The test achieved 90% coverage.', SOURCES)[1]


def test_unsupported_numeric_answer_cannot_resurface_as_partial(monkeypatch):
    answer=('The test suite covered 97.3% of the corpus, and this result shows the '
            'retrieval component performs well, but answer correctness is different.')
    row, _, _=_run(monkeypatch,
        'Explain why retrieval coverage cannot guarantee correct answers.',
        json.dumps({'answer':answer,'citations':[1]}))
    assert row['status']=='partial'
    assert '97.3%' not in row['answer']
    assert db.decode(row['citations']) == []
    stats=db.decode(row['metrics'])
    assert '97.3%' in stats['numeric_grounding_issue']
    assert stats['abstained_after_validation'] is True


def test_english_partial_notice_has_no_chinese(monkeypatch):
    from app.agent.workflow import _partial_answer_notice, _unverified_answer_notice
    assert 'Partial answer' in _partial_answer_notice('en')
    assert 'partial' not in _partial_answer_notice('zh').lower()
    assert 'verified answer' in _unverified_answer_notice('en')
    assert not any('\u4e00' <= c <= '\u9fff' for c in _partial_answer_notice('en'))


def test_focus_is_extracted_only_from_user_stated_gap():
    gap=learning_focus_gaps([{'content':'I know how retrieval works but not familiar with set union.'}])
    assert gap and 'set union' in gap[0]
    assert not learning_focus_visible('Coverage is a percentage of chunks and repeats do not help.', gap)
    assert learning_focus_visible('Set union combines {A, B} and {B, C} into {A, B, C}.', gap)
    assert learning_focus_gaps([{'content':'I have read about set union.'}])==[]


class SequenceTeachingModel:
    def __init__(self, answers):
        self.answers=answers
        self.generation_calls=[]
    async def chat(self,messages,**kwargs):
        if 'Judge if the cited passages' in messages[0]['content']:
            return {'content':json.dumps({'sufficient':True,'missing':''}), 'latency_ms':1}
        self.generation_calls.append(messages)
        content=self.answers[min(len(self.generation_calls)-1,len(self.answers)-1)]
        return {'content':content, 'latency_ms':1}
    async def stream(self,*args,**kwargs):
        raise AssertionError('Analytical path must use JSON chat')
        yield ''


def test_personalization_rewrite_is_visible_and_source_bounded(monkeypatch):
    fake_langgraph(monkeypatch)
    question=('Explain what Chunk Coverage measures and why retrieving the same chunk '
              'again does not increase coverage.')
    rid,req=create_run(monkeypatch,teaching=True,use_memory=True,question=question)
    db.execute('UPDATE memories SET content=? WHERE owner_id=?',
               ('I understand retrieval, but not familiar with set union.',db.settings.owner_id))
    plain=('Chunk Coverage measures the fraction of distinct corpus chunks retrieved across '
           'a test suite. Repeated retrieval does not increase the fraction because only '
           'unique chunks count.')
    learned=(plain+' Set union combines the results. For example, if the first query '
             'retrieves {A, B} and the second retrieves {B, C}, their union is {A, B, C}.')
    model=SequenceTeachingModel([json.dumps({'answer':plain,'citations':[1]}),
                                json.dumps({'answer':learned,'citations':[1]})])
    asyncio.run(workflow.AgentRunner(model).run(rid,req))
    row=db.one('SELECT * FROM runs WHERE id=?',(rid,))
    stats=db.decode(row['metrics'])
    assert row['status']=='succeeded', (row['answer'],stats)
    assert len(model.generation_calls)==2
    assert 'PERSONALIZED TEACHING TARGET' in model.generation_calls[-1][0]['content']
    assert 'set union' not in model.generation_calls[-1][0]['content'].lower()
    assert 'set union' in model.generation_calls[-1][-1]['content'].lower()
    assert stats['learning_focus_rewrite_attempted'] is True
    assert stats['learning_focus_covered'] is True
    assert 'set union' in row['answer'].lower()


def test_rewrite_failure_keeps_verified_answer_without_invented_numbers(monkeypatch):
    fake_langgraph(monkeypatch)
    question='Explain what Chunk Coverage measures, and why repeating retrieval does not help.'
    rid,req=create_run(monkeypatch,teaching=True,use_memory=True,question=question)
    db.execute('UPDATE memories SET content=? WHERE owner_id=?',
               ('I know retrieval, but not familiar with set union.',db.settings.owner_id))
    plain=('Chunk Coverage measures the fraction of distinct corpus chunks retrieved across '
           'the test suite. A repeated chunk is counted only once, not multiple times.')
    model=SequenceTeachingModel([json.dumps({'answer':plain,'citations':[1]}),
                                '{"answer":"","citations":[]}'])
    asyncio.run(workflow.AgentRunner(model).run(rid,req))
    row=db.one('SELECT * FROM runs WHERE id=?',(rid,))
    stats=db.decode(row['metrics'])
    assert row['status']=='succeeded', (row['answer'],stats)
    assert plain in row['answer']
    assert stats['learning_focus_rewrite_failed'] is True
    assert stats['learning_focus_covered'] is False
    assert '97.3%' not in row['answer']

"""RC4 regression: dependent instructions and explicit user sets must produce math.

Uses a local model/retriever fixture. No PDF content or QASPER gold is hardcoded
in production code, and no real Ollama calls occur here.
"""
import asyncio
from app import db
from app.agent.workflow import _split_question_parts, _question_intent, AgentRunner
from app.agent.learning_context import learning_focus_gaps
from app.skills.task_outputs import user_set_calculation
from test_english_preferences_and_memory import create_run
from test_day1_learning_memory import fake_langgraph

QUESTION = (
    'Suppose a corpus has 6 chunks in total. Query 1 retrieves {c1, c2}, '
    'Query 2 retrieves {c2, c3}, and Query 3 retrieves {c1, c3}. '
    'What is the Chunk Coverage? Please explain it step by step for a beginner.'
)
NOTE = (
    'I understand the basic RAG retrieval pipeline, but I often confuse set union '
    'with simply counting all retrieved chunks. Please explain Chunk Coverage step by step.'
)


class AssessmentOnlyModel:
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        system = messages[0]['content']
        if 'Judge if the cited passages' in system:
            return {'content': '{"sufficient": true, "missing": ""}', 'latency_ms': 1}
        # A deterministic user-given computation must not need LLM-generated
        # mathematical steps. Planning calls may legitimately take place.
        return {'content': '{"sufficient": true, "missing": ""}', 'latency_ms': 1}

    async def stream(self, *args, **kwargs):
        raise AssertionError('Adaptive explicit arithmetic must not stream unscreened text')
        yield ''


def test_dependent_english_formatting_request_is_not_an_independent_question():
    assert _split_question_parts(QUESTION) == [
        QUESTION.replace('? Please', '；Please').rstrip('.')
    ]
    assert _question_intent(QUESTION) == 'calculation'
    assert _split_question_parts('What is CC? What is code coverage?') == [
        'What is CC', 'What is code coverage'
    ]
    assert len(_split_question_parts('What is CC? Explain why code coverage is different?')) == 2


def test_computes_user_given_natural_language_sets():
    text = user_set_calculation(QUESTION, language='en')
    assert 'Q1={c1,c2}' in text and 'Q2={c2,c3}' in text and 'Q3={c1,c3}' in text
    assert 'set union is {c1,c2,c3}' in text
    assert '3 unique retrieved chunks' in text
    assert '3/6 = 50%' in text
    assert '2 + 2 + 2' not in text
    detailed = user_set_calculation(QUESTION, language='en', explain_union=True)
    assert '2 + 2 + 2' in detailed
    assert 'NOT the sum' in detailed


def test_does_not_invent_missing_corpus_or_unavailable_chunk_members():
    assert user_set_calculation('Query 1 retrieves {c1,c2}, Query 2 retrieves {c2,c3}.') == ''
    assert user_set_calculation('Suppose a corpus has 2 chunks. Query 1 retrieves {c1,c2}, Query 2 retrieves {c2,c3}.') == ''
    assert user_set_calculation('Suppose a corpus has 5 chunks. Query 1 retrieves {c1,c2}, Query 1 retrieves {c2,c3}.') == ''
    assert user_set_calculation('C={c1,c2};q1={c3},q2={c1}') == ''


def test_original_explicit_corpus_and_formula_request_still_work():
    q = ('请写出论文中 Chunk Coverage 的正式数学公式，解释每个符号，'
         '并用 C={c1,c2,c3,c4,c5};q1={c1,c2};q2={c2,c3} 计算。')
    assert _question_intent(q) == 'formula'
    assert '3/5 = 60%' in user_set_calculation(q)


def test_confusion_memory_records_the_actual_learning_gap():
    assert learning_focus_gaps([{'content': NOTE}]) == ['set union']


def _execute(monkeypatch, *, memory):
    fake_langgraph(monkeypatch)
    run_id, req = create_run(monkeypatch, teaching=True,
                             use_memory=memory, question=QUESTION)
    db.execute('UPDATE memories SET content=? WHERE owner_id=? AND topic=?',
               (NOTE, db.settings.owner_id, 'Chunk Coverage'))
    model = AssessmentOnlyModel()
    asyncio.run(AgentRunner(model).run(run_id, req))
    result = db.one('SELECT * FROM runs WHERE id=?', (run_id,))
    return result, db.decode(result['metrics']), db.decode(result['citations']), model


def test_memory_off_correct_stepwise_answer_with_real_paper_citation(monkeypatch):
    record, metrics, citations, model = _execute(monkeypatch, memory=False)
    assert record['status'] == 'succeeded', record.get('error')
    assert metrics['question_intent'] == 'calculation'
    assert 'subquestions_total' not in metrics
    assert metrics['deterministic_calculation'] is True
    assert metrics['learning_memory_applied'] is False
    assert '3/6 = 50%' in record['answer']
    assert '{c1,c2,c3}' in record['answer']
    assert '2 + 2 + 2' not in record['answer']
    assert len(citations) == 1
    assert citations[0]['evidence_id'].endswith(':0')
    assert '[1]' in record['answer']


def test_memory_on_explains_double_counting_without_changing_math(monkeypatch):
    record, metrics, citations, model = _execute(monkeypatch, memory=True)
    assert record['status'] == 'succeeded', record.get('error')
    assert metrics['question_intent'] == 'calculation'
    assert 'subquestions_total' not in metrics
    assert metrics['learning_memory_candidates'] == 1
    assert metrics['learning_memory_applied'] is True
    assert metrics['learning_focus_requested'] is True
    assert metrics['learning_focus_covered'] is True
    assert '3/6 = 50%' in record['answer']
    assert '2 + 2 + 2' in record['answer']
    assert 'NOT the sum' in record['answer']
    assert len(citations) == 1

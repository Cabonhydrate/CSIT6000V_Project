"""Regression checks for model formulas and multi-intent Skill selection.

No CC-specific logic in production code; paper-related examples are here only.
"""
import json

from app import db
from app.skills import select_answer_skills, normalize_skill_output, validate_skill_output
from test_routing_analytic import _run


FORMULA_QUESTION = (
    '请写出论文中 Chunk Coverage 的数学公式，并逐一解释 '
    'T、C、R_k(q)、k 和并集符号，最后举一个例子。'
)
CLAIM_QUESTION = (
    '有人说：同一个 Chunk 被重复检索三次应当计算三次。'
    '请判断是否正确，并结合 Chunk Coverage 的数学公式解释原因。'
)


def test_single_letter_formula_is_not_accepted_just_because_variables_exist():
    obj = {
        'formula': 'T',
        'variables': [
            {'symbol': 'T', 'meaning': '测试查询集合'},
            {'symbol': 'C', 'meaning': '全部文档块'},
            {'symbol': 'R_k(q)', 'meaning': 'top-k retrieval'},
            {'symbol': 'k', 'meaning': 'top-k size'},
        ],
        'explanation': '先对检索结果取并集，然后去重计数。',
        'citations': [1],
    }
    result = normalize_skill_output(obj, 'explainFormula', 1)
    assert result.structured and result.complete
    assert '公式字段' in validate_skill_output(result, 'explainFormula', FORMULA_QUESTION)


def test_json_escaped_latex_control_character_is_rejected():
    data = {'formula': 'CC(T)=\x0crac{|E|}{|C|}',
            'variables': [{'symbol': 'T', 'meaning': 'queries'}],
            'explanation': 'coverage ratio', 'citations': [1]}
    result = normalize_skill_output(data, 'explainFormula', 1)
    assert '控制字符' in validate_skill_output(result, 'explainFormula', '写出公式')


def test_claim_with_requested_math_requires_composed_task_skills():
    plan = select_answer_skills('claim_check', teaching=True, question=CLAIM_QUESTION)
    assert plan.names == ('verifyClaims', 'explainFormula', 'explainForBeginner')
    assert select_answer_skills('claim_check', teaching=True, question='判断下列说法对错').names == (
        'verifyClaims', 'explainForBeginner')
    assert select_answer_skills('formula', teaching=True, question=FORMULA_QUESTION).names == (
        'explainFormula', 'explainForBeginner')


def test_composed_claim_cannot_succeed_without_requested_equation():
    obj = {'claims': [{
        'claim': '重复检索按多次覆盖统计', 'verdict': 'refuted',
        'explanation': '每个块被检索到即覆盖，不按频率计数。',
        'evidence_ids': [1]
    }]}
    output = normalize_skill_output(obj, 'verifyClaims', 1)
    assert '缺少' in validate_skill_output(output, 'verifyClaims', CLAIM_QUESTION)
    obj['formula'] = 'T'
    assert '公式字段' in validate_skill_output(
        normalize_skill_output(obj, 'verifyClaims', 1), 'verifyClaims', CLAIM_QUESTION)
    obj['formula'] = r'CC(T)=\frac{|\bigcup_{q\in T}R_k(q)|}{|C|}'
    assert validate_skill_output(normalize_skill_output(obj, 'verifyClaims', 1),
                                 'verifyClaims', CLAIM_QUESTION) == ''
    legacy = normalize_skill_output({
        'answer': '说法错误。同一个块只算一次。', 'citations': [1]}, 'verifyClaims', 1)
    assert '数学等式' in validate_skill_output(legacy, 'verifyClaims', CLAIM_QUESTION)


def test_structured_formula_t_cannot_mark_run_succeeded(monkeypatch):
    obj = {
        'formula': 'T',
        'variables': [
            {'symbol': 'T', 'meaning': '测试查询集合'},
            {'symbol': 'C', 'meaning': '语料库'},
            {'symbol': 'R_k(q)', 'meaning': '单次检索结果'},
            {'symbol': 'k', 'meaning': '返回条数'},
        ],
        'explanation': '对检索结果取并集并去重。', 'citations': [1],
    }
    record, _, _ = _run(monkeypatch, FORMULA_QUESTION, json.dumps(obj, ensure_ascii=False))
    assert record['status'] == 'partial'
    assert '公式' in db.decode(record['metrics'])['answer_validation_issue']


def test_composed_claim_with_valid_formula_is_complete(monkeypatch):
    obj = {
        'claims': [{
            'claim': '重复检索三次要计数三次', 'verdict': 'refuted',
            'explanation': '重复出现的 Chunk 只在集合并集中出现一次。',
            'evidence_ids': [1],
        }],
        'formula': r'CC(T)=\frac{|\bigcup_{q\in T} R_k(q)|}{|C|}',
        'variables': [{'symbol': 'T', 'meaning': '测试查询集合'},
                      {'symbol': 'C', 'meaning': '全部 Chunk'}],
    }
    record, model, _ = _run(monkeypatch, CLAIM_QUESTION, json.dumps(obj, ensure_ascii=False))
    assert record['status'] == 'succeeded', db.decode(record['metrics'])
    assert r'\bigcup' in record['answer']
    assert '**错误**' in record['answer'] or '**refuted**' in record['answer']
    assert db.decode(record['metrics'])['answer_skills'] == [
        'verifyClaims', 'explainFormula', 'explainForBeginner'
    ]
    assert db.decode(record['citations'])[0]['page_numbers'] == [7]
    assert 'explainFormula' in model.prompts[-1][0]['content']


def test_bad_equation_is_repaired_on_second_answer_attempt(monkeypatch):
    from test_routing_analytic import MockModel

    bad = {
        'formula': 'T',
        'variables': [{'symbol': 'T', 'meaning': '问题集合'},
                      {'symbol': 'C', 'meaning': '全部块'},
                      {'symbol': 'R_k(q)', 'meaning': 'Top-k 块'},
                      {'symbol': 'k', 'meaning': '查询返回数'}],
        'explanation': '用去重后的数量计算比例。',
        'citations': [1]
    }
    fixed = {**bad, 'formula': r'CC(T)=\frac{|\bigcup_{q\in T} R_k(q)|}{|C|}'}
    original_chat = MockModel.chat
    attempts = []

    async def patched_chat(self, messages, **kwargs):
        if 'Judge if the cited passages' in messages[0]['content']:
            return await original_chat(self, messages, **kwargs)
        attempts.append(messages)
        obj = bad if len(attempts) == 1 else fixed
        return {'content': json.dumps(obj, ensure_ascii=False), 'latency_ms': 1}

    monkeypatch.setattr(MockModel, 'chat', patched_chat)
    record, _, _ = _run(monkeypatch, FORMULA_QUESTION, '')
    assert record['status'] == 'succeeded', db.decode(record['metrics'])
    assert len(attempts) == 2
    assert '公式字段' in attempts[1][-1]['content']
    assert r'\bigcup' in record['answer']
    assert '\nT\n' not in record['answer']
    assert db.decode(record['metrics'])['answer_complete'] is True

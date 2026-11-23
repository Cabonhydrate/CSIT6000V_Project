"""Regression for a real local-model JSON shape and PDF math extraction.

All inputs are synthetic or from known parser behavior; no benchmark gold answers
are injected into model prompts. The evaluation suite remains external.
"""
from app.skills.outputs import normalize_skill_output, validate_skill_output
from app.skills.math_support import _recover_multiline_union_fraction, _literal_equation, safe_math_fallback
from app.skills.quality import formal_equation_issue

GLYPH_DAMAGE = (
    'A paper defines MQ(𝑇) =\n\x0c\x0cÐ\n𝑞∈𝑇𝑅𝑘(𝑞)\n\x0c\x0c\n|C|\n'
    'where T is a test query suite, C contains all chunks.')
E = [{'content': GLYPH_DAMAGE, 'paper_id': 'p', 'evidence_id': 'p:c1',
      'page_numbers': [3]}]


def test_map_variables_do_not_crash_verify_claim_plus_formula():
    obj = {
        'claims': [{'claim': 'The proposed expression is false',
                    'verdict': 'refuted', 'explanation': 'Set overlap changes the numerator.',
                    'evidence_ids': [1]}],
        'formula': r'MQ(T)=\frac{|A\cup B|}{|C|}',
        'variables': {'T': 'test queries', 'C': {'meaning': 'all chunks'},
                      'bad': None},
        'formula_citations': [1],
    }
    out = normalize_skill_output(obj, 'verifyClaims', 1)
    assert 'T\\)：test queries' in out.answer
    assert 'C\\)：all chunks' in out.answer
    assert out.structured and out.formula and out.claims_count == 1
    assert not validate_skill_output(out, 'verifyClaims', '请判断并写出正确公式')


def test_unexpected_variable_type_never_causes_runtime_error():
    for value in ('unexpected', 8, None, {'T': 'test queries'}, []):
        obj = {'formula': 'MQ(T)=|A|/|C|', 'variables': value,
               'explanation': 'Explained', 'citations': [1]}
        # This is a shape mismatch, not an executable crash.
        out = normalize_skill_output(obj, 'verifyClaims', 1)
        assert isinstance(out.answer, str)


def test_source_math_normalization_recovers_mangled_fraction():
    expected = 'MQ(T) = |∪_{q∈T} R_k(q)| / |C|'
    assert _recover_multiline_union_fraction(GLYPH_DAMAGE) == expected
    q = 'MQ的定义是否为错误的 T=∪_{i=1}^k R_k(q)？请写出正确公式。'
    assert _literal_equation(q, E) == (1, expected)
    partial = safe_math_fallback(q, E)
    assert partial and expected in partial.text and partial.citation == 1


def test_no_reconstruction_from_ambiguous_or_inconsistent_source():
    damaged = GLYPH_DAMAGE.replace('𝑞∈𝑇𝑅𝑘(𝑞)', '𝑞∈𝑇𝑅𝑘(𝑧)')
    assert not _recover_multiline_union_fraction(damaged)
    damaged = GLYPH_DAMAGE.replace('𝑞∈𝑇𝑅𝑘(𝑞)', '𝑞∈𝑍𝑅𝑘(𝑞)')
    assert not _recover_multiline_union_fraction(damaged)
    assert not _recover_multiline_union_fraction('MQ(T)=a+b')


def test_formal_equation_check_requires_the_original_metric_parameter():
    q = '写出正式的 MQ(T) 公式'
    assert '正式定义遗漏原文指标的参数' in formal_equation_issue(q, 'MQ=|A|/|C|', E, 'MQ=|A|/|C|')
    assert not formal_equation_issue(q,
        'MQ(T) = |∪_{q∈T} R_k(q)| / |C|', E,
        'MQ(T) = |∪_{q∈T} R_k(q)| / |C|')


def test_source_retrieval_prioritizes_mangled_formal_definition(monkeypatch):
    from app import db
    from app.agent.workflow import _ensure_formula_definition_evidence
    # Simulates the original PDF page being indexed while a different page
    # (e.g. Algorithm 1) was initially chosen by the retriever.
    monkeypatch.setattr(db, 'rows', lambda sql, args: [{
        'id': 'p:c1', 'paper_id': 'p', 'seq': 1,
        'section': 'Page 3', 'content': GLYPH_DAMAGE,
        'pages': '[3]', 'paragraph_refs': '[]',
    }])
    earlier = [{'evidence_id': 'p:c2', 'paper_id': 'p',
                'page_numbers': [4], 'content': 'Background only.'}]
    results = _ensure_formula_definition_evidence(
        earlier, '请写出正式的 MQ(T) 公式', ['p'])
    assert results[0]['evidence_id'] == 'p:c1'
    assert results[0]['page_numbers'] == [3]


def test_agent_workflow_handles_qwen_mapping_variables(monkeypatch):
    import json
    from test_safe_math_recovery_v025 import _seed
    obj = {
        'claims': [{'claim': 'Two sets can overlap', 'verdict': 'supported',
                    'explanation': 'The same element can appear twice in a sum.',
                    'evidence_ids': [1]}],
        'formula': r'CR(T)=\frac{|A\cup B|}{|D|}',
        'variables': {'A': 'first set', 'B': 'second set'},
        'formula_citations': [1],
    }
    question = '判断 CR 的求和公式是否总相等，并给出反例。'
    row = _seed(monkeypatch, question, json.dumps(obj))
    assert row['status'] != 'failed', row.get('error')
    assert 'slice(None, 16, None)' not in str(row.get('error') or '')

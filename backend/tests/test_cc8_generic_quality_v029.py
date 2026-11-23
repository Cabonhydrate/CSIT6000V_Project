"""Offline guardrail tests from CC8 failure *classes*, not benchmark answers.

Mocks cannot establish real semantic correctness. The real CC8 rerun remains
mandatory. Tests never import expected paper answers into production code.
"""
import json
from app import db
from app.agent.workflow import (
    _quoted_atomic_claims, _claim_coverage_issue,
    _unsupported_precise_numbers, _enumerative_paper_topic,
    _enumeration_evidence_issue,
)
from app.skills.outputs import normalize_skill_output
from test_routing_analytic import _run


def _claim_payload(parts):
    return {'claims': [
        {'claim': p, 'verdict': 'refuted', 'explanation': '引用文献说明该断言不成立',
         'evidence_ids': [1]}
        for p in parts
    ]}


def test_paraphrased_claim_groups_preserve_atomic_scope():
    questions = [
        '有人声称：“覆盖越多一定越准确，而且系统发布前必须覆盖全部内容。”请逐条判断。',
        '一位读者说：“方法 X 永远最优，并且指标 Y 越大越可靠。”请核查。',
        '请评价：“所有实验都有效；每种模型都比基线好。”并解释理由。',
    ]
    for q in questions:
        clauses = _quoted_atomic_claims(q)
        assert len(clauses) == 2
        assert all(len(x) >= 5 for x in clauses)
    assert _quoted_atomic_claims('请判断：“系统可能表现更好。”') == []


def test_claim_count_and_original_assertions_required():
    q = '有人说：“模型 A 一定胜过模型 B，而且结论无需任何实验支持。”请逐项判断。'
    clauses = _quoted_atomic_claims(q)
    missing = _claim_payload(clauses[:1])
    output = normalize_skill_output(missing, 'verifyClaims', 1)
    assert '1/2' in _claim_coverage_issue(q, missing, output)
    unrelated = _claim_payload([clauses[0], clauses[0]])
    assert '不一致' in _claim_coverage_issue(
        q, unrelated, normalize_skill_output(unrelated, 'verifyClaims', 1))
    good = _claim_payload(clauses)
    assert _claim_coverage_issue(q, good, normalize_skill_output(good, 'verifyClaims', 1)) == ''


def test_precise_numbers_need_explicit_source_or_hypothetical_label():
    q = '作者为何不使用指标 A？'
    evidence = [{'content': 'The authors did not select A because it is not actionable.'}]
    assert '0.94' in _unsupported_precise_numbers(q,
        '研究发现指标 A 的取值是 0.94，因此作者不采用它。', evidence)
    assert _unsupported_precise_numbers(q,
        '假设某指标是 0.94，这只是说明性的假设例子。', evidence) == ''
    evidence = [{'content': 'The reported A score is 0.94 for this setting.'}]
    assert _unsupported_precise_numbers(q, '报告 A=0.94。', evidence) == ''


def test_t22_style_full_agent_cannot_mark_one_of_two_claims_complete(monkeypatch):
    q = '有人说：“指标 A 越大结果一定越准确，而且上线前必须达到满覆盖。”请根据论文逐条判断。'
    first = _quoted_atomic_claims(q)[0]
    reply = json.dumps(_claim_payload([first]), ensure_ascii=False)
    row, _, _ = _run(monkeypatch, q, reply)
    assert row['status'] == 'partial', row['answer']
    metrics = db.decode(row['metrics'])
    assert metrics['answer_complete'] is False
    assert '1/2' in metrics['answer_validation_issue']


def test_enumeration_cannot_be_replaced_with_generic_definition(monkeypatch):
    q = '这篇论文提出了哪些量子计算算法？请列出名称。'
    assert _enumerative_paper_topic(q) == '量子计算算法'
    generic = {'definition': 'RAG 是利用外部检索信息生成答案的方法。',
               'example': '假设系统处理一个查询。',
               'takeaway': '检索帮助减少幻觉。', 'citations': [1]}
    row, _, _ = _run(monkeypatch, q, json.dumps(generic, ensure_ascii=False))
    assert row['status'] == 'partial', row['answer']
    assert '量子计算算法' in row['answer']
    metrics = db.decode(row['metrics'])
    assert metrics['answer_complete'] is False
    assert 'enumeration_validation_issue' in metrics


def test_list_requires_real_source_span_not_arbitrary_citation():
    q = '这篇论文介绍了哪些排序算法？'
    paper = [{'content': 'This paper studies retrieval models and their evaluation.'}]
    bad = {'answer': '排序算法包括排序模型。', 'citations': [1],
           'topic_english': 'sorting algorithms',
           'source_quote': 'sorting algorithms are discussed here and evaluated.'}
    assert '原文' in _enumeration_evidence_issue(q, bad, paper)
    quoted = 'The sorting algorithms are insertion sort and merge sort.'
    good = {'answer': '排序算法包括 insertion sort 和 merge sort。', 'citations': [1],
            'topic_english': 'sorting algorithms', 'source_quote': quoted}
    assert _enumeration_evidence_issue(q, good, [{'content': quoted}]) == ''


def test_procedure_prompt_distinguishes_require_inputs_from_initial_assignments(monkeypatch):
    q = '请根据 Algorithm 1 逐步说明如何执行测试，什么时候停止？'
    reply = {'answer': '算法先读取初始输入，按预算逐次运行，每执行一次查询扣除一次预算；满足停止条件后退出。',
             'citations':[1]}
    row, model, _ = _run(monkeypatch, q, json.dumps(reply, ensure_ascii=False))
    prompts = '\n'.join(m['content'] for sequence in model.prompts for m in sequence)
    assert 'Require 中的输入' in prompts
    assert '总预算和每轮执行的次数' in prompts
    assert row['status'] in ('succeeded','partial')


def test_all_quoted_claims_are_accepted_when_individually_covered(monkeypatch):
    q = '有人说：“检索越多一定越准，而且所有测试都能保证质量。”请逐条判断并纠正。'
    reply = json.dumps(_claim_payload(_quoted_atomic_claims(q)), ensure_ascii=False)
    row, _, _ = _run(monkeypatch, q, reply)
    assert row['status'] == 'succeeded', row['answer']
    assert db.decode(row['metrics'])['answer_complete'] is True
    assert row['answer'].count('1. **') == 1
    assert row['answer'].count('2. **') == 1


def test_source_missing_decimal_must_not_be_succeeded(monkeypatch):
    q = '作者为什么不直接使用准确率作为全面评价指标？'
    reply = {'answer': '研究表明准确率是 0.94，但该指标并不能反映未测试的区域。',
             'citations':[1]}
    row, _, _ = _run(monkeypatch, q, json.dumps(reply,ensure_ascii=False))
    assert row['status'] == 'partial',row['answer']
    assert '0.94' in db.decode(row['metrics'])['answer_validation_issue']

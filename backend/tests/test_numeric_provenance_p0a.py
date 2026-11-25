"""P0-A: number provenance regressions independent of the CC paper's answers.

The tested numbers are synthetic. Models are not called, and no Gold Answers
are used as rules. Full frontend and Qwen tests remain separate.
"""
from app.agent.workflow import _unsupported_precise_numbers
from app.skills.numeric_provenance import audit_numeric_provenance

NO_NUMBERS=[{'content':'The paper describes retrieval-based evaluation.'}]
RATIOS=[{'content':'Table 5: treatment 0.839, control 0.669.'}]


def check(question,answer,evidence=NO_NUMBERS):
    return audit_numeric_provenance(question,answer,evidence)


def test_hypothetical_count_ratio_is_allowed_and_audited():
    records, issue=check('请比较两个覆盖指标。',
        '例如：若语料库有10个片段，检索到6个片段，CC=60%。')
    assert not issue
    assert records == [{'value':'60%','kind':'hypothetical','verified':True}]


def test_hypothetical_comparison_like_browser_r03():
    text=('例如：RAG系统测试时，若语料库有10个片段，CC=60%表示测试套件检索了6个片段；'
          '而代码覆盖率60%表示60%的代码行被测试执行过。')
    _,issue=check('比较文档检索覆盖率与代码覆盖率。',text)
    assert not issue, issue


def test_hypothetical_ratio_wrong_arithmetic_fails():
    _,issue=check('解释比例指标。','例如：若语料库有10个块，检索到6个块，CC=70%。')
    assert '不一致' in issue


def test_ungrounded_reported_percentage_still_fails():
    _,issue=check('论文的实验结果如何？','论文实验报告成功率达到 97.3%。')
    assert '97.3%' in issue


def test_hypothetical_does_not_waive_next_reported_claim():
    _,issue=check('作者有什么结论？',
        '假设指标 A=0.94，这只是例子。论文报告指标 B=0.97。')
    assert '0.97' in issue


def test_hypothetical_phrase_is_not_universal_numeric_bypass():
    _,issue=check('作者有什么结论？',
        '假设论文报告了一个结果：该研究报告准确率达到 0.94。')
    assert '0.94' in issue


def test_bare_numeric_example_requires_marking():
    _,issue=check('解释比例指标。','研究的 CC=60%，这是一个很好的结果。')
    assert '60%' in issue


def test_grounded_paper_fact_is_accepted():
    records,issue=check('实验得分如何？','报告 A=0.839。',RATIOS)
    assert not issue and records[0]['kind']=='paper_fact'


def test_question_number_is_accepted():
    records,issue=check('为什么给出 0.51？','用户提到 0.51，我们不能仅据此认定它是论文结果。')
    assert not issue and records[0]['kind']=='question_input'


def test_derived_ratio_from_grounded_inputs():
    records,issue=check('计算两指标的相对提高。',
        '根据数字计算：(0.839-0.669)/0.669 × 100% ≈ 25.41%。',RATIOS)
    assert not issue, issue
    assert any(rec['value']=='25.41%' and rec['kind']=='calculated' and rec['verified']
               for rec in records)
    assert any(rec['value']=='100%' and rec['kind']=='calculated' for rec in records)


def test_wrong_derived_ratio_is_rejected():
    _,issue=check('计算两指标的相对提高。',
        '计算：(0.839-0.669)/0.669 × 100% ≈ 30.41%。',RATIOS)
    assert '不一致' in issue


def test_formula_with_unsourced_operands_is_not_evidence():
    _,issue=check('计算提升比例。','计算：530/248≈2.14 倍。')
    assert '输入数字' in issue


def test_simple_derived_ratio_with_sourced_inputs():
    paper=[{'content':'Random required 530 runs, selected tests required 248 runs.'}]
    records,issue=check('根据原文计算加速比。','530/248≈2.14 倍。',paper)
    assert not issue and records[0]['kind']=='calculated'


def test_numeric_token_must_not_match_substring_of_other_value():
    _,issue=check('结果是什么？','结果为 60%。', [{'content':'The result was 160%.'}])
    assert '60%' in issue


def test_compatibility_wrapper_preserved():
    answer='假设有10个片段，检索到6个片段，则CC=60%。'
    assert _unsupported_precise_numbers('定义CC',answer,NO_NUMBERS)==''
    assert '0.94' in _unsupported_precise_numbers(
        '研究为什么?', '作者在实验中报告 0.94。', NO_NUMBERS)


def test_synthetic_set_counterexample_across_two_sentences():
    answer=('假设同一语料 D={a,b}，两个查询都只命中 a。'
            '逐次求和2/2=1，而并集去重为1/2=0.5，所以两者不等价。')
    records,issue=check('构造反例说明差异。',answer)
    assert not issue, issue
    assert any(r['value']=='0.5' and r['verified'] for r in records)


def test_two_sentence_hypothetical_ratio_correct_and_incorrect():
    good='假设语料库有10个块，检索到6个块。于是 CC=60%。'
    bad='假设语料库有10个块，检索到6个块。于是 CC=70%。'
    assert check('说明定义。',good)[1]==''
    assert '不一致' in check('说明定义。',bad)[1]


def test_two_sentence_hypothesis_never_covers_reported_experiment_result():
    answer='假设有10个文档块，检索6个。论文实验结果准确率达到 0.97。'
    assert '0.97' in check('概括论文。',answer)[1]


def test_agent_runner_accepts_supported_comparison_with_synthetic_example(monkeypatch):
    import json
    from app import db
    from test_routing_analytic import _run
    question='请比较 Chunk Coverage 和传统的代码覆盖率，举一个假设的比例例子。'
    answer=('CC 衡量测试问题检索到的不同文档片段比例，而代码覆盖率衡量测试执行的程序代码比例。'
            '例如：若语料库有10个片段，检索到6个片段，CC=60%。这只是教学假设。')
    row,_,_=_run(monkeypatch,question,json.dumps({'answer':answer,'citations':[1]},ensure_ascii=False))
    assert row['status']=='succeeded',row['answer']
    metrics=db.decode(row['metrics'])
    assert metrics['answer_complete'] is True
    assert any(x['value']=='60%' and x['kind']=='hypothetical'
               for x in metrics['numeric_provenance'])


def test_agent_runner_still_rejects_unsourced_experiment_score(monkeypatch):
    import json
    from app import db
    from test_routing_analytic import _run
    question='请比较 Chunk Coverage 和传统的代码覆盖率。'
    answer=('CC 衡量检索过的文档片段，代码覆盖率衡量程序代码的执行范围。'
            '论文实验显示前者达到 97.3%，因此更可靠。')
    row,_,_=_run(monkeypatch,question,json.dumps({'answer':answer,'citations':[1]},ensure_ascii=False))
    assert row['status']=='partial',row['answer']
    assert '97.3%' in db.decode(row['metrics'])['answer_validation_issue']


def test_derived_percentage_when_multiplying_by_100_without_percent_sign():
    records,issue=check('计算相对提升。',
        '相对提升 = (0.839-0.669)/0.669 × 100 ≈ 25.41%。', RATIOS)
    assert not issue,issue
    assert any(r['value']=='25.41%' and r['verified'] for r in records)


def test_question_input_is_not_a_paper_result_without_paper_source():
    records,issue=check('有人说准确率为97.3%，你怎么看？',
        '论文报告准确率为97.3%。', NO_NUMBERS)
    assert '97.3%' in issue
    assert records[0]['verified'] is False


# RC3: actual 2026-11-25 Memory-OFF regression. A comma after "For example"
# must not make a valid synthetic ratio look like an unsourced paper statistic.
def test_english_comma_if_example_calculated_decimal_is_hypothetical():
    question = 'Can you explain how Chunk Coverage is calculated in this paper?'
    answer = (
        'Chunk Coverage (CC) measures the fraction of corpus chunks retrieved '
        'at least once across all tests in a RAG system. For example, if a test '
        'suite runs 3 queries that retrieve 5 unique chunks from a total corpus '
        'of 20 chunks, CC = 5/20 = 0.25. CC focuses purely on the retrieved '
        'unique chunks, not answer correctness.'
    )
    records, issue = check(question, answer)
    assert not issue, issue
    assert records == [{'value': '0.25', 'kind': 'hypothetical', 'verified': True}]


def test_english_comma_if_example_wrong_decimal_still_rejected():
    _, issue = check(
        'How is coverage calculated?',
        'For example, if 5 unique chunks are returned out of 20, CC = 5/20 = 0.35.',
    )
    assert '不一致' in issue


def test_english_for_example_real_paper_fact_is_not_excused():
    _, issue = check(
        'What were the empirical results?',
        'For example, the study reports the test accuracy reached 0.97.',
    )
    assert '0.97' in issue


def test_english_for_example_does_not_waive_next_sentence():
    _, issue = check(
        'What were the empirical results?',
        'For example, if 5 chunks are found out of 20, CC = 5/20 = 0.25. '
        'The experiment reports score 0.97.',
    )
    assert '0.97' in issue


def test_english_decimal_example_memory_off_agent_end_to_end(monkeypatch):
    import json
    from app import db
    from test_routing_analytic import _run
    question = 'Can you explain how Chunk Coverage is calculated in this paper?'
    answer = (
        'Chunk Coverage is the proportion of distinct document chunks returned '
        'by all test queries. For example, if the corpus has 20 chunks and '
        'queries retrieve 5 unique chunks in total, CC = 5/20 = 0.25. '
        'Repeatedly finding a chunk does not increase this fraction.'
    )
    row, model, _ = _run(monkeypatch, question, json.dumps({'answer': answer, 'citations': [1]}))
    stats = db.decode(row['metrics'])
    assert row['status'] == 'succeeded', stats
    assert stats['answer_complete'] is True
    assert stats['learning_memory_applied'] is False
    assert stats['llm_calls'] == 2  # evidence check + initial answer, no repair
    assert stats['numeric_provenance'] == [
        {'value': '0.25', 'kind': 'hypothetical', 'verified': True}]

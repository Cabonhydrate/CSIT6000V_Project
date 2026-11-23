"""P2.1: evidence for *refuting* a quoted quantitative premise.

No CC paper gold, special question ID, or hard-coded answer is used here.
The example evidence has the same source structure as ordinary indexed PDFs.
"""
from app import db
from app.agent.workflow import (
    _claim_specific_source_evidence,
    _claim_citation_grounding_issue,
    _percentage_rebuttal_strength,
    _unsupported_precise_numbers,
)
from test_routing_analytic import _seed_run
from test_cc8_evidence_binding_v030 import _run_with_sources


def _claim_payload(second_ref, second_verdict='refuted'):
    return {'claims': [
        {'claim': '指标越高答案一定正确', 'verdict': 'refuted',
         'explanation': '指标不衡量回答正确性。', 'evidence_ids': [2]},
        {'claim': '系统发布前必须达到100%覆盖', 'verdict': second_verdict,
         'explanation': '完整覆盖不是普遍的强制门槛。', 'evidence_ids': [second_ref]},
    ]}


def test_same_percentage_is_not_sufficient_for_normative_refutation():
    claim='系统发布前必须达到100%覆盖'
    unrelated='The maximum observed coverage remains strictly below 100% in Table 2.'
    direct='The method does not imply that full (100%) coverage is always desirable.'
    assert _percentage_rebuttal_strength(unrelated,'100%',claim)==1
    assert _percentage_rebuttal_strength(direct,'100%',claim)==3
    q='有人说：“指标越高答案一定正确，而且系统发布前必须达到100%覆盖。”'
    evidence=[{'content':'The metric does not measure answer correctness.'},
              {'content':unrelated},{'content':direct}]
    assert '第 2 条' in _claim_citation_grounding_issue(q,_claim_payload(2),evidence)
    assert _claim_citation_grounding_issue(q,_claim_payload(3),evidence)==''


def test_full_coverage_denial_without_literal_100_is_valid_counterevidence():
    q='有人说：“指标越高答案一定正确，而且系统发布前必须达到100%覆盖。”'
    evidence=[{'content':'This measures retrieval behavior, not answer correctness.'},
              {'content':'For open knowledge tasks, full coverage is neither required nor expected.'}]
    payload=_claim_payload(2)
    assert _claim_citation_grounding_issue(q,payload,evidence)==''
    # A positively supported numerical assertion has different requirements.
    payload['claims'][1]['verdict']='supported'
    assert '第 2 条' in _claim_citation_grounding_issue(q,payload,evidence)


def test_unrelated_denial_in_another_sentence_is_not_support():
    claim='系统发布前必须达到100%覆盖'
    bad='The method reached 100% coverage. Separately, a baseline model is not required.'
    assert _percentage_rebuttal_strength(bad,'100%',claim)==1
    assert _percentage_rebuttal_strength('100% is reported without explanation.',
                                         '100%',claim)==1


def test_recover_direct_rebuttal_even_when_an_unrelated_100_is_already_selected():
    q='有人说：“指标越高答案一定正确，而且系统发布前必须达到100%覆盖。”'
    paper,_,_=_seed_run(q)
    with db.connect() as conn:
        conn.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) '
                     'VALUES(?,?,?,?,?,?,?)',
                     (f'{paper}:c1', paper,1,'Measured results',
                      'The largest possible value is below 100% for this data.',
                      db.encode(['s1']),db.encode([3])))
        conn.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) '
                     'VALUES(?,?,?,?,?,?,?)',
                     (f'{paper}:c2', paper,2,'Scope of metric',
                      'The metric does not imply that full (100%) coverage is always desirable.',
                      db.encode(['s2']),db.encode([4])))
    selected=[{'evidence_id':f'{paper}:c1','paper_id':paper,
               'content':'The largest possible value is below 100% for this data.',
               'page_numbers':[3]}]
    upgraded=_claim_specific_source_evidence(selected,q,[paper])
    assert upgraded[0]['evidence_id']==f'{paper}:c2'
    assert upgraded[0]['page_numbers']==[4]
    assert upgraded[1]['evidence_id']==f'{paper}:c1'
    # An unrelated paper cannot be pulled in by the background search.
    assert _claim_specific_source_evidence(selected,q,['not-this-paper'])==selected


def test_numeric_factual_claim_remains_literal_strict():
    q='有人说：“模型很准确，而且实验达到98.5%精确率。”'
    payload={'claims':[{'claim':'模型很准确','verdict':'supported',
                        'explanation':'支持','evidence_ids':[1]},
                       {'claim':'实验达到98.5%精确率','verdict':'supported',
                        'explanation':'支持','evidence_ids':[1]}]}
    assert '98.5%' in _claim_citation_grounding_issue(q,payload,[{'content':'Precision is 97%.'}])
    assert _claim_citation_grounding_issue(q,payload,[{'content':'Precision is 98.5%.'}])==''


def test_unrelated_100pct_accuracy_cannot_be_justified_by_full_coverage():
    q='有人说：“软件非常好，而且任何分类器必须实现100%准确率。”'
    payload={'claims':[{'claim':'软件非常好','verdict':'refuted','explanation':'不成立','evidence_ids':[1]},
                       {'claim':'任何分类器必须实现100%准确率','verdict':'refuted',
                        'explanation':'不是强制条件','evidence_ids':[1]}]}
    assert '第 2 条' in _claim_citation_grounding_issue(
        q,payload,[{'content':'Full coverage is neither required nor expected.'}])


def test_unverified_precise_generated_numbers_remain_blocked():
    issue=_unsupported_precise_numbers('Why not use entropy?',
        'The entropy is 0.94.',[{'content':'Entropy is not an actionable selection signal.'}])
    assert '0.94' in issue


def test_full_agent_answers_two_assertions_with_different_sources(monkeypatch):
    question='有人说：“指标越高答案一定正确，而且系统发布前必须达到100%覆盖。”请逐条判断。'
    source1=('The metric characterizes retrieval behavior and does not guarantee answer correctness. '
             'The maximum attainable coverage stays strictly below 100%.')
    source2=('The authors explain that it is misleading to imply that full (100%) coverage '
             'is always desirable. Full coverage is neither required nor expected.')
    payload={'claims':[
        {'claim':'指标越高答案一定正确', 'verdict':'refuted',
         'explanation':'覆盖率不保证答案正确性。','evidence_ids':[2],
         'source_quote':'does not guarantee answer correctness'},
        {'claim':'系统发布前必须达到100%覆盖','verdict':'refuted',
         'explanation':'完整覆盖不是普遍要求。','evidence_ids':[1],
         'source_quote':'full (100%) coverage is always desirable'},
    ]}
    row=_run_with_sources(monkeypatch,question,[(9,source1),(7,source2)],payload)
    assert row['status']=='succeeded',db.decode(row['metrics']).get('answer_validation_issue')
    citations=db.decode(row['citations'])
    assert {c['page_numbers'][0] for c in citations}=={7,9}
    metrics=db.decode(row['metrics'])
    assert metrics.get('claim_specific_evidence_added'),metrics
    assert metrics.get('answer_complete') is True

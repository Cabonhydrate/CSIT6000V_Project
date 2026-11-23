"""Regression: a grounded multi-table answer must not fail because Qwen omitted [2]."""
from app import db
from app.agent.workflow import _reconcile_claim_citations
from test_multi_table_claims_v016 import TABLE_A, TABLE_B, execute_case


def complete_answer():
    return ('1. Table 3 的 Average 行报告平均加速 1.6×，此项正确。\n'
            '2. Atlas 不是 1.6 倍：Table 3 中 Random 20 次、Coverage-guided 10 次，20/10=2.0 倍。\n'
            '3. 不是所有 APFD 都更好。Table 9 的 Response Groundedness，'
            'Random=0.670，CC-guided=0.610，覆盖引导在这项上更低。')


def test_grounded_claim_repaired_when_llm_returns_no_citations(monkeypatch):
    rec,_=execute_case(monkeypatch,complete_answer(),citations=[])
    assert rec['status']=='succeeded',rec['answer']
    assert rec['answer'].count('[1]')>=1
    assert rec['answer'].count('[2]')>=1
    assert [c['page_numbers'] for c in db.decode(rec['citations'])]==[[2],[4]]
    assert db.decode(rec['metrics'])['verified_claim_tables']==[3,9]


def test_grounded_claim_repaired_when_llm_returns_only_first_citation(monkeypatch):
    rec,_=execute_case(monkeypatch,complete_answer(),citations=[1])
    assert rec['status']=='succeeded',rec['answer']
    assert [c['page_numbers'] for c in db.decode(rec['citations'])]==[[2],[4]]


def test_bad_inline_citations_replaced_with_verified_sources(monkeypatch):
    claimed=complete_answer().replace('1.6×，此项正确。','1.6×，此项正确。[2]')
    claimed=claimed.replace('覆盖引导在这项上更低。','覆盖引导在这项上更低。[1]')
    rec,_=execute_case(monkeypatch,claimed,citations=[1])
    assert rec['status']=='succeeded',rec['answer']
    assert '正确。 [1]' in rec['answer']
    assert '更低。 [2]' in rec['answer']


def test_a_table_name_alone_does_not_fabricate_citation(monkeypatch):
    partial='Table 3 的 Atlas 20/10=2.0 倍。Table 9 没有别的信息可以解释。'
    rec,_=execute_case(monkeypatch,partial,citations=[1,2])
    assert rec['status']=='partial'
    assert 'Table 9' in db.decode(rec['metrics'])['answer_validation_issue']


def test_wrong_numeric_row_still_rejected_after_auto_citation(monkeypatch):
    wrong=complete_answer().replace('Random=0.670，CC-guided=0.610',
                                    'Random=0.610，CC-guided=0.670')
    rec,_=execute_case(monkeypatch,wrong,citations=[])
    assert rec['status']=='partial'
    assert 'Random' in db.decode(rec['metrics'])['numeric_fact_conflict']


def test_generic_table_facts_match_without_paper_specific_vocabulary():
    a=dict(TABLE_A,paper_id='p')
    b=dict(TABLE_B,paper_id='p')
    selected=[{'paper_id':'p','page_numbers':[2],'content':'Table 3: data'},
              {'paper_id':'p','page_numbers':[4],'content':'Table 9: data'}]
    text,indices,tables=_reconcile_claim_citations(complete_answer(),selected,[a,b])
    assert indices==[1,2]
    assert tables=={3,9}
    assert '[1]' in text and '[2]' in text


def test_reference_only_tail_does_not_leave_broken_sentence(monkeypatch):
    """Qwen may append `引用 [1] 和 [2]。` after the conclusion."""
    rec,_=execute_case(monkeypatch,complete_answer()+' 引用 [1] 和 [2]。',citations=[])
    assert rec['status']=='succeeded',rec['answer']
    assert '引用 和 。' not in rec['answer']
    assert '引用 [1] 和 [2]' not in rec['answer']
    assert rec['answer'].count('[1]')>=1
    assert rec['answer'].count('[2]')>=1
    assert '覆盖引导在这项上更低' in rec['answer']


def test_reference_only_tail_with_colon_and_no_spaces(monkeypatch):
    rec,_=execute_case(monkeypatch,complete_answer()+'\n来源：[1]、[2]。',citations=[])
    assert rec['status']=='succeeded',rec['answer']
    assert '来源：' not in rec['answer']
    assert '覆盖引导在这项上更低' in rec['answer']

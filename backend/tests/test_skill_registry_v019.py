"""Skill-registry regression tests; no Ollama or internet needed."""
import asyncio
import json
from app import db
from app.agent.workflow import AgentRunner
from app.knowledge import retrieval
from app.schemas import ChatRequest
from app.skills import (select_answer_skills, skill_instructions,
                        normalize_skill_output, validate_skill_output)
from test_multi_question import _fake_langgraph
from test_multi_table_claims_v016 import prepare_case, QUESTION


def test_select_task_and_style_skills_independently():
    assert select_answer_skills('claim_check', teaching=True).names == (
        'verifyClaims', 'explainForBeginner')
    assert select_answer_skills('formula', teaching=False).names == ('explainFormula',)
    assert select_answer_skills('comparison', teaching=True).names == ('explainForBeginner',)
    assert select_answer_skills('claim_check', teaching=False).names == ('verifyClaims',)
    assert select_answer_skills('general', teaching=False).names == ()


def test_skill_instructions_loaded_from_modular_files():
    text=skill_instructions(select_answer_skills('claim_check',teaching=True))
    assert 'Skill: verifyClaims' in text
    assert 'Style Skill: explainForBeginner' in text
    assert 'evidence_ids' in text


def test_structured_claims_are_rendered_in_order_and_only_valid_refs():
    obj={'claims':[
        {'claim':'平均加速是 1.6×', 'verdict':'supported', 'explanation':'Table 3 Average 为 1.6×',
         'evidence_ids':[1,1,7]},
        {'claim':'所有指标均更好', 'verdict':'refuted',
         'explanation':'Table 9 的 Random=0.670 高于 CC=0.610', 'evidence_ids':[2]}
    ]}
    out=normalize_skill_output(obj,'verifyClaims',2)
    assert out.structured and out.complete
    assert out.citations == (1,2)
    assert '1. **正确**' in out.answer
    assert '2. **错误**' in out.answer
    assert '[7]' not in out.answer
    assert validate_skill_output(out,'verifyClaims','请核验两个判断')==''


def test_claim_skill_rejects_missing_structured_claims():
    obj={'claims':[{'claim':'平均快','verdict':'supported','explanation':'来源 Average 行','evidence_ids':[1]}]}
    out=normalize_skill_output(obj,'verifyClaims',2)
    assert '1/3' in validate_skill_output(out,'verifyClaims','请核验三个判断')
    broken=normalize_skill_output({'claims':[{'verdict':'refuted'}]},'verifyClaims',3)
    assert validate_skill_output(broken,'verifyClaims','核验')


def test_formula_skill_renders_math_and_variables():
    data={'formula':r'CC(T)=\frac{|E|}{|C|}', 'variables':[
        {'symbol':'T','meaning':'测试查询集合'}, {'symbol':'E','meaning':'已覆盖的块'},
        {'symbol':'C','meaning':'全部块'}],
          'explanation':'对不同块去重后计算覆盖比例。', 'citations':[1]}
    out=normalize_skill_output(data,'explainFormula',1)
    assert out.structured and out.complete and out.citations==(1,)
    assert r'\[' in out.answer and r'\frac' in out.answer
    assert r'\(T\)' in out.answer
    assert validate_skill_output(out,'explainFormula','解释每个变量')==''


def test_legacy_json_supported_during_ollama_migration():
    out=normalize_skill_output({'answer':'旧版模型的回答 [1]','citations':[1]},'verifyClaims',2)
    assert not out.structured and out.answer.startswith('旧版')
    assert out.citations==(1,)


def test_unknown_citations_are_never_mapped_to_sources():
    out=normalize_skill_output({'answer':'测试', 'citations':[0,7,'1',True]},'explainFormula',2)
    assert out.citations==()


def test_end_to_end_structured_claims_with_real_table_rows(monkeypatch):
    _fake_langgraph(monkeypatch)
    paper,conversation,run,sources=prepare_case()
    monkeypatch.setattr(retrieval,'search',lambda *a,**kw: {'evidences':sources,'latency_ms':1})
    structured={'claims':[
        {'claim':'Table 3 平均加速 1.6 倍', 'verdict':'supported',
         'explanation':'Average 行的 Speedup(R) 是 1.6×。', 'evidence_ids':[1]},
        {'claim':'Atlas 也加速 1.6 倍', 'verdict':'refuted',
         'explanation':'Table 3 Atlas 的 Random 是 20，Coverage-guided 是 10，故 20/10=2.0×。',
         'evidence_ids':[1]},
        {'claim':'Table 9 中所有 APFD 指标更好', 'verdict':'refuted',
         'explanation':'Table 9 的 Atlas Response Groundedness，Random=0.670，CC-guided=0.610。',
         'evidence_ids':[2]},
    ]}
    class MockModel:
        async def chat(self,messages,**kwargs):
            return {'content':json.dumps(structured,ensure_ascii=False),'latency_ms':1}
        async def stream(self,*args,**kw):
            raise AssertionError('claim skill should not stream raw content')
            yield ''
    asyncio.run(AgentRunner(MockModel()).run(run,ChatRequest(
        conversation_id=conversation,question=QUESTION,paper_ids=[paper],mode='adaptive',teaching=True)))
    rec=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert rec['status']=='succeeded',rec['answer']
    assert '**错误**' in rec['answer'] and '**正确**' in rec['answer']
    assert [x['page_numbers'] for x in db.decode(rec['citations'])]==[[2],[4]]
    metrics=db.decode(rec['metrics'])
    assert metrics['structured_skill_output'] is True
    assert metrics['structured_claims_count']==3
    assert metrics['answer_skills']==['verifyClaims','explainForBeginner']
    assert any(ev['event_type']=='skill_selected' for ev in db.rows('SELECT * FROM run_events WHERE run_id=?',(run,)))


def test_explicit_formula_symbols_must_be_explained_individually():
    question='逐一解释 T、C、R_k(q)、k 和并集符号'
    data={'formula':r'CC(T)=\frac{|E|}{|C|}',
          'variables':[{'symbol':'T','meaning':'query set'},
                       {'symbol':'C','meaning':'corpus'}],
          'explanation':'ratio','citations':[1]}
    out=normalize_skill_output(data,'explainFormula',1)
    issue=validate_skill_output(out,'explainFormula',question)
    assert 'R_k(q)' in issue and 'k' in issue
    data['variables'] += [
        {'symbol':'R_k(q)','meaning':'retrieved results'},
        {'symbol':'k','meaning':'Top-k size'},
    ]
    assert validate_skill_output(normalize_skill_output(data,'explainFormula',1),
                                 'explainFormula',question)==''


def test_structured_formula_reaches_existing_grounded_citation_path(monkeypatch):
    import json
    from test_routing_analytic import _run, FORMULA
    reply={'formula':r'CC(T)=\frac{|\bigcup_{q\in T} R(q)|}{|D|}',
           'variables':[{'symbol':'T','meaning':'测试查询集合'},
                        {'symbol':'R(q)','meaning':'每次查询检索的块'},
                        {'symbol':'D','meaning':'语料库全部块'}],
           'explanation':'并集表示同一个块只计算一次。','citations':[1]}
    rec,model,_=_run(monkeypatch,FORMULA,json.dumps(reply,ensure_ascii=False))
    assert rec['status']=='succeeded',rec['error']
    assert r'\frac' in rec['answer'] and r'\(T\)' in rec['answer']
    assert db.decode(rec['metrics'])['structured_skill_output'] is True
    assert db.decode(rec['metrics'])['answer_skills']==['explainFormula','explainForBeginner']
    assert db.decode(rec['citations'])[0]['page_numbers']==[7]

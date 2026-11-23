"""Day1 skill manifest, teaching evidence, and realistic small-model failures."""
import json

from app.skills.registry import get_skill_definition, skill_instructions, SkillPlan
from app.skills.teaching import (validate_and_render_teaching,
                                 prioritize_teaching_evidence, teaching_evidence_quote)
from app.agent.skills import select_skill
from test_day1_learning_memory import prepare, do_run, CaptureModel

QUESTION = '请根据我的学习背景，解释论文中的 Chunk Coverage。重点帮助我理解尚不熟悉的部分，并给一个简单的例子。'
EVIDENCE = [
    {'content':'Chunk Coverage (CC) measures the fraction of distinct document chunks retrieved across test queries, relative to all chunks in the corpus.',
     'evidence_id':'paper:c1','page_numbers':[6], 'paper_id':'p', 'chunk_id':'paper:c1'},
    {'content':'Table 3 results compare Chunk Coverage to retrieval diversity measurements and report dataset outcomes.',
     'evidence_id':'paper:c2','page_numbers':[13], 'paper_id':'p', 'chunk_id':'paper:c2'},
]
VALID = {
    'definition':'Chunk Coverage 衡量测试查询检索到的不同文档片段占整个语料文档片段的比例。',
    'example':'假设论文分成5个文档块，多次检索总共命中3个不同的块，则 CC=3/5=60%。',
    'takeaway':'同一个文档块命中多次仍只计一次，不代表答案一定准确。',
    'citations':[1],
}


def test_all_seven_standardized_skill_manifests_load():
    names = ('verifyClaims', 'explainFormula', 'constructCounterexample',
             'explainForBeginner', 'guidePaperReading', 'comparePapers', 'suggestLearningPath')
    for name in names:
        spec = get_skill_definition(name)
        assert spec.name == name and spec.version == 1
        assert spec.output_contract and spec.description and spec.instructions
    assert get_skill_definition('explainForBeginner').type == 'style'
    assert get_skill_definition('guidePaperReading').type == 'task'
    assert 'Procedure' in skill_instructions(SkillPlan(None, ('guidePaperReading',)))


def test_compatibility_skill_selector_uses_same_file_backed_prompts():
    profile={'domains':['ML'], 'explanation_level':'beginner'}
    for question,expected in (
            ('请导读论文', 'guidePaperReading'),
            ('比较两篇论文', 'comparePapers'),
            ('给我一个学习路径', 'suggestLearningPath'),
            ('请解释术语', 'explainForBeginner')):
        name, text=select_skill(question,profile,True)
        assert name==expected and text==get_skill_definition(name).instructions


def test_teaching_validation_renders_source_grounded_short_answer():
    result=validate_and_render_teaching(VALID,EVIDENCE,QUESTION)
    assert not result.issue
    assert result.citation_ids==(1,)
    assert '**核心定义**' in result.text
    assert '**举个例子（假设）**' in result.text
    assert '3/5=60%' in result.text
    assert '代码块' not in result.text
    assert '[1]' in result.text


def test_teaching_rejects_document_code_confusion_and_unsupported_metrics():
    wrong={**VALID,'definition':'Chunk Coverage 表示测试执行的代码块比例。'}
    assert '代码' in validate_and_render_teaching(wrong,EVIDENCE,QUESTION).issue
    wrong={**VALID,'takeaway':'它与多样性指标负相关，因此模型答案更准确。'}
    assert '负相关' in validate_and_render_teaching(wrong,EVIDENCE,QUESTION).issue
    wrong={**VALID,'takeaway':'Table 3 证明模型输出变好了。'}
    assert '实验' in validate_and_render_teaching(wrong,EVIDENCE,QUESTION).issue


def test_teaching_rejects_missing_example_or_invalid_citations():
    assert '数字' in validate_and_render_teaching({**VALID,'example':'例如两个查询检索到同样的块。'},
                                                EVIDENCE,QUESTION).issue
    assert '来源' in validate_and_render_teaching({**VALID,'citations':[8]},EVIDENCE,QUESTION).issue
    assert '概念' in validate_and_render_teaching({**VALID,'definition':'Context Precision 是相关性指标。'},
                                                EVIDENCE,QUESTION).issue


def test_teaching_prioritizes_definitions_and_source_quote():
    long = 'Lots of preliminary discussion. ' * 90 + EVIDENCE[0]['content']
    src={**EVIDENCE[0], 'content':long}
    evidence=[EVIDENCE[1],src]
    ranked=prioritize_teaching_evidence(evidence,QUESTION)
    assert ranked[0]['evidence_id']=='paper:c1'
    quote=teaching_evidence_quote(src,QUESTION,max_chars=400)
    assert 'Chunk Coverage' in quote and 'fraction' in quote
    assert len(quote)<=400


def test_full_agent_bad_concept_gets_bounded_repair(monkeypatch):
    paper=prepare(monkeypatch)

    class OneBadThenCorrect(CaptureModel):
        def __init__(self):
            super().__init__();self.generated=0

        async def chat(self,messages,**kwargs):
            if '"definition"' not in messages[0]['content']:
                return await super().chat(messages,**kwargs)
            self.chat_inputs.append(messages)
            self.generated+=1
            if self.generated==1:
                data={**VALID,'definition':'Chunk Coverage 是被测试用例执行的代码块比例。'}
            else:
                data=VALID
            return {'content':json.dumps(data,ensure_ascii=False),'latency_ms':1}

    model=OneBadThenCorrect()
    metrics=do_run(paper,model)
    assert model.generated==2
    assert metrics['teaching_answer_repaired'] is True
    assert metrics['teaching_skill_structured'] is True
    assert '代码' in metrics['teaching_validation_attempts'][0]['issue']
    assert metrics['learning_context_applied'] is True

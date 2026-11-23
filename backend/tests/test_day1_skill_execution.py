"""Regressions from six real Day 1 acceptance prompts (synthetic model outputs)."""
import asyncio
import json
from uuid import uuid4

from app import db
from app.schemas import ChatRequest
from app.agent.skills import select_skill
from app.agent.learning_context import choose_learning_memories
from app.agent.workflow import AgentRunner,_split_question_parts
from app.agent.facts import direct_table_answer
from app.skills.task_outputs import (task_mode,render_task_output,render_claim_verdicts,
                                     user_set_calculation)
from test_day1_learning_memory import prepare, CaptureModel, NOTE

T1=('请带我阅读这篇论文。假设我是第一次接触学术论文的研究生，按照'
    '“研究背景 → 现有方法的问题 → 核心创新 → 实验设计 → 主要结论”的顺序解释。每部分尽量简单，并指出对应的论文证据。')
T2=('请根据论文判断以下三个说法是否正确，并分别给出证据和解释：\n'
    '1. Chunk Coverage 越高，RAG 生成的答案就一定越准确。\n'
    '2. 同一个 Chunk 被不同查询检索到多次，只计算一次覆盖。\n'
    '3. Chunk Coverage 可以在不依赖人工标准答案的情况下衡量检索测试的覆盖程度。')
T3=('请写出论文中 Chunk Coverage 的正式数学公式，解释公式中的每个符号，并用下面的具体数据一步步计算：\n'
    '语料库 C = {c1,c2,c3,c4,c5}。\n'
    '查询 q1 检索到 {c1,c2}，查询 q2 检索到 {c2,c3}。\n'
    '两次查询一共覆盖了多少个不同 Chunk？CC 等于多少？为什么 c2 不能重复计数？')
T5=('根据我已有的学习记录，假设我了解 RAG 的基本流程，但不熟悉集合运算和检索评价指标，'
    '请为我设计一个学习 Chunk Coverage 的三阶段学习路径。每个阶段说明需要掌握什么、为什么学习、如何判断自己已经理解。')
T6=('我现在准备继续学习这篇论文。请结合你已经保存的我的学习情况，告诉我哪些知识我已经接触过、'
    '哪些知识还需要重点补充，并推荐我下一步阅读论文的哪个部分。不要假设我已经掌握了所有学过的知识。')
EVIDENCE=[{'evidence_id':'doc:c0','paper_id':'doc','chunk_id':'doc:c0','content':
          'Chunk Coverage (CC) is the fraction of distinct document chunks retrieved at least once.'}]


def test_routes_and_subquestion_borders_from_user_prompts():
    profile={'domains':['RAG'],'explanation_level':'beginner'}
    assert select_skill(T1,profile,True)[0]=='guidePaperReading'
    assert select_skill(T5,profile,True)[0]=='suggestLearningPath'
    assert select_skill(T6,profile,True)[0]=='suggestLearningPath'
    assert task_mode('guidePaperReading',T1,1)=='paper_guide'
    assert task_mode('suggestLearningPath',T5,1)=='learning_path'
    assert task_mode('suggestLearningPath',T6,1)=='learning_progress'
    assert len(_split_question_parts(T2))==3
    assert _split_question_parts(T2)[0].startswith('Chunk Coverage 越高')
    assert _split_question_parts(T3)==[T3]


def test_user_set_arithmetic_is_deterministic():
    value=user_set_calculation(T3)
    assert '3/5 = 60%' in value
    assert 'q1={c1,c2}' in value
    assert 'q2={c2,c3}' in value
    assert '={c1,c2,c3}' in value
    assert user_set_calculation('请说明公式的含义')==''
    assert user_set_calculation('C={c1,c2};q1={c3},q2={c1}')==''


def test_guide_needs_five_evidenced_sections():
    data={'sections':[{'heading':h,'content':'根据论文的这一段落介绍研究内容与相应证据。','citations':[1]}
                      for h in ('研究背景','现有方法的问题','核心创新','实验设计','主要结论')]}
    result=render_task_output('paper_guide',data,EVIDENCE)
    assert not result.issue
    assert result.text.count('**')==10
    assert result.citations==(1,)
    data['sections'][2]['content']=''
    assert '核心创新' in render_task_output('paper_guide',data,EVIDENCE).issue


def test_three_stage_learning_path_is_not_a_definition():
    data={'steps':[{'topic':f'第{i}主题','learn':'学习文档集合检索指标',
                    'why':'为后续读论文方法做准备', 'check':'能否手工计算覆盖率'}
                   for i in range(1,4)],'citations':[1]}
    r=render_task_output('learning_path',data,EVIDENCE)
    assert not r.issue and r.text.count('阶段')==3
    assert '自我检查' in r.text
    data['steps']=data['steps'][:1]
    assert '三个阶段' in render_task_output('learning_path',data,EVIDENCE).issue


def test_saved_learning_recap_requires_actual_memory():
    records=[{'topic':'Chunk Coverage','domain':'RAG','content':NOTE,'mastery':'studied'}]
    selected=choose_learning_memories(T6,records)
    assert len(selected)==1
    data={'studied':'曾经接触过 RAG 基本流程',
          'not_yet_mastered':'还不熟悉集合并集与检索覆盖指标',
          'next_reading':'继续阅读论文的 Chunk Coverage 定义部分',
          'citations':[1]}
    result=render_task_output('learning_progress',data,EVIDENCE,memories=selected)
    assert not result.issue
    assert '不等于掌握' in result.text
    assert '继续阅读' in result.text
    empty=render_task_output('learning_progress',data,EVIDENCE,memories=[])
    assert '没有找到' in empty.text


def test_claim_check_keeps_original_assertions_and_complete_verdicts():
    claims=_split_question_parts(T2)
    data={'verdicts':[{'verdict':v,'reason':'论文的原文证据支持或反驳了该项具体命题','citations':[1]}
                      for v in ('错误','正确','正确')]}
    r=render_claim_verdicts(data,claims,EVIDENCE)
    assert not r.issue
    assert 'RAG 生成的答案就一定越准确' in r.text
    assert r.text.count('**判断：')==3
    assert r.citations==(1,)
    assert '必须逐一判断' in render_claim_verdicts({'verdicts':[]},claims,EVIDENCE).issue


def test_multi_strategy_table_answer_lists_every_requested_value_and_correct_rounding():
    fact={'table':{'table_number':4,'page':15,
                   'caption':'Number of tests required to reach 50% of maximum reachable coverage',
                   'headers':['Dataset','Coverage-guided','Overlap-biased','Random','Speedup (R)']},
          'row':{'Dataset':'MIMIC-IV','Coverage-guided':'248','Overlap-biased':'1,301',
                 'Random':'530','Speedup (R)':'2.1×'}}
    q=('根据 Table 4，找出 MIMIC-IV 使用 CC-guided、Random、Overlap-biased 三种策略需要多少次查询。'
       '计算 Random 相比 CC-guided 需要的查询次数倍数，并解释这一结果说明了什么。')
    answer=direct_table_answer(q,fact)
    assert answer and all(x in answer for x in ('248','530','1,301','2.14'))


def _run(paper,model,question):
    run=uuid4().hex;conv=uuid4().hex
    with db.connect() as c:
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                  (conv,db.settings.owner_id,'Skill test',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                  (run,conv,'pending','adaptive',question,db.encode([paper]),db.now()))
    asyncio.run(AgentRunner(model).run(run,ChatRequest(conversation_id=conv,question=question,
              paper_ids=[paper],mode='adaptive',teaching=True,use_memory=True)))
    row=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert row['status']=='succeeded',row.get('error')
    return row['answer'],db.decode(row['metrics']),db.decode(row['citations'])


def test_end_to_end_task_specific_rendering_and_memory(monkeypatch):
    paper=prepare(monkeypatch)
    class TaskModel(CaptureModel):
        async def chat(self,messages,**kwargs):
            self.chat_inputs.append(messages)
            mode=messages[0]['content']
            if 'paper_guide' in mode:
                data={'sections':[{'heading':h,'content':'根据论文提出的检索覆盖研究与实验结果进行简明概括。','citations':[1]}
                                  for h in ('研究背景','现有方法的问题','核心创新','实验设计','主要结论')]}
            elif 'learning_path' in mode:
                data={'steps':[{'topic':f'基础主题 {i}','learn':'学习集合的并集运算',
                                'why':'理解检索结果去重计算', 'check':'计算两个集合的并集'} for i in (1,2,3)],'citations':[1]}
            elif 'learning_progress' in mode:
                data={'studied':'已经接触过 RAG 基本流程',
                      'not_yet_mastered':'集合并集和 Chunk Coverage 尚不熟悉',
                      'next_reading':'下一步阅读论文 Chunk Coverage 定义部分','citations':[1]}
            else:
                return await super().chat(messages,**kwargs)
            return {'content':json.dumps(data,ensure_ascii=False),'latency_ms':1}

    for q,mode,needle in ((T1,'paper_guide','研究背景'),
                          (T5,'learning_path','阶段 1'),
                          (T6,'learning_progress','已经接触过')):
        model=TaskModel()
        text,metrics,cites=_run(paper,model,q)
        assert metrics['task_skill_mode']==mode
        assert needle in text
        assert cites and cites[0]['number']==1
        if q==T6:
            assert metrics['learning_memory_candidates']>=1
            assert any(NOTE in str(call) for call in model.chat_inputs)


def test_end_to_end_three_numbered_claims_are_single_complete_verdict_task(monkeypatch):
    paper=prepare(monkeypatch)
    class VerdictModel(CaptureModel):
        async def chat(self,messages,**kwargs):
            self.chat_inputs.append(messages)
            if '原始断言' in messages[-1]['content'] and 'verdicts' in messages[0]['content']:
                data={'verdicts':[{'verdict':v,'reason':'这是论文中不同指标或定义可以支持的判断',
                                   'citations':[1]} for v in ('错误','正确','正确')]}
                return {'content':json.dumps(data,ensure_ascii=False),'latency_ms':1}
            return await super().chat(messages,**kwargs)
    model=VerdictModel()
    answer,metrics,cites=_run(paper,model,T2)
    assert metrics['subquestions_total']==3
    assert metrics['structured_verdicts'] is True
    assert metrics['subquestions_answered']==3
    assert answer.count('**原断言：')==3
    assert '答案就一定越准确' in answer
    assert '错误' in answer and len(cites)==1

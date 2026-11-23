"""Source-grounded regressions for cross-passage experiments and atomic claims.

Artificial papers/models only; these tests never contain benchmark gold answers.
"""
import json
from app import db
from app.agent.workflow import (
    _claim_specific_source_evidence, _claim_citation_grounding_issue,
    _rebind_claim_source_quotes, _cited_followup_experiment,
)
from app.skills.quality import matching_experiment_citation
from test_routing_analytic import _seed_run


def test_followup_model_identifier_can_touch_chinese_characters():
    evidence=[
        {'content': 'A table lists earlier results for the baseline model.'},
        {'content': 'To test the result, we repeated the experiment using a stronger generator (Alpha-2-mini). '
                    'In this setup the pattern reversed.'},
    ]
    answer='随后进行了补充实验：使用Alpha-2-mini生成新查询，对比了原始结果。'
    assert matching_experiment_citation(answer,evidence)==2
    # The same token with an ASCII letter suffix is a DIFFERENT name.
    assert matching_experiment_citation('使用Alpha-2-miniPlus重新运行。',evidence) is None


def test_followup_cannot_attach_unmentioned_configuration():
    evidence=[{'content': 'We repeated the experiment using Beta-3 but the model changed.'}]
    assert matching_experiment_citation('重新实验，没有说明用哪个模型。',evidence) is None
    assert matching_experiment_citation('Beta-3的原始性能很好。',evidence) is None


def test_followup_reference_placed_near_matching_sentence():
    text='第一阶段结果为 0.3。随后进行了补充实验，换用 Alpha-2-mini 生成问题。'
    rendered=_cited_followup_experiment(text,[1,3],3)
    assert '0.3。 [1]' in rendered
    assert 'Alpha-2-mini 生成问题。 [3]' in rendered
    assert rendered.count('[3]')==1


def test_literal_percentage_anchor_recovers_real_scoped_chunk_only():
    question='请判定：“指标很有用，而且所有项目必须达到100%覆盖率。”'
    paper,_,_=_seed_run(question)
    with db.connect() as conn:
        conn.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) '
                     'VALUES(?,?,?,?,?,?,?)',
                     (f'{paper}:c1',paper,1,'Limitations',
                      'Not all systems need full (100%) coverage; external retrieval is optional in some settings.',
                      db.encode(['s2-p1']),db.encode([12])))
    selected=[{'evidence_id':f'{paper}:c0','paper_id':paper,
               'content':'The paper defines a coverage metric.', 'page_numbers':[2]}]
    upgraded=_claim_specific_source_evidence(selected,question,[paper])
    assert upgraded[0]['evidence_id']==f'{paper}:c1'
    assert upgraded[0]['page_numbers']==[12]
    assert upgraded[1]['evidence_id']==f'{paper}:c0'
    # Strictly scoped to selected papers; cannot pull unrelated data.
    assert _claim_specific_source_evidence(selected,question,['nonexistent-paper'])==selected


def test_atomic_percentage_citation_must_refer_to_numbered_paper_source():
    question='有人说：“测试越多越好，而且上线前必须达到100%覆盖。”'
    claims=[{'claim':'测试越多越好','verdict':'refuted','explanation':'不一定', 'evidence_ids':[1]},
            {'claim':'上线前必须达到100%覆盖','verdict':'refuted',
             'explanation':'论文不规定通用上线标准','evidence_ids':[1]}]
    sources=[{'content':'The retrieval metric measures how many chunks have been exercised.'},
             {'content':'Not all settings need full (100%) coverage.'}]
    bad={'claims':claims}
    assert '第 2 条' in _claim_citation_grounding_issue(question,bad,sources)
    good={'claims':[claims[0],dict(claims[1],evidence_ids=[2])]}
    assert _claim_citation_grounding_issue(question,good,sources)==''


def test_claim_quote_rebinds_only_to_real_source():
    cited=[{'content':'A metric counts retrieved blocks.'},
           {'content':'The authors state that full coverage is not required for open settings.'}]
    quote='full coverage is not required for open settings'
    payload={'claims':[{'claim':'All settings must cover everything',
                        'verdict':'refuted','explanation':'This is not universal.',
                        'evidence_ids':[1],'source_quote':quote}]}
    repaired,issue,n=_rebind_claim_source_quotes(payload,cited)
    assert issue=='' and n==1
    assert repaired['claims'][0]['evidence_ids']==[2]
    assert payload['claims'][0]['evidence_ids']==[1]  # pure copy
    bad={'claims':[dict(payload['claims'][0],source_quote='Unrelated text that the paper never contained')]}
    _,issue,n=_rebind_claim_source_quotes(bad,cited)
    assert '不存在' in issue and n==0


def test_claim_quote_rebinding_does_not_require_optional_field():
    payload={'claims':[{'claim':'A', 'evidence_ids':[1]}]}
    assert _rebind_claim_source_quotes(payload,[{'content':'A source.'}])==(payload,'',0)


def _run_with_sources(monkeypatch, question, passages, payload):
    """Real AgentRunner + SQLite with a fake deterministic local model."""
    from uuid import uuid4
    import asyncio
    from app.agent.workflow import AgentRunner
    from app.knowledge import retrieval
    from app.schemas import ChatRequest
    from app.config import settings
    from test_multi_question import _fake_langgraph

    _fake_langgraph(monkeypatch)
    paper,conversation,run=(uuid4().hex for _ in range(3))
    with db.connect() as conn:
        conn.execute('INSERT INTO papers(id,owner_id,title,source_type,status,created_at) VALUES(?,?,?,?,?,?)',
                     (paper,settings.owner_id,'Synthetic source for binding tests','pdf','ready',db.now()))
        conn.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                     (conversation,settings.owner_id,'Cited QA regression',db.now()))
        conn.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                     (run,conversation,'pending','adaptive',question,db.encode([paper]),db.now()))
        for i,(page,content) in enumerate(passages):
            conn.execute('INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages) '
                         'VALUES(?,?,?,?,?,?,?)',
                         (f'{paper}:c{i}',paper,i,f'Page {page}',content,
                          db.encode([f'p{page}']),db.encode([page])))
    src=dict(evidence_id=f'{paper}:c0',paper_id=paper,chunk_id=f'{paper}:c0',
             section='Page '+str(passages[0][0]),page_numbers=[passages[0][0]],
             paragraph_refs=['p0'],content=passages[0][1])
    monkeypatch.setattr(retrieval,'search',lambda *a,**kw:{'evidences':[src],'latency_ms':2})

    class FakeModel:
        async def chat(self,messages,**kwargs):
            if messages[0]['content'].startswith('Judge if the cited passages'):
                return {'content':'{"sufficient":true,"missing":""}', 'latency_ms':1}
            return {'content':json.dumps(payload,ensure_ascii=False),'latency_ms':1}
        async def stream(self,*args,**kwargs):
            raise AssertionError('This path should be JSON non-streaming')
            yield ''
    asyncio.run(AgentRunner(FakeModel()).run(run,ChatRequest(
        conversation_id=conversation,question=question,paper_ids=[paper],
        mode='adaptive',teaching=True,use_memory=False,allow_web=False)))
    return db.one('SELECT * FROM runs WHERE id=?',(run,))


def test_full_agent_repairs_missing_experiment_citation_without_inventing_source(monkeypatch):
    passages=[
        (5,'Table 5. One setting shows an unexpected comparison in the evaluation.'),
        (4,'To investigate the exception, we repeated the experiment using a '
           'stronger generator (Alpha-2-mini). The surprising comparison disappeared.')
    ]
    q='根据 Table 5，这个例外是什么？论文又用什么补充实验解释这一例外？'
    reply={'answer':'Table 5 中出现一次例外。论文通过补充实验使用Alpha-2-mini生成新查询，结果例外消失。',
           'citations':[1]}  # model overlooked the supplement source
    row=_run_with_sources(monkeypatch,q,passages,reply)
    assert row['status']=='succeeded',row['answer']
    refs=db.decode(row['citations'])
    assert {c['page_numbers'][0] for c in refs}=={4,5}
    assert 'Alpha-2-mini生成新查询，结果例外消失。 [2]' in row['answer']
    assert db.decode(row['metrics']).get('experiment_citation_repaired') is True


def test_full_agent_recovers_second_claim_source_and_rejects_wrong_numeric_citation(monkeypatch):
    q='有人说：“指标 A 越高答案一定越准，而且系统发布前必须达到100%覆盖。”请逐条判断。'
    first='指标 A 越高答案一定越准'
    second='系统发布前必须达到100%覆盖'
    passages=[
        (9,'The proposed measure only characterizes retrieval behavior, '
           'rather than downstream answer correctness.'),
        (10,'In some deployments, full (100%) coverage is neither required nor expected.')
    ]
    good={'claims':[
        {'claim':first,'verdict':'refuted','explanation':'覆盖率不保证输出正确性。',
         'evidence_ids':[1], 'source_quote':'rather than downstream answer correctness'},
        {'claim':second,'verdict':'refuted','explanation':'完整覆盖不是普遍的强制要求。',
         'evidence_ids':[2], 'source_quote':'full (100%) coverage is neither required nor expected'},
    ]}
    row=_run_with_sources(monkeypatch,q,passages,good)
    assert row['status']=='succeeded',row['answer']
    assert {tuple(c['page_numbers']) for c in db.decode(row['citations'])}=={(9,),(10,)}
    assert '[1]' in row['answer'].split('2. **')[1]
    assert db.decode(row['metrics']).get('claim_specific_evidence_added')
    assert db.decode(row['metrics']).get('claim_source_quote_rebindings')==2


def test_full_agent_wrong_percentage_claim_citation_stays_partial(monkeypatch):
    q='有人说：“指标 A 越高答案一定越准，而且系统发布前必须达到100%覆盖。”请逐条判断。'
    passages=[
        (9,'The proposed measure only characterizes retrieval behavior, '
           'rather than downstream answer correctness.'),
        (10,'In some deployments, full (100%) coverage is neither required nor expected.')
    ]
    bad={'claims':[
        {'claim':'指标 A 越高答案一定越准','verdict':'refuted',
         'explanation':'输出正确性不一定增加。','evidence_ids':[2]},
        {'claim':'系统发布前必须达到100%覆盖','verdict':'refuted',
         'explanation':'100%不是强制上线条件。','evidence_ids':[2]},
    ]}
    row=_run_with_sources(monkeypatch,q,passages,bad)
    assert row['status']=='partial',row['answer']
    assert '100%' in db.decode(row['metrics'])['answer_validation_issue']

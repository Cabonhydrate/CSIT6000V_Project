"""Table row integrity and numeric grounding tests; no model/API calls.

Fixtures intentionally use imaginary datasets and unfamiliar table numbers.
"""
import asyncio
from pathlib import Path
from uuid import uuid4
import fitz
from app import db
from app.config import settings
from app.knowledge.tables import extract_pdf_tables, load_tables, store_tables
from app.agent.facts import select_fact_row, requested_columns, direct_table_answer, numeric_row_conflict
from app.agent.workflow import AgentRunner, _question_intent
from app.schemas import ChatRequest


def make_pdf(path):
    doc=fitz.open()
    page=doc.new_page(width=560, height=720)
    page.insert_text((45, 80), 'Table 8. Normalized quality values by dataset and evaluation metric.')
    for x,text in [(70,'Dataset'),(170,'Metric'),(305,'R'),(365,'CC')]:
        page.insert_text((x,120), text, fontsize=10)
    for y, dataset, metric, r, cc in [
        (145,'Alpha','Accuracy','0.400','0.750'),
        (160,'','Faithfulness','0.500','0.800'),
        (180,'Beta','Accuracy','0.510','0.900'),
    ]:
        for x,txt in [(70,dataset),(170,metric),(305,r),(365,cc)]:
            if txt:page.insert_text((x,y),txt,fontsize=10)
    page.insert_text((45,235),'The numbers show normalized values for different measures.')
    doc.save(path)
    doc.close()


def seed_pdf(tmp_path):
    paper=uuid4().hex
    pdf_path=tmp_path / 'demo.pdf'
    make_pdf(pdf_path)
    with db.connect() as c:
        c.execute('''INSERT INTO papers(id,owner_id,title,source_type,document_path,status,created_at)
                     VALUES(?,?,?,?,?,?,?)''',
                  (paper,settings.owner_id,'Imaginary benchmark','pdf',str(pdf_path),'ready',db.now()))
        c.execute('''INSERT INTO chunks(id,paper_id,seq,section,content,paragraph_refs,pages)
                     VALUES(?,?,?,?,?,?,?)''',
                  (paper+':c0',paper,0,'Page 1',fitz.open(pdf_path)[0].get_text(),db.encode(['p1']),db.encode([1])))
    return paper,pdf_path


def test_positioned_pdf_cells_and_row_label_inheritance(tmp_path):
    paper,path=seed_pdf(tmp_path)
    tables=extract_pdf_tables(path)
    assert len(tables)==1
    t=tables[0]
    assert t['table_number']==8 and t['page']==1
    assert t['headers']==['Dataset','Metric','R','CC']
    assert t['rows'][1]=={'Dataset':'Alpha','Metric':'Faithfulness','R':'0.500','CC':'0.800'}
    store_tables(paper,tables)
    assert load_tables([paper],8)[0]['rows'][2]['Dataset']=='Beta'


def test_selects_exact_row_not_other_metric_or_dataset(tmp_path):
    paper,_=seed_pdf(tmp_path)
    q='根据 Table 8，Alpha 的 Accuracy：Random 与 CC-guided 分别是多少？相对提升百分比和百分点？'
    f=select_fact_row(q,[paper])
    assert f['row']['R']=='0.400' and f['row']['CC']=='0.750'
    assert requested_columns(q,f)==['R','CC']
    answer=direct_table_answer(q,f)
    assert '0.400' in answer and '0.750' in answer
    assert '87.50%' in answer and '35.0 个百分点' in answer
    assert '0.800' not in answer
    assert numeric_row_conflict('Random=0.400 and CC-guided=0.750',f)==''
    assert '0.800' in numeric_row_conflict('CC-guided is 0.800',f)


def test_ambiguous_row_abstains_instead_of_picking_wrong_metric(tmp_path):
    paper,_=seed_pdf(tmp_path)
    # Without a dataset, both Accuracy rows match equally.
    assert select_fact_row('Table 8 的 Accuracy 在 CC-guided 下是多少？',[paper]) is None
    # Without identifying which Alpha metric, the row cannot be chosen.
    assert select_fact_row('Table 8 Alpha 的 CC-guided 是多少？',[paper]) is None


class ModelMustNotRun:
    async def chat(self,*a,**kw):
        raise AssertionError('Numeric table lookup should not ask the model to select cells')
    async def stream(self,*a,**kw):
        raise AssertionError('Numeric table lookup should not stream model output')
        yield ''


def test_real_agent_persists_correct_verified_table_result(tmp_path):
    paper,_=seed_pdf(tmp_path)
    conv, run=uuid4().hex,uuid4().hex
    q='Table 8 的 Beta Accuracy：Random 和 CC-guided 分别是多少？计算相对提升百分比。'
    with db.connect() as c:
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                  (conv,settings.owner_id,'Synthetic table',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                  (run,conv,'pending','adaptive',q,db.encode([paper]),db.now()))
    asyncio.run(AgentRunner(ModelMustNotRun()).run(run,ChatRequest(
        conversation_id=conv,question=q,paper_ids=[paper],mode='adaptive',teaching=True)))
    result=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert result['status']=='succeeded',result['error']
    assert '0.510' in result['answer'] and '0.900' in result['answer']
    assert '76.47%' in result['answer']
    assert db.decode(result['metrics'])['structured_fact_used'] is True
    citation=db.decode(result['citations'])[0]
    assert citation['page_numbers']==[1]
    assert citation['chunk_id']==paper+':c0'


def test_claim_checking_is_not_definition_or_blanket_refusal():
    q='有人说平均 1.7 倍并不是论文结果，且所有指标都提高了。请逐条判断和纠正。'
    assert _question_intent(q)=='claim_check'


def test_table_diagnostics_endpoint_exposes_pdf_source_coordinates(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app
    paper,_=seed_pdf(tmp_path)
    with TestClient(app) as client:
        response=client.get(f'/api/papers/{paper}/tables?number=8')
        assert response.status_code==200
        table=response.json()['tables'][0]
        assert table['page']==1
        assert table['rows'][0]['Metric']=='Accuracy'
        assert table['rows'][0]['CC']=='0.750'


def test_explanatory_answer_cannot_use_a_number_from_the_wrong_row(tmp_path,monkeypatch):
    from test_multi_question import _fake_langgraph
    from app.knowledge import retrieval
    _fake_langgraph(monkeypatch)
    paper,path=seed_pdf(tmp_path)
    q='为什么 Table 8 的 Beta Accuracy 中 Random 为 0.750 而 CC-guided 为 0.900？'
    conv,run=uuid4().hex,uuid4().hex
    with db.connect() as c:
        c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                  (conv,settings.owner_id,'Table audit',db.now()))
        c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                  (run,conv,'pending','adaptive',q,db.encode([paper]),db.now()))
    class WrongRowModel:
        def __init__(self):self.prompts=[]
        async def chat(self,messages,**kwargs):
            self.prompts.append(messages)
            return {'content':'{"answer":"Beta Accuracy 的 Random 为 0.750，CC-guided 为 0.900。", "citations":[1]}',
                    'latency_ms':1}
        async def stream(self,*args,**kwargs):
            raise AssertionError('Not a free-form stream')
            yield ''
    model=WrongRowModel()
    def retrieve(*args,**kwargs):
        chunk=db.one('SELECT * FROM chunks WHERE paper_id=?',(paper,))
        return {'evidences':[{'evidence_id':chunk['id'],'paper_id':paper,'chunk_id':chunk['id'],
                'section':'Page 1','page_numbers':[1], 'paragraph_refs':['p1'],
                'content':chunk['content']}], 'latency_ms':1}
    monkeypatch.setattr(retrieval,'search',retrieve)
    asyncio.run(AgentRunner(model).run(run,ChatRequest(
        conversation_id=conv,question=q,paper_ids=[paper],mode='adaptive')))
    output=db.one('SELECT * FROM runs WHERE id=?',(run,))
    assert output['status']=='partial', output['error']
    assert '0.750' not in output['answer']
    assert '0.510' in output['answer'] and '0.900' in output['answer']
    assert '0.750' in db.decode(output['metrics'])['numeric_fact_conflict']
    assert any('Beta' in msg['content'] and '0.510' in msg['content']
               for prompt in model.prompts for msg in prompt)


def test_wrong_column_is_detected_even_when_both_numbers_exist_in_same_row(tmp_path):
    paper,_=seed_pdf(tmp_path)
    f=select_fact_row('Table 8 Beta Accuracy: Random and CC-guided values',[paper])
    assert f
    # Both numbers appear in the correct row; their column association is wrong.
    issue=numeric_row_conflict('Random 为 0.900；CC-guided 为 0.510。',f)
    assert issue and 'Random' in issue


def test_claims_use_table_rows_and_find_a_counterexample():
    from app.agent.facts import claim_table_sources, claim_table_context
    # No fixed answer values appear in this helper; source rows drive the audit.
    source={
        'table_number':22,'page':3,'caption':'Table 22. APFD of two approaches.',
        'headers':['Dataset','Metric','R','CC'],
        'rows':[
            {'Dataset':'Group-A','Metric':'Faithfulness','R':'0.400','CC':'0.800'},
            {'Dataset':'Group-A','Metric':'Grounding','R':'0.750','CC':'0.700'},
        ]
    }
    statement='有人说所有 APFD 指标下 CC-guided 都高于 Random。'
    context=claim_table_context(statement,[source])
    assert 'Group-A / Grounding' in context
    assert 'CC=0.700 <= R=0.750' in context
    assert 'Faithfulness' in context
    assert claim_table_sources('Chunk Coverage 必须达到100%吗？',[])==[]

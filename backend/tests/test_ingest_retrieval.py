import json
from pathlib import Path
from app import db
from app.knowledge.ingest import import_qasper_file, paragraphs_to_chunks
from app.knowledge.retrieval import search, get_passage
from app.evaluation import f1, evidence_metrics, list_cases


DEMO=Path(__file__).resolve().parents[2]/'examples'/'qasper-mini.json'


def test_paragraph_overlap_and_provenance():
    sections=[{'name':'Intro','paragraphs':[{'id':f'p{i}','text':f'content{i}'*20,'pages':[i+1]} for i in range(4)]}]
    chunks=paragraphs_to_chunks('x',sections,target_chars=170,overlap=1)
    assert chunks[0]['paragraph_refs']==['p0']
    assert chunks[1]['paragraph_refs']==['p1'] or 'p0' in chunks[1]['paragraph_refs']
    assert chunks[0]['pages']==[1]


def test_qasper_ingest_and_evidence_mapping():
    counts=import_qasper_file(DEMO,split='dev')
    assert counts['imported']==1
    assert counts['questions']==1
    assert import_qasper_file(DEMO)['skipped']==1
    case=list_cases('dev',10)[0]
    assert case['gold_evidence_refs']==['s1-p0']
    assert 'BM25' in case['answers'][0]
    rows=db.rows('SELECT * FROM chunks')
    assert not any('What retrieval methods are combined?' in x['content'] for x in rows)
    result=search('BM25 reciprocal rank fusion','qasper-demo-paper-001'.split(',') ,'keyword',top_k=5)
    assert result['evidences']
    assert 's1-p0' in result['evidences'][0]['paragraph_refs']
    passage=get_passage(result['evidences'][0]['chunk_id'],['qasper-demo-paper-001'])
    assert passage['paper_id']=='qasper-demo-paper-001'
    assert get_passage(result['evidences'][0]['chunk_id'],['another-paper']) is None


def test_metrics_handle_unmapped_evidence():
    assert f1('dense bm25','dense bm25')==1
    assert evidence_metrics(['p1'],['p1','p2'])['recall']==0.5
    assert evidence_metrics(['p1'],[])['recall'] is None

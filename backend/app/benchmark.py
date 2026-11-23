"""Run real model predictions: same papers, model, retrieval corpus across all modes."""
import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from . import db
from .agent.workflow import AgentRunner
from .evaluation import answer_f1,evidence_metrics,list_cases
from .schemas import ChatRequest


async def benchmark(split,limit,modes,output):
    cases=list_cases(split,limit)
    if not cases:raise RuntimeError('No evaluation cases imported for this split')
    result=[]
    runner=AgentRunner()
    for case in cases:
        for mode in modes:
            conv=uuid4().hex
            run=uuid4().hex
            with db.connect() as c:
                c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                    (conv,db.settings.owner_id,'Benchmark',db.now()))
                c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                    (run,conv,'pending',mode,case['question'],db.encode([case['paper_id']]),db.now()))
            req=ChatRequest(conversation_id=conv,question=case['question'],paper_ids=[case['paper_id']],
                            mode=mode,teaching=False,use_memory=False,allow_web=False)
            await runner.run(run,req)
            r=db.one('SELECT * FROM runs WHERE id=?',(run,))
            citations=db.decode(r['citations'],[])
            run_metrics=db.decode(r['metrics'],{})
            retrieved=run_metrics.get('retrieved_evidence',[])
            refs=[ref for step in retrieved for chunk in step.get('chunks',[]) for ref in chunk.get('paragraph_refs',[])]
            result.append({'question_id':case['question_id'],'mode':mode,'status':r['status'],
                'answer':r['answer'],'reference_answers':case['answers'], 'answer_type':case['answer_type'],
                'answer_f1':answer_f1(r['answer'] or '',case['answers']) if r['status'] in ('succeeded','partial') else None,
                'evidence_metrics':evidence_metrics(refs,case['gold_evidence_refs']),
                'gold_mapping_available':bool(case['gold_evidence_refs']),
                'metrics':run_metrics,'error':r['error']})
            print(case['question_id'],mode,r['status'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'split':split,'model':runner.model.model,'results':result},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved',output)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split',default='dev')
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--modes',nargs='+',choices=['basic','hybrid','adaptive'],default=['basic','hybrid','adaptive'])
    p.add_argument('--output',type=Path,default=Path('results/qasper-dev.json'))
    a=p.parse_args()
    db.initialize()
    asyncio.run(benchmark(a.split,a.limit,a.modes,a.output))


if __name__=='__main__':main()

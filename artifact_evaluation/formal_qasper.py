#!/usr/bin/env python3
"""Phase-2 controlled QASPER experiment: standard-library harness, no Agent edits.

Usage from repository root:
  python artifact_evaluation/formal_qasper.py prepare --archive PATH --out PATH
  python artifact_evaluation/formal_qasper.py preflight --dataset PATH --live
  python artifact_evaluation/formal_qasper.py run --dataset PATH --out PATH --trial smoke --max-runs 3
  python artifact_evaluation/formal_qasper.py run --dataset PATH --out PATH --trial official
  python artifact_evaluation/formal_qasper.py report --dataset PATH --run-dir PATH
  python artifact_evaluation/formal_qasper.py finalize --run-dir PATH --review-csv PATH
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from uuid import uuid4
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('ARAG_REPO', str(HERE.parent))).resolve()
MODES = ('basic', 'hybrid', 'adaptive')
SEED = 62026
FROZEN_APP_SHA256 = '0c4c648a988faf1df14ebd3251ac2d8eb4bc89163465617e784953650132f872'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def project_script():
    p = HERE / 'day2_qasper.py'
    if not p.is_file():
        raise FileNotFoundError(f'Cannot find {p}; install this add-on into the v0.3.18-rc1 repository.')
    spec = importlib.util.spec_from_file_location('qasper_day2_frozen', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prior_ids(path):
    obj = load(path)
    return set(obj['question_ids'])


def classify_answer(qa):
    cats = set()
    for annotation in qa.get('answers', []):
        a = annotation.get('answer', annotation)
        if a.get('unanswerable'): cats.add('unanswerable')
        elif a.get('yes_no') is not None: cats.add('yes_no')
        elif a.get('extractive_spans'): cats.add('extractive')
        elif a.get('free_form_answer'): cats.add('free_form')
    return ','.join(sorted(cats)) or 'unknown'


def select_unseen(dev, *, count, per_paper, seed, old_ids):
    """Deterministic blind selection; no model outputs or reference answers used for ranking.

    A paper is excluded if ANY question was used previously. Require >=2 mapped
    gold questions within the hashed-random selected questions. This criterion
    biases towards mappable evidence and is explicitly disclosed.
    """
    dq = project_script()
    eligible = []
    excluded = []
    for pid, doc in dev.items():
        if not isinstance(doc, dict) or not isinstance(doc.get('qas'), list):
            continue
        ids = {str(qa.get('question_id', '')) for qa in doc['qas']}
        overlap = sorted(ids & old_ids)
        if overlap:
            excluded.append({'paper_id':pid, 'title':doc.get('title',''), 'historical_question_matches':len(overlap)})
            continue
        paragraphs = dq.paragraph_inventory(doc)
        chars = sum(len(t) for t, _ in paragraphs)
        if len(paragraphs)<8 or chars<2500:
            continue
        questions=[]
        for qa in doc['qas']:
            if len(str(qa.get('question', '')).strip()) < 12 or not qa.get('answers'):
                continue
            if not any(dq.is_gold_answer(a) for a in qa['answers']):
                continue
            refs=dq.map_gold_evidence(qa,paragraphs)
            questions.append({'qa':qa, 'refs':refs,'question_id':str(qa.get('question_id',''))})
        if len(questions) < per_paper:
            continue
        questions.sort(key=lambda q:dq.seeded_rank(seed, q['question_id'] or str(q['qa']['question'])))
        chosen = questions[:per_paper]
        if sum(bool(q['refs']) for q in chosen)<2:
            continue
        eligible.append((pid,doc,chosen,chars))
    eligible.sort(key=lambda e:dq.seeded_rank(seed,e[0]))
    if len(eligible)<count:
        raise ValueError(f'Only {len(eligible)} never-tested eligible papers; requested {count}.')
    selected = {}
    papers = []
    for pid,doc,chosen,chars in eligible[:count]:
        record=dict(doc)
        record['qas']=[q['qa'] for q in chosen]
        selected[pid]=record
        papers.append({'paper_id':f'qasper-{pid}', 'dataset_paper_id':pid,'title':doc.get('title',''),
                       'full_text_chars':chars,'questions':[{
                           'question_id':q['question_id'],'question':q['qa']['question'],
                           'answer_type':classify_answer(q['qa']),
                           'gold_evidence_refs':q['refs'], 'gold_mapping_available':bool(q['refs'])}
                           for q in chosen]})
    if any(q['question_id'] in old_ids for p in papers for q in p['questions']):
        raise AssertionError('Historical question leaked into selection.')
    return selected, papers, excluded, len(eligible)


def dataset_checked(folder):
    folder=Path(folder).resolve()
    manifest=load(folder/'selection_manifest.json')
    selected=folder/'qasper_selected_dev.json'
    if not selected.exists() or sha(selected)!=manifest.get('selected_json_sha256'):
        raise ValueError('Dataset changed since selection: SHA-256 mismatch.')
    if manifest.get('phase')!='formal-v1' or manifest.get('total_questions')!=20:
        raise ValueError('Expect a Phase-2 formal sample of 4 papers × 5 questions (20 total).')
    papers=load(selected)
    seen={str(q.get('question_id')) for p in papers.values() for q in p.get('qas',[])}
    expected={q['question_id'] for p in manifest['papers'] for q in p['questions']}
    if seen!=expected or len(expected)!=20 or len(papers)!=4:
        raise ValueError('Selection manifest and actual QASPER subset disagree.')
    old=prior_ids(HERE/'prior_exposed_qasper_ids.json')
    if seen & old:
        raise ValueError('Formal dataset includes historically used QASPER question(s).')
    return manifest


def cmd_prepare(a):
    if a.seed!=SEED:
        print('NOTE: alternative predeclared seed; disclose it in your report, do not vary based on results.')
    dq=project_script()
    source_args=argparse.Namespace(archive=a.archive,dev_json=a.dev_json)
    raw, source=dq.load_dev_json(source_args,a.out)
    if source['source_kind']=='local_dev_json' and not a.trust_local_dev:
        raise ValueError('Local dev JSON lacks official archive SHA verification; use --archive, or explicitly --trust-local-dev.')
    dev=json.loads(raw.decode('utf-8-sig'))
    if not isinstance(dev,dict):
        raise ValueError('Expected official QASPER mapping of paper IDs to records.')
    old=prior_ids(HERE/'prior_exposed_qasper_ids.json')
    selected,papers,excluded,n_eligible=select_unseen(dev,count=4,per_paper=5,seed=a.seed,old_ids=old)
    a.out.mkdir(parents=True,exist_ok=True)
    target=a.out/'qasper_selected_dev.json'
    save(target,selected)
    manifest={'phase':'formal-v1','dataset':'QASPER','version':'0.3','split':'dev',
              'created_at_utc':stamp(), 'seed':a.seed,'papers_requested':4,
              'questions_per_paper':5,'total_questions':20,'eligible_unseen_papers':n_eligible,
              'historical_question_ids_sha256':sha(HERE/'prior_exposed_qasper_ids.json'),
              'previously_used_papers_excluded':excluded,
              'sampling_policy':'Exclude full papers containing any previously used QASPER development questions (21 known question IDs). Deterministic SHA-256 hash rank selects papers and questions; require each selected paper to contain at least two evidence-mappable gold questions among its five sampled questions. This is a conditional sample, not representative random QASPER.',
              'source':source,'selected_json_sha256':sha(target),'papers':papers}
    save(a.out/'selection_manifest.json',manifest)
    dataset_checked(a.out)
    types=Counter(q['answer_type'] for p in papers for q in p['questions'])
    print(f'Selected UNSEEN 4 papers / 20 questions; eligible={n_eligible}; excluded previously tested papers={len(excluded)}')
    for p in papers:
        print(f"  {p['dataset_paper_id']} | mapped={sum(q['gold_mapping_available'] for q in p['questions'])}/5 | {p['title'][:78]}")
    print('Types:',dict(types))
    print('Manifest:',a.out/'selection_manifest.json')
    print('Selected QASPER raw text:',target)
    print('NOTE: do not inspect gold answers or use selected questions to tune the model before the official run.')


def app_fingerprint():
    h=hashlib.sha256()
    base=REPO/'backend'/'app'
    if not base.is_dir():
        raise FileNotFoundError(base)
    for path in sorted(base.rglob('*.py')):
        rel=path.relative_to(base).as_posix()
        h.update(rel.encode()+b'\0'+bytes.fromhex(sha(path)))
    return h.hexdigest()


def git_info():
    try:
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
        changes=subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
        return {'commit':commit,'tracked_modifications':bool(changes),'status':changes[:1000]}
    except (OSError,subprocess.CalledProcessError):
        return {'commit':None,'tracked_modifications':None,'status':'No Git repository available'}


def live_model_check(model):
    req=urllib.request.Request('http://127.0.0.1:11434/api/tags')
    with urllib.request.urlopen(req,timeout=8) as resp:
        models=json.loads(resp.read().decode()).get('models',[])
    available={m.get('name') or m.get('model') for m in models}
    if model not in available:
        raise RuntimeError(f'Ollama reachable but {model!r} not installed; available={sorted(available)}')
    print(f'Ollama model available: {model}')


def assert_frozen_code():
    current=app_fingerprint()
    if current!=FROZEN_APP_SHA256:
        raise RuntimeError(f'Application code differs from v0.3.18-rc1: {current}; expected {FROZEN_APP_SHA256}. Restore baseline before evaluating.')


def cmd_preflight(a):
    manifest=dataset_checked(a.dataset)
    assert_frozen_code()
    info=git_info()
    print('Dataset hash:',manifest['selected_json_sha256'])
    print('App code hash:',app_fingerprint())
    print('Git:',info)
    if info['commit'] is None or info['tracked_modifications']:
        raise RuntimeError('Require a clean Git commit before formal experiment.')
    if a.live:live_model_check(a.model)
    print('PREFLIGHT PASSED. This does not verify live model answer correctness.')


def clean_run_records(ledger):
    records=[]; keys=set()
    if not ledger.exists():return records,keys
    for line_n,line in enumerate(ledger.read_text(encoding='utf8').splitlines(),1):
        if not line.strip():continue
        try:r=json.loads(line)
        except json.JSONDecodeError as ex:raise ValueError(f'Malformed journal line {line_n}: {ex}') from ex
        key=(r['question_id'],r['mode'])
        if key in keys:raise ValueError(f'Duplicate experiment observation: {key}')
        keys.add(key);records.append(r)
    return records,keys


def run_order(manifest):
    questions=[q['question_id'] for p in manifest['papers'] for q in p['questions']]
    for index,qid in enumerate(questions):
        for j in range(3):yield qid,MODES[(index+j)%3]


def experiment_metadata(manifest,model):
    return {'experiment':'formal-v1', 'created_at_utc':stamp(), 'dataset_sha256':manifest['selected_json_sha256'],
            'selection_seed':manifest['seed'],'app_code_sha256':app_fingerprint(),
            'git':git_info(),'model':model,
            'model_base_url':'http://127.0.0.1:11434/v1','model_provider':'ollama',
            'temperature_default':0.0, 'model_http_repetition_retry':'unchanged v0.3.18-rc1 policy',
            'embedding_model':os.environ.get('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2'),
            'enable_rerank':False,'memory':False,'teaching':False,'allow_web':False,
            'execution_order':'balanced rotation of basic/hybrid/adaptive across sequential questions',
            'runner_sha256':sha(Path(__file__))}


async def real_run(a,manifest,ledger,metadata):
    backend=REPO/'backend'
    runtime=(a.out/'runtime').resolve()
    source_data=(backend/'data').resolve()
    if runtime==source_data:
        raise ValueError('Refusing to modify the normal app backend/data database.')
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ['APP_DATA_DIR']=str(runtime)
    os.environ['MODEL_NAME']=a.model
    os.environ['MODEL_BASE_URL']='http://127.0.0.1:11434/v1'
    os.environ['MODEL_PROVIDER']='ollama'
    os.environ['ENABLE_RERANK']='false'
    os.environ['PYTHONUTF8']='1'
    sys.path.insert(0,str(backend))
    from app import db
    from app.agent.workflow import AgentRunner
    from app.evaluation import answer_f1, evidence_metrics, list_cases
    from app.knowledge.ingest import import_qasper_file
    from app.schemas import ChatRequest
    db.initialize()
    rows=db.rows('SELECT id FROM papers')
    if not rows:
        result=import_qasper_file(a.dataset/'qasper_selected_dev.json','dev')
        print('Imported into isolated evaluation DB:',result,flush=True)
        if result['imported']!=4 or result['questions']!=20:
            raise RuntimeError('QASPER import unexpected; abort before inference.')
    if db.one('SELECT COUNT(*) AS n FROM memories')['n']:
        raise RuntimeError('Experiment DB contains memories. Use a new run directory.')
    with db.connect() as c:
        c.execute("UPDATE profile SET language='en', explanation_level='beginner',memory_enabled=0,domains='[]',goals='[]'")
    cases=list_cases('dev',100000)
    index={case['question_id']:case for case in cases}
    expected={q['question_id'] for p in manifest['papers'] for q in p['questions']}
    if set(index)!=expected or len(cases)!=20:
        raise RuntimeError('Eval DB has unexpected questions; use a fresh run dir.')
    seen_records,completed=clean_run_records(ledger)
    runner=AgentRunner()
    if runner.model.model!=a.model:
        raise RuntimeError('Loaded model mismatch.')
    todo=[key for key in run_order(manifest) if key not in completed]
    if a.max_runs is not None:todo=todo[:a.max_runs]
    print(f'Runs finished={len(completed)} remaining_this_command={len(todo)} expected_total=60',flush=True)
    for ix,(qid,mode) in enumerate(todo,1):
        case=index[qid]
        conv,run=uuid4().hex,uuid4().hex
        with db.connect() as c:
            c.execute('INSERT INTO conversations(id,owner_id,title,created_at) VALUES(?,?,?,?)',
                      (conv,db.settings.owner_id,'Phase-2 formal evaluation',db.now()))
            c.execute('INSERT INTO runs(id,conversation_id,status,mode,question,paper_ids,created_at) VALUES(?,?,?,?,?,?,?)',
                      (run,conv,'pending',mode,case['question'],db.encode([case['paper_id']]),db.now()))
        req=ChatRequest(conversation_id=conv,question=case['question'],paper_ids=[case['paper_id']],
                        mode=mode,teaching=False,use_memory=False,allow_web=False)
        started=time.perf_counter()
        try:
            await runner.run(run,req)
        except Exception as ex:
            # Make an exception auditable and move on to the next paired task.
            with db.connect() as c:
                c.execute("UPDATE runs SET status='failed',error=?,completed_at=? WHERE id=?",
                          (f'Runner exception: {type(ex).__name__}: {str(ex)[:300]}',db.now(),run))
        record=db.one('SELECT * FROM runs WHERE id=?',(run,))
        if not record:raise RuntimeError(f'Run vanished from DB: {run}')
        metrics=db.decode(record['metrics'],{})
        citations=db.decode(record['citations'],[])
        evidence=metrics.get('retrieved_evidence',[])
        predicted=[ref for rd in evidence for chunk in rd.get('chunks',[]) for ref in chunk.get('paragraph_refs',[])]
        result={'run_id':run,'question_id':qid,'paper_id':case['paper_id'],'question':case['question'],
                'mode':mode,'status':record['status'],'answer':record['answer'] or '',
                'gold_reference_answers':case['answers'], 'answer_type':case['answer_type'],
                'gold_evidence_refs':case['gold_evidence_refs'],
                'answer_f1_proxy':answer_f1(record['answer'] or '',case['answers']) if record['status'] in ('succeeded','partial') else 0.0,
                'evidence_metrics':evidence_metrics(predicted,case['gold_evidence_refs']),
                'gold_mapping_available':bool(case['gold_evidence_refs']),
                'retrieved_paragraph_refs':list(dict.fromkeys(predicted)),
                'citations':citations,'metrics':metrics,'error':record['error'],
                'elapsed_wall_s':round(time.perf_counter()-started,3),'completed_at_utc':stamp(),
                'experiment_config_hash':hashlib.sha256(json.dumps({k:v for k,v in metadata.items() if k!='created_at_utc'},sort_keys=True).encode()).hexdigest()}
        with ledger.open('a',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(result,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())
        print(f'[{len(completed)+ix}/60] {qid[:9]} {mode:8} {record["status"]:10} F1={result["answer_f1_proxy"]:.3f} {result["elapsed_wall_s"]:.1f}s',flush=True)
    print(f'Journal: {ledger}',flush=True)


def cmd_run(a):
    manifest=dataset_checked(a.dataset)
    assert_frozen_code()
    info=git_info()
    if info['commit'] is None or info['tracked_modifications']:
        raise RuntimeError('Record a clean Git commit before inference; source must remain frozen.')
    live_model_check(a.model)
    a.out=(a.out/a.trial).resolve()
    a.out.mkdir(parents=True,exist_ok=True)
    ledger=a.out/'runs.jsonl'
    meta=a.out/'experiment_manifest.json'
    current=experiment_metadata(manifest,a.model)
    if meta.exists():
        prev=load(meta)
        for key in ('dataset_sha256','app_code_sha256','model','embedding_model','enable_rerank','runner_sha256'):
            if prev[key]!=current[key]:
                raise RuntimeError(f'Cannot resume with changed {key}: {prev[key]} != {current[key]}')
        if prev['git']['commit']!=current['git']['commit']:
            raise RuntimeError('Cannot resume with changed Git commit.')
        current=prev
    else:
        if ledger.exists():
            raise RuntimeError('Journal already exists but experiment_manifest missing; refusing to resume.')
        save(meta,current)
    asyncio.run(real_run(a,manifest,ledger,current))


def numeric(x):
    try:return float(x)
    except (TypeError,ValueError):return None


def avg(xs):
    ys=[x for x in xs if x is not None]
    return statistics.mean(ys) if ys else None


def percentile(xs,p):
    xs=sorted(v for v in xs if v is not None)
    if not xs:return None
    x=(len(xs)-1)*p/100
    lo=math.floor(x);hi=math.ceil(x)
    return xs[lo]+(xs[hi]-xs[lo])*(x-lo)


def fmt(x,n=3):return '—' if x is None else f'{x:.{n}f}'


def validate_ledger(rows,manifest):
    expected=set(run_order(manifest))
    seen=Counter((r['question_id'],r['mode']) for r in rows)
    extra=sorted(set(seen)-expected)
    missing=sorted(expected-set(seen))
    duplicates=[key for key,n in seen.items() if n>1]
    return expected,missing,extra,duplicates


def summaries(rows):
    stats=[]
    for mode in MODES:
        rr=[r for r in rows if r['mode']==mode]
        good=[r for r in rr if r['status']=='succeeded']
        times=[numeric(r['metrics'].get('total_ms')) for r in rr]
        latencies=[(t/1000 if t is not None else numeric(r.get('elapsed_wall_s'))) for t,r in zip(times,rr)]
        stats.append({'mode':mode,'count':len(rr),'succeeded':len(good),
                     'partial':sum(r['status']=='partial' for r in rr),
                     'failed':sum(r['status'] not in ('succeeded','partial') for r in rr),
                     'answer_f1_all':avg([numeric(r.get('answer_f1_proxy')) or 0 for r in rr]),
                     'answer_f1_succeeded':avg([numeric(r.get('answer_f1_proxy')) for r in good]),
                     'evidence_f1_gold_mapped':avg([numeric(r['evidence_metrics'].get('f1')) for r in rr if r['gold_mapping_available']]),
                     'evidence_mapped_n':sum(bool(r['gold_mapping_available']) for r in rr),
                     'citation_rate':avg([1.0 if r.get('citations') else 0.0 for r in rr]),
                     'latency_mean_s':avg(latencies),'latency_p50_s':percentile(latencies,50),
                     'latency_p95_s':percentile(latencies,95),
                     'model_time_mean_s':avg([(numeric(r['metrics'].get('model_ms')) or 0)/1000 for r in rr]),
                     'retrieval_time_mean_s':avg([(numeric(r['metrics'].get('retrieval_ms')) or 0)/1000 for r in rr]),
                     'llm_calls_mean':avg([numeric(r['metrics'].get('llm_calls')) for r in rr]),
                     'retrieval_rounds_mean':avg([numeric(r['metrics'].get('retrieval_rounds')) for r in rr]),
                     'memory_leak_n':sum(bool(r['metrics'].get('learning_memory_applied')) for r in rr),
                     'web_leak_n':sum(bool(r['metrics'].get('external_sources_found')) for r in rr)})
    return stats


def paired_comparisons(rows):
    grouped=defaultdict(dict)
    for r in rows:
        grouped[r['question_id']][r['mode']]=r
    output={}
    for left,right in (('hybrid','basic'),('adaptive','hybrid'),('adaptive','basic')):
        paired=[(pair[left],pair[right]) for pair in grouped.values() if left in pair and right in pair]
        delta_f1=[float(a.get('answer_f1_proxy') or 0)-float(b.get('answer_f1_proxy') or 0) for a,b in paired]
        delta_calls=[float(a['metrics'].get('llm_calls') or 0)-float(b['metrics'].get('llm_calls') or 0) for a,b in paired]
        delta_time=[float(a['metrics'].get('total_ms') or a.get('elapsed_wall_s',0)*1000)/1000-
                    float(b['metrics'].get('total_ms') or b.get('elapsed_wall_s',0)*1000)/1000 for a,b in paired]
        output[f'{left}_minus_{right}']={'matched_questions':len(paired),
            'answer_f1_mean_delta':avg(delta_f1),'calls_mean_delta':avg(delta_calls),
            'latency_mean_delta_s':avg(delta_time),
            'answer_f1_wins':sum(d>1e-9 for d in delta_f1),
            'answer_f1_ties':sum(abs(d)<=1e-9 for d in delta_f1),
            'answer_f1_losses':sum(d< -1e-9 for d in delta_f1)}
    return output


def write_csv(path, rows, headers):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def cmd_report(a):
    manifest=dataset_checked(a.dataset)
    run_dir=a.run_dir.resolve()
    exp=load(run_dir/'experiment_manifest.json')
    if exp['dataset_sha256']!=manifest['selected_json_sha256']:
        raise ValueError('Experiment dataset hash differs from current manifest.')
    rows,_=clean_run_records(run_dir/'runs.jsonl')
    expected,missing,extra,duplicates=validate_ledger(rows,manifest)
    stats=summaries(rows)
    complete=not(missing or extra or duplicates)
    warning=[]
    if missing:warning.append(f'{len(missing)} observations missing')
    if extra:warning.append(f'{len(extra)} unexpected observations')
    if duplicates:warning.append(f'{len(duplicates)} duplicated observations')
    leak=sum(x['memory_leak_n']+x['web_leak_n'] for x in stats)
    if leak:warning.append(f'Memory/web isolation violations: {leak}')
    results={'experiment':'formal-v1','source_commit':exp['git']['commit'],
             'source_clean':not exp['git']['tracked_modifications'],
             'app_code_sha256':exp['app_code_sha256'],
             'dataset_sha256':manifest['selected_json_sha256'],
             'observed_runs':len(rows),'expected_runs':len(expected),
             'complete':complete,'missing':missing,'extras':extra,'duplicates':duplicates,
             'warnings':warning,'results':stats,'paired_mode_differences':paired_comparisons(rows),
             'evaluation_limitations':["Answer F1 is a simplified lexical proxy, not the official QASPER metric.",
               "Evidence F1 uses all retrieved paragraph refs across rounds; it is not citation entailment or equal-budget Recall@K.",
               "The 20-question sample is conditional on minimum evidence mapping; it is not representative of all QASPER.",
               "Human assessment of correctness, completeness, and citation support is required before claims about groundedness."]}
    save(run_dir/'metrics_summary.json',results)
    lookup={q['question_id']:(p,q) for p in manifest['papers'] for q in p['questions']}
    flat=[]
    for r in rows:
        p,q=lookup[r['question_id']]
        flat.append({'question_id':r['question_id'],'paper_title':p['title'],'answer_type':r['answer_type'],
                     'mode':r['mode'],'status':r['status'],'answer_f1_proxy':r['answer_f1_proxy'],
                     'evidence_f1':r['evidence_metrics'].get('f1'),
                     'gold_mapping_available':r['gold_mapping_available'],
                     'latency_s':r.get('elapsed_wall_s'),'llm_calls':r['metrics'].get('llm_calls'),
                     'question':r['question'],'gold_references':' | '.join(r['gold_reference_answers']),
                     'model_answer':r['answer'],'citation_count':len(r.get('citations') or []),
                     'run_id':r['run_id'],'error':r.get('error') or ''})
    fields=['question_id','paper_title','answer_type','mode','status','answer_f1_proxy','evidence_f1','gold_mapping_available','latency_s','llm_calls','citation_count','question','gold_references','model_answer','run_id','error']
    write_csv(run_dir/'per_question_results.csv',flat,fields)
    # Blinded review order is reproducible and does NOT expose mode or F1 in the review form.
    rng=random.Random(manifest['seed'])
    blind=[dict(r) for r in rows]
    rng.shuffle(blind)
    review=[]; keys=[]
    for i,r in enumerate(blind,1):
        review_id=f'QA{i:03d}'
        citation_view=' | '.join(f"[{c.get('number')}] {c.get('evidence_id')} : {str(c.get('quote') or '')[:850]}" for c in (r.get('citations') or []))
        review.append({'review_id':review_id,'question_id':r['question_id'],
                       'question':r['question'],'answer_type':r['answer_type'],
                       'gold_reference_answers':' | '.join(r['gold_reference_answers']),
                       'model_answer':r['answer'],'cited_evidence_preview':citation_view,
                       'correctness_0_2':'','completeness_0_2':'','citation_support_0_2':'',
                       'notes':''})
        keys.append({'review_id':review_id,'question_id':r['question_id'],'mode':r['mode'],'run_id':r['run_id']})
    write_csv(run_dir/'blind_review.csv',review,list(review[0]) if review else [])
    save(run_dir/'blind_review_mapping.json',keys)
    md=['# Phase 2 — QASPER v0.3 Formal Comparison','',
        f'- Source commit: `{exp["git"]["commit"]}`',
        f'- Source application SHA-256: `{exp["app_code_sha256"]}`',
        f'- Selected dataset SHA-256: `{manifest["selected_json_sha256"]}`',
        f'- Model: `{exp["model"]}`, fixed local Ollama; Memory/Skills/Web/Rerank disabled.',
        f'- Unseen subset: 4 papers × 5 questions; {len(rows)} / 60 observations complete.',
        '- Run order rotated across modes to reduce first-in-order warm-cache confounding.',
        '', '## Status', '',
        ('**COMPLETE** — ready for blinded manual review.' if complete else '**INCOMPLETE** — not eligible as a final published result.'), '']
    if warning:md += ['Warnings: '+ '; '.join(warning),'']
    md += ['## Quantitative results (descriptive only)','',
           '| Mode | Succeeded | Partial | Failed | Answer F1 proxy (all) | Evidence F1 (mapped) | Citation rate | Latency P50 / P95 (s) | Mean calls |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in stats:
        md.append(f"| {s['mode']} | {s['succeeded']}/{s['count']} | {s['partial']} | {s['failed']} | {fmt(s['answer_f1_all'])} | {fmt(s['evidence_f1_gold_mapped'])} | {fmt(s['citation_rate'])} | {fmt(s['latency_p50_s'],2)} / {fmt(s['latency_p95_s'],2)} | {fmt(s['llm_calls_mean'],2)} |")
    md += ['', '## Paired mode differences (first mode minus second, descriptive)', '',
           '| Comparison | Matched questions | Δ Answer F1 proxy | Δ Mean calls | Δ Mean latency (s) | F1 W/T/L |',
           '|---|---:|---:|---:|---:|---:|']
    for name,delta in results['paired_mode_differences'].items():
        md.append(f"| {name} | {delta['matched_questions']} | {fmt(delta['answer_f1_mean_delta'])} | {fmt(delta['calls_mean_delta'],2)} | {fmt(delta['latency_mean_delta_s'],2)} | {delta['answer_f1_wins']}/{delta['answer_f1_ties']}/{delta['answer_f1_losses']} |")
    md += ['', '## What is and is not measured','',
           'Automatic Answer F1 is **not official QASPER F1**. It is a lower-level token overlap proxy; a technically correct paraphrase can score poorly.',
           'Evidence F1 compares all retrieved paragraph references against mappable gold paragraphs. Since Adaptive may retrieve multiple rounds, this is **not a fixed-budget comparison**.',
           'A citation link or a succeeded status is not proof of correct or supported claims.',
           'After the blinded review, analyze quality, citations, partial/failures and compute cost together. Do not claim statistical significance from n=20.', '']
    (run_dir/'formal_report.md').write_text('\n'.join(md),encoding='utf-8')
    print(f'Report complete={complete}, observations={len(rows)}/60')
    print('Wrote:',run_dir/'formal_report.md',run_dir/'metrics_summary.json',run_dir/'per_question_results.csv',run_dir/'blind_review.csv')
    if not complete:print('Missing:',len(missing),'  Checkpoint is resumable with the identical run command.')


def cmd_finalize(a):
    run_dir=a.run_dir.resolve()
    summary=load(run_dir/'metrics_summary.json')
    if not summary['complete'] or summary['warnings']:
        raise ValueError('Do not finalize an incomplete/contaminated benchmark.')
    keys={r['review_id']:r for r in load(run_dir/'blind_review_mapping.json')}
    with a.review_csv.open('r',encoding='utf-8-sig',newline='') as f:
        ratings=list(csv.DictReader(f))
    if len(ratings)!=len(keys):raise ValueError('Missing blinded ratings.')
    values=defaultdict(list);used=set()
    for r in ratings:
        rid=r['review_id']
        if rid not in keys or rid in used:raise ValueError(f'Duplicate/unknown review id: {rid}')
        used.add(rid)
        each=[]
        for col in ('correctness_0_2','completeness_0_2','citation_support_0_2'):
            v=numeric(r.get(col))
            if v is None or v not in (0,1,2):raise ValueError(f'{rid}: score {col} must be 0,1,2')
            each.append(v)
        values[keys[rid]['mode']].append(each)
    out={mode:{name:avg([row[j] for row in values[mode]]) for j,name in enumerate(('correctness','completeness','citation_support'))} for mode in MODES}
    save(run_dir/'human_review_summary.json',{'scores_range':'0–2','n_per_mode':{m:len(values[m]) for m in MODES},'means':out,'finalized_at':stamp()})
    lines=['','## Blinded human assessment (0–2 per criterion)','',
           '| Mode | Correctness | Completeness | Citation support | N |', '|---|---:|---:|---:|---:|']
    for m in MODES:lines.append(f"| {m} | {fmt(out[m]['correctness'],2)} | {fmt(out[m]['completeness'],2)} | {fmt(out[m]['citation_support'],2)} | {len(values[m])} |")
    with (run_dir/'formal_report.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print('Blinded review summary:',run_dir/'human_review_summary.json')


def main():
    p=argparse.ArgumentParser(description='QASPER Phase-2 formal experiment; source runtime is never patched')
    sub=p.add_subparsers(dest='command',required=True)
    pp=sub.add_parser('prepare',help='Select fixed 4 × 5 unseen QASPER development questions')
    src=pp.add_mutually_exclusive_group()
    src.add_argument('--archive',type=Path,help='Official QASPER v0.3 train/dev tgz, verified by SHA-256')
    src.add_argument('--dev-json',type=Path,help='Local extracted dev JSON (requires explicit trust flag)')
    pp.add_argument('--trust-local-dev',action='store_true')
    pp.add_argument('--seed',type=int,default=SEED)
    pp.add_argument('--out',type=Path,default=HERE/'formal_dataset')
    pp.set_defaults(go=cmd_prepare)
    pf=sub.add_parser('preflight',help='Check frozen hash, manifest and optional local Ollama service')
    pf.add_argument('--dataset',type=Path,required=True)
    pf.add_argument('--live',action='store_true')
    pf.add_argument('--model',default='qwen3:4b')
    pf.set_defaults(go=cmd_preflight)
    rr=sub.add_parser('run',help='Run a checkpointed, sequential, balanced-order comparison')
    rr.add_argument('--dataset',type=Path,required=True)
    rr.add_argument('--out',type=Path,default=HERE/'formal_runs')
    rr.add_argument('--trial',required=True,help='Use smoke or official; never reuse an existing trial name for a new configuration')
    rr.add_argument('--model',default='qwen3:4b')
    rr.add_argument('--max-runs',type=int,default=None,help='For a separate smoke trial only; official should run all 60')
    rr.set_defaults(go=cmd_run)
    re=sub.add_parser('report',help='Produce English report, per-question CSV, and blind review sheets')
    re.add_argument('--dataset',type=Path,required=True)
    re.add_argument('--run-dir',type=Path,required=True)
    re.set_defaults(go=cmd_report)
    ff=sub.add_parser('finalize',help='Validate completed blind review scores; append results')
    ff.add_argument('--run-dir',type=Path,required=True)
    ff.add_argument('--review-csv',type=Path,required=True)
    ff.set_defaults(go=cmd_finalize)
    a=p.parse_args()
    try:a.go(a)
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':sys.exit(main())

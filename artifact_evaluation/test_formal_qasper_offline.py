"""Evaluation harness checks; synthetic fixtures ONLY, no model inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT=Path(__file__).with_name('formal_qasper.py')
spec=importlib.util.spec_from_file_location('formal_qasper_offline',SCRIPT)
fq=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fq)


def doc(i, historical=None):
    paragraphs=[f'Article {i}. Section {j}. '+'This section describes the measured system and evaluation evidence. '*8 for j in range(10)]
    qas=[]
    for j in range(7):
        qas.append({'question_id':historical if historical and j==0 else f'new-doc-{i}-q{j}',
                    'question':f'What is the experimental method employed for task {i}-{j}?',
                    'answers':[{'answer':{'unanswerable':False,'yes_no':None,'extractive_spans':[],
                                          'free_form_answer':f'The system runs experiment {i}-{j}.',
                                          'evidence':[paragraphs[j]]}}]})
    return {'title':f'Paper {i}', 'abstract':'Overview',
            'full_text':[{'section_name':'Methodology','paragraphs':paragraphs}], 'qas':qas}


class FormalEvaluationHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_fixed_sample_excludes_entire_historical_paper(self):
        old=fq.prior_ids(fq.HERE/'prior_exposed_qasper_ids.json')
        data={f'P{i}':doc(i,next(iter(old)) if i==0 else None) for i in range(9)}
        sel,papers,excluded,eligible=fq.select_unseen(data,count=4,per_paper=5,seed=fq.SEED,old_ids=old)
        self.assertEqual(len(sel),4)
        self.assertEqual(sum(len(p['questions']) for p in papers),20)
        self.assertTrue(excluded and excluded[0]['paper_id']=='P0')
        self.assertNotIn('P0',sel)
        self.assertEqual([p['dataset_paper_id'] for p in papers],
                         [p['dataset_paper_id'] for p in fq.select_unseen(data,count=4,per_paper=5,seed=fq.SEED,old_ids=old)[1]])

    def test_balanced_execution_order(self):
        m={'papers':[{'questions':[{'question_id':f'q{k*5+i}'} for i in range(5)]} for k in range(4)]}
        pairs=list(fq.run_order(m))
        self.assertEqual(len(pairs),60)
        self.assertEqual(len(set(pairs)),60)
        self.assertEqual(set(mode for qid,mode in pairs),set(fq.MODES))
        c=fq.Counter(mode for _,mode in pairs[:21])
        self.assertEqual(set(c.values()),{7})

    def test_checkpoint_rejects_duplicate_rows(self):
        ledger=self.root/'runs.jsonl'
        item={'question_id':'abc','mode':'basic'}
        ledger.write_text(json.dumps(item)+'\n'+json.dumps(item)+'\n',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            fq.clean_run_records(ledger)

    def test_summary_tracks_partial_and_memory_violation(self):
        rows=[{'mode':'basic','status':'succeeded','answer_f1_proxy':0.5,'gold_mapping_available':True,
               'evidence_metrics':{'f1':0.3},'citations':[{'number':1}],
               'metrics':{'total_ms':1000,'llm_calls':1,'retrieval_rounds':1,'model_ms':700,'retrieval_ms':100}},
              {'mode':'adaptive','status':'partial','answer_f1_proxy':0.0,'gold_mapping_available':False,
               'evidence_metrics':{'f1':None},'citations':[],
               'metrics':{'total_ms':2000,'llm_calls':3,'retrieval_rounds':2,'learning_memory_applied':True}}]
        stats={r['mode']:r for r in fq.summaries(rows)}
        self.assertEqual(stats['basic']['succeeded'],1)
        self.assertEqual(stats['adaptive']['partial'],1)
        self.assertEqual(stats['adaptive']['memory_leak_n'],1)
        self.assertEqual(stats['basic']['evidence_mapped_n'],1)

    def test_rc4_baseline_fingerprint_is_recorded(self):
        self.assertEqual(fq.app_fingerprint(),fq.FROZEN_APP_SHA256)
        self.assertEqual(fq.check_source(),fq.FROZEN_APP_SHA256)

    def test_local_source_change_warns_but_does_not_block(self):
        from unittest.mock import patch
        changed='1'*64
        with patch.object(fq,'app_fingerprint',return_value=changed):
            self.assertEqual(fq.check_source(),changed)
            with self.assertRaisesRegex(RuntimeError,'Disable --strict-source'):
                fq.check_source(strict=True)

    def test_per_trial_manifest_keeps_actual_source_hash(self):
        from unittest.mock import patch
        changed='2'*64
        with patch.object(fq,'app_fingerprint',return_value=changed):
            metadata=fq.experiment_metadata({'selected_json_sha256':'DUMMY','seed':62026},'qwen3:4b')
        self.assertEqual(metadata['app_code_sha256'],changed)
        self.assertEqual(metadata['model'],'qwen3:4b')
        self.assertIn('runner_sha256',metadata)

    def test_unfiltered_reasoning_fails_formal_eval_integrity_gate(self):
        fq.assert_final_output_integrity('The dataset contains 171 entries [1].')
        with self.assertRaisesRegex(RuntimeError,'reasoning tag'):
            fq.assert_final_output_integrity('private reasoning </think> Final text [1]')

    def test_original_driver_matches_original_source_hash(self):
        # Eval add-on must NOT change the existing Day2 runner.
        self.assertEqual(fq.sha(fq.HERE/'day2_qasper.py'),
                         fq.sha(fq.REPO/'artifact_evaluation'/'day2_qasper.py'))

if __name__=='__main__':unittest.main()

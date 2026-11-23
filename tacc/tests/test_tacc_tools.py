"""TACC tooling smoke tests; no GPU, TACC account or internet needed."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import smoke
import jobctl


class MockVllm(BaseHTTPRequestHandler):
    results=[]
    token='fake-key'
    def log_message(self,*args):pass
    def _send(self,body,ctype='application/json'):
        raw=body if isinstance(body,bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Type',ctype)
        self.send_header('Content-Length',str(len(raw)))
        self.end_headers();self.wfile.write(raw)
    def _auth(self):
        if self.headers.get('Authorization')!='Bearer '+self.token:
            self.send_error(401)
            return False
        return True
    def do_GET(self):
        if not self._auth():return
        if self.path!='/v1/models':return self.send_error(404)
        self._send({'data':[{'id':'qwen3.8-27b'}]})
    def do_POST(self):
        if not self._auth():return
        if self.path!='/v1/chat/completions':return self.send_error(404)
        data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.results.append(data)
        if data.get('stream'):
            stream=''.join('data: '+json.dumps(x)+'\n\n' for x in [
                {'choices':[{'delta':{'content':'TACC '}}]},
                {'choices':[{'delta':{'content':'ready'}}]},
            ])+'data: [DONE]\n\n'
            self._send(stream.encode(),'text/event-stream')
        elif data.get('response_format'):
            self._send({'choices':[{'message':{'content':'{"ready":true}'},'finish_reason':'stop'}]})
        else:
            self._send({'choices':[{'message':{'content':'TACC ready'},'finish_reason':'stop'}],
                        'usage':{'prompt_tokens':5,'completion_tokens':2}})


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),MockVllm)
        cls.server_thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.server_thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.server_thread.join(timeout=3)
    def test_smoke_chat_json_stream(self):
        url=f'http://127.0.0.1:{self.server.server_address[1]}/v1'
        result=smoke.check(url,'qwen3.8-27b','fake-key',timeout=3)
        self.assertTrue(result['chat']['ok'])
        self.assertTrue(result['json_mode']['ok'])
        self.assertTrue(result['stream']['ok'])
        self.assertEqual(result['stream']['content_chars'],10)
        self.assertEqual(result['chat']['usage']['completion_tokens'],2)
    def test_unlisted_model_rejected(self):
        url=f'http://127.0.0.1:{self.server.server_address[1]}/v1'
        with self.assertRaisesRegex(RuntimeError,'not exposed'):
            smoke.check(url,'hallucinated-model','fake-key',timeout=3)
    def test_invalid_url(self):
        with self.assertRaises(ValueError):
            smoke.check('file:///etc/passwd','qwen','fake-key')
    def test_wrong_key(self):
        url=f'http://127.0.0.1:{self.server.server_address[1]}/v1'
        with self.assertRaisesRegex(RuntimeError,'401'):
            smoke.check(url,'qwen3.8-27b','wrong-key',timeout=3)


class JobCtlTests(unittest.TestCase):
    def test_submit_requires_actual_files_and_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory,'job.sbatch');script.write_text('#!/bin/bash\n')
            config=Path(directory,'serving.env');config.write_text('TACC_MODEL_PATH=/real/path\n')
            args=argparse.Namespace(script=str(script),config=str(config),
                partition='',account='',gpus=2,gpu_syntax='gres',dry_run=True)
            with self.assertRaisesRegex(ValueError,'partition'):
                jobctl.submit(args)
            args.partition='approved-partition';args.account='approved-account'
            self.assertEqual(jobctl.submit(args),0)
    @patch('jobctl.subprocess.run')
    def test_status_calls_squeue(self,mock_run):
        mock_run.return_value.stdout='123|RUNNING|gpu001|00:01|01:00\n'
        self.assertEqual(jobctl.status(argparse.Namespace(job='123')),0)
        self.assertEqual(mock_run.call_args.args[0][0],'squeue')
    @patch('jobctl.subprocess.run')
    def test_cancel_requires_consent(self,mock_run):
        with self.assertRaisesRegex(ValueError,'--yes'):
            jobctl.cancel(argparse.Namespace(job='42',yes=False))
        mock_run.assert_not_called()
    def test_launcher_refuses_login_node(self):
        env={**os.environ}
        env.pop('SLURM_JOB_ID',None)
        proc=subprocess.run(['bash',str(ROOT/'serve_in_allocation.sh')],
                            env=env,capture_output=True,text=True,timeout=4)
        self.assertEqual(proc.returncode,2)
        self.assertIn('allocated Slurm job',proc.stderr)


if __name__=='__main__':unittest.main()

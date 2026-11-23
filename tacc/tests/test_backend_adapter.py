"""Check the existing app.agent.ModelClient against an OpenAI/vLLM mock server."""
import asyncio
import json
from pathlib import Path
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[2]
BACKEND=ROOT/'backend'
if not (BACKEND/'app/agent/model.py').exists():
    raise unittest.SkipTest('Backend snapshot not present; extract test only in complete repo')
sys.path.insert(0,str(BACKEND))
from app.agent.model import ModelClient
from test_tacc_tools import MockVllm


class BackendAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),MockVllm)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(3)
    def test_real_model_client_supports_chat_json_and_stream(self):
        endpoint=f'http://127.0.0.1:{self.server.server_address[1]}/v1'
        client=ModelClient(base_url=endpoint,model='qwen3.8-27b',api_key='fake-key',provider='openai')
        self.assertTrue(asyncio.run(client.is_available()))
        result=asyncio.run(client.chat([{'role':'user','content':'Are you ready?'}],json_mode=True))
        self.assertEqual(json.loads(result['content']),{'ready':True})
        async def collect():return ''.join([part async for part in client.stream(
            [{'role':'user','content':'Say hello'}])])
        self.assertEqual(asyncio.run(collect()),'TACC ready')
        self.assertEqual(client.last_stream_meta['finish_reason'],None)

if __name__=='__main__':unittest.main()

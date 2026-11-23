#!/usr/bin/env python3
"""Standard-library-only OpenAI-compatible vLLM smoke test.

No backend or commercial LLM SDK required. Intended for inside an allocated
TACC client job, or local Windows via an explicitly permitted SSH tunnel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from urllib import error, request, parse


def http_json(url: str, key: str, payload=None, *, timeout: int = 30):
    headers = {'Accept': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    if body is not None:
        headers['Content-Type'] = 'application/json'
    req = request.Request(url, data=body, headers=headers, method='POST' if body else 'GET')
    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read(4_000_000)
            return json.loads(raw), round((time.perf_counter()-start)*1000, 2)
    except error.HTTPError as exc:
        detail = exc.read(300).decode('utf-8','replace')
        raise RuntimeError(f'HTTP {exc.code} from vLLM at {parse.urlsplit(url).path}: {detail}') from exc
    except error.URLError as exc:
        raise RuntimeError(f'Network error (tunnel/job/port): {exc.reason}') from exc


def stream_request(url: str, key: str, payload: dict, *, timeout: int = 60):
    headers={'Accept':'text/event-stream','Content-Type':'application/json'}
    if key:headers['Authorization']='Bearer '+key
    req=request.Request(url,data=json.dumps(payload).encode(),headers=headers,method='POST')
    n_delta=0
    content=''
    usage={}
    t0=time.perf_counter()
    try:
        with request.urlopen(req,timeout=timeout) as resp:
            for raw in resp:
                line=raw.decode('utf-8','replace').strip()
                if not line.startswith('data: '):continue
                item=line[6:].strip()
                if item=='[DONE]':break
                try:obj=json.loads(item)
                except ValueError:continue
                if isinstance(obj,dict):
                    usage=obj.get('usage') or usage
                    for choice in obj.get('choices') or []:
                        delta=choice.get('delta') or {}
                        if isinstance(delta.get('content'),str):
                            content+=delta['content'];n_delta+=1
    except error.HTTPError as exc:
        raise RuntimeError(f'Stream HTTP {exc.code}: {exc.read(300).decode("utf-8","replace")}') from exc
    except error.URLError as exc:
        raise RuntimeError(f'Stream network error: {exc.reason}') from exc
    return {'delta_chunks':n_delta,'content_chars':len(content),
            'latency_ms':round((time.perf_counter()-t0)*1000,2),'usage':usage}


def check(base_url: str, model: str, key: str, *, timeout:int=60, stream=True, json_mode=True):
    parsed=parse.urlsplit(base_url)
    if parsed.scheme not in ('http','https') or not parsed.hostname:
        raise ValueError('Base URL must be an http(s) URL')
    root=base_url.rstrip('/')
    if not root.endswith('/v1'):root+='/v1'
    models,latency=http_json(root+'/models',key,timeout=timeout)
    ids=[item.get('id') for item in models.get('data',[]) if isinstance(item,dict)]
    if model not in ids:
        raise RuntimeError(f'Model {model!r} not exposed by vLLM; available={ids}')
    tests={'endpoint':root,'model':model,'models_latency_ms':latency}
    base_payload={'model':model,'messages':[{'role':'user','content':'Reply with exactly: TACC ready'}],
                  'temperature':0,'max_tokens':64}
    answer,latency=http_json(root+'/chat/completions',key,base_payload,timeout=timeout)
    choice=(answer.get('choices') or [{}])[0]
    content=(choice.get('message') or {}).get('content') or ''
    if not isinstance(content,str) or not content.strip():
        raise RuntimeError('vLLM returned empty answer; inspect thinking mode and token budget')
    tests['chat']={'ok':True,'response_chars':len(content),'latency_ms':latency,
                   'usage':answer.get('usage') or {},'finish_reason':choice.get('finish_reason')}
    if json_mode:
        payload={**base_payload,
                 'messages':[{'role':'user','content':'Return only JSON: {"ready": true}'}],
                 'response_format':{'type':'json_object'}}
        try:
            obj,jlat=http_json(root+'/chat/completions',key,payload,timeout=timeout)
            msg=((obj.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
            parsed_answer=json.loads(msg)
            if not isinstance(parsed_answer,dict):
                raise RuntimeError('JSON response was not an object')
            tests['json_mode']={'ok':True,'latency_ms':jlat}
        except (ValueError,RuntimeError) as exc:
            tests['json_mode']={'ok':False,'error':str(exc)[:300]}
    if stream:
        try:
            out=stream_request(root+'/chat/completions',key,{**base_payload,'stream':True},timeout=timeout)
            tests['stream']={'ok':out['content_chars']>0,**out}
        except RuntimeError as exc:
            tests['stream']={'ok':False,'error':str(exc)[:300]}
    return tests


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default=os.environ.get('MODEL_BASE_URL','http://127.0.0.1:8001/v1'))
    p.add_argument('--model',default=os.environ.get('MODEL_NAME','qwen3.8-27b'))
    p.add_argument('--api-key-env',default='MODEL_API_KEY',help='Environment variable name, never inline secret')
    p.add_argument('--timeout',type=int,default=90)
    p.add_argument('--no-stream',action='store_true')
    p.add_argument('--no-json',action='store_true')
    p.add_argument('--output',help='Optional JSON results file')
    a=p.parse_args(argv)
    try:
        results=check(a.base_url,a.model,os.environ.get(a.api_key_env,''),
                      timeout=a.timeout,stream=not a.no_stream,json_mode=not a.no_json)
    except (ValueError,RuntimeError,TimeoutError) as exc:
        print('TACC SMOKE FAILED:',exc,file=sys.stderr)
        return 1
    print(json.dumps(results,ensure_ascii=False,indent=2))
    if a.output:
        from pathlib import Path
        path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    ok=results['chat']['ok'] and all(results[k]['ok'] for k in ('json_mode','stream') if k in results)
    print('TACC SMOKE:', 'PASS' if ok else 'PARTIAL (unsupported JSON/stream)')
    return 0 if ok else 3

if __name__=='__main__':raise SystemExit(main())

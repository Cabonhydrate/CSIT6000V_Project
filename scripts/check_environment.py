"""Run from backend directory with python ../scripts/check_environment.py."""
import importlib.util
import os
import sys
import urllib.request

print('Python:',sys.version.split()[0])
for lib in ['fastapi','multipart','fitz','numpy','langgraph','sentence_transformers','faiss','docling']:
    print(f'{lib:24s}:', 'available' if importlib.util.find_spec(lib) else 'not installed')
base=os.environ.get('MODEL_BASE_URL','http://127.0.0.1:11434/v1').rstrip('/')
try:
    with urllib.request.urlopen(urllib.request.Request(base+'/models'),timeout=3) as res:
        print('Local model endpoint:',res.status,base)
except Exception as e:
    print('Local model endpoint is offline/unreachable:',str(e)[:160])
print('No TACC settings are required.')

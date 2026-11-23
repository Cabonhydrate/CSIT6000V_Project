#!/usr/bin/env python3
"""Verify source hashes for the standalone TACC adapter kit."""
from pathlib import Path
from hashlib import sha256
import json
import sys


def verify(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    problems=[]
    for relative,expected in manifest['sha256'].items():
        path=root/relative
        if not path.is_file():
            problems.append(f'missing: {relative}')
        elif sha256(path.read_bytes()).hexdigest()!=expected:
            problems.append(f'hash mismatch: {relative}')
    return problems


if __name__=='__main__':
    issues=verify()
    for issue in issues:print('FAIL',issue,file=sys.stderr)
    print('TACC kit integrity:', 'PASS' if not issues else 'FAILED')
    raise SystemExit(0 if not issues else 1)

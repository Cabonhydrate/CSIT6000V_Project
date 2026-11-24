#!/usr/bin/env python3
"""Verify packaged source files against SOURCE_SHA256.txt (stdlib only)."""
from pathlib import Path
import hashlib
import sys


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    manifest = root / 'SOURCE_SHA256.txt'
    failures = []
    checked = 0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        digest, sep, filename = line.partition('  ')
        if not sep or len(digest) != 64:
            failures.append('Invalid manifest entry: ' + line[:100])
            continue
        target = root / filename
        # All manifest paths are relative; avoid traversing outside the root.
        if not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
            failures.append('File missing/invalid: ' + filename)
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            failures.append('Checksum mismatch: ' + filename)
        checked += 1
    if failures:
        for item in failures:
            print('[FAIL]', item)
        return 1
    print(f'[PASS] Verified {checked} source files against {manifest.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Merge course TACC documentation into an existing Paper Tutor repository.

Preserves current README content outside the managed block, backs up documents,
verifies package hashes and does not edit program code.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys

START = '<!-- TACC-COURSE-SDK-START -->'
END = '<!-- TACC-COURSE-SDK-END -->'


def _valid_package(root: Path):
    manifest = json.loads((root / 'docs_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = root / entry['path']
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Package integrity check failed: {entry["path"]}')


def merge_readme(original: str, replacement: str) -> str:
    a, b = original.find(START), original.find(END)
    if (a >= 0) != (b >= 0):
        raise ValueError('Only one TACC managed marker found; do not overwrite broken README')
    if a >= 0:
        if b <= a:
            raise ValueError('README TACC markers are reversed')
        result = original[:a] + replacement.strip() + original[b + len(END):]
    else:
        anchor = '## 3. Test without downloading real QASPER'
        i = original.find(anchor)
        if i < 0:
            i = original.find('## References')
        if i >= 0:
            result = original[:i] + replacement.strip() + '\n\n' + original[i:]
        else:
            result = original.rstrip() + '\n\n' + replacement.strip() + '\n'
    result = result.replace('### Local model (no TACC)',
                            '### Local model (Ollama; see TACC deployment below)')
    result = result.replace(
        '9. **TACC (intentionally omitted):** future stage adds vLLM GPU deployment/configuration and verified model endpoint routing, without modifying the Agent workflow.',
        '9. **TACC SDK documentation added:** See the course TACC section above and `tacc/README_TACC.md`; the actual GPU and network verification is pending.')
    result = result.replace(
        '> **Scope of this release:** Local application implementation only. No TACC configuration, cluster scheduler, TACC SSH tunnel, or remote GPU deployment.',
        '> **Historical Day1 scope:** The initial package omitted TACC. Current course SDK instructions are documented below; actual TACC GPU deployment is not yet verified.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', default='.', type=Path,
                        help='Inner repo root containing backend, frontend, README.md')
    parser.add_argument('--dry-run', action='store_true', help='Check docs and repo without writing')
    args = parser.parse_args(argv)
    package_root = Path(__file__).resolve().parent
    repo = args.repo_root.expanduser().resolve()
    _valid_package(package_root)
    if not (repo / 'backend/app/main.py').is_file() or not (repo / 'frontend/package.json').is_file():
        raise ValueError('The repository must contain backend/app/main.py and frontend/package.json')
    readme_path = repo / 'README.md'
    if not readme_path.is_file():
        raise ValueError('Cannot find README.md at the specified repo root')
    original_bytes = readme_path.read_bytes()
    has_bom = original_bytes.startswith(b'\xef\xbb\xbf')
    text = original_bytes.decode('utf-8-sig')
    replacement = (package_root / 'root_tacc_section.md').read_text(encoding='utf-8')
    merged = merge_readme(text, replacement)
    targets = [
        (readme_path, (b'\xef\xbb\xbf' if has_bom else b'') + merged.encode('utf-8')),
        (repo / 'tacc/README_TACC.md', (package_root / 'tacc/README_TACC.md').read_bytes()),
        (repo / 'tacc/TACC_QUICKSTART.md', (package_root / 'tacc/TACC_QUICKSTART.md').read_bytes()),
    ]
    print('Checking repository:', repo)
    for target, content in targets:
        current = target.read_bytes() if target.exists() else None
        print(f'  {"unchanged" if current == content else "update"}: {target.relative_to(repo)}')
    if args.dry_run:
        print('DRY RUN OK; no files modified.')
        return 0
    if all(target.exists() and target.read_bytes() == content for target, content in targets):
        print('README and TACC guides already match; nothing to update.')
        return 0

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = repo / 'patch_backups' / f'tacc_docs_{stamp}'
    for target, _ in targets:
        if target.exists():
            archive = backup / target.relative_to(repo)
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, archive)
    for target, content in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.tacc_tmp')
        temporary.write_bytes(content)
        os.replace(temporary, target)
    print('Documentation updated; backups:', backup)
    print('No application source files or course scripts were modified.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, UnicodeError, KeyError, json.JSONDecodeError) as exc:
        print('TACC README update failed:', exc, file=sys.stderr)
        raise SystemExit(1)

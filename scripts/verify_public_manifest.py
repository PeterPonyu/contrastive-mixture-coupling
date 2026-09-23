#!/usr/bin/env python3
"""Check the published scientific input files using only the Python library."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', action='store_true',
                        help='Also require the optional checkpoint archive in this repository')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'validation/recomputed/input-integrity.json')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'public-manifest.json').read_text())
    files = dict(manifest['files'])
    if args.checkpoints:
        files.update(manifest['checkpoint_files'])
    for relative, expected in files.items():
        source = (ROOT/relative).resolve()
        if not source.is_relative_to(ROOT):
            raise SystemExit(f'Nonlocal input in manifest: {relative}')
        if not source.is_file():
            raise SystemExit(f'Missing input: {relative}')
        with source.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected['sha256'] or source.stat().st_size != expected['bytes']:
            raise SystemExit(f'Input bytes differ from the public manifest: {relative}')
    report = dict(gate='PASS', verified_files=len(files),
                  checkpoint_files_required=args.checkpoints,
                  new_neural_updates=0, new_mixture_fits=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

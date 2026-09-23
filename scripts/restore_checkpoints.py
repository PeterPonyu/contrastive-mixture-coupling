"""Download optional checkpoint volumes, verify them, and reconstruct one ZIP."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-dir', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Retrieve missing volumes from this study\'s public Zenodo record.')
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced' / 'checkpoints.zip')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'checkpoint-downloads.json').read_text())
    args.download_dir.mkdir(parents=True, exist_ok=True)
    def ensure(part):
        path = args.download_dir / part['name']
        if path.is_file() and path.stat().st_size == part['bytes'] and digest(path) == part['sha256']:
            return path
        if path.exists():
            raise ValueError(f'Existing volume differs; preserve or remove it explicitly: {path}')
        if not args.download:
            raise FileNotFoundError(f'Missing {path}; download this study\'s volumes or add --download.')
        temporary = path.with_name(path.name + '.partial')
        if temporary.exists():
            raise FileExistsError(f'Incomplete prior download retained at {temporary}; inspect/remove it before retrying.')
        with urllib.request.urlopen(part['url'], timeout=180) as source, temporary.open('xb') as destination:
            while block := source.read(1024 * 1024):
                destination.write(block)
        if temporary.stat().st_size != part['bytes'] or digest(temporary) != part['sha256']:
            raise ValueError(f'Download failed checksum: {temporary}')
        temporary.replace(path)
        print('Verified', part['name'], flush=True)
        return path
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(ensure, manifest['parts']))
    if args.output.exists():
        if digest(args.output) == manifest['archive']['sha256']:
            print('Verified existing archive:', args.output)
            return
        raise FileExistsError(f'Refusing to overwrite a different output: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.partial')
    with temporary.open('xb') as destination:
        for path in files:
            with path.open('rb') as source:
                while block := source.read(1024 * 1024):
                    destination.write(block)
    if temporary.stat().st_size != manifest['archive']['bytes'] or digest(temporary) != manifest['archive']['sha256']:
        raise ValueError('Reassembled archive checksum mismatch; partial output preserved.')
    temporary.replace(args.output)
    print('Verified checkpoint ZIP:', args.output)
    print('Extract at the repository root to restore package-relative checkpoint paths.')
    print('No checkpoint is deserialized, and no model or training is executed.')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Audit CPU checkpoint metadata from the optional Zenodo checkpoint archive."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint-root', type=Path, default=ROOT/'evidence/run-archive',
                    help='Directory containing seed0, seed1 and seed2 from the checkpoint archive')
parser.add_argument('--output', type=Path,
                    default=ROOT/'validation/recomputed/checkpoint-metadata.json')
args = parser.parse_args()
manifest = json.loads((ROOT/'public-manifest.json').read_text())
missing = [f'seed{seed}/{name}.pt' for seed in range(3)
           for name in ('initial', 'warmup', 'w0', 'w1')
           if not (args.checkpoint_root/f'seed{seed}/{name}.pt').is_file()]
if missing:
    parser.error('Optional checkpoint files are absent: ' + ', '.join(missing) +
                 '. Extract the corresponding Zenodo checkpoint archive and pass its '
                 'evidence/run-archive directory with --checkpoint-root. '
                 'The small-input audit is scripts/verify_saved_results.py.')


def state_digest(state):
    """Retain the historical registered-model state digest convention."""
    digest = hashlib.sha256()

    def add(value):
        if torch.is_tensor(value):
            digest.update(str(value.dtype).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                add(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        else:
            digest.update(str(value).encode())

    add(state)
    return digest.hexdigest()


# These historical files contain RNG state, so they need the pickle loader.
# Verify all bytes against the published archive identities before loading any.
for seed in range(3):
    for name in ('initial', 'warmup', 'w0', 'w1'):
        relative = f'evidence/run-archive/seed{seed}/{name}.pt'
        checkpoint = args.checkpoint_root/f'seed{seed}/{name}.pt'
        expected = manifest['checkpoint_files'][relative]
        assert checkpoint.stat().st_size == expected['bytes'], relative
        with checkpoint.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert actual == expected['sha256'], f'checkpoint hash mismatch: {relative}'

results = json.loads((ROOT/'evidence/new_results.json').read_text())['rows']
rows=[]
for seed in range(3):
    base=args.checkpoint_root/f'seed{seed}'
    states={name:torch.load(base/f'{name}.pt',map_location='cpu',weights_only=False) for name in ('initial','warmup','w0','w1')}
    b=states['warmup']['model']
    for w in (0, 1):
        result = next(row for row in results if (row['seed'], row['w']) == (seed, w))
        assert state_digest(b) == result['warmup_sha']
    assert states['warmup']['epochs_completed']==180
    assert all(x in b for x in ('ae.moco.queue','ae.moco.queue_ptr','ae.moco.query_projector.1.num_batches_tracked','ae.moco.key_projector.1.num_batches_tracked'))
    assert all(torch.equal(b[k],states[a]['model'][k]) is False for a in ('w0','w1') for k in ('ae.moco.queue',))
    row={'seed':seed,'states':{}}
    for name,state in states.items():
        sd=state['model']; prefix='ae.moco.'
        diff=max((sd[prefix+'query_projector.'+key]-sd[prefix+'key_projector.'+key]).abs().max().item()
                 for key in ('0.weight','0.bias','1.weight','1.bias','3.weight','3.bias'))
        row['states'][name]={'epochs_completed':state.get('epochs_completed'),
            'queue_ptr':int(sd[prefix+'queue_ptr'].item()),'queue_shape':list(sd[prefix+'queue'].shape),
            'query_bn_batches':int(sd[prefix+'query_projector.1.num_batches_tracked'].item()),
            'key_bn_batches':int(sd[prefix+'key_projector.1.num_batches_tracked'].item()),
            'query_key_max_parameter_difference':diff}
    assert row['states']['initial']['query_key_max_parameter_difference']>0
    assert row['states']['warmup']['query_bn_batches']>row['states']['warmup']['key_bn_batches']
    rows.append(row)
assert rows == json.loads((ROOT/'evidence/checkpoint-metadata.json').read_text())['rows']
out={'protocol':'CPU checkpoint metadata and registered-model digest verification; no model import, optimizer step or mixture fit',
     'gate':'PASS', 'checkpoint_files_verified':12, 'new_neural_updates':0, 'rows':rows}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

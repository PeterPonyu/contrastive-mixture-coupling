#!/usr/bin/env python3
"""Non-training, deterministic probes of the bundled frozen DPMM code path.

Both required code files are hash-pinned, including for an alternative source root.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

EXPECTED = 'f1514e701b139a64e214ac0c40e32107853ebbbac1b0a9823f5affaac0e7f8ea'
EXPECTED_BASE_MODEL = '3bf3e923fa7f4c25f5e78747174b7c88b210cf7a41d36e2e01d54314c0c7a7f1'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / 'evidence' / 'frozen-source'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'validation/recomputed/invariant-probe.json')
    args = parser.parse_args()
    source = args.source_root / 'models' / 'dpmm_contrastive.py'
    base_model = args.source_root / 'utils' / 'base_model.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    base_digest = hashlib.sha256(base_model.read_bytes()).hexdigest()
    if digest != EXPECTED:
        raise SystemExit(f'source hash changed: {digest} != {EXPECTED}')
    if base_digest != EXPECTED_BASE_MODEL:
        raise SystemExit(f'base model hash changed: {base_digest} != {EXPECTED_BASE_MODEL}')
    sys.path.insert(0, str(args.source_root))
    from models.dpmm_contrastive import MomentumContrast, DPMMODEContrastiveModel

    results = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        m = MomentumContrast(10, embedding_dim=16, queue_size=16, device=torch.device('cpu'))
        q = list(m.query_projector.parameters())
        k = list(m.key_projector.parameters())
        init_max_diff = max((a-b).abs().max().item() for a,b in zip(q,k))
        initial_buffers_equal = all(torch.equal(m.query_projector[1].state_dict()[n], m.key_projector[1].state_dict()[n]) for n in m.query_projector[1].state_dict())
        assert init_max_diff > 1e-6
        assert initial_buffers_equal
        before_q = [p.detach().clone() for p in q]
        before_k = [p.detach().clone() for p in k]
        before_queue = m.queue.detach().clone()
        x1 = torch.randn(4,10); x2 = torch.randn(4,10)
        logits, labels = m(x1,x2)
        expected_key = [m.momentum * old + (1-m.momentum) * query for old,query in zip(before_k,before_q)]
        ema_error = max((a-b).abs().max().item() for a,b in zip(k,expected_key))
        changed_queue_cols = torch.nonzero(torch.any(m.queue != before_queue, dim=0)).flatten().tolist()
        assert ema_error < 1e-6 and changed_queue_cols == [0,1,2,3] and m.queue_ptr.item() == 4
        assert logits.shape == (4,17) and torch.equal(labels, torch.zeros(4,dtype=torch.long))
        assert all(torch.equal(a,b) for a,b in zip(q,before_q))
        # The forward's negative bank must be pre-enqueue, not today's keys.
        # Only the four destination columns change in this single-forward check.
        results.append(dict(seed=seed, initialization_max_parameter_difference=init_max_diff,
                            initialization_bn_buffers_equal=initial_buffers_equal,
                            one_forward_ema_max_error=ema_error, one_forward_queue_changed_columns=changed_queue_cols,
                            one_forward_queue_ptr=m.queue_ptr.item(), one_forward_logit_shape=list(logits.shape)))

    torch.manual_seed(17)
    model = DPMMODEContrastiveModel(input_dim=12, latent_dim=10, encoder_dims=[16], decoder_dims=[16],
                                   dropout_rate=0., moco_embedding_dim=16, moco_queue_size=16,
                                   aug_noise_prob=0., aug_mask_prob=0.)
    model.train()
    m = model.ae.moco
    calls = {'query':0, 'key':0}
    hq = m.query_projector[1].register_forward_hook(lambda *a: calls.__setitem__('query',calls['query']+1))
    hk = m.key_projector[1].register_forward_hook(lambda *a: calls.__setitem__('key',calls['key']+1))
    batch = torch.randn(4,12)
    torch.manual_seed(29)
    outputs = model(batch)
    after_forward = dict(calls)
    before_loss_ptr = m.queue_ptr.item()
    losses = model.compute_loss(batch,outputs)
    after_loss = dict(calls)
    after_loss_ptr = m.queue_ptr.item()
    hq.remove(); hk.remove()
    assert after_forward == {'query':1,'key':1}
    assert after_loss == {'query':4,'key':1}
    assert before_loss_ptr == after_loss_ptr == 4
    contrastive = losses['moco_loss'] + .5*losses['symmetric_cl_loss'] + .3*losses['prototype_cl_loss']
    assert torch.isfinite(contrastive) and contrastive.item() > 0
    original_total = losses['total_loss'].detach()
    model.moco_weight = 0.
    zero = model.compute_loss(batch,outputs)
    assert m.queue_ptr.item() == after_loss_ptr
    error = abs((original_total-zero['total_loss']).item()-contrastive.item())
    assert error < 1e-5
    # Loss-weight zero still ran forward, enqueue and all auxiliary projector calls.
    # It suppresses the scalar contribution/gradient, not the stateful route.
    report = dict(source='evidence/frozen-source/models/dpmm_contrastive.py',source_sha256=digest,
                  base_model='evidence/frozen-source/utils/base_model.py',base_model_sha256=base_digest,torch=torch.__version__,
                  protocol='CPU synthetic four-row batches, no optimizer step or biological data; seeds 0/1/2 for initialization/forward; seed 17 model and seed 29 augmentation stream',
                  isolated=results, integrated=dict(query_bn_calls_after_forward=after_forward['query'],
                  query_bn_calls_after_loss=after_loss['query'],key_bn_calls_after_loss=after_loss['key'],
                  queue_ptr_after_forward=before_loss_ptr,queue_ptr_after_loss=after_loss_ptr,
                  moco_weight_one_to_zero_total_delta=float((original_total-zero['total_loss']).item()),
                  contrastive_composite=float(contrastive.item()),difference_error=error,
                  moco_weight_zero_queue_ptr_after_recompute=m.queue_ptr.item()),
                  gate='PASS: assertions executed; does not certify full training or biological endpoints')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()

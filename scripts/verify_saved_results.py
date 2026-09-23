#!/usr/bin/env python3
"""Verify coupling contrasts, state pairing and saved mixture probabilities."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'validation/recomputed/robustness-audit.json')
    args = parser.parse_args()
    original=json.loads((ROOT/'evidence/new_results.json').read_text())
    assert original == json.loads((ROOT/'evidence/run-archive/results.json').read_text())
    rows=original['rows']; assert len(rows)==6
    contrasts=[]; max_error=0.
    for seed in range(3):
        folder=ROOT/'evidence/run-archive'/f'seed{seed}'
        a,b=[next(r for r in rows if (r['seed'],r['w'])==(seed,w)) for w in (0,1)]
        assert a['warmup_sha']==b['warmup_sha'] and a['batch_sha']==b['batch_sha']
        permutations=np.load(folder/'permutations.npy')
        assert sha_array(permutations)==a['batch_sha']
        assert all(np.array_equal(np.sort(p),np.arange(2100)) for p in permutations)
        for row in (a,b):
            w=row['w']; gate=row['gate']
            assert gate[0]['model']==gate[1]['model'] and gate[0]['optimizer']==gate[1]['optimizer']
            endpoint=json.loads((folder/f'w{w}.json').read_text())
            assert endpoint['metrics']==row['metrics']
            assert [x['epoch'] for x in row['training_refits']]==[180,190]
            for refit in row['training_refits']:
                assert refit['subset_sha']==sha_array(permutations[refit['epoch'],:2048])
            with np.load(folder/f'w{w}.npz') as fit:
                z=fit['latent']; assert z.shape==(450,10)
                for prefix,target in (('train','training_occupancy'),('observer','observer_occupancy')):
                    p=fit[prefix+'_weights'].astype(float);p=np.maximum(p,1e-12);p/=p.sum()
                    occ=float(np.exp(-np.dot(p,np.log(p))))
                    np.testing.assert_allclose(occ,row[target]['effective'],atol=1e-6,rtol=0)
                    c=fit[prefix+'_precisions_cholesky']; mu=fit[prefix+'_means']
                    logits=np.log(np.maximum(fit[prefix+'_weights'],1e-12))[None]+np.log(c).sum(1)[None]-.5*(10*np.log(2*np.pi)+(((z[:,None]-mu[None])*c[None])**2).sum(2))
                    probs=np.exp(logits-logsumexp(logits,axis=1)[:,None])
                    error=float(np.abs(probs-fit[prefix+'_responsibilities']).max());max_error=max(max_error,error)
                    assert error<1e-5
        contrasts.append(dict(seed=seed,edge_delta=b['metrics']['edge_survival']-a['metrics']['edge_survival'],
                              branch_delta=b['metrics']['branch_knn_mean_r2']-a['metrics']['branch_knn_mean_r2'],
                              occupancy_delta=b['training_occupancy']['effective']-a['training_occupancy']['effective']))
    assert all(r['edge_delta']<0 and r['branch_delta']<0 for r in contrasts)
    sensitivity={}
    for field in ('edge_delta','branch_delta'):
        values=np.array([r[field] for r in contrasts])
        sensitivity[field]=dict(mean=float(values.mean()),sample_sd=float(values.std(ddof=1)),
                                range=[float(values.min()),float(values.max())],
                                leave_one_seed_out_means=[float(np.delete(values,i).mean()) for i in range(3)])
    historical=[]
    for budget,foldername in ((200,'dpmm-contrastive-prior-20260907'),(400,'dpmm-contrastive-prior-400ep-setty-20260908')):
        runs=json.loads((ROOT/'evidence'/foldername/'sweep_scores.json').read_text())['runs']
        for seed in range(3):
            a,b=[next(r for r in runs if (r['dataset'],r['seed'],r['axis'],r['value'])==('setty',seed,'mixture_weight',w)) for w in (0,1)]
            historical.append(dict(epochs=budget,seed=seed,branch_delta=b['branch_knn_mean_r2']-a['branch_knn_mean_r2']))
    assert all(r['branch_delta']<0 for r in historical)
    report=dict(gate='PASS',paired_seed_count=3,endpoint_count=6,paired_contrasts=contrasts,
                sensitivity=sensitivity,historical_branch_contrasts=historical,
                max_responsibility_reconstruction_error=max_error,
                new_neural_updates=0,new_mixture_fits=0,
                limitation='Scores independently read and differenced; PCA input/branch targets are not bundled, so endpoint metrics are not recomputed from raw data.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

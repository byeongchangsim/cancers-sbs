"""RBF kernel ridge on the unchanged, frozen SBS6 attention representation.

Saunders, Gammerman & Vovk, ICML 1998, pp. 515-521; sklearn KernelRidge.
Adaptation: original continuous SBS6 target and original patient partitions.
Train-fitted feature scaling/target centering; original inner SBS6 Pearson-r
selection; validation-standardised components and fixed equal averaging.
This is one RBF procedure, without mixing favourable linear/RBF results.
No MSI labels, outer-test outcomes or CPTAC outcomes enter selection.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from sbs6_probe import representation, training, original

ALPHAS = [.01, .1, 1., 10., 100., 1000.]
MULTIPLIERS = [.25, 1., 4.]
PROTOCOL = dict(
    method='sklearn KernelRidge: squared loss + L2; RBF kernel',
    publication='Saunders, Gammerman & Vovk. ICML 1998, pp.515-521',
    publication_url='https://eprints.soton.ac.uk/258942/1/Dualrr_ICML98.pdf',
    implementation_url='https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html',
    target='Original continuous SBS6 exposure', alpha_grid=ALPHAS,
    gamma_multiplier_grid=MULTIPLIERS, gamma_rule='multiplier / 256',
    selection='Maximum original TCGA inner-validation SBS6 Pearson r',
    feature_scaling='StandardScaler fitted only on original train patients',
    target_centering='Subtract train SBS6 mean for fitting; add it for prediction',
    component_scaling='Original inner-validation prediction mean/population SD',
    aggregation='Fixed 1:1 bag-scale mean; TCGA 3 OOF seeds; CPTAC 15 contexts',
    MSI_labels_read=False, TCGA_outer_targets_used_for_selection=False,
    CPTAC_outcomes_read=False, linear_RBF_selection_or_mixing=False)


def features(seed):
    """One seed cache holds ten frozen-representation contexts, not outcomes."""
    path = OUT / f'kread6_features_{seed}.npz'
    selected = pd.read_csv(original.SELECTION, sep='\t').query(
        'cancer == "UCEC" and seed_index == @seed')
    assert len(selected) == 10
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files if k != 'metadata'}
            metadata = json.loads(str(saved['metadata'].item()))
        for fold in range(5):
            for scale in [1024, 2048]:
                key = f'{fold}_{scale}'
                row = selected.query('fold == @fold and scale == @scale').iloc[0]
                assert metadata[key]['checkpoint_sha256'] == row.checkpoint_sha256
                assert metadata[key]['selected_variant'] == row.selected_variant
        return arrays, metadata
    arrays, metadata = {}, {}
    for fold in range(5):
        for scale in [1024, 2048]:
            key = f'{fold}_{scale}'
            row = selected.query('fold == @fold and scale == @scale').iloc[0]
            mod = training.implementation(str(scale))
            mode, hybrid = mod.variant_parts(row.selected_variant)
            directory = mod.output_dir(row.selected_variant, row.cancer_dir, seed, fold)
            checkpoint_path = directory / 'best.pt'
            assert original.sha256(checkpoint_path) == row.checkpoint_sha256
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            prefix = mod.cache_prefix('TCGA', 'Ucec', mode)
            cp_prefix = mod.cache_prefix('CPTAC', 'Ucec', mode)
            meta = pd.read_csv(prefix.with_suffix('.meta.csv'))
            cpmeta = pd.read_csv(cp_prefix.with_suffix('.meta.csv'))
            assert len(meta) == 422 and len(cpmeta) == 82
            roles = {r: [str(p) for p in checkpoint[r + '_patients']]
                     for r in ['train', 'validation', 'test']}
            sets = {r: set(p) for r, p in roles.items()}
            assert not (sets['train'] & sets['validation'] or
                        sets['train'] & sets['test'] or
                        sets['validation'] & sets['test'])
            assert set(meta.PATIENT) == set.union(*sets.values())
            model = mod.TargetAttentionMIL(hybrid).cuda().eval()
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            x = torch.tensor(np.load(prefix.with_suffix('.bags.npy'), mmap_mode='r'),
                             dtype=torch.float16, device='cuda')
            cx = torch.tensor(np.load(cp_prefix.with_suffix('.bags.npy'), mmap_mode='r'),
                              dtype=torch.float16, device='cuda')
            f = representation(model, x, meta.lens.to_numpy())
            cf = representation(model, cx, cpmeta.lens.to_numpy())
            assert f.shape == (422, 256) and cf.shape == (82, 256)
            assert np.isfinite(f).all() and np.isfinite(cf).all()
            # Same existing representation/forward check used by sbs6_probe.
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, logit = model(x[:mod.BATCH_SIZE], torch.tensor(
                    meta.lens.to_numpy()[:mod.BATCH_SIZE], device='cuda'))
            head = (f[:mod.BATCH_SIZE] @ model.presence_weight[5].detach().float().cpu().numpy()
                    + float(model.presence_bias[5].detach().cpu()))
            np.testing.assert_allclose(head, logit[:, 5].float().cpu().numpy(), atol=.1, rtol=.04)
            arrays[key + '_tcga'] = f
            arrays[key + '_cptac'] = cf
            metadata[key] = dict(seed=seed, fold=fold, scale=scale,
                tcga_patients=meta.PATIENT.tolist(), cptac_patients=cpmeta.PATIENT.tolist(),
                roles=roles, checkpoint_sha256=row.checkpoint_sha256,
                selected_variant=row.selected_variant, checkpoint=str(checkpoint_path),
                tcga_bag=str(prefix.with_suffix('.bags.npy')),
                cptac_bag=str(cp_prefix.with_suffix('.bags.npy')))
            print('cached representation', seed, fold, scale, flush=True)
            del model, x, cx
            torch.cuda.empty_cache()
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, metadata=np.array(json.dumps(metadata)), **arrays)
    temporary.replace(path)
    return arrays, metadata


def run(seed):
    assert seed in range(3)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    arrays, metadata = features(seed)
    labels = pd.read_csv(original.FROZEN / 'ucec_tcga_oof.tsv', sep='\t',
                         usecols=['PATIENT', 'SBS6']).set_index('PATIENT')
    audit, chosen = [], []
    for fold in range(5):
        outputs, contexts = [], []
        for scale in [1024, 2048]:
            key = f'{fold}_{scale}'
            m = metadata[key]
            assert m['tcga_patients'] == metadata[f'{fold}_1024']['tcga_patients']
            assert m['cptac_patients'] == metadata[f'{fold}_1024']['cptac_patients']
            ids, cpids, roles = m['tcga_patients'], m['cptac_patients'], m['roles']
            lookup = {p: i for i, p in enumerate(ids)}
            ix = {r: np.array([lookup[p] for p in roles[r]]) for r in roles}
            f, cf = arrays[key + '_tcga'].astype(float), arrays[key + '_cptac'].astype(float)
            y = labels.loc[ids, 'SBS6'].to_numpy(float)
            assert np.isfinite(y).all() and np.std(y[ix['validation']]) > 0
            scaler = StandardScaler().fit(f[ix['train']])
            xt, xv = scaler.transform(f[ix['train']]), scaler.transform(f[ix['validation']])
            mean = float(y[ix['train']].mean())
            best = None
            for alpha in ALPHAS:
                for multiplier in MULTIPLIERS:
                    gamma = multiplier / 256
                    model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma).fit(
                        xt, y[ix['train']] - mean)
                    validation = model.predict(xv) + mean
                    r = float(pearsonr(y[ix['validation']], validation)[0])
                    audit.append(dict(seed=seed, fold=fold, scale=scale, alpha=alpha,
                                      gamma=gamma, multiplier=multiplier, validation_r=r))
                    if np.isfinite(r) and (best is None or r > best[0]):
                        best = (r, model, alpha, gamma, validation)
            assert best is not None
            r, model, alpha, gamma, validation = best
            vm, vs = float(validation.mean()), float(validation.std(ddof=0))
            assert np.isfinite(vs) and vs > 0
            outputs.append(dict(
                validation=(validation - vm) / vs,
                tcga=(model.predict(scaler.transform(f[ix['test']])) + mean - vm) / vs,
                cptac=(model.predict(scaler.transform(cf)) + mean - vm) / vs))
            contexts.append(dict(scale=scale, alpha=alpha, gamma=gamma, validation_r=r,
                                 train_target_mean=mean, validation_mean=vm, validation_sd=vs,
                                 checkpoint_sha256=m['checkpoint_sha256']))
            assert roles == metadata[f'{fold}_1024']['roles']
        for co, patients in [('validation', roles['validation']),
                             ('tcga', roles['test']), ('cptac', cpids)]:
            values = np.mean([o[co] for o in outputs], axis=0)
            assert np.isfinite(values).all()
            pd.DataFrame(dict(PATIENT=patients, prediction_SBS6=values)).to_csv(
                OUT / f'kread6_{seed}_{fold}_{co}.tsv', sep='\t', index=False)
        chosen.append(dict(seed=seed, fold=fold, scales=contexts))
        print('selected continuous SBS6 KRR', seed, fold, contexts, flush=True)
    pd.DataFrame(audit).to_csv(OUT / f'kread6_{seed}_selection.tsv', sep='\t', index=False)
    (OUT / f'kread6_{seed}_protocol.json').write_text(json.dumps(
        dict(PROTOCOL, seed=seed, feature_cache=f'kread6_features_{seed}.npz',
             selected_contexts=chosen), indent=2))


def aggregate():
    for co, expected, repeats in [('tcga', 422, 3), ('cptac', 82, 15)]:
        parts = []
        for seed in range(3):
            seen = set()
            for fold in range(5):
                frame = pd.read_csv(OUT / f'kread6_{seed}_{fold}_{co}.tsv', sep='\t')
                assert frame.PATIENT.is_unique and np.isfinite(frame.prediction_SBS6).all()
                if co == 'tcga':
                    assert not (seen & set(frame.PATIENT))
                    seen.update(frame.PATIENT)
                parts.append(frame)
            if co == 'tcga':
                assert len(seen) == 422
        combined = pd.concat(parts, ignore_index=True)
        assert combined.groupby('PATIENT').size().eq(repeats).all()
        result = combined.groupby('PATIENT', as_index=False).prediction_SBS6.mean()
        assert len(result) == expected
        result.to_csv(OUT / f'kread6_{co}_scores.tsv', sep='\t', index=False)
        print(co, 'fixed KRR context average', len(result), flush=True)
    (OUT / 'kread6_protocol.json').write_text(json.dumps(PROTOCOL, indent=2))


if __name__ == '__main__':
    if sys.argv[1] == 'aggregate':
        aggregate()
    else:
        run(int(sys.argv[1]))

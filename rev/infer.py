"""Revision sensitivity: fixed primary models, CPTAC tumour-labelled slides.

Run with cenv Python, one cancer per GPU. No model/variant/scale selection.
The deterministic original PCA-ridge fit is replayed on unchanged TCGA folds;
its original all-slide output must reproduce before restricted inputs are used.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import torch, h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import build_full210_equal_model_average_20260904 as original
import run_full210_exact25_multiscale_fusion_20260904 as training
import run_full210_exact25_nested_stack_20260904 as stack

ROOT=original.BASE; OUT=ROOT/'rev'; SBS=list(original.SIGNATURES)

def run(cancer):
    started=time.monotonic(); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    builder=original.import_file(ROOT/'legacy30_retccl_ctranspath_feature_fusion_20260829/build_fused_bags.py','revision_bags')
    ridge_module=original.import_file(stack.RIDGE_SCRIPT,'revision_ridge')
    tp,tx,ty,*_=ridge_module.load('TCGA',cancer)
    cp,cx,cy,*_=ridge_module.load('CPTAC',cancer)
    slides=pd.read_csv(builder.SOURCE/'CPTAC'/cancer/'slides.tsv',sep='\t',dtype={'PATIENT':str})
    annotations=pd.read_csv(ROOT/'jpcr/slide_manifest.tsv',sep='\t').query('cohort == "CPTAC" and cancer == @cancer')
    slides=slides.merge(annotations[['PATIENT','FILENAME','specimen_type']],on=['PATIENT','FILENAME'],validate='one_to_one')
    assert len(slides)==len(annotations) and set(slides.PATIENT)==set(cp)
    parts={}; means={}; counts=[]; replay_checked=False
    for patient,d in slides.groupby('PATIENT',sort=True):
        all_f=[];all_q=[];tum_f=[];tum_q=[];retall=[];rettum=[]
        for row in d.itertuples():
            f,q,_,_=builder.read_slide('CPTAC',row.FILENAME)
            with h5py.File(builder.RETCCL/'CPTAC'/f'{row.FILENAME}.h5') as h:
                r=h['feats'][:].astype(np.float64)
            retall.append(r);all_f.append(f);all_q.append(q)
            if row.specimen_type=='tumor_tissue':
                tum_f.append(f);tum_q.append(q);rettum.append(r)
        allmean=np.concatenate(retall).mean(axis=0)
        k=np.flatnonzero(cp==patient)[0]
        np.testing.assert_allclose(allmean,cx[k],atol=2e-5,rtol=2e-5)
        af=np.concatenate(all_f);aq=np.concatenate(all_q)
        # One patient replay verifies the reused published sampling implementation.
        if not replay_checked:
            for scale,seed in [(1024,20260907),(2048,20260911)]:
                builder.BAG_SIZE=scale;builder.BASE_SEED=seed
                rng=np.random.default_rng(builder.stable_seed('CPTAC',cancer,patient))
                ix={'uniform':builder.select_uniform(len(af),rng),'qhybrid':builder.select_qhybrid(aq,rng)}
                mod=training.implementation(str(scale))
                for mode in ix:
                    old=np.load(mod.cache_prefix('CPTAC',builder.CACHE_CANCER[cancer],mode).with_suffix('.bags.npy'),mmap_mode='r')
                    np.testing.assert_array_equal(af[ix[mode]],old[k,:len(ix[mode])])
            replay_checked=True
        counts.append(dict(cancer=cancer,PATIENT=patient,all_slides=len(d),tumour_slides=len(tum_f),all_aligned_tiles=len(af),tumour_aligned_tiles=sum(len(f) for f in tum_f)))
        if tum_f:
            parts[patient]=(np.concatenate(tum_f),np.concatenate(tum_q))
            means[patient]=np.concatenate(rettum).mean(axis=0).astype(cx.dtype)
        if len(counts)%25==0: print(cancer,'slides_read',len(counts),flush=True)
    ids=np.array([p for p in cp if p in parts]); lookup={p:i for i,p in enumerate(cp)}
    ci=np.array([lookup[p] for p in ids]); restricted=np.stack([means[p] for p in ids])
    buffers={}
    for scale,seed in [(1024,20260907),(2048,20260911)]:
        builder.BAG_SIZE=scale;builder.BASE_SEED=seed
        arrays={m:np.zeros((len(ids),scale,2816),np.float16) for m in ('uniform','qhybrid')}
        lens=[]
        for k,p in enumerate(ids):
            f,q=parts[p];rng=np.random.default_rng(builder.stable_seed('CPTAC',cancer,p))
            ix={'uniform':builder.select_uniform(len(f),rng),'qhybrid':builder.select_qhybrid(q,rng)}
            for m in ix: arrays[m][k,:len(ix[m])]=f[ix[m]]
            lens.append(min(len(f),scale))
        buffers[scale]=({m:torch.tensor(x,device='cuda',dtype=torch.float16) for m,x in arrays.items()},np.array(lens))
    del parts
    selected=pd.read_csv(original.SELECTION,sep='\t').query('cancer == @cancer')
    predictions=[];folds=[];errors=[]
    for seed in range(3):
        for fold in range(5):
            mil=[];base_mil=[];roles=None
            old_context=original.OLD_STACK/'contexts'/cancer/f'seed-{seed}'/f'fold-{fold}'
            for scale,name in [(1024,'fused1024'),(2048,'fused2048')]:
                row=selected.query('scale == @scale and seed_index == @seed and fold == @fold').iloc[0]
                mod=training.implementation(str(scale));directory=mod.output_dir(row.selected_variant,row.cancer_dir,seed,fold)
                checkpoint_path=directory/'best.pt'
                assert original.sha256(checkpoint_path)==row.checkpoint_sha256
                checkpoint=torch.load(checkpoint_path,map_location='cpu')
                current={r:[str(p) for p in checkpoint[f'{r}_patients']] for r in ('train','validation','test')}
                if roles is None: roles=current
                else: assert roles==current
                sampling,hybrid=mod.variant_parts(row.selected_variant)
                model=mod.TargetAttentionMIL(hybrid).cuda().eval();model.load_state_dict(checkpoint['state_dict'],strict=True)
                bags,lens=buffers[scale]
                values=mod.predict(model,bags[sampling],lens,np.arange(len(ids)))['composition'].astype(float)
                mean,sd=original.component_moments(old_context,name)
                mil.append((values-mean)/sd)
                base=pd.read_csv(directory/'cptac_label_free_predictions.tsv',sep='\t').set_index('PATIENT').reindex(cp)
                base_mil.append((base[[f'composition_{s}' for s in SBS]].to_numpy(float)-mean)/sd)
                del model
            train_index=stack.indices_for(tp,roles['train'])
            scaler=StandardScaler();xtrain=scaler.fit_transform(tx[train_index])
            pca=PCA(n_components=16,random_state=20260901);xtrain=pca.fit_transform(xtrain)
            model=Ridge(alpha=100.0).fit(xtrain,np.sqrt(np.clip(ty[train_index],0,None)))
            def ridge(x): return np.square(np.clip(model.predict(pca.transform(scaler.transform(x))),0,None))
            mean,sd=original.component_moments(old_context,'retccl')
            base_ridge=(ridge(cx)-mean)/sd
            with np.load(original.CONTEXTS/cancer/f'seed-{seed}'/f'fold-{fold}'/'predictions.npz') as z:
                assert z['cptac_patients'].astype(str).tolist()==cp.tolist()
                reproduced=(base_ridge+base_mil[0]+base_mil[1])/3
                error=float(np.max(np.abs(reproduced-z['cptac_prediction'])))
                # Original float32 PCA replay differs slightly across BLAS runtimes.
                # Bound this well below the reported 0.001 score precision.
                np.testing.assert_allclose(reproduced,z['cptac_prediction'],atol=5e-4,rtol=3e-5)
                errors.append(error)
            pred=((ridge(restricted)-mean)/sd+mil[0]+mil[1])/3
            predictions.append(pred)
            for role,patients in roles.items():
                folds.extend(dict(cancer=cancer,seed_index=seed,seed=20260829+seed,fold=fold,role=role,PATIENT=p) for p in patients)
            print(cancer,'context',seed,fold,'baseline_max_error',error,flush=True)
    pd.DataFrame({'PATIENT':ids,**{f'prediction_{s}':np.mean(predictions,axis=0)[:,j] for j,s in enumerate(SBS)}}).to_csv(OUT/f'{cancer.lower()}_tumour_scores.tsv',sep='\t',index=False)
    pd.DataFrame(counts).to_csv(OUT/f'{cancer.lower()}_input_counts.tsv',sep='\t',index=False)
    pd.DataFrame(folds).to_csv(OUT/f'{cancer.lower()}_folds.tsv',sep='\t',index=False)
    report=dict(cancer=cancer,all_patients=len(cp),tumour_labelled_patients=len(ids),excluded_no_tumour_label=len(cp)-len(ids),all_slides=len(slides),tumour_slides=int(sum(r['tumour_slides'] for r in counts)),baseline_max_absolute_error=max(errors),original_bag_replay=True,original_checkpoint_hashes_verified=True,seconds=time.monotonic()-started,MIL_training=False,ridge='Original deterministic TCGA fit replay, verified against stored primary ensemble for every context',scaling='Original inner-validation means and SD retained',scope='CPTAC inference restriction; slide labels do not define pixel-level tumour regions')
    (OUT/f'{cancer.lower()}_inference.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__': run(sys.argv[1])

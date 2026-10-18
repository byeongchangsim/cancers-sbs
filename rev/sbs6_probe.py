"""Targeted scalar ridge readout of the unchanged primary SBS6 representation.

Reuse published regularised least-squares and the manuscript's trained
RetCCL/CTransPath target-attention representation. Change only the SBS6
regression readout; no MSI labels. Select alpha using original inner SBS6 r.
Average the two bag-scale regressors and the same original seed/fold outputs.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'rev';sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import run_full210_exact25_multiscale_fusion_20260904 as training
import build_full210_equal_model_average_20260904 as original
@torch.inference_mode()
def representation(model,x,lens,batch=20):
 model.eval();result=[]
 for start in range(0,len(x),batch):
  v=x[start:start+batch];ll=torch.as_tensor(lens[start:start+batch],device='cuda')
  with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
   a=model.retccl_encoder(v[:,:,:2048]);b=model.ctranspath_encoder(v[:,:,2048:]);h=model.encoder_fusion(torch.cat([a,b],dim=2))
   valid=torch.arange(v.shape[1],device='cuda')[None,:]<ll[:,None]
   logits=model.attention_w(model.attention_v(h)*model.attention_u(h));logits=torch.where(valid[:,:,None],logits,torch.full_like(logits,-1e4))
   attention=torch.softmax(logits,dim=1);pooled=torch.einsum('btk,bth->bkh',attention,h)
   if model.hybrid:
    mask=valid[:,:,None].to(h.dtype);den=ll.clamp_min(1).to(h.dtype)[:,None];mu=(h*mask).sum(1)/den
    variance=((h-mu[:,None,:]).square()*mask).sum(1)/den;sd=torch.sqrt(variance.clamp_min(1e-6))
    pooled=model.stat_fusion(torch.cat([pooled,mu[:,None,:].expand(-1,30,-1),sd[:,None,:].expand(-1,30,-1)],dim=2))
  result.append(pooled[:,5,:].float().cpu().numpy())
 return np.concatenate(result)
def run(seed):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 selected=pd.read_csv(original.SELECTION,sep='\t').query('cancer == "UCEC" and seed_index == @seed')
 labels=pd.read_csv(original.FROZEN/'ucec_tcga_oof.tsv',sep='\t').set_index('PATIENT');audit=[]
 for fold in range(5):
  outputs=[];contexts=[];ids=None;cids=None
  for scale in [1024,2048]:
   row=selected.query('scale == @scale and fold == @fold').iloc[0];mod=training.implementation(str(scale));mode,hybrid=mod.variant_parts(row.selected_variant)
   directory=mod.output_dir(row.selected_variant,row.cancer_dir,seed,fold);checkpoint=torch.load(directory/'best.pt',map_location='cpu');assert original.sha256(directory/'best.pt')==row.checkpoint_sha256
   meta=pd.read_csv(mod.cache_prefix('TCGA','Ucec',mode).with_suffix('.meta.csv'));cm=pd.read_csv(mod.cache_prefix('CPTAC','Ucec',mode).with_suffix('.meta.csv'));ids=meta.PATIENT.to_numpy(str);cids=cm.PATIENT.to_numpy(str)
   model=mod.TargetAttentionMIL(hybrid).cuda().eval();model.load_state_dict(checkpoint['state_dict'],strict=True)
   x=torch.tensor(np.load(mod.cache_prefix('TCGA','Ucec',mode).with_suffix('.bags.npy'),mmap_mode='r'),dtype=torch.float16,device='cuda');cx=torch.tensor(np.load(mod.cache_prefix('CPTAC','Ucec',mode).with_suffix('.bags.npy'),mmap_mode='r'),dtype=torch.float16,device='cuda')
   f=representation(model,x,meta.lens.to_numpy());cf=representation(model,cx,cm.lens.to_numpy())
   # Check that the extracted representation matches the original forward.
   with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
    comp,logit=model(x[:mod.BATCH_SIZE],torch.tensor(meta.lens.to_numpy()[:mod.BATCH_SIZE],device='cuda'))
   from_features=f[:mod.BATCH_SIZE] @ model.presence_weight[5].detach().float().cpu().numpy()+float(model.presence_bias[5].detach().cpu())
   np.testing.assert_allclose(from_features,logit[:,5].float().cpu().numpy(),atol=.1,rtol=.04)
   roles={r:[str(p) for p in checkpoint[r+'_patients']] for r in ['train','validation','test']};lookup={p:i for i,p in enumerate(ids)};ix={r:np.array([lookup[p] for p in roles[r]]) for r in roles}
   y=labels.loc[ids,'SBS6'].to_numpy(float);scaler=StandardScaler().fit(f[ix['train']]);xt=scaler.transform(f[ix['train']]);xv=scaler.transform(f[ix['validation']]);best=None
   for alpha in [.01,.1,1.,10.,100.,1000.]:
    ridge=Ridge(alpha=alpha).fit(xt,y[ix['train']]);v=ridge.predict(xv);r=float(pearsonr(y[ix['validation']],v).statistic)
    audit.append(dict(seed=seed,fold=fold,scale=scale,alpha=alpha,validation_r=r))
    if best is None or r>best[0]:best=(r,ridge,alpha,v)
   r,ridge,alpha,v=best;vm=float(v.mean());vs=float(v.std());assert vs>0
   # Preserve the manuscript's established score definition: each component
   # is standardised using its original inner-validation patients, then averaged.
   outputs.append(dict(test=(ridge.predict(scaler.transform(f[ix['test']]))-vm)/vs,cptac=(ridge.predict(scaler.transform(cf))-vm)/vs,validation=(v-vm)/vs,roles=roles))
   for co,patient_list,key in [('tcga',roles['test'],'test'),('validation',roles['validation'],'validation'),('cptac',cids,'cptac')]:
    pd.DataFrame(dict(PATIENT=patient_list,prediction_SBS6=outputs[-1][key])).to_csv(OUT/f'probe6_{seed}_{fold}_{scale}_{co}.tsv',sep='\t',index=False)
   contexts.append(dict(scale=scale,alpha=alpha,validation_r=r,validation_mean=vm,validation_sd=vs,checkpoint_sha256=row.checkpoint_sha256));del model,x,cx;torch.cuda.empty_cache()
  assert outputs[0]['roles']==outputs[1]['roles'];roles=outputs[0]['roles']
  for co,patient_list,key in [('tcga',roles['test'],'test'),('validation',roles['validation'],'validation'),('cptac',cids,'cptac')]:
   values=np.mean([o[key] for o in outputs],axis=0);pd.DataFrame(dict(PATIENT=patient_list,prediction_SBS6=values)).to_csv(OUT/f'probe6_{seed}_{fold}_{co}.tsv',sep='\t',index=False)
   pick=int(np.argmax([z['validation_r'] for z in contexts]));pd.DataFrame(dict(PATIENT=patient_list,prediction_SBS6=outputs[pick][key])).to_csv(OUT/f'probe6_selected_{seed}_{fold}_{co}.tsv',sep='\t',index=False)
  (OUT/f'probe6_{seed}_{fold}.json').write_text(json.dumps(dict(seed=seed,fold=fold,scales=contexts,objective='Original continuous SBS6 exposures; scalar ridge readout of frozen target-specific256-dimensional representation; TCGA inner r selection; fixed1:1 bag-scale mean; no MSI or CPTAC-outcome fitting'),indent=2));print(seed,fold,contexts,flush=True)
 pd.DataFrame(audit).to_csv(OUT/f'probe6_{seed}_selection.tsv',sep='\t',index=False)
def aggregate():
 """Fixed averaging of generated contexts; no further fitting or selection."""
 for co,expected,repeats in [('tcga',422,3),('cptac',82,15)]:
  parts=[]
  for seed in range(3):
   for fold in range(5):
    frame=pd.read_csv(OUT/f'probe6_{seed}_{fold}_{co}.tsv',sep='\t')
    assert not frame.PATIENT.duplicated().any()
    parts.append(frame)
  combined=pd.concat(parts,ignore_index=True)
  assert combined.groupby('PATIENT').size().eq(repeats).all()
  result=combined.groupby('PATIENT',as_index=False).prediction_SBS6.mean()
  assert len(result)==expected
  path=OUT/f'probe6_{co}_scores.tsv'
  if path.exists():
   old=pd.read_csv(path,sep='\t').set_index('PATIENT').reindex(result.PATIENT)
   np.testing.assert_allclose(result.prediction_SBS6,old.prediction_SBS6,atol=1e-12)
  result.to_csv(path,sep='\t',index=False)
  print(co,'fixed context average',len(result),flush=True)

if __name__=='__main__':
 if sys.argv[1]=='aggregate':aggregate()
 else:run(int(sys.argv[1]))

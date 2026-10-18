"""Actual primary-model attention, replayed onto original input-tile coordinates.

Illustrative cases: median positive exposure, without selecting by prediction.
TCGA: seed-0 held-out fold. CPTAC: fixed seed-0/fold-0. One original 1024-tile
MIL component is displayed; this is not attribution of the full ensemble.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'rev';sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch,h5py
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import build_full210_equal_model_average_20260904 as primary
import run_full210_exact25_multiscale_fusion_20260904 as training

Image.MAX_IMAGE_PIXELS=None  # Known local study mosaics, not untrusted uploads.
PRE=Path('/home/end2end-WSI-preprocessing-1.0.0-preprocessing')

def raw_path(co,fn):
    for cache in (['cache_testa','cache_test','cache_testa_test'] if co=='TCGA' else ['cache_cptac']):
        p=PRE/cache/fn/'slide.jpg'
        if p.exists():return p
    raise FileNotFoundError(fn)

def aligned_coordinates(builder,co,fn):
    with h5py.File(builder.RETCCL/co/(fn+'.h5')) as r,h5py.File(builder.ctranspath_path(co,fn)) as c:
        rc=r['coords'][:];cc=c['coords'][:]
        ra=r['augmented'][:] if 'augmented' in r else np.zeros(len(rc),bool)
        ca=c['augmented'][:] if 'augmented' in c else np.zeros(len(cc),bool)
    assert not ra.any() and not ca.any(),'Augmented tiles need separate geometry handling'
    if np.array_equal(rc,cc):return rc
    keys=set(builder.keys(rc,ra))
    return cc[[k in keys for k in builder.keys(cc,ca)]]

def main():
    torch.set_num_threads(1)
    builder=primary.import_file(ROOT/'legacy30_retccl_ctranspath_feature_fusion_20260829/build_fused_bags.py','attention_bags')
    mod=training.implementation('1024');builder.BAG_SIZE=1024;builder.BASE_SEED=20260907
    selection=pd.read_csv(primary.SELECTION,sep='\t')
    scores=pd.read_csv(OUT/'patient_scores.tsv',sep='\t')
    annotations=pd.read_csv(OUT/'slide_manifest.tsv',sep='\t')
    folds=pd.read_csv(OUT/'fold_assignments.tsv',sep='\t')
    cases=[];all_tiles=[]
    pdf_path=OUT/'Attention.pdf'
    with PdfPages(pdf_path) as pdf:
      for cancer,sig in [('UCEC','SBS13'),('UCEC','SBS3'),('BRCA','SBS3'),('LUAD','SBS4')]:
       for co in ['TCGA','CPTAC']:
        eligible=annotations[(annotations.cohort==co)&(annotations.cancer==cancer)&annotations.specimen_type.isin(['tumor_tissue','Primary tumour (sample type 01)'])].PATIENT.unique()
        pos=scores[(scores.cohort==co)&(scores.cancer==cancer)&(scores[sig]>0)&scores.PATIENT.isin(eligible)].sort_values([sig,'PATIENT'])
        case=pos.iloc[(len(pos)-1)//2];patient=case.PATIENT
        fold=int(folds[(folds.cancer==cancer)&(folds.seed_index==0)&(folds.role=='test')&(folds.PATIENT==patient)].fold.iloc[0]) if co=='TCGA' else 0
        row=selection[(selection.cancer==cancer)&(selection.seed_index==0)&(selection.fold==fold)&(selection.scale==1024)].iloc[0]
        mode,hybrid=mod.variant_parts(row.selected_variant);directory=mod.output_dir(row.selected_variant,row.cancer_dir,0,fold)
        sha=primary.sha256(directory/'best.pt');assert sha==row.checkpoint_sha256
        checkpoint=torch.load(directory/'best.pt',map_location='cpu')
        if co=='TCGA':assert patient in checkpoint['test_patients'] and patient not in checkpoint['train_patients'] and patient not in checkpoint['validation_patients']
        slides=pd.read_csv(builder.SOURCE/co/cancer/'slides.tsv',sep='\t');slides=slides[slides.PATIENT==patient]
        fs=[];qs=[];coords=[];names=[]
        for sr in slides.itertuples():
            f,q,_,_=builder.read_slide(co,sr.FILENAME);xy=aligned_coordinates(builder,co,sr.FILENAME)
            assert len(f)==len(xy)
            fs.append(f);qs.append(q);coords.append(xy);names.extend([sr.FILENAME]*len(f))
        features=np.concatenate(fs);qc=np.concatenate(qs);xy=np.concatenate(coords);names=np.asarray(names)
        rng=np.random.default_rng(builder.stable_seed(co,cancer,patient))
        selected={'uniform':builder.select_uniform(len(features),rng),'qhybrid':builder.select_qhybrid(qc,rng)}[mode]
        meta=pd.read_csv(mod.cache_prefix(co,row.cancer_dir,mode).with_suffix('.meta.csv'));idx=int(np.flatnonzero(meta.PATIENT.to_numpy()==patient)[0])
        bag=np.load(mod.cache_prefix(co,row.cancer_dir,mode).with_suffix('.bags.npy'),mmap_mode='r')[idx:idx+1].copy()
        np.testing.assert_array_equal(features[selected],bag[0,:len(selected)])
        model=mod.TargetAttentionMIL(hybrid).cuda().eval();model.load_state_dict(checkpoint['state_dict'],strict=True)
        captured=[];hook=model.attention_w.register_forward_hook(lambda module,inputs,output:captured.append(output.detach()))
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            comp,_=model(torch.tensor(bag,device='cuda'),torch.tensor([len(selected)],device='cuda'))
            logits=captured[0];valid=torch.arange(1024,device='cuda')[None,:]<len(selected)
            weights=torch.softmax(torch.where(valid[:,:,None],logits,torch.full_like(logits,-1e4)),dim=1)[0,:len(selected),int(sig[3:])-1].float().cpu().numpy()
        hook.remove();assert abs(float(weights.sum())-1)<.01 and np.all(weights>=0)
        saved=pd.read_csv(directory/('tcga_test_predictions.tsv' if co=='TCGA' else 'cptac_label_free_predictions.tsv'),sep='\t').set_index('PATIENT').loc[patient,'composition_'+sig]
        np.testing.assert_allclose(float(comp[0,int(sig[3:])-1]),saved,atol=1e-4,rtol=.03)
        t=pd.DataFrame(dict(cohort=co,cancer=cancer,signature=sig,PATIENT=patient,FILENAME=names[selected],x=xy[selected,0],y=xy[selected,1],attention=weights,qc=qc[selected]))
        t=t.merge(annotations[['cohort','PATIENT','FILENAME','specimen_type','embedding']],on=['cohort','PATIENT','FILENAME'],validate='many_to_one')
        tumour=t[t.specimen_type.isin(['tumor_tissue','Primary tumour (sample type 01)'])]
        fn=tumour.groupby('FILENAME').size().sort_values(ascending=False).index[0]
        show=t[t.FILENAME==fn].copy();path=raw_path(co,fn)
        with Image.open(path) as im:
            size=im.size
            assert (show.x+224<=size[0]).all() and (show.y+224<=size[1]).all() and (show[['x','y']]>=0).all().all()
            thumb=im.copy();thumb.thumbnail((1200,850));thumb=thumb.convert('RGB')
            ordering=show.sort_values(['attention','x','y'])
            ranked=pd.concat([ordering.iloc[-3:].iloc[::-1].assign(group='High attention'),ordering.iloc[:3].assign(group='Low attention')])
            crops=[im.crop((int(r.x),int(r.y),int(r.x)+224,int(r.y)+224)).convert('RGB') for r in ranked.itertuples()]
        layer=Image.new('RGBA',thumb.size,(0,0,0,0));draw=ImageDraw.Draw(layer)
        # Display scaling only: each sampled tile's within-patient attention rank.
        ranks=t.attention.rank(method='average',pct=True);show['attention_percentile']=ranks.loc[show.index]
        sx=thumb.width/size[0];sy=thumb.height/size[1];cmap=plt.get_cmap('inferno')
        for r in show.itertuples():
            colour=tuple(int(255*v) for v in cmap(r.attention_percentile)[:3])+(145,)
            draw.rectangle((int(r.x*sx),int(r.y*sy),max(int((r.x+224)*sx),int(r.x*sx)+1),max(int((r.y+224)*sy),int(r.y*sy)+1)),fill=colour)
        overlay=Image.alpha_composite(thumb.convert('RGBA'),layer).convert('RGB')
        fig=plt.figure(figsize=(11.7,8.3));gs=fig.add_gridspec(3,6,height_ratios=[2.5,1,1],left=.035,right=.965,top=.84,bottom=.12,hspace=.5,wspace=.3)
        for block,img,title in [(gs[0,:3],thumb,'Original resampled H&E'),(gs[0,3:],overlay,'Attention overlay: sampled tiles only')]:
            ax=fig.add_subplot(block);ax.imshow(img);ax.set_title(title,fontsize=12);ax.axis('off')
        for k,(r,crop) in enumerate(zip(ranked.itertuples(),crops)):
            ax=fig.add_subplot(gs[1+k//3,2*(k%3):2*(k%3)+2]);ax.imshow(crop);ax.axis('off')
            ax.set_title(f'{r.group} {k%3+1} | weight {r.attention:.4g}\n({int(r.x)}, {int(r.y)})',fontsize=9)
            ax.plot([10,80],[210,210],color='black',lw=3);ax.text(10,195,'80 µm',fontsize=8,color='black',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        tier='joint FDR-supported target' if cancer=='UCEC' else 'nominal cross-cohort target'
        fig.suptitle(f'Figure S9 | {co} {cancer}–{sig} | {patient}\n{tier}; observed exposure {case[sig]:.3f}',fontsize=15,y=.98)
        fig.text(.04,.89,fn,fontsize=9)
        colour_ax=fig.add_axes([.70,.875,.24,.012])
        colour_bar=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap=cmap),cax=colour_ax,orientation='horizontal',ticks=[0,.5,1])
        colour_bar.ax.set_xticklabels(['0','50','100']);colour_bar.ax.tick_params(labelsize=8,pad=1)
        colour_bar.set_label('Within-patient attention percentile',fontsize=8,labelpad=1)
        fig.text(.04,.045,f'Original selected 1,024-tile component; seed 0, fold {fold}; {mode}; {len(show)}/{len(t)} sampled tiles on this slide.\nColour: within-patient attention percentile; unselected regions remain uncoloured. Each crop spans 256 × 256 µm.\nIllustrative cases selected by median positive exposure; no expert pathology labels or complete-ensemble attribution.',fontsize=8.5)
        pdf.savefig(fig);fig.savefig(OUT/f'attention_{co.lower()}_{cancer.lower()}_{sig.lower()}.png',dpi=150);plt.close(fig)
        all_tiles.append(t)
        cases.append(dict(cohort=co,cancer=cancer,signature=sig,PATIENT=patient,observed_exposure=float(case[sig]),primary_ensemble_score=float(case['prediction_'+sig]),seed_index=0,fold=fold,scale=1024,variant=row.selected_variant,checkpoint_sha256=sha,raw_image=str(path),displayed_slide=fn,displayed_specimen_type=show.specimen_type.iloc[0],sampled_patient_tiles=len(t),displayed_slide_tiles=len(show),normal_attention_mass=float(t.loc[t.specimen_type=='normal_tissue','attention'].sum()),unspecified_attention_mass=float(t.loc[~t.specimen_type.isin(['normal_tissue','tumor_tissue','Primary tumour (sample type 01)']),'attention'].sum()),selection='Lower-median positive exposure among tumour-labelled eligible patients; no selection by image score or appearance',bag_replay_exact=True,held_out_tcga=co=='TCGA',component_prediction_replayed=True,expert_review=False))
        print(co,cancer,sig,patient,fn,'sampled',len(show),flush=True)
        del model;torch.cuda.empty_cache()
    pd.DataFrame(cases).to_csv(OUT/'attention_cases.tsv',sep='\t',index=False)
    pd.concat(all_tiles,ignore_index=True).to_csv(OUT/'attention_tiles.tsv',sep='\t',index=False)
    (OUT/'attention_protocol.json').write_text(json.dumps(dict(targets=['UCEC-SBS13','UCEC-SBS3','BRCA-SBS3','LUAD-SBS4'],patients=8,component='Original selected 1024-tile MIL; seed0; TCGA held-out fold / CPTAC fixed fold0',selection_rule=cases[0]['selection'],display='Actual sampled coordinates, no spatial interpolation; raw resampled H&E; within-patient attention percentile',ensemble_attribution=False,expert_review=False,bag_replay=True),indent=2))

if __name__=='__main__':main()

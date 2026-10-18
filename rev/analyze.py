"""Auditable major-revision statistics from the submitted primary predictions."""
import sys,json,hashlib,re
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import pearsonr,spearmanr,mannwhitneyu
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'rev'
sys.path.insert(0,str(ROOT))
import build_full210_equal_model_average_20260904 as primary
SBS=list(primary.SIGNATURES);CANCERS=list(primary.CANCERS)
LOCAL=dict(BRCA='Brca',HNSC='Hnscc',KIRC='Ccrcc',LUAD='Luad',LUSC='Lscc',OV='Ov',UCEC='Ucec')

def bh(p):
    p=np.asarray(p,float);finite=np.isfinite(p);v=np.where(finite,p,1)
    order=np.argsort(v);q=np.empty(len(v));q[order]=np.minimum(1,np.minimum.accumulate((v[order]*len(v)/np.arange(1,len(v)+1))[::-1])[::-1]);q[~finite]=np.nan
    return q

def metrics(y,s):
    y=np.asarray(y,float);s=np.asarray(s,float)
    if len(y)<3 or np.ptp(y)==0 or np.ptp(s)==0:return dict(n=len(y),r=np.nan,p=np.nan,ci_low=np.nan,ci_high=np.nan)
    r,p=pearsonr(y,s);z=np.arctanh(np.clip(r,-1+1e-15,1-1e-15));ci=np.tanh(z+np.array([-1,1])*1.95996398454/np.sqrt(len(y)-3)) if len(y)>3 else np.array([np.nan,np.nan])
    return dict(n=len(y),r=float(r),p=float(p/2 if r>=0 else 1-p/2),ci_low=ci[0],ci_high=ci[1])

def binary(y,s,seed=20261013):
    y=np.asarray(y,int);s=np.asarray(s,float);pos=s[y==1];neg=s[y==0]
    if not len(pos) or not len(neg):return dict(n=len(y),positive_n=len(pos),auc=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan)
    # Stratified patient bootstrap; rank comparison is exactly ROC AUC.
    wins=(pos[:,None]>neg[None,:]).astype(float)+.5*(pos[:,None]==neg[None,:])
    rng=np.random.default_rng(seed);vals=[]
    for _ in range(10000):
        ip=rng.integers(len(pos),size=len(pos));inn=rng.integers(len(neg),size=len(neg));vals.append(wins[np.ix_(ip,inn)].mean())
    lo,hi=np.quantile(vals,[.025,.975]);a=roc_auc_score(y,s)
    return dict(n=len(y),positive_n=len(pos),auc=float(a),ci_low=float(lo),ci_high=float(hi),p=float(mannwhitneyu(pos,neg,alternative='two-sided').pvalue))

def main():
    original=pd.read_csv(primary.RESULTS/'all_420_cohort_metrics.tsv',sep='\t')
    long=original.copy()
    tables={};distributions=[];samples=[];data={};input_hashes={}
    for co in ['TCGA','CPTAC']:
        mask=long.dataset.eq(co);long.loc[mask,'q_bh210']=bh(long.loc[mask,'pearson_p_positive_direction'])
    for cancer in CANCERS:
        for co in ['TCGA','CPTAC']:
            path=primary.FROZEN/f'{cancer.lower()}_{"tcga_oof" if co=="TCGA" else "cptac_label_free"}.tsv'
            df=pd.read_csv(path,sep='\t',dtype={'PATIENT':str});input_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            if co=='CPTAC':
                lab=pd.read_excel(ROOT/f'process_endpoint_inputs_legacy30_min50/CPTAC/{LOCAL[cancer]}/CPTAC_{LOCAL[cancer]}_process_endpoints.xlsx')
                df=df.merge(lab[['PATIENT',*SBS]],on='PATIENT',validate='one_to_one')
            data[co,cancer]=df
            for sig in SBS:
                m=metrics(df[sig],df['prediction_'+sig]);row=long[(long.cancer==cancer)&(long.signature==sig)&(long.dataset==co)].index[0]
                assert m['n']==long.loc[row,'n'];np.testing.assert_allclose(m['r'],long.loc[row,'pearson_r'],atol=1e-12,equal_nan=True)
                for k in ['ci_low','ci_high']:long.loc[row,k]=m[k]
                for kind,col in [('Exposure',sig),('Score','prediction_'+sig)]:
                    v=df[col];distributions.append(dict(cohort=co,cancer=cancer,signature=sig,variable=kind,n=len(v),nonzero=int((v!=0).sum()),zero_fraction=float((v==0).mean()),minimum=v.min(),q1=v.quantile(.25),median=v.median(),q3=v.quantile(.75),maximum=v.max(),mean=v.mean(),sd=v.std(ddof=0)))
            samples.append(df.assign(cohort=co,cancer=cancer))
    assert sum(len(data['TCGA',c]) for c in CANCERS)==2741 and sum(len(data['CPTAC',c]) for c in CANCERS)==580
    wide=long.pivot(index=['cancer','signature'],columns='dataset',values=['n','nonzero','pearson_r','ci_low','ci_high','pearson_p_positive_direction','q_bh210'])
    wide.columns=['_'.join(c) for c in wide.columns];wide=wide.reset_index()
    wide['p_both_positive']=np.maximum(wide.pearson_p_positive_direction_TCGA,wide.pearson_p_positive_direction_CPTAC)
    wide['q_both_positive_bh210']=bh(wide.p_both_positive)
    both=wide.pearson_r_TCGA.notna()&wide.pearson_r_CPTAC.notna()
    wide['direction']=np.select([~both,(wide.pearson_r_TCGA>0)&(wide.pearson_r_CPTAC>0),(wide.pearson_r_TCGA<0)&(wide.pearson_r_CPTAC<0)],['Non-estimable','Positive-positive','Negative-negative'],default='Discordant')
    wide['nominal_positive_both']=wide.p_both_positive.lt(.05)
    wide['bh_positive_both']=wide.q_both_positive_bh210.lt(.05)
    summary=dict(pairs=210,estimable_both=int(both.sum()),directions=wide.direction.value_counts().to_dict(),nominal_both=int(wide.nominal_positive_both.sum()),bh_both=int(wide.bh_positive_both.sum()),tcga_bh=int(long.query('dataset == "TCGA"').q_bh210.lt(.05).sum()),cptac_bh=int(long.query('dataset == "CPTAC"').q_bh210.lt(.05).sum()),joint_q_min=float(wide.q_both_positive_bh210.min()),joint_q_max_nominal=float(wide.loc[wide.nominal_positive_both,'q_both_positive_bh210'].max()))
    long.to_csv(OUT/'cohort_metrics.tsv',sep='\t',index=False);wide.to_csv(OUT/'paired_metrics.tsv',sep='\t',index=False)
    tables['cohort_metrics']=long;tables['paired_metrics']=wide;tables['distributions']=pd.DataFrame(distributions);tables['patient_scores']=pd.concat(samples,ignore_index=True)
    counts=[]
    for co in ['TCGA','CPTAC']:
        for c in CANCERS:
            base=f'tcga_sigprof_paper_sweep/by_config/legacy30_min50/{LOCAL[c]}/TCGA_{c}' if co=='TCGA' else f'cptac_sigprof_legacy30_min50/by_cancer/{LOCAL[c]}/CPTAC_{LOCAL[c]}'
            ct=pd.read_csv(ROOT/(base+'_legacy30_min50_sample_snv_count.csv'));ct['cohort']=co;ct['cancer']=c;ct['eligible_sample']=ct.snv_count.gt(50);ct['included_patient']=ct.PATIENT.isin(data[co,c].PATIENT);counts.append(ct)
    tables['mutation_counts']=pd.concat(counts,ignore_index=True)
    mut=[]
    for (co,c),d in tables['mutation_counts'].groupby(['cohort','cancer']):
        matched=d[d.eligible_sample & d.included_patient].snv_count
        mut.append(dict(cohort=co,cancer=c,source_samples=len(d),source_patients=d.PATIENT.nunique(),samples_over50=int(d.eligible_sample.sum()),included_patients=data[co,c].PATIENT.nunique(),included_eligible_samples=len(matched),minimum=int(matched.min()),median=float(matched.median()),q1=float(matched.quantile(.25)),q3=float(matched.quantile(.75)),maximum=int(matched.max())))
    tables['mutation_summary']=pd.DataFrame(mut)
    for name in ['patient_flow','eligibility_audit','slide_manifest','slide_summary','cohort_summary']:
        tables[name]=pd.read_csv(ROOT/f'rev/{name}.tsv',sep='\t')
    # Context-preserving slide restriction, paired to identical available patients.
    sensitivity=[];restricted_scores=[]
    for c in CANCERS:
        new=pd.read_csv(OUT/f'{c.lower()}_tumour_scores.tsv',sep='\t');old=data['CPTAC',c]
        pair=old.merge(new,on='PATIENT',suffixes=('_all','_tumour'),validate='one_to_one')
        restricted_scores.append(new.assign(cancer=c,cohort='CPTAC'))
        for s in SBS:
            a=metrics(pair[s],pair['prediction_'+s+'_all']);t=metrics(pair[s],pair['prediction_'+s+'_tumour'])
            sensitivity.append(dict(cancer=c,signature=s,**{'all_'+k:v for k,v in a.items()},**{'tumour_'+k:v for k,v in t.items()},delta_r=t['r']-a['r'],score_correlation=float(pearsonr(pair['prediction_'+s+'_all'],pair['prediction_'+s+'_tumour']).statistic)))
    tables['tumour_sensitivity']=pd.DataFrame(sensitivity)
    ts=tables['tumour_sensitivity']
    ts['tumour_q_bh210']=bh(ts.tumour_p)
    tcga_p=wide.set_index(['cancer','signature']).pearson_p_positive_direction_TCGA
    ts['p_both_positive']=np.maximum(ts.tumour_p,[tcga_p.loc[(c,s)] for c,s in zip(ts.cancer,ts.signature)])
    ts['q_both_positive_bh210']=bh(ts.p_both_positive)
    tables['tumour_scores']=pd.concat(restricted_scores,ignore_index=True)
    tables['input_counts']=pd.concat([pd.read_csv(OUT/f'{c.lower()}_input_counts.tsv',sep='\t') for c in CANCERS],ignore_index=True)
    tables['fold_assignments']=pd.concat([pd.read_csv(OUT/f'{c.lower()}_folds.tsv',sep='\t') for c in CANCERS],ignore_index=True)
    subset=[];flags=[]
    for (co,c,p),d in tables['slide_manifest'].groupby(['cohort','cancer','PATIENT']):
        x=dict(cohort=co,cancer=c,PATIENT=p,has_normal=bool(d.specimen_type.eq('normal_tissue').any()),has_tumour=bool(d.specimen_type.isin(['tumor_tissue','Primary tumour (sample type 01)']).any()),has_ffpe=bool(d.embedding.isin(['FFPE','FFPE diagnostic']).any()),has_frozen=bool(d.embedding.isin(['OCT','Frozen tissue section']).any()))
        x['embedding_group']='Mixed' if x['has_ffpe'] and x['has_frozen'] else 'FFPE only' if x['has_ffpe'] else 'Frozen/OCT only' if x['has_frozen'] else 'Unspecified';flags.append(x)
    flags=pd.DataFrame(flags);tables['patient_input_groups']=flags
    for (co,c),f in flags.groupby(['cohort','cancer']):
        base=data[co,c].merge(f,on='PATIENT',validate='one_to_one')
        for group,dd in base.groupby('embedding_group'):
            for s in SBS:subset.append(dict(cohort=co,cancer=c,signature=s,stratum=group,**metrics(dd[s],dd['prediction_'+s])))
        if co=='CPTAC':
            for flag in [True,False]:
                dd=base[base.has_normal.eq(flag)]
                for s in SBS:subset.append(dict(cohort=co,cancer=c,signature=s,stratum='Normal slide present' if flag else 'No normal-labelled slide',**metrics(dd[s],dd['prediction_'+s])))
    tables['specimen_strata']=pd.DataFrame(subset)
    clinical=[];clinical_scores=[]
    u=data['TCGA','UCEC'];ext=data['CPTAC','UCEC']
    directory=ROOT/'external_hrd_sources/msi_mmr_published_high_impact_20260604'
    raw=json.loads((directory/'ucec_tcga_pan_can_atlas_2018_sample_clinical_data.json').read_text())
    if isinstance(raw,dict):raw=raw.get('data',raw.get('clinicalData',[]))
    mc=pd.DataFrame(raw);mm=mc[mc.clinicalAttributeId.eq('MSI_SCORE_MANTIS')].copy();mm['PATIENT']=mm.patientId;mm['mantis']=pd.to_numeric(mm.value,errors='coerce')
    assert mm.PATIENT.is_unique
    mc=u.merge(mm[['PATIENT','mantis']],on='PATIENT',validate='one_to_one').dropna(subset=['mantis'])
    for rule in ['Original gray-zone exclusion','Published 0.4 threshold']:
        d=mc[(mc.mantis<.4)|(mc.mantis>.6)].copy() if rule.startswith('Original') else mc.copy()
        d['label']=d.mantis.gt(.6 if rule.startswith('Original') else .4).astype(int)
        d['score']=d.prediction_SBS6;clinical.append(dict(cohort='TCGA',endpoint='MSI',rule=rule,signature='SBS6',**binary(d.label,d.score)))
        clinical_scores.append(d[['PATIENT','score','label','mantis']].assign(cohort='TCGA',endpoint='MSI',rule=rule))
    labels=pd.read_csv(directory/'cptac_ucec_cell2020_msi_status_published_labels.csv')
    d=ext.merge(labels,on='PATIENT',validate='one_to_one');assert len(d)==len(ext) and d.msi_binary.isin([0,1]).all()
    d['score']=d.prediction_SBS6;d['label']=d.msi_binary
    clinical.append(dict(cohort='CPTAC',endpoint='MSI',rule='Source-published MSI-H versus MSS',signature='SBS6',**binary(d.label,d.score)))
    clinical_scores.append(d[['PATIENT','score','label','MSI_STATUS']].assign(cohort='CPTAC',endpoint='MSI',rule='Source-published MSI-H versus MSS'))
    rules=pd.read_excel(ROOT/'can/Supplementary_Data_S2.xlsx',sheet_name='polymerase_rules',header=3)
    allowed={g:set(d.iloc[:,1].astype(str).str.replace('p.','',regex=False)) for g,d in rules.groupby(rules.columns[0])}
    # Use the complete sequenced profile, not the narrower all-sample list.
    audit=json.loads((OUT/'cptac_endpoint_source.json').read_text())
    raw=audit['direct_api_full_sequenced_mutation_rows'];records=[]
    for patient in ext.PATIENT:
        row=dict(PATIENT=patient)
        for gene in ['POLE','POLD1']:
            events=[e for e in raw[gene] if e['patientId']==patient]
            row[gene.lower()+'_positive']=int(any(e['proteinChange'].replace('p.','') in allowed[gene] and e['mutationType']=='Missense_Mutation' for e in events))
            row[gene+'_EVENT_DETAILS']=';'.join(e['proteinChange']+'|'+e['mutationType'] for e in events) or np.nan
        row['label']=max(row['pole_positive'],row['pold1_positive']);records.append(row)
    source=pd.DataFrame(records);assert len(source)==82 and source.label.sum()==6
    d=ext.merge(source[['PATIENT','label','pole_positive','pold1_positive','POLE_EVENT_DETAILS','POLD1_EVENT_DETAILS']],on='PATIENT',validate='one_to_one');assert len(d)==len(ext)
    d['score']=d.prediction_SBS10
    clinical.append(dict(cohort='CPTAC',endpoint='POLE/POLD1',rule='Same exact proofreading-domain missense list',signature='SBS10',**binary(d.label,d.score)))
    clinical_scores.append(d[['PATIENT','score','label','pole_positive','pold1_positive','POLE_EVENT_DETAILS','POLD1_EVENT_DETAILS']].assign(cohort='CPTAC',endpoint='POLE/POLD1',rule='Same exact proofreading-domain missense list'))
    cl=pd.DataFrame(clinical);p=cl.loc[cl.cohort.eq('CPTAC'),'p'].to_numpy();order=np.argsort(p);adjusted=np.empty(2);adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.array([2,1])));cl.loc[cl.cohort.eq('CPTAC'),'p_holm_two_endpoints']=adjusted
    tables['clinical_revision']=cl;tables['clinical_revision_scores']=pd.concat(clinical_scores,ignore_index=True)
    tables['polymerase_rules']=rules
    tables['analysis_notes']=pd.DataFrame([
        ('FDR','BH separately over 210 planned pairs in each cohort; non-estimable tests enter adjustment as P=1 but their published P/q stay missing. Positive-direction P-values retain the submitted test direction.'),
        ('Two-cohort test','Full conjunction P=max(P_TCGA,P_CPTAC); BH over all 210 conjunction tests. This distinguishes nominal recurrence from multiplicity-adjusted cross-cohort evidence.'),
        ('Tumour restriction','CPTAC only: remove all non-tumour-labelled slides, retain original eligible targets. Original MIL checkpoints, variants, seeds, component scaling and fixed thirds remain; deterministic original PCA-ridge replay is checked against stored all-slide predictions.'),
        ('Specimen strata','Descriptive fixed-score patient subgroups, not FFPE-only/frozen-only re-inference. Mixed patients appear only in the mixed stratum. No confounder-adjusted causal effect is claimed.'),
        ('MANTIS','Original <0.4/>0.6 with intermediate exclusion compared with >0.4 positive and <=0.4 negative, all available continuous labels. No score/model changes or optimized cutoff.'),
        ('External endpoints','CPTAC source MSI-H versus MSS; source mutation-event annotations filtered to the same exact missense list. No outcome fitting. Genomic-scar HRD >=42 not available in the checked matched clinical sources; BRCA1/2 mutation flags are not substituted.'),
        ('Unperformed','Alternative deconvolution/contemporary signature reanalysis; attention overlays/expert pathology review; complete public repository release are not claimed completed.'),
        ('Versions','Current revision runtime versions are supplied; these are not represented as a complete archived original-training environment.'),
    ],columns=['item','definition'])
    for key in ['numpy','pandas','scipy','torch','sklearn','matplotlib']:
        module=__import__(key);tables.setdefault('runtime_versions',[]).append(dict(package=key,current_revision_version=module.__version__))
    tables['runtime_versions']=pd.DataFrame(tables['runtime_versions'])
    for name,df in tables.items():df.to_csv(OUT/f'{name}.tsv',sep='\t',index=False)
    with pd.ExcelWriter(OUT/'Revision_Data.xlsx',engine='xlsxwriter') as writer:
        for name,df in tables.items():
            df.to_excel(writer,sheet_name=name[:31],index=False);ws=writer.sheets[name[:31]];ws.freeze_panes(1,0);ws.autofilter(0,0,len(df),len(df.columns)-1);ws.set_column(0,len(df.columns)-1,18)
    summary['clinical']=json.loads(cl.to_json(orient='records'));summary['tumour_patients']=len(tables['tumour_scores']);summary['excluded_no_tumour_label']=580-summary['tumour_patients'];summary['baseline_verified']=True
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'input_hashes.json').write_text(json.dumps(input_hashes,indent=2));plots(tables,wide,summary,data)
    print(json.dumps(summary,indent=2))

def save(fig,name,pdf):
    fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight',facecolor='white');
    if name!='nominal_forest':pdf.savefig(fig,bbox_inches='tight')
    plt.close(fig)

def plots(tables,wide,summary,data):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    with PdfPages(OUT/'Revision_Figures.pdf') as pdf:
        fig,axes=plt.subplots(2,1,figsize=(13,7),constrained_layout=True)
        for ax,co in zip(axes,['TCGA','CPTAC']):
            matrix=wide.pivot(index='cancer',columns='signature',values='pearson_r_'+co).reindex(index=CANCERS,columns=SBS)
            cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#cccccc')
            im=ax.imshow(matrix,vmin=-.5,vmax=.5,cmap=cmap,aspect='auto');ax.set_yticks(range(7),CANCERS);ax.set_xticks(range(30),[str(i) for i in range(1,31)]);ax.set_xlabel('SBS signature');ax.set_title(co+' — all 210 planned pairs (grey: non-estimable)',loc='left')
            for i,c in enumerate(CANCERS):
                for j,s in enumerate(SBS):
                    r=wide[(wide.cancer==c)&(wide.signature==s)].iloc[0]
                    if r['q_bh210_'+co]<.05:ax.text(j,i,'*',ha='center',va='center',fontsize=13)
            fig.colorbar(im,ax=ax,label='Pearson r',shrink=.7)
        fig.suptitle('Figure S3. Complete signature landscape; * = cohort BH q < 0.05');save(fig,'Figure_5',pdf)
        chosen=wide[wide.nominal_positive_both].copy();chosen['rank']=chosen.signature.str[3:].astype(int);chosen=chosen.sort_values(['cancer','rank']);fig,axes=plt.subplots(1,2,figsize=(11,6),sharey=True,constrained_layout=True)
        for ax,co in zip(axes,['TCGA','CPTAC']):
            y=np.arange(len(chosen));r=chosen['pearson_r_'+co];lo=chosen['ci_low_'+co];hi=chosen['ci_high_'+co]
            ax.errorbar(r,y,xerr=[r-lo,hi-r],fmt='o',color='#187c91',capsize=3);ax.axvline(0,color='#999999',ls='--');ax.set_yticks(y,chosen.cancer+'–'+chosen.signature);ax.set_title(co);ax.set_xlabel('Pearson r (95% Fisher-z CI)');ax.invert_yaxis()
        fig.suptitle(f"14 nominal positive recurrences; {summary['bh_both']} pass joint BH q < 0.05");save(fig,'nominal_forest',pdf)
        for c in CANCERS:
            d=wide[wide.cancer.eq(c)].set_index('signature').reindex(SBS).reset_index()
            fig,axes=plt.subplots(1,2,figsize=(12,10),sharey=True,constrained_layout=True)
            for ax,co in zip(axes,['TCGA','CPTAC']):
                y=np.arange(30);r=d['pearson_r_'+co];lo=d['ci_low_'+co];hi=d['ci_high_'+co]
                ax.errorbar(r,y,xerr=[r-lo,hi-r],fmt='o',color='#187c91',capsize=2,ms=4)
                ax.axvline(0,color='#aaaaaa',ls='--');ax.set_yticks(y,SBS);ax.set_title(co);ax.set_xlim(-1,1);ax.set_xlabel('Pearson r (95% Fisher-z CI)')
                for j in range(30):
                    if pd.isna(r.iloc[j]):ax.text(.02,j,'NE',fontsize=8,color='#777777')
            axes[0].invert_yaxis();fig.suptitle('Figure S4. Complete effects and confidence intervals — '+c);save(fig,'forest_'+c.lower(),pdf)
        for co in ['TCGA','CPTAC']:
            for variable in ['Exposure','Score']:
                fig,axes=plt.subplots(7,1,figsize=(14,15),constrained_layout=True)
                for ax,c in zip(axes,CANCERS):
                    df=data[co,c];vals=[df[s if variable=='Exposure' else 'prediction_'+s] for s in SBS]
                    ax.boxplot(vals,whis=(5,95),showfliers=False,patch_artist=True,boxprops={'facecolor':'#b9d8e3'},medianprops={'color':'#233646'},widths=.55);ax.set_title(c+' (n='+str(len(df))+')',loc='left');ax.set_xticks(range(1,31),range(1,31));ax.set_ylabel(variable)
                fig.suptitle('Figure S5. '+co+' '+variable.lower()+' distributions: all 30 signatures; whiskers 5th–95th percentiles');save(fig,co.lower()+'_'+variable.lower()+'_distributions',pdf)
        fig,axes=plt.subplots(2,1,figsize=(13,6),constrained_layout=True)
        for ax,co in zip(axes,['TCGA','CPTAC']):
            df=tables['distributions'];m=df[(df.cohort==co)&(df.variable=='Exposure')].pivot(index='cancer',columns='signature',values='zero_fraction').reindex(index=CANCERS,columns=SBS)
            im=ax.imshow(m,vmin=0,vmax=1,cmap='YlOrBr',aspect='auto');ax.set_yticks(range(7),CANCERS);ax.set_xticks(range(30),range(1,31));ax.set_title(co,loc='left');fig.colorbar(im,ax=ax,label='Zero-exposure fraction')
        fig.suptitle('Figure S6. Target sparsity after submitted signature fitting and cutoff');save(fig,'target_sparsity',pdf)
        ss=tables['tumour_sensitivity'].merge(chosen[['cancer','signature']],on=['cancer','signature']);ss['label']=ss.cancer+'–'+ss.signature;fig,ax=plt.subplots(figsize=(11,7),constrained_layout=True)
        y=np.arange(len(ss));ax.errorbar(ss.all_r,y-.13,xerr=[ss.all_r-ss.all_ci_low,ss.all_ci_high-ss.all_r],fmt='o',ms=5,capsize=2,label='All slides: matched subset',color='#86959d');ax.errorbar(ss.tumour_r,y+.13,xerr=[ss.tumour_r-ss.tumour_ci_low,ss.tumour_ci_high-ss.tumour_r],fmt='o',ms=5,capsize=2,label='Tumour-labelled slides',color='#187c91');ax.set_yticks(y,ss.label);ax.axvline(0,color='#999999',ls='--');ax.set_xlabel('CPTAC Pearson r (95% Fisher-z CI)');ax.set_title('Figure S7. Fixed-model input sensitivity; paired patients');ax.legend();save(fig,'Figure_6',pdf)
        cl=tables['clinical_revision'];fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True);y=np.arange(len(cl));ax.errorbar(cl.auc,y,xerr=[cl.auc-cl.ci_low,cl.ci_high-cl.auc],fmt='o',capsize=3,color='#187c91');ax.axvline(.5,ls='--',color='#999999');ax.set_yticks(y,[f'{r.cohort} {r.endpoint}\n{r.rule} (n={r.n}, positive={r.positive_n})' for r in cl.itertuples()]);ax.set_xlabel('AUC (95% stratified patient-bootstrap CI)');ax.set_xlim(.3,.95);ax.set_title('Figure S8. Exploratory genomic associations — unchanged signature scores');save(fig,'clinical_revision',pdf)

if __name__=='__main__':main()

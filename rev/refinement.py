"""Paired evaluation of targeted SBS6 readout; no further fitting/selection."""
import sys,json
from pathlib import Path
import pandas as pd,numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'rev';sys.path.insert(0,str(OUT))
from analyze import binary
labels=pd.read_csv(OUT/'clinical_revision_scores.tsv',sep='\t');rows=[];patients=[]
for co in ['TCGA','CPTAC']:
 d=pd.read_csv(OUT/f'probe6_{co.lower()}_scores.tsv',sep='\t').rename(columns={'prediction_SBS6':'refined_score'})
 original=pd.read_csv(ROOT/f'full210_equal_model_average_20260904/frozen_predictions/ucec_{"tcga_oof" if co=="TCGA" else "cptac_label_free"}.tsv',sep='\t')
 if co=='TCGA':genomic=original[['PATIENT','SBS6']]
 else:genomic=pd.read_excel(ROOT/'process_endpoint_inputs_legacy30_min50/CPTAC/Ucec/CPTAC_Ucec_process_endpoints.xlsx')[['PATIENT','SBS6']]
 for rule in (['Original gray-zone exclusion','Published 0.4 threshold'] if co=='TCGA' else ['Source-published MSI-H versus MSS']):
  a=labels[(labels.cohort==co)&(labels.endpoint=='MSI')&(labels.rule==rule)].merge(d,on='PATIENT',validate='one_to_one').merge(genomic,on='PATIENT',validate='one_to_one')
  for name,col in [('Original SBS6 score','score'),('Targeted SBS6 readout','refined_score'),('Observed genomic SBS6','SBS6')]:
   rows.append(dict(cohort=co,definition=rule,model=name,**binary(a.label,a[col])))
  pos=np.flatnonzero(a.label.to_numpy()==1);neg=np.flatnonzero(a.label.to_numpy()==0)
  def wins(v):
   return (v[pos,None]>v[None,neg]).astype(float)+.5*(v[pos,None]==v[None,neg])
  difference=wins(a.refined_score.to_numpy())-wins(a.score.to_numpy());rng=np.random.default_rng(20261013);boot=[]
  for _ in range(10000):boot.append(difference[np.ix_(rng.choice(len(pos),len(pos)),rng.choice(len(neg),len(neg)))].mean())
  lo,hi=np.quantile(boot,[.025,.975]);delta=roc_auc_score(a.label,a.refined_score)-roc_auc_score(a.label,a.score)
  rows.append(dict(cohort=co,definition=rule,model='Paired AUC difference: targeted minus original',n=len(a),positive_n=len(pos),auc=delta,ci_low=lo,ci_high=hi,p=np.nan))
  patients.append(a.assign(definition=rule));print(co,rule,'original',roc_auc_score(a.label,a.score),'targeted',roc_auc_score(a.label,a.refined_score),'difference',delta,'CI',lo,hi)
 g=original[['PATIENT','prediction_SBS6']].merge(d,on='PATIENT',validate='one_to_one').merge(genomic,on='PATIENT',validate='one_to_one')
 print(co,'SBS6 target r original/targeted',pearsonr(g.SBS6,g.prediction_SBS6).statistic,pearsonr(g.SBS6,g.refined_score).statistic)
pd.DataFrame(rows).to_csv(OUT/'refinement_metrics.tsv',sep='\t',index=False);pd.concat(patients,ignore_index=True).to_csv(OUT/'refinement_patients.tsv',sep='\t',index=False)
# Add the two sheets without modifying any existing spreadsheet cell.
from openpyxl import load_workbook
book=load_workbook(OUT/'Revision_Data.xlsx')
for name,df in [('SBS6_refinement',pd.DataFrame(rows)),('SBS6_refinement_patients',pd.concat(patients,ignore_index=True))]:
 if name in book:del book[name]
 ws=book.create_sheet(name);ws.append(df.columns.tolist())
 for row in df.itertuples(index=False,name=None):ws.append([None if pd.isna(v) else v for v in row])
 ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
book.save(OUT/'Revision_Data.xlsx')
(OUT/'refinement_summary.json').write_text(json.dumps(json.loads(pd.DataFrame(rows).to_json(orient='records')),indent=2))

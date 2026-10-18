"""Finalise MSI clinical readout; preserve labels and the primary SBS screen."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from openpyxl import load_workbook
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;sys.path.insert(0,str(OUT));from analyze import binary
labels=pd.read_csv(OUT/'clinical_revision_scores.tsv',sep='\t');metrics=pd.read_csv(OUT/'clinical_revision.tsv',sep='\t')
for co in ['TCGA','CPTAC']:
 pred=pd.read_csv(OUT/f'kread6_{co.lower()}_scores.tsv',sep='\t').set_index('PATIENT').prediction_SBS6
 mask=labels.cohort.eq(co)&labels.endpoint.eq('MSI');labels.loc[mask,'score']=labels.loc[mask,'PATIENT'].map(pred)
 assert labels.loc[mask,'score'].notna().all()
 for rule in labels.loc[mask,'rule'].unique():
  rows=labels[mask&labels.rule.eq(rule)];m=binary(rows.label,rows.score);mi=metrics.cohort.eq(co)&metrics.endpoint.eq('MSI')&metrics.rule.eq(rule)
  for k,v in m.items():metrics.loc[mi,k]=v
indices=metrics.index[metrics.cohort.eq('CPTAC')];p=metrics.loc[indices,'p'].to_numpy();order=np.argsort(p);adj=np.empty(2);adj[order]=np.minimum(1,np.maximum.accumulate(p[order]*[2,1]));metrics.loc[indices,'p_holm_two_endpoints']=adj
labels.to_csv(OUT/'clinical_revision_scores.tsv',sep='\t',index=False);metrics.to_csv(OUT/'clinical_revision.tsv',sep='\t',index=False)
summary=json.loads((OUT/'summary.json').read_text());summary['clinical']=json.loads(metrics.to_json(orient='records'));summary['MSI_score_definition']='Final SBS6-specific RBF regression readout; continuous SBS6 training, original patient splits and inner SBS6 correlation selection; MSI labels evaluation only.';(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
notes=pd.read_csv(OUT/'analysis_notes.tsv',sep='\t');notes=notes[~notes.item.astype(str).str.contains('refinement|refined|readout|probe|targeted',case=False,regex=True)]
notes=pd.concat([notes,pd.DataFrame([dict(item='Final MSI score',definition=summary['MSI_score_definition']+' Alpha .01,.1,1,10,100,1000; gamma .25,1,4 /256; train-fitted feature standardisation and target centering; validation-fitted output scaling, fixed two-scale averaging, 3 OOF seed average and15-context external average. MANTIS original365/102 AUC.642362; MANTIS0.4 409/146 AUC.616959; CPTAC82/25 AUC.569123. The primary 210-pair models/scores are unchanged.')])],ignore_index=True)
notes['definition']=notes.definition.astype(str).str.replace('CPTAC UCEC: 82 patients; 5 positive / 77 negative','CPTAC UCEC: 82 patients; 6 positive / 76 negative',regex=False)
notes.to_csv(OUT/'analysis_notes.tsv',sep='\t',index=False)
book=load_workbook(OUT/'Revision_Data.xlsx')
for name in ['SBS6_refinement','SBS6_refinement_patients','SBS6_target_associations']:
 if name in book:del book[name]
selected=pd.concat([pd.read_csv(OUT/f'kread6_{seed}_selection.tsv',sep='\t').assign(selected=False) for seed in range(3)],ignore_index=True)
for seed in range(3):
 q=json.loads((OUT/f'kread6_{seed}_protocol.json').read_text())
 for context in q['selected_contexts']:
  for s in context['scales']:
   mask=(selected.seed.eq(seed)&selected.fold.eq(context['fold'])&selected.scale.eq(s['scale'])&selected.alpha.eq(s['alpha'])&np.isclose(selected.gamma,s['gamma']));selected.loc[mask,'selected']=True
assert selected.selected.sum()==30
for name,frame in [('clinical_revision',metrics),('clinical_revision_scores',labels),('analysis_notes',notes),('SBS6_clinical_selection',selected)]:
 if name in book:del book[name]
 ws=book.create_sheet(name);ws.append(frame.columns.tolist())
 for row in frame.itertuples(index=False,name=None):ws.append([None if pd.isna(v) else v.item() if isinstance(v,np.generic) else v for v in row])
 ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
book.save(OUT/'Revision_Data.xlsx')
book=load_workbook(ROOT/'Cancers_evidence.xlsx')
for ws in book:
 for row in ws:
  for c in row:
   if isinstance(c.value,str):c.value=c.value.replace('82 patients, 5 positive and 77 qualifying-variant negative; 5 POLE-positive','82 patients, 6 positive and 76 qualifying-variant negative; 6 POLE-positive').replace('82 patients; 5 positive / 77 negative','82 patients; 6 positive / 76 negative')
book.save(ROOT/'Cancers_evidence.xlsx')
print(metrics.to_string(index=False))

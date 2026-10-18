"""Descriptive patient influence for all 14 original nominal cross-cohort pairs.

Delete one patient at a time from the correlation calculation only. No model
refitting, primary exclusions, or new discovery thresholds are introduced.
"""
from pathlib import Path
import pandas as pd,numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'rev'
scores=pd.read_csv(OUT/'patient_scores.tsv',sep='\t')
pairs=pd.read_csv(OUT/'paired_metrics.tsv',sep='\t').query('nominal_positive_both')
details=[];summary=[]
for pair in pairs.itertuples():
 for co in ['TCGA','CPTAC']:
  d=scores[(scores.cancer==pair.cancer)&(scores.cohort==co)].reset_index(drop=True)
  y=d[pair.signature].to_numpy();s=d['prediction_'+pair.signature].to_numpy()
  full=float(pearsonr(y,s).statistic);values=[]
  for i in range(len(d)):
   keep=np.arange(len(d))!=i;r=float(pearsonr(y[keep],s[keep]).statistic) if np.ptp(y[keep]) else np.nan
   values.append(r);details.append(dict(cohort=co,cancer=pair.cancer,signature=pair.signature,omitted_PATIENT=d.PATIENT.iloc[i],omitted_exposure=float(y[i]),full_r=full,leave_one_out_r=r))
  v=np.asarray(values);minimum=int(np.nanargmin(v));maximum=int(np.nanargmax(v));pos=y>0
  summary.append(dict(cohort=co,cancer=pair.cancer,signature=pair.signature,n=len(d),nonzero=int(pos.sum()),full_r=full,leave_one_out_min_r=float(np.nanmin(v)),leave_one_out_max_r=float(np.nanmax(v)),min_r_omitted_PATIENT=d.PATIENT.iloc[minimum],max_r_omitted_PATIENT=d.PATIENT.iloc[maximum],positive_direction_fraction=float(np.mean(v>0)),positive_exposure_omission_min_r=float(np.nanmin(v[pos])),positive_exposure_omission_max_r=float(np.nanmax(v[pos]))))
summary=pd.DataFrame(summary);summary.to_csv(OUT/'influence_summary.tsv',sep='\t',index=False)
pd.DataFrame(details).to_csv(OUT/'influence_patients.tsv',sep='\t',index=False)
fig,axes=plt.subplots(1,2,figsize=(11.7,8.3),sharey=True)
labels=[r.cancer+'–'+r.signature for r in pairs.itertuples()]
for ax,co in zip(axes,['TCGA','CPTAC']):
 d=summary[summary.cohort==co];yy=np.arange(len(d))
 ax.hlines(yy,d.leave_one_out_min_r,d.leave_one_out_max_r,color='#187C91',lw=3)
 ax.scatter(d.full_r,yy,color='#243746',s=25,zorder=3)
 ax.axvline(0,color='grey',lw=.8);ax.set_title(co);ax.set_xlabel('Pearson r');ax.set_yticks(yy);ax.set_yticklabels(labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
fig.suptitle('Figure S10. Patient influence in the 14 original nominal pairs',fontsize=15)
fig.text(.08,.045,'Points: full-cohort correlations. Lines: minimum–maximum after omitting one patient at a time.\nDescriptive post hoc assessment; original patients, models and primary discovery criteria are unchanged.',fontsize=10)
fig.subplots_adjust(left=.16,right=.96,top=.9,bottom=.14,wspace=.15)
fig.savefig(OUT/'Influence.pdf');fig.savefig(OUT/'influence.png',dpi=170);plt.close(fig)
print(summary[(summary.cancer=='UCEC')&summary.signature.isin(['SBS3','SBS13'])].to_string(index=False))

"""Plot saved JSON diagnostics only; never load row data, fit a probe or recompute WP."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'analysis-01/results.json'
FITS=HERE/'analysis-01-fit-diagnostics.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads(RESULTS.read_text());fits=json.loads(FITS.read_text())
assert r['status']=='COMPLETE_FIXED_ERROR_DIAGNOSTICS_NO_FORECAST_CANDIDATE'
assert fits['status']=='COMPLETE_POSTREADOUT_FIXED_COEFFICIENT_AUDIT_NO_REFIT'
assert fits['provenance'][next(k for k in fits['provenance'] if k.endswith('/analysis-01/results.json'))]==sha(RESULTS)
b=r['budget']['models']['current_student4'];total_deficit=1-b['metric']['score']
slices=[
 ('Class 3\n(retrospective)','class',{'3'}),
 ('Price-code event age ≥64 rows','i0px_age_rows',{'[64,256)','[256,1024)','[1024,inf)'}),
 ('Price-code event rate₂₅₆ <0.05','i0px_rate256',{'[-inf,0.01)','[0.01,0.05)'}),
 ('First 2,000 rows','position',{'[-inf,2000)'}),
]
weight=[];deficit=[]
for _,partition,labels in slices:
 pieces=[[v for v in b['partitions'][partition][f'target{t}'] if v['label'] in labels] for t in (0,1)]
 assert all({v['label'] for v in z}==labels for z in pieces)
 weight.append(50*sum(v['weight_share'] for z in pieces for v in z))
 deficit.append(100*sum(v['budget'] for z in pieces for v in z)/total_deficit)
blocks=r['probe']['blocks'];fmap={v['block']:v for v in fits['pooled']}
train=100*np.array([fmap[k]['fit_skill_mean'] for k in blocks])
held=100*np.array(r['probe']['summaries']['pooled']['equal_target_mean_skill'])
assert np.allclose(held,100*np.array([fmap[k]['heldout_skill_mean'] for k in blocks]),rtol=0,atol=1e-14)
ci=100*np.array(r['probe']['conditional_chain_bootstrap']['intervals_95']['skill_equal_target_mean'])
assert ci.shape==(4,2) and np.all(ci[:,0]<=held) and np.all(ci[:,1]>=held)
labels=['Prediction confidence (7)', '+ Current inputs (211)', '+ Summaries / history (30)', '+ Member deviations (10)']
blue='#37688E';orange='#BD5B3F';muted='#59636D';ink='#202C35'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':12,'axes.titleweight':'semibold','axes.edgecolor':'#AAB3BB','axes.labelcolor':ink,'text.color':ink,'xtick.color':muted,'ytick.color':ink,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
fig,(ax,bx)=plt.subplots(1,2,figsize=(14,5.65),gridspec_kw={'width_ratios':[1,1.05]})
fig.subplots_adjust(left=.18,right=.98,top=.74,bottom=.28,wspace=.78)
fig.text(.035,.945,'Current student: error concentration and residual predictability',fontsize=17,weight='semibold')
fig.text(.035,.891,f"Previously exposed OOF discovery population  ·  {r['rows']:,} scored rows  ·  {r['chains']} whole chains",fontsize=10.5,color=muted)
y=np.arange(4)
for a in (ax,bx):
 a.spines[['top','right','left']].set_visible(False)
 a.tick_params(axis='y',length=0,pad=10)
 a.set_axisbelow(True);a.grid(axis='x',color='#E4E8EB',linewidth=.7)
ax.barh(y-.16,weight,height=.28,color=blue,label='Metric-weight share')
ax.barh(y+.16,deficit,height=.28,color=orange,label='Exact deficit share')
for yy,ww,dd in zip(y,weight,deficit):
 ax.text(ww+.8,yy-.16,f'{ww:.1f}%',va='center',fontsize=9.5,color=blue)
 ax.text(dd+.8,yy+.16,f'{dd:.1f}%',va='center',fontsize=9.5,color=orange)
ax.set(yticks=y,yticklabels=[z[0] for z in slices],xlim=(0,67),xticks=[0,20,40,60],xlabel='Share of total (%)')
ax.invert_yaxis();ax.set_title('A   Where the metric deficit concentrates',loc='left',pad=43)
ax.legend(loc='lower left',bbox_to_anchor=(-.02,1.015),frameon=False,ncol=1,fontsize=9.5,handlelength=1.3,labelspacing=.3,borderaxespad=0)
# Connecting the two fixed estimates visualizes the fitting-to-heldout gap only.
for yy,tr,ho in zip(y,train,held):bx.plot([ho,tr],[yy,yy],color='#D3D8DE',lw=1.3,zorder=1)
bx.axvline(0,color='#87939D',lw=1,ls=(0,(3,3)),zorder=1)
bx.scatter(train,y,s=40,marker='o',color=blue,label='Fitting chains',zorder=3)
bx.errorbar(held,y,xerr=np.vstack([held-ci[:,0],ci[:,1]-held]),fmt='s',markersize=5,color=orange,elinewidth=1.4,capsize=3,label='Heldout chains · 95% CI',zorder=3)
for yy,tr,ho in zip(y,train,held):
 bx.annotate(f'{tr:+.2f}%',(tr,yy),xytext=((4 if yy==0 else 0),-17),textcoords='offset points',ha=('left' if yy==0 else 'center'),fontsize=9,color=blue)
 bx.annotate(f'{ho:+.2f}%',(ho,yy),xytext=((-4 if yy==0 else 0),-17),textcoords='offset points',ha=('right' if yy==0 else 'center'),fontsize=9,color=orange)
bx.set(yticks=y,yticklabels=labels,xlim=(-4.55,3.55),xticks=[-4,-2,0,2],xlabel='q-SSE skill (%) — not a WP gain',ylim=(3.6,-.55))
bx.set_title('B   Fixed probes lose skill on heldout chains',loc='left',pad=43)
bx.legend(loc='lower left',bbox_to_anchor=(-.02,1.015),frameon=False,ncol=1,fontsize=9.5,labelspacing=.3,borderaxespad=0)
fig.text(.035,.157,'A  Overlapping marginal slices; their percentages must not be added. Weight shares average the two targets equally.',fontsize=9,color=muted)
fig.text(.035,.122,'    Deficit share is the exact contribution to 1 − WP. “Price-code event” refers to the observed encoded i0 input update.',fontsize=9,color=muted)
fig.text(.035,.077,'B  Nested feature blocks; q-SSE skill = 1 − SSE / constant-baseline SSE, averaged equally over targets. Ridge λ = 0.01.',fontsize=9,color=muted)
fig.text(.035,.042,'    Heldout intervals: 2,000 paired-chain bootstrap draws, conditional on the fitted probes; no selection or hidden-shift guarantee.',fontsize=9,color=muted)
provenance={'results_sha256':sha(RESULTS),'fixed_fit_diagnostics_sha256':sha(FITS),'source_sha256':sha(Path(__file__)),'values':{'weight_share_percent':weight,'deficit_share_percent':deficit,'fit_skill_percent':train.tolist(),'heldout_skill_percent':held.tolist(),'heldout_CI95_percent':ci.tolist()},'new_data_reads':False,'new_fits':False}
fig.savefig(HERE/'diagnosis.png',dpi=220,metadata={'Title':'Current student error diagnosis','Description':json.dumps(provenance)})
fig.savefig(HERE/'diagnosis.pdf',metadata={'Title':'Current student error diagnosis','Subject':json.dumps(provenance),'Creator':'Matplotlib; saved JSON results only'})
print(json.dumps(provenance,indent=2))
plt.close(fig)

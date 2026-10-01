"""Visualize the completed immutable trace; no inference or new timing."""
from pathlib import Path
import hashlib,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/wunder-runtime-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
f=P/'execution/sequences.jsonl';assert sha(f)=='810e83fe5d2cd7d5395681e3f5bbf4d61fe1fd3304ca9a75237016d7f80f33d9'
a=json.loads((P/'execution/launch-result.json').read_text());assert sha(P/'execution/launch-result.json')=='1202f368ce48d2f187179feea68286ffa28d831a94355f587eccedc2626c5ed1'
rows=[json.loads(line) for line in f.read_text().splitlines()];assert len(rows)==1873
callbacks=sum(r['callbacks'] for r in rows);assert callbacks==37460000
per=np.array([r['callback_loop_wall_seconds']/r['callbacks']*1e6 for r in rows]);x=np.arange(1,1874)*.02
callback=sum(r['callback_loop_wall_seconds'] for r in rows);io=sum(r['io_decode_conversion_seconds'] for r in rows);checks=sum(r['validation_checksum_seconds'] for r in rows)
hashes=13.598222130909562+13.481270264834166
other=a['process_wall_seconds']-callback-io-checks-hashes;assert other>=0
factor=39400000/callbacks/60;parts=np.array([callback,io,checks,hashes,other])*factor
assert abs(parts.sum()-a['full_process_projection_minutes_39_4M'])<1e-10
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(left,right)=plt.subplots(1,2,figsize=(12,4.9),gridspec_kw={'width_ratios':[2.2,1]},layout='constrained')
fig.suptitle('Fixed-three submission: complete 37.46M-callback runtime audit',fontsize=16,fontweight='bold')
left.plot(x,per,color='#294f70',linewidth=.65,alpha=.8,label='Each 20,000-row sequence')
left.axhline(50*60/39400000*1e6,color='#bd642b',linestyle='--',linewidth=1.4,label='50-min callback gate at 39.4M rows')
left.set(xlabel='Completed callbacks (millions)',ylabel='Microseconds per complete callback',xlim=(0,37.46));left.grid(alpha=.16)
left.legend(loc='upper left',bbox_to_anchor=(0,.88),frameon=False,fontsize=9)
colors=['#294f70','#69a89f','#b1c88b','#d6ad72','#bd8a94'];labels=['Callbacks','Read/decode','Validation/hash outputs','Input file hashes','Startup / remaining overhead']
bottom=0
for value,color,label in zip(parts,colors,labels):right.bar(0,value,bottom=bottom,width=.55,color=color,label=label);bottom+=value
right.axhline(60,color='#bd642b',linestyle='--',linewidth=1.4);right.text(.31,60,'60-min limit',color='#bd642b',fontsize=9,va='bottom')
right.set(xticks=[0],xticklabels=['Projected complete process'],ylabel='Minutes at 39.4M rows',ylim=(0,64),xlim=(-.6,.95));right.text(0,parts.sum()+1,f'{parts.sum():.2f} min',ha='center',fontweight='bold')
right.legend(loc='upper left',bbox_to_anchor=(0,-.10),fontsize=8,frameon=False)
fig.supxlabel('One v3 trace on beleriand: 1-CPU cgroup, CPU18, warmed input cache. Public hardware not measured.',fontsize=10)
fig.savefig(P/'runtime-trace.png',dpi=170);fig.savefig(P/'runtime-trace.svg')
(P/'runtime-trace-inputs.json').write_text(json.dumps({'source_sha256':sha(__file__),'sequence_trace_sha256':sha(f),'launch_result_sha256':sha(P/'execution/launch-result.json'),'callbacks':callbacks,'observed_process_minutes':a['process_wall_seconds']/60,'projected_components_minutes':dict(zip(labels,parts.tolist())),'new_inference':0,'new_timing':0},indent=2)+'\n')
print(json.dumps({'saved':str(P/'runtime-trace.png'),'projected_process_minutes':float(parts.sum())}))

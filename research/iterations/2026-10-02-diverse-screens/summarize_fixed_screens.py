"""Summarize immutable completed graders; never scores or selects predictions."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
specs=[('mingru-forecast','minGRU 256 × 2',None),('distill07','Дистилляция α=0.7',None),
 ('atom-consistent-training','Исправленный atom head','primary'),('interval-memory','Память между событиями','primary'),
 ('ordinary-warm','Дополнительная эпоха GRU','primary'),('ordinary-warm-rng-control','Только повторная настройка головы','explanatory'),
 ('temporal-screen','Detach 500, update 2000','primary'),('warm-midpoint','Середина весов 0.5','primary')]
rows=[];inputs={};total_fits=total_OOF=0
for branch,label,contrast in specs:
 p=ROOT/f'research/iterations/2026-10-02-{branch}/evaluation-01/report.json';r=json.loads(p.read_text());inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 q=r['scopes']['all'] if contrast is None else r['comparisons'][contrast]['scopes']['all']
 assert abs(q['control']['WP']-.7023796676142182)<2e-12
 rows.append(dict(branch=branch,label=label,WP=q['candidate']['WP'],delta_WP=q['delta_WP'],conditional95=q['paired_chain_CI']['interval95'],fold_deltas={k:v['delta_WP'] for k,v in q['folds'].items()},fits=r['fits'],OOF=r['OOF_jobs']))
 total_fits+=r['fits'];total_OOF+=r['OOF_jobs']
p=ROOT/'research/iterations/2026-10-02-ordinary-warm-seed4/evaluation-01/report.json';r=json.loads(p.read_text());inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();total_fits+=r['fits'];total_OOF+=r['OOF_jobs']
record=dict(status='SUMMARY_OF_FIXED_COMPLETED_GPU_SCREENS',rows=rows,input_sha256=inputs,forecast_fits=total_fits,target_free_GPU_OOF=total_OOF,
 exclusions='Final full-data2fits and native/runtime qualification tracked separately; no constructed checkpoint counted as training.',
 limitations='Plot is seed3 only on same860exposedpuresequences/250chains. Conditional95% chain intervals omit adaptive selection, training-seed population uncertainty and testshift. Unshrunk warm failed paired seed4 gate. Midpoint seed4 +.000028922 has interval crossingzero; midpoint was designed after seeing both warmseed results. Selected midpoint remains pending serving qualification.')
summary=HERE/'fixed-screen-summary.json'
if summary.exists():assert json.loads(summary.read_text())==record
else:
 with summary.open('x') as f:json.dump(record,f,indent=2,ensure_ascii=False);f.write('\n')
fig,ax=plt.subplots(figsize=(10.5,6.1));fig.subplots_adjust(left=.36,right=.96,top=.85,bottom=.28)
for i,row in enumerate(rows):
 x=row['delta_WP']*1e4;lo,hi=[v*1e4 for v in row['conditional95']];color='#3168AD' if row['branch']=='warm-midpoint' else ('#277E6B' if x>0 else '#B45D4A')
 ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color=color,capsize=4,lw=1.7,markersize=6)
ax.set_yticks(range(len(rows)),[r['label'] for r in rows]);ax.invert_yaxis();ax.axvline(0,color='#555555',lw=1,ls='--');ax.grid(axis='x',alpha=.15)
ax.set_xlabel('Изменение pooled WP относительно исходного ансамбля, ×10⁻⁴')
ax.set_title('Разные ветки — один фиксированный ансамблевый baseline',loc='left',fontsize=14,pad=20)
for side in ('top','right','left'):ax.spines[side].set_visible(False)
fig.text(.04,.055,'Seed 3, одинаковые 860 последовательностей. Интервалы условны на обученных моделях.\nWarm не подтвердился на seed 4; midpoint: малый положительный эффект на обоих seed,\nно это адаптивный выбор на уже изученных данных. Качество и время native проверяются отдельно.',fontsize=9,color='#444444')
fig.savefig(HERE/'fixed-screen-deltas.png',dpi=160,facecolor='white');plt.close(fig)
print(json.dumps(dict(forecast_fits=total_fits,target_free_GPU_OOF=total_OOF,figure=str(HERE/'fixed-screen-deltas.png'))))

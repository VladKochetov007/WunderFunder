"""Render a fixed CLOCK2 readout; no new data or empirical queries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent
d=json.loads((r/'clock-output/clock-result.json').read_text())
b=d['by_class']['3']['interval_EDA_fixed_A']['i0_volume_code']['by_gap_bin'];total=sum(x['intervals'] for x in b)
values=[100*x['intervals']/total for x in b]
fig,ax=plt.subplots(figsize=(7,4.3))
fig.subplots_adjust(left=.12,right=.98,top=.88,bottom=.22)
ax.bar(['1','2–4','5–16','17–64','65+'],values,color='#315c85',width=.68)
for i,v in enumerate(values):ax.text(i,v+.7,f'{v:.2f}%',ha='center',fontsize=10)
ax.set(ylim=(0,46),xlabel='Gap between successive learned-clock writes (rows)',ylabel='Share of complete intervals (%)',title='Class 3: learned-clock gap distribution')
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.text(.98,.025,'456,824 complete intervals · 64 audit chains · fixed fresh A clock',ha='right',fontsize=8,color='#555555')
fig.savefig(r/'clock-gap-distribution.svg');fig.savefig(r/'clock-gap-distribution.png',dpi=160)

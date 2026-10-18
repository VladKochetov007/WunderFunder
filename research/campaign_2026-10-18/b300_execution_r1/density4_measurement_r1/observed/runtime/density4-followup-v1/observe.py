"""Observe actual cohort progress without changing training processes."""
import json,re,time,subprocess
from pathlib import Path
ROOT=Path('/data/epanov_s/tabdata')
DEST=ROOT/'state/density4-followup-v1'
PAT=re.compile(r'ep ([0-9]+) batch ([0-9]+)/([0-9]+).*?seq/s ([0-9.]+)')
DEST.mkdir(exist_ok=False)
start=time.monotonic()
with (DEST/'progress.jsonl').open('x',buffering=1) as out:
 while time.monotonic()-start<1800:
  row={'observed_unix':time.time(),'progress':{}}
  for state in ['r01-execution-v4','r01-density4-v1','pr15-execution-v2']:
   for p in (ROOT/'state'/state/'logs').glob('*.log'):
    try:
     matches=PAT.findall(p.read_text(errors='replace'))
     if matches:
      ep,b,total,reported=matches[-1]
      row['progress'][p.stem]={'state':state,'batch_index':int(ep)*int(total)+int(b),'log_mtime':p.stat().st_mtime,'reported_seq_s':float(reported)}
    except FileNotFoundError:pass
  out.write(json.dumps(row,separators=(',',':'))+'\n')
  time.sleep(5)

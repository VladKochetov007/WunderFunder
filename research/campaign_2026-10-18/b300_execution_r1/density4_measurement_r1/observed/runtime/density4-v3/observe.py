import json,time,subprocess
from pathlib import Path
S=Path('/data/epanov_s/tabdata/state/density4-v3')
rows={str(g):{} for g in (2,7)}
deadline=time.time()+800
while time.time()<deadline:
 now=time.time()
 for g,r in rows.items():
  p=S/'logs'/('gpu'+g+'.log')
  if not p.exists(): continue
  text=p.read_text(errors='replace')
  if 'loader workers' in text and 'training_seen' not in r: r['training_seen']=now
  if 'PROFILE_UPDATE_EVIDENCE ' in text and 'completed_seen' not in r: r['completed_seen']=now
 (S/'OBSERVER.json.tmp').write_text(json.dumps(dict(rows=rows,time_unix=now),indent=2))
 (S/'OBSERVER.json.tmp').replace(S/'OBSERVER.json')
 if all('completed_seen' in r for r in rows.values()) or (S/'FAILED.json').exists(): break
 time.sleep(.5)

from pathlib import Path
import json,datetime
b=Path('/home/grinchenkoeg/computer-vision-homework-20261018')
paths=[b,b/'research',b/'research/full_gru_tweaks_r1',b/'research/full_gru_tweaks_r1/teacher_cache_v1']
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'paths':[{'path':str(p),'exists':p.exists(),'is_dir':p.is_dir(),'is_symlink':p.is_symlink(),'resolved':str(p.resolve())} for p in paths]}))

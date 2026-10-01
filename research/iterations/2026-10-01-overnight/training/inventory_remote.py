"""Read-only source inventory; output one JSON record to stdout."""
from pathlib import Path
import hashlib,json,torch
r=Path('/home/grinchenkoeg/WunderFunder')
files={}
names=[]
for f in [0,1]:
 for base in ['ref','refs4','d256','c500']:
  names.append((f'{base}_f{f}','model_ep3.pt'))
 for base in ['ft128','ft128s4','ft256d','ftc500']:
  names.append((f'{base}_f{f}h','model_ep0.pt'))
for base in ['fin128_s3','fin128_s4','find256_s3','finc500b_s3']:
 for suffix,ep in [('',3),('h2',1)]: names.append((base+suffix,f'model_ep{ep}.pt'))
for name,fn in names:
 p=r/'artifacts/runs'/name/fn
 if not p.exists():
  files[str(p)]={'exists':False};continue
 c=torch.load(p,map_location='cpu',weights_only=False)
 files[str(p)]={'exists':True,'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'keys':list(c),'args':c.get('args'), 'epoch':c.get('epoch'), 'model_shapes':{k:list(v.shape) for k,v in c['model'].items()},'history':json.loads((p.parent/'history.json').read_text())}
print(json.dumps({'files':files},indent=2))

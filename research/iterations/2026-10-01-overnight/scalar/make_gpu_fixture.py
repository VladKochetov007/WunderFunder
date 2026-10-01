"""Generate a bounded synthetic fixture on CPU; no competition data access."""
from pathlib import Path
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;D=P/'generated';D.mkdir(exist_ok=True)
with np.load(D/'knots.npz') as z:k=z['knots'];counts=z['counts']
steps=np.arange(20000);rng=np.random.default_rng(2026100137)
raw=rng.normal(0,.5,size=(20000,112)).astype(np.float32)
indices=list(range(22))+list(range(52,74))
for j,index in enumerate(indices):raw[:,index]=k[j,((steps//64+j)%counts[j]).astype(np.int64)]
quant=np.asarray([.12543,.24133,.46854,.65687],np.float32)
y=np.empty((4,20000,2),np.float32)
for j,q in enumerate(quant):
 y[j,:,0]=q*((steps//37)%5-2);y[j,:,1]=-q*((steps//53)%4+1)
prob=np.linspace(.1,.9,20000,dtype=np.float32);mask=(steps%7)==0
path=D/'gpu_fixture.npz';assert not path.exists();np.savez(path,raw=raw,targets=y,probability=prob,mask=mask,knots=k,counts=counts)
receipt={'generated_only':True,'competition_inputs_read':False,'seed':2026100137,'rows':20000,'raw_features':112,'target_banks':4,'training_batch':32,'chunk':2000,'repeated_exact_knots':True,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(D/'gpu_fixture.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

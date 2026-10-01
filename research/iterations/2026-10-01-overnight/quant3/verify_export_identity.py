"""Read-only original checkpoint/ZIP identity proof, then verified isolated copies."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import hashlib,io,json,shutil,zipfile
import numpy as np
import torch
ROOT=Path('/home/grinchenkoeg/WunderFunder-creative');P=ROOT/'research/iterations/2026-10-01-overnight/quant3';ORIG=Path('/home/grinchenkoeg/WunderFunder/artifacts/runs');ZIP=Path('/home/grinchenkoeg/WunderFunder-cpp/submissions/candV1_ft2/candV1_ft2.zip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (P/'checkpoint-identity.json').exists()
assert sha(ZIP)=='ea494921497d9f3cf53b19b1b68285133983560a249b38ef40738b3d846d319d'
receipt=[];spec=[]
with zipfile.ZipFile(ZIP)as z:
 members=json.loads(z.read('members.json'))
 for index,name in [(0,'find256_s3h2'),(1,'fin128_s3h2'),(3,'finc500b_s3h2')]:
  assert members[index]['source']==name
  src=ORIG/name/'model_ep1.pt';digest=sha(src);ck=torch.load(src,map_location='cpu',weights_only=False)
  assert ck['epoch']==1 and ck['n_in']==211
  arrays={k:v.detach().cpu().float().numpy()for k,v in ck['model'].items()};arrays['head.2.weight']=arrays['head.2.weight'][:2].copy();arrays['head.2.bias']=arrays['head.2.bias'][:2].copy()
  with np.load(io.BytesIO(z.read(members[index]['weights'])),allow_pickle=False)as archived:
   assert set(arrays)==set(archived.files),(name,set(arrays)^set(archived.files));checks={}
   for key,a in arrays.items():
    b=archived[key];assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),(name,key,a.dtype,b.dtype)
    checks[key]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest(),'bitexact':True}
  dst=ROOT/'artifacts/runs'/name/'model_ep1.pt';assert not dst.parent.is_symlink();dst.parent.mkdir(parents=True,exist_ok=True)
  copied=not dst.exists()
  if copied:
   temp=dst.with_suffix('.copying');assert not temp.exists();shutil.copy2(src,temp);assert sha(temp)==digest;os.rename(temp,dst)
  assert sha(dst)==digest
  receipt.append({'source':str(src),'source_sha256':digest,'isolated_copy':str(dst),'copied_now':copied,'archived_member':members[index],'epoch':ck['epoch'],'n_in':ck['n_in'],'arrays':checks})
  spec.append({'checkpoint':str(dst),'coef':1/3})
(P/'members.json').write_text(json.dumps(spec,indent=2)+'\n');(P/'checkpoint-identity.json').write_text(json.dumps({'status':'PASS','original_zip':str(ZIP),'original_zip_sha256':sha(ZIP),'members':receipt,'targets_read':False,'WP_queries':0},indent=2)+'\n');print(json.dumps({'status':'PASS','members':[m['archived_member']['source']for m in receipt],'all_export_arrays_bitexact':True}))

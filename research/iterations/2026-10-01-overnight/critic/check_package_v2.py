"""Synthetic packaging smoke; no trained competition checkpoint is packaged."""
from pathlib import Path
import sys,json,subprocess,zipfile,hashlib
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT/'src'))
from wnn.nn import RNNModel

def main():
 dst=HERE/'synthetic-package-v2';assert not dst.exists();dst.mkdir();torch.manual_seed(924381)
 model=RNNModel(211,hidden=128,layers=2,n_out=23)
 ck={'model':model.state_dict(),'n_in':211,'epoch':1,'args':{'arch':'gru','cell':'gru','enc':'linear','hidden':128,'layers':2,'lookahead':'','synthetic_fixture':True}}
 path=dst/'untrained-synthetic.pt';torch.save(ck,path)
 result=[]
 for count in [2,3,4]:
  spec=dst/f'members{count}.json';spec.write_text(json.dumps([{'checkpoint':str(path),'coef':1/count} for _ in range(count)]));out=dst/f'synthetic{count}.zip'
  subprocess.run([sys.executable,str(HERE/'package_fixed_v2.py'),'--members',str(spec),'--native-dir',str(HERE/'native_order_v2/build'),'--output',str(out)],check=True)
  with zipfile.ZipFile(out) as z:
   assert z.testzip() is None and 'solution.py' in z.namelist();assert len(json.loads(z.read('members.json')))==count;z.extractall(dst/f'extracted{count}')
  result.append({'members':count,'zip_bytes':out.stat().st_size,'zip_sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
 (dst/'receipt.json').write_text(json.dumps({'status':'packaged; CP311 smoke pending','cases':result,'forecasting_fits':0,'trained_checkpoints_packaged':0},indent=2)+'\n')
if __name__=='__main__':main()

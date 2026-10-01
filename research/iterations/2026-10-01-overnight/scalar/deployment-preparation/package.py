"""Preparation only: package allocated actual exports with already tested native binaries."""
from pathlib import Path
import argparse,hashlib,json,zipfile
import numpy as np
P=Path(__file__).resolve().parents[1];ROOT=P.parents[3]
RUNTIME_FILES={'avx2/wnn_cpp.cpython-311-x86_64-linux-gnu.so': '1e7a120dfd250a7b3f468be4a6385d6624cc0094051f775bc58637f3f33f89a3', 'solution.py': '7a7027decfb374af2be9b631e60339fbfc75953e5fe84db255d14f37ead9bfa2', 'v2/wnn_cpp.cpython-311-x86_64-linux-gnu.so': 'b64bb8ab124337fe9e3539226dc8fc000d099ab089ee509e25a479040d80b0cd', 'v3/wnn_cpp.cpython-311-x86_64-linux-gnu.so': '75c631b26ee8f989340603b50bfc3959c2778cce91df6350b440ef2b49ca77b3'}
NATIVE_TEMPLATE_SHA256="0769792d712df731c70a90a21c217d066042141623fd142d0a58e6346059a7a9"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);a=ap.parse_args();alloc=json.loads(a.allocation.read_text());assert alloc.get('SCALARPLE_native_package_authorized') is True
 for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
 ep=ROOT/alloc['exports']['path'];assert sha(ep)==alloc['exports']['sha256'];export=json.loads(ep.read_text());assert export['status']=='EXACT_FIXED_CHECKPOINT_EXPORT_NO_INFERENCE';assert [c['checkpoint_sha256'] for c in export['children']]==alloc['child_checkpoint_sha256']
 prototype=ROOT/alloc['native_template']['path'];assert sha(prototype)==alloc['native_template']['sha256']==NATIVE_TEMPLATE_SHA256;assert alloc['runtime_files']==RUNTIME_FILES;contents={}
 with zipfile.ZipFile(prototype) as z:
  for name in z.namelist():
   assert not Path(name).is_absolute() and '..' not in Path(name).parts
   if name=='solution.py' or name.endswith('.so'):contents[name]=z.read(name)
 assert len(contents)==4 and set(contents)==set(alloc['runtime_files'])
 for name,h in alloc['runtime_files'].items():assert hashlib.sha256(contents[name]).hexdigest()==h,name
 members=[]
 for slot,child in enumerate(export['children']):
  path=ROOT/child['export'];assert sha(path)==child['export_sha256']
  with np.load(path,allow_pickle=False) as data:
   assert set(data.files)==set(child['arrays'])
   for key,meta in child['arrays'].items():
    value=data[key];assert list(value.shape)==meta['shape'] and str(value.dtype)==meta['dtype'] and hashlib.sha256(value.tobytes()).hexdigest()==meta['sha256']
  name=f'weights_{slot}.npz';contents[name]=path.read_bytes();members.append({'weights':name,'coef':1/3})
 contents['members.json']=(json.dumps(members,indent=2)+'\n').encode();out=ROOT/alloc['zip_path'];assert out.resolve().is_relative_to((ROOT/'submissions/overnight_scalar').resolve());out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists() and not out.with_suffix('.manifest.json').exists()
 with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for name,blob in sorted(contents.items()):
   info=zipfile.ZipInfo(name,date_time=(2026,10,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,blob)
 assert out.stat().st_size<=20_000_000
 receipt={'status':'PACKAGED_PENDING_FRESH_INTERFACE_PARITY_RUNTIME_AND_QUALITY','zip_sha256':sha(out),'zip_bytes':out.stat().st_size,'files':{n:{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)} for n,b in contents.items()},'exports_path':str(ep.relative_to(ROOT)),'exports_sha256':sha(ep),'children':export['children'],'allocation_sha256':sha(a.allocation),'script_sha256':sha(__file__),'runtime':'samefixed3 members/order/equalfloat64thirds on allISAs; nativeFMA/nonFMAengine follows compiledISA; no wallclockgovernor','scalar_numeric':'F32 table/projection/knots and separated ordered interpolation; original native GRU halfweights/fastactivations unchanged','targets_read':False,'WP_queries':0}
 out.with_suffix('.manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'zip_sha256':receipt['zip_sha256'],'zip_bytes':receipt['zip_bytes']}))
if __name__=='__main__':main()

"""Export only warm final member0; preserve every other qualified ZIP entry bytewise."""
from pathlib import Path
import argparse,hashlib,io,json,zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BASE_SHA='eec571629b0829aeb9c73db80a1170fcb7cb3ccfd132f72f3ddd51f779d2fdd4'
BASE_MANIFEST=ROOT/'research/iterations/2026-10-01-continuation/student-runtime/package/student_fixed4.manifest.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base-zip',type=Path,required=True);ap.add_argument('--base-manifest',type=Path,required=True)
 ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--checkpoint-receipt',type=Path,required=True);ap.add_argument('--completion',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 for k,v in vars(a).items():setattr(a,k,v.resolve())
 assert a.output.is_relative_to(HERE/'package') and not a.output.exists() and not a.output.with_suffix('.manifest.json').exists()
 inventory=json.loads((HERE/'legacy-bindings.json').read_text());assert a.base_manifest==BASE_MANIFEST.resolve()
 assert sha(a.base_manifest)==inventory['source_sha256'][str(BASE_MANIFEST.relative_to(ROOT))]
 assert a.base_zip==(ROOT/inventory['incumbent']['zip']).resolve() and sha(a.base_zip)==BASE_SHA
 base=json.loads(a.base_manifest.read_text());assert base['zip_sha256']==BASE_SHA
 commands=json.loads((HERE/'commands.json').read_text());job=commands['jobs'][1]
 assert a.checkpoint==(ROOT/job['endpoint']).resolve() and a.checkpoint_receipt==(HERE/'execution-01'/f'{job["name"]}.check.json').resolve()
 assert a.completion==(HERE/'execution-01/result.json').resolve();completion=json.loads(a.completion.read_text())
 allocation=HERE/'training-allocation.json';alloc=json.loads(allocation.read_text())
 assert json.loads((HERE/'execution-01/allocation.json').read_text())==alloc
 assert completion['allocation_sha256']==sha(allocation) and alloc['sources'][str((HERE/'commands.json').relative_to(ROOT))]==sha(HERE/'commands.json')
 assert completion['status']==commands['success_status'] and completion['fits']==2 and completion['OOF_jobs']==0 and completion['metric_calls']==0
 assert [j['name'] for j in completion['jobs']]==[j['name'] for j in commands['jobs']]
 for j in completion['jobs']:
  assert j['exit_code']==0 and not j.get('cleanup_error') and not j['remaining_scope_pids']
  p=ROOT/j['worker_result']['path'];assert sha(p)==j['worker_result']['sha256'];r=json.loads(p.read_text());assert r['status']=='COMPLETE'
  for rel,b in r['outputs'].items():assert sha(ROOT/rel)==b['sha256']
 check=json.loads(a.checkpoint_receipt.read_text());digest=sha(a.checkpoint)
 assert check['status']=='PASS_ORDINARY_WARM_FIXED_ENDPOINT_AND_LINEAGE' and check['checkpoint']['sha256']==digest
 assert check['normalization_matches_parent'] and check['all103_args_match'] and check['model_pt_state_byte_equals_endpoint']
 assert check['commands_sha256']==sha(HERE/'commands.json') and check['validator_sha256']==sha(HERE/'check_checkpoint.py')
 for rel,h in commands['source_sha256'].items():assert sha(ROOT/rel)==h
 import numpy as np
 import torch
 torch.set_num_threads(1)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 assert ck['epoch']==1 and ck['n_in']==211 and ck['args']==job['resolved_args'] and ck['readout'] is None and ck['readout_tilted'] is False
 from check_checkpoint import SHAPES
 assert set(ck['model'])==set(SHAPES)
 arrays={k:v.detach().cpu().numpy() for k,v in ck['model'].items()}
 for k,v in arrays.items():assert v.shape==SHAPES[k] and v.dtype==np.float32 and np.isfinite(v).all(),k
 assert (arrays['std']>0).all();arrays['head.2.weight']=arrays['head.2.weight'][:2].copy();arrays['head.2.bias']=arrays['head.2.bias'][:2].copy()
 buffer=io.BytesIO();np.savez(buffer,**arrays);new=buffer.getvalue()
 a.output.parent.mkdir(parents=True,exist_ok=True);files={};unchanged=[]
 with zipfile.ZipFile(a.base_zip) as old,zipfile.ZipFile(a.output,'x') as z:
  entries=old.infolist();assert len(entries)==len(base['files']) and {e.filename for e in entries}==set(base['files'])
  for e in entries:
   p=Path(e.filename);assert not p.is_absolute() and '..' not in p.parts and not e.is_dir() and (e.external_attr>>16)&0o170000!=0o120000
   body=old.read(e);b=base['files'][e.filename];assert len(body)==b['bytes'] and hashlib.sha256(body).hexdigest()==b['sha256']
   if e.filename=='weights_0.npz':body=new
   else:unchanged.append(e.filename)
   z.writestr(e,body);files[e.filename]=dict(sha256=hashlib.sha256(body).hexdigest(),bytes=len(body))
 assert a.output.stat().st_size<=20000000 and sha(a.checkpoint)==digest and sha(a.base_zip)==BASE_SHA
 assert set(unchanged)==set(base['files'])-{'weights_0.npz'} and all(files[n]==base['files'][n] for n in unchanged)
 result=dict(output=str(a.output),zip_sha256=sha(a.output),zip_bytes=a.output.stat().st_size,runtime_policy='fixed4',members=base['members'],
  checkpoints=[dict(checkpoint=str(a.checkpoint),sha256=digest,epoch=1,args=ck['args']),*base['checkpoints'][1:]],files=files,
  unchanged_files_byte_exact=unchanged,base_zip_sha256=BASE_SHA,base_manifest_sha256=sha(a.base_manifest),
  completion_sha256=sha(a.completion),checkpoint_receipt_sha256=sha(a.checkpoint_receipt),commands_sha256=sha(HERE/'commands.json'),
  packager_sha256=sha(Path(__file__)),status='PACKAGED_REQUIRES_FRESH_PARITY_AND_RUNTIME',metrics=0)
 with a.output.with_suffix('.manifest.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ['status','zip_sha256','zip_bytes','unchanged_files_byte_exact']}))
if __name__=='__main__':main()

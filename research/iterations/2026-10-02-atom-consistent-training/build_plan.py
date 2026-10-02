"""Source and checkpoint-metadata only. No Torch/data imports or execution allocation."""
from __future__ import annotations
import argparse,ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parser():
 p=argparse.ArgumentParser(add_help=False)
 for n in ast.walk(ast.parse((ROOT/'scripts/train_rnn.py').read_text())):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument':
   args=[ast.literal_eval(x) for x in n.args];kw={}
   for k in n.keywords:kw[k.arg]={'int':int,'float':float,'str':str}[k.value.id] if k.arg=='type' else ast.literal_eval(k.value)
   p.add_argument(*args,**kw)
 p.add_argument('--atom-decoder',choices=['original','corrected'],required=True)
 return p

def resolve(p,argv):
 a=vars(p.parse_args(argv))
 if a['fast']:
  for k in ('sync_free','fused_adam','pin_memory','prop_cache'):a[k]=True
 return a

def main():
 p=parser();defaults=resolve(p,['--name','none','--atom-decoder','original'])
 meta=json.loads((HERE/'original-metadata.json').read_text());records={r['run']:r for r in meta['records']}
 assert len(records)==4 and len(set(r['atom_tensor_sha256'] for r in records.values()))==1
 jobs=[];rel=str(HERE.relative_to(ROOT))
 for fold in (0,1):
  for arm in ('original','corrected'):
   parent=f'oct02_ac_{arm}_s3_f{fold}';head=f'oct02_ac_{arm}_s3h1_f{fold}'
   for stage,original,name,ep in [('parent',f'ah128_f{fold}',parent,3),('H2',f'ftah128_f{fold}h',head,0)]:
    old=records[original];assert set(old['args'])<=set(defaults)
    a={**defaults,**old['args'],'name':name,'atom_decoder':arm}
    if stage=='H2':a['init']=parent+'@ep3'
    av=[]
    for k,v in a.items():
     if k not in ('name','atom_decoder') and v==defaults[k]:continue
     flag='--'+k.replace('_','-')
     if isinstance(v,bool):assert v;av.append(flag)
     else:av += [flag,str(v)]
    assert resolve(p,av)==a and a['epochs']==ep+1
    diff={k:[v,a[k]] for k,v in old['args'].items() if v!=a[k]}
    assert set(diff)==({'name'} if stage=='parent' else {'name','init'})
    endpoint=f'artifacts/runs/{name}/model_ep{ep}.pt'
    job=dict(name=name,stage=stage,fold=fold,arm=arm,argv=['.venv/bin/python',rel+'/train.py',*av],resolved_args=a,endpoint=endpoint,expected_n_in=211,timeout_seconds=1800 if stage=='parent' else 360,require_absent=[f'artifacts/runs/{name}'],required_outputs=[endpoint,f'artifacts/runs/{name}/atom-receipt.json',f'artifacts/runs/{name}/history.json'],metadata_reference=old['path'],historical_checkpoint_sha256=old['sha256'],inherited_argument_changes=diff,new_parser_defaults={k:v for k,v in a.items() if k not in old['args']},scientific_stage=f'{ep+1} epochs, fixed ep{ep}')
    jobs.append(job)
   oof=f'{rel}/cache/pred_{head}_valid.npy'
   jobs.append(dict(name=head,stage='OOF',fold=fold,arm=arm,argv=['.venv/bin/python',rel+'/predict_fixed.py',head,'--fold',str(fold),'--epoch','0','--batch','16','--output-dir',rel+'/cache'],timeout_seconds=300,require_absent=[oof,oof.replace('.npy','.npy.partial'),oof.replace('.npy','.inputs.json')],required_outputs=[oof,oof.replace('.npy','.inputs.json')]))
 sources=[*HERE.glob('*.py'),HERE/'contract.md',HERE/'original-metadata.json',ROOT/'scripts/train_rnn.py',*sorted((ROOT/'src/wnn').glob('*.py'))]
 result=dict(status='SOURCE_ONLY_FIXED_ORIGINAL_VS_CORRECTED_ATOM',execution_authorized=False,source_sha256={str(x.relative_to(ROOT)):sha(x) for x in sources},counts=dict(fits=8,OOF=4),jobs=jobs,environment={'PYTHONPATH':'src','WNN_THREADS':'4','WNN_LOADER_WORKERS':'2','WNN_LOADER_RAM_GB':'4','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','ARROW_NUM_THREADS':'4','WNN_GPU_DUTY':'1.0'},note='H2 stage label is generic runner bookkeeping; actual head epochs1, fixed ep0. No empirical allocation.')
 (HERE/'commands.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(status=result['status'],jobs=len(jobs),fits=8,OOF=4,sha256=sha(HERE/'commands.json'))))
if __name__=='__main__':main()

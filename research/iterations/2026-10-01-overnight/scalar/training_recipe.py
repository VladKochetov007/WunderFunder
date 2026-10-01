"""Parser-only fixed recipe checks; no dataset/model construction."""
from pathlib import Path
import argparse,ast,datetime,hashlib,json
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def parsed(argv):
 tree=ast.parse((P.parent/'training/train_aux.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');nodes=[]
 for node in main.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in node.targets):break
  nodes.append(node)
 namespace={'argparse':argparse};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<scalar-parser-only>','exec'),namespace)
 args=namespace['ap'].parse_args(argv)
 if args.fast:args.sync_free=args.fused_adam=args.pin_memory=args.prop_cache=True
 return vars(args)
def verify_job(job,name):
 args=parsed(job['base_argv']);assert args==job['parsed_args'] and args['name']==name
 assert args['seed']==job['seed'] and args['valid_fold']==job['fold']
 assert args['full_valid'] and args['fast'] and args['n_valid_folds']==4
 assert args['arch']=='gru' and args['enc']=='linear' and args['cell']=='gru' and args['hidden']==128 and args['layers']==2 and args['dropout']==0
 assert args['aux_struct']==args['aux_next']==.3 and args['aux_loss_scale']==1 and args['precision']=='fp16'
 assert args['valid_fold'] in [0,1,-1] and args['seed'] in [3,4]
 if job['stage']=='parent':
  assert args['train_source']=='train+valid' and args['batch']==32 and args['chunk']==2000 and args['epochs']==4 and args['loss']=='wmse' and args['lr']==.001
  assert args['ema_teacher']==1 and args['ema_decay']==.999 and args['eval_ema'] and args['since_weight']==32 and args['since_floor']==.25
  assert args['propensity']==f"prop_f{max(0,args['valid_fold'])}" and args['prop_power']==2 and args['prop_eps']==.03 and args['valid_true_mask']==.03
  assert not args['init'] and not args['freeze']
 else:
  assert args['train_source']=='valid' and args['batch']==16 and args['epochs']==2 and args['chunk']==2000 and args['loss']=='gcorr' and args['lr']==.0005 and args['mask_weight']
  assert not args['propensity'] and args['ema_teacher']==0 and not args['eval_ema'] and args['since_weight']==0
  assert args['freeze']==('inp,gru,scalar' if job['arm']=='ple' else 'inp,gru')
  assert args['init']==job['parent_run']+'@ep3'
 if job['fold']>=0:assert args['eval_source']=='vfold' and not args['include_heldout_valid']
 else:assert args['eval_source']=='holdout' and args['include_heldout_valid'] and args['seed']==3
 return args
def enforce_cutoff(job):
 if job['stage']!='parent':return
 cutoff=datetime.datetime(2026,10,1,6 if job['fold']>=0 else 7,tzinfo=datetime.timezone.utc)
 assert datetime.datetime.now(datetime.timezone.utc)<cutoff,'fixed long-parent start cutoff exceeded'
def verify_input_snapshot(allocation):
 entry=allocation['input_identity'];path=ROOT/entry['receipt_path'];assert sha(path)==entry['receipt_sha256'];receipt=json.loads(path.read_text());assert receipt['status']=='PASS'
 for rel,expected in receipt['files'].items():
  if not rel.startswith(('datasets/','cache/','artifacts/runs/prop_')):continue
  f=ROOT/rel;stat=f.stat();assert stat.st_size==expected['bytes'] and stat.st_mtime_ns==expected['mtime_ns'],rel
  if stat.st_size<2_000_000:assert sha(f)==expected['sha256'],rel
 return receipt

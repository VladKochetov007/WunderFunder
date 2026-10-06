"""Isolated factory adapter; immutable train_aux owns all losses and update clocks."""
import argparse,hashlib,importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
TRAIN_SHA='f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'
class Parser(argparse.ArgumentParser):
 def parse_args(self,*a,**kw):
  self.add_argument('--gradient-horizon',type=int,choices=(500,2000),required=True)
  args=super().parse_args(*a,**kw)
  if not args.name.startswith('oct02_temporal_'):self.error('Fresh oct02_temporal_ run name required')
  if args.arch!='gru' or args.enc!='linear' or args.cell!='gru' or args.layers!=1 or args.hidden!=256 or args.dropout or args.class_heads:self.error('Only current ordinary shallow256 GRU')
  if args.chunk!=2000 or args.batch!=32 or args.epochs!=4 or args.seed!=3 or args.freeze or args.init:self.error('Only parent4/seed3/batch32/chunk2000 from initialization')
  if args.valid_fold not in (0,1) or args.n_valid_folds!=4:self.error('Fixed four-way chain folds0/1 only')
  if (ROOT/'artifacts/runs'/args.name).exists():self.error('Refuse existing run')
  return args
def load_trainer():
 p=ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==TRAIN_SHA
 spec=importlib.util.spec_from_file_location('temporal_fixed_clock_trainer',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.argparse=SimpleNamespace(**{**vars(argparse),'ArgumentParser':Parser})
 def factory(config,n_in,n_out):
  from temporal import SegmentedRNN
  assert config['arch']=='gru' and config['enc']=='linear' and config['cell']=='gru' and not config['class_heads'] and config['hidden']==256 and config['layers']==1 and config['dropout']==0
  assert n_in==211 and n_out==23 and config['gradient_horizon'] in (500,2000)
  config['temporal_source_sha256']=hashlib.sha256((HERE/'temporal.py').read_bytes()).hexdigest()
  return SegmentedRNN(n_in,config['hidden'],config['layers'],config['dropout'],n_out=n_out,gradient_horizon=config['gradient_horizon'])
 m.make_model=factory;return m
if __name__=='__main__':load_trainer().main()

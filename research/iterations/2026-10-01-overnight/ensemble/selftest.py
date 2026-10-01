"""Synthetic ENSHEAD01 checks; CPU only, no competition data or scores."""
from __future__ import annotations
import hashlib,json,sys,tempfile,types
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from support import ScaleModel,PartnerCache,export_state,sha
from wnn.nn import RNNModel,weighted_pooled_corr_loss
import train_ensemble as trainer


def gradient_checks():
 torch.manual_seed(893);p=torch.randn(3,19,2,requires_grad=True);y=torch.randn_like(p);v=torch.rand(3,19)>.2
 target=weighted_pooled_corr_loss(p,y,v);plain=torch.autograd.grad(target,p,retain_graph=True)[0]
 ensemble=4*weighted_pooled_corr_loss((p+3*p.detach())/4,y,v);actual=torch.autograd.grad(ensemble,p)[0]
 torch.testing.assert_close(actual,plain,rtol=2e-6,atol=2e-8)
 own=torch.randn(2,23,2,requires_grad=True);partners=torch.randn_like(own,requires_grad=True);labels=torch.randn_like(own);mask=torch.rand(2,23)>.3
 loss=4*weighted_pooled_corr_loss((own+partners.detach())/4,labels,mask);loss.backward()
 assert partners.grad is None and own.grad.abs().sum()>0
 clipped_y=labels.clamp(-2,2);weights=clipped_y.abs()*mask[...,None];q=(own.detach()+partners.detach())/4
 scores=[]
 for k in range(2):
  w=weights[...,k];a=clipped_y[...,k];b=q[...,k];total=w.sum();ma=(w*a).sum()/total;mb=(w*b).sum()/total
  cov=(w*(a-ma)*(b-mb)).sum()/total;va=(w*(a-ma)**2).sum()/total;vb=(w*(b-mb)**2).sum()/total
  scores.append(cov/torch.sqrt(va*vb+1e-12))
 torch.testing.assert_close(loss.detach(),4*(1-torch.stack(scores).mean()),rtol=2e-6,atol=2e-7)
 return {'identical_partner_limit_matches_standalone_gradients':True,'partners_stop_gradient':True,'weighted_masked_target_clipped_unclipped_ensemble_formula':True,'identical_limit_max_abs':float(abs(actual-plain).max())}


def scale_checks():
 torch.manual_seed(943);base=RNNModel(11,hidden=7,layers=2,n_out=23);rng=torch.get_rng_state().clone()
 torch.manual_seed(943);model=ScaleModel(11,hidden=7,layers=2,n_out=23);assert torch.equal(rng,torch.get_rng_state());model.load_state_dict(base.state_dict(),strict=True)
 
 for prm in base.parameters():prm.requires_grad_(False)
 x=torch.randn(2,21,11);y=torch.randn(2,21,2);v=torch.ones(2,21,dtype=torch.bool);partners=torch.randn(2,21,2)
 want,state=base(x);actual,other=model(x);assert torch.equal(actual,want) and torch.equal(state,other)
 params=[(n,p) for n,p in model.named_parameters() if p.requires_grad];assert [(n,tuple(p.shape)) for n,p in params]==[('ensemble_log_scale',(2,))]
 before={k:t.clone() for k,t in model.state_dict().items() if k!='ensemble_log_scale'}
 opt=torch.optim.AdamW([p for _,p in params],lr=.0005,weight_decay=.01)
 opt.zero_grad(set_to_none=True);out,_=model(x);loss=4*weighted_pooled_corr_loss((out[...,:2]+partners)/4,y,v);loss.backward();assert model.ensemble_log_scale.grad.abs().sum()>0;opt.step()
 assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())
 with torch.no_grad():model.ensemble_log_scale.copy_(torch.tensor([.18,-.11]))
 folded=RNNModel(11,hidden=7,layers=2,n_out=23);folded.load_state_dict(export_state(model),strict=True)
 with torch.no_grad():a,ha=model(x);b,hb=folded(x)
 torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7);torch.testing.assert_close(ha,hb,rtol=1e-6,atol=1e-7)
 assert set(export_state(model))==set(base.state_dict())
 return {'initial_prediction_state_and_RNG_exact':True,'two_logscale_parameters_only_offsets_fixed_zero':True,'base_entirely_frozen_after_AdamW':True,'export_strict_plain_model_load':True,'export_float32_max_abs':float((a-b).abs().max()),'native_fp16_parity':'requires separate real validation before packaging'}


def cache_checks():
 with tempfile.TemporaryDirectory(prefix='enshead-selftest-') as td:
  root=Path(td);cache=root/'cache';cache.mkdir();dest=root/'research/iterations/2026-10-01-overnight/ensemble/cache';dest.mkdir(parents=True)
  groups=np.array([10,20,21]);np.savez(cache/'valid_targets.npz',row_group=groups);np.savez(cache/'chain_folds.npz',split=np.array(['valid']*3),row_group=groups,fold=np.array([0,1,1]))
  entries=[]
  for index in range(3):
   checkpoint=root/f'checkpoint{index}.pt';checkpoint.write_bytes(f'synthetic{index}'.encode());pred=dest/f'partner{index}.npy';values=np.full((3,20000,2),np.nan,np.float32);values[1:]=index+1;np.save(pred,values)
   entries.append({'run':f'control{index}','fold':0,'epoch':1,'role':'eligible_training_rows','prediction_path':str(pred),'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'prediction_sha256':sha(pred),'row_groups':[20,21],'rows':[1,2],'source_row_group_order_sha256':hashlib.sha256(groups.tobytes()).hexdigest(),'chain_folds_sha256':sha(cache/'chain_folds.npz')})
  meta={'schema':'enshead01-partners-v1','fold':0,'partners':entries,'source_row_group_order_sha256':hashlib.sha256(groups.tobytes()).hexdigest(),'chain_folds_sha256':sha(cache/'chain_folds.npz')};manifest=dest/'manifest.json';manifest.write_text(json.dumps(meta));obj=PartnerCache(manifest,root,0)
  assert np.array_equal(obj.sum_for_group(20),np.full((20000,2),6,np.float32))
  try:obj.sum_for_group(10)
  except ValueError:pass
  else:raise AssertionError('held row not rejected')
  for key,badvalue in [('fold',1),('epoch',0),('role','held_out')]:
   bad=json.loads(json.dumps(meta));bad['partners'][0][key]=badvalue;manifest.write_text(json.dumps(bad))
   try:PartnerCache(manifest,root,0)
   except ValueError:pass
   else:raise AssertionError(f'{key} mismatch accepted')
  bad=json.loads(json.dumps(meta));bad['partners'][0]['row_groups']=[10,20];manifest.write_text(json.dumps(bad))
  try:PartnerCache(manifest,root,0)
  except ValueError:pass
  else:raise AssertionError('wrong training set accepted')
  manifest.write_text(json.dumps(meta));Path(entries[0]['checkpoint']).write_bytes(b'changed')
  try:PartnerCache(manifest,root,0)
  except ValueError:pass
  else:raise AssertionError('changed checkpoint accepted')
 return {'same_fold_epoch1_three_partner_sum':True,'held_training_row_rejected':True,'wrong_fold_epoch_role_rows_checkpoint_rejected':True}


def dataset_and_source_checks():
 class Cache:
  def sum_for_group(self,g):assert g==73;return np.full((5,2),4,np.float32)
 class Reader:
  def read(self,g):assert g==73;return types.SimpleNamespace(features=np.zeros((5,112),np.float32),targets=np.full((5,2),-1,np.float32),scored=np.array([1,0,1,1,0],bool))
 trainer.PARTNER_CACHE=Cache();data=trainer.Sequences('valid',[73]);data._file=Reader();x,y,m=data[0]
 assert x.shape==(5,112) and torch.equal(y[:,:2],torch.full((5,2),-1.)) and torch.equal(y[:,2:],torch.full((5,2),4.))
 assert m.tolist()==[True,False,True,True,False]
 source=json.loads((HERE/'trainer-source.json').read_text());text=(ROOT/source['base']).read_text();assert hashlib.sha256(text.encode()).hexdigest()==source['base_sha256']
 for edit in source['changes']:
  assert text.count(edit['before'])==edit['count'];text=text.replace(edit['before'],edit['after'])
 assert text==(HERE/'train_ensemble.py').read_text()
 return {'dataset_targets_unchanged_partner_channels_separate':True,'scoring_mask_unchanged':True,'declared_source_transform_exact':True}


def main():
 torch.set_num_threads(2);result={'status':'PASS','gradients':gradient_checks(),'scale':scale_checks(),'cache':cache_checks(),'source':dataset_and_source_checks(),'empirical_data_read':False,'GPU_jobs':0,'WP_queries':0,'code_sha256':{p.name:sha(p) for p in sorted(HERE.glob('*.py'))}}
 print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()

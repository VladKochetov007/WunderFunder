"""Synthetic-only complete trainer main: original loss, AMP, fused AdamW, schedule."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import torch
import train_ctx as trainer
from wnn.nn import RNNModel
from torch.utils.data import Dataset
from recipe import plan,HERE


class SyntheticSequences(Dataset):
    def __init__(self,split,indices,*args,**kwargs):self.indices=list(indices)
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        rng=np.random.default_rng(713000+int(self.indices[index]))
        x=rng.normal(0,.2,(20000,112)).astype(np.float32)
        y=(rng.integers(-3,4,(20000,2)).astype(np.float32)*np.float32(.12543))
        m=np.ones(20000,dtype=bool);m[:99]=False
        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(m)


def main():
    started=time.monotonic();assert torch.cuda.is_available();torch.set_num_threads(4)
    frozen=plan();jobs=[j for j in frozen['jobs'] if j['seed']==3 and j['fold']==0];assert len(jobs)==2
    torch.manual_seed(7329);parent=RNNModel(211,128,2,0,n_out=23)
    initial={key:value.detach().clone() for key,value in parent.state_dict().items()};records=[]
    original_load=np.load;original_adam=torch.optim.AdamW;original_scaler=torch.amp.GradScaler
    # Replace only filesystem/dataset endpoints. Real model, loss, autocast, optimizer,
    # scaler, clipping, state detach, schedule and epoch serialization are unchanged.
    with tempfile.TemporaryDirectory(prefix='ctx-synthetic-wrapper-') as tmp:
        trainer.ROOT=Path(tmp)
        trainer.Sequences=SyntheticSequences
        trainer.valid_row_groups=lambda path:(list(range(36)),[])
        trainer.dev_halves=lambda path:(list(range(18)),list(range(18,36)))
        trainer.dataloader_budget=lambda *args,**kwargs:(0,None)
        trainer.verify_inputs=lambda args:None
        def load(path,*args,**kwargs):
            assert Path(path)==Path(tmp)/'cache/chain_folds.npz','Unexpected data read'
            return {'split':np.array(['valid']*36),'row_group':np.arange(36),'fold':np.array([0]*4+[1]*32)}
        np.load=load
        def forbidden(*args,**kwargs):raise AssertionError('Validation/metric path executed')
        trainer.evaluate=trainer.global_wp=trainer.score=trainer.block_scores=forbidden
        for job in jobs:
            target=Path(tmp)/job['parent_checkpoint'];target.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'model':initial},target)
            optimizers=[];scalers=[]
            class RecordingAdam(original_adam):
                def __init__(self,*args,**kwargs):
                    assert kwargs.get('fused') is True
                    super().__init__(*args,**kwargs);optimizers.append(self)
            class RecordingScaler(original_scaler):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs);assert self.is_enabled();scalers.append(self)
            torch.optim.AdamW=RecordingAdam;torch.amp.GradScaler=RecordingScaler
            sys.argv=job['train_argv'][1:]
            trainer.main()
            torch.cuda.synchronize();assert len(optimizers)==len(scalers)==1
            state=optimizers[0].state_dict();steps=[int(s['step']) for s in state['state'].values()]
            assert steps and 0<min(steps)==max(steps)<=40
            assert all(torch.isfinite(value).all() for s in state['state'].values() for value in s.values() if torch.is_tensor(value))
            run=Path(tmp)/'artifacts/runs'/job['name'];history=json.loads((run/'history.json').read_text())
            assert len(history)==2 and all(r['evaluation_skipped'] for r in history)
            assert all('weighted_pearson' not in r for r in history)
            ck=torch.load(run/'model_ep1.pt',map_location='cpu',weights_only=False)
            final=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
            assert ck['epoch']==final['epoch']==1
            assert all(torch.equal(value,final['model'][key]) for key,value in ck['model'].items())
            assert all(torch.equal(value,ck['model'][key]) for key,value in initial.items() if not key.startswith('head.'))
            assert torch.count_nonzero(ck['model']['head_extra.weight'])>0
            assert all(torch.isfinite(value).all() for value in ck['model'].values())
            assert ck['args']['fused_adam'] and ck['args']['pin_memory'] and ck['args']['precision']=='fp16'
            if job['arm']=='current':assert torch.equal(ck['model']['head_extra.weight'][:,:16],ck['model']['head_extra.weight'][:,16:])
            records.append({'arm':job['arm'],'synthetic_sequences':32,'rows_per_sequence':20000,'epochs':2,
                            'optimizer_step_attempts':40,'successful_optimizer_steps_each_parameter':min(steps),
                            'overflow_skipped_steps':40-min(steps),'fused_AdamW':True,'GradScaler_enabled':True,
                            'scaler_scale_final':scalers[0].get_scale(),'all_finite':True,
                            'frozen_backbone_normalization_bitexact':True,'extra_learned':True,
                            'fixed_ep1_equals_model_pt':True,'evaluation_never_called':True,
                            'updated_model_digest':hashlib.sha256(b''.join(v.numpy().tobytes() for v in ck['model'].values())).hexdigest()})
            del ck,final,optimizers,scalers;torch.cuda.empty_cache()
    np.load=original_load;torch.optim.AdamW=original_adam;torch.amp.GradScaler=original_scaler
    result={'status':'PASS','synthetic_only':True,'real_input_or_checkpoint_reads':False,'forecast_fits':0,'WP_queries':0,
            'trainer_sha256':hashlib.sha256((HERE/'train_ctx.py').read_bytes()).hexdigest(),
            'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records,
            'seconds':time.monotonic()-started,'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    output=HERE/'synthetic-wrapper-result.json'
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Inspect existing generated probe inputs and loss-side labels, no model execution."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
import torch
torch.set_num_threads(1)
P=Path(__file__).resolve().parent.parent;ROOT=P.parents[2]
sys.path[:0]=[str(ROOT/'src')]
spec=importlib.util.spec_from_file_location('ordinary',P/'training/train_aux.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
from wnn.outcomes import structure_labels,sequence_class
from wnn.features import FEATURE_NAMES
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(t):
 t=t.detach().cpu();v=t.double()
 return {'shape':list(t.shape),'finite':bool(torch.isfinite(t).all()),'min':float(v.min()),'max':float(v.max()),'nonfinite_count':int((~torch.isfinite(t)).sum())}
fixture=P/'scalar/generated/gpu_fixture.npz'
with np.load(fixture,allow_pickle=False) as z:
 raw=torch.from_numpy(z['raw'].copy()).unsqueeze(0)
 y=torch.from_numpy(z['targets'].copy())[torch.arange(32)%4]
 scored=torch.from_numpy(z['mask'].copy()).repeat(32,1);scored[:16]=False
 prob=torch.from_numpy(z['probability'].copy()).repeat(32,1)
inputs=trainer.build_inputs(raw)
# Feature builder acts independently per sequence; repeated fixture has 32 identical sequences.
full=inputs.repeat(32,1,1);sample=full.reshape(-1,211)[::7]
mean=sample.mean(0);std=sample.std(0).clamp_min(1e-3)
normalized=(inputs-mean)/std
steps=torch.arange(20000)[None].expand(32,-1);valid=steps>=99
weight=trainer.true_mask_weights(valid.float()*(prob.square()+.03),valid,scored,.03,True)
since=torch.expm1(full[...,112+FEATURE_NAMES.index('since_i0px')].float())
weight=weight*(1/(1+since/32)).clamp_min(.25)
sp,branch,cls=structure_labels(y,sequence_class(y))
nx_rows,nx_up,nx_ok=trainer.next_change_labels(y,valid);nw=weight*nx_ok.float()
arrays={'raw':raw,'inputs':inputs,'normalization_mean':mean,'normalization_std':std,'normalized_inputs':normalized,'targets':y,'main_weights':weight,'since':since,'structure_labels':sp,'branch_labels':branch,'class_labels':cls,'next_rows':nx_rows,'next_up':nx_up,'next_weights':nw}
report={name:stats(value) for name,value in arrays.items()}
assert all(v['finite'] for v in report.values())
chunks=[]
for start in range(0,20000,2000):
 sl=slice(start,start+2000)
 # These are denominators and targets only: no model, objective or optimizer evaluation.
 chunks.append({'start':start,'main_weight_sum':float(weight[:,sl].sum()),'main_target_abs_weight_sum':float((y[:,sl].clamp(-2,2).abs()*weight[:,sl,None]).sum()),'next_weight_sum':float(nw[:,sl].sum()),'next_standardized_label_min':float(((nx_rows[:,sl]-3)/2).min()),'next_standardized_label_max':float(((nx_rows[:,sl]-3)/2).max())})
out={'status':'GENERATED_INPUTS_AND_LOSS_LABELS_FINITE','checker_sha256':sha(__file__),'fixture_sha256':sha(fixture),'ordinary_trainer_sha256':sha(P/'training/train_aux.py'),'inspected_arrays':report,'chunk_denominators':chunks,'normalization_clamped_coordinate_count':int((std==1e-3).sum()),'feature_names':[f'raw{i}' for i in range(112)]+list(FEATURE_NAMES),'normalized_coordinate_abs_max':normalized.abs().amax((0,1)).tolist(),'model_forward_calls':0,'model_backward_calls':0,'optimizer_steps':0,'GPU_initialized':torch.cuda.is_initialized(),'empirical_data_reads':0,'forecast_fits':0,'WP_queries':0,'limitation':'CPU generated feature/normalizer/label inspection does not reconstruct the missing CUDA loss, gradient, step index or scaler state.'}
(P/'critic/scalar-probe-failure-generated-inputs.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['feature_names','normalized_coordinate_abs_max']}))

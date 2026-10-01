"""Generated-weight native/Python integration check, with true FP16 weight policy."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch
ap=argparse.ArgumentParser();ap.add_argument('--actor',type=Path,required=True);ap.add_argument('--native',type=Path,required=True);ap.add_argument('--mode',choices=['baseline','ctx'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--baseline',type=Path);a=ap.parse_args()
assert not a.output.exists();torch.set_num_threads(2);torch.set_num_interop_threads(1)
sys.path.insert(0,str(a.actor.resolve()));from model import ContextMemoryModel
from memory import metadata
from wnn.features import event_features
sys.path.insert(0,str(a.native.resolve()));import wnn_cpp
torch.manual_seed(185927);model=ContextMemoryModel(arm='memory').eval();state={k:v.detach().clone() for k,v in model.state_dict().items()}
gen=torch.Generator().manual_seed(731579);state['head_extra.weight']=torch.randn(state['head_extra.weight'].shape,generator=gen)*.03
rng=np.random.default_rng(271357);raw=rng.normal(0,.3,(701,112)).astype(np.float32);raw[:2]=0;raw[29:139]=raw[28];raw[340:400,0]=raw[339,0];raw[411:512,11]=raw[410,11]
inputs=torch.from_numpy(np.concatenate([raw,event_features(raw)],-1)[None])
plain={k:v.numpy() for k,v in state.items() if k not in ['head_extra.weight','ctx_projection']}
outputs={};checks=[]
for half in [False,True]:
    label='half' if half else 'float'
    if a.mode=='baseline':
        outputs[label]=wnn_cpp.Predictor(plain,half_weights=half).run(raw)
        continue
    old=np.load(a.baseline,allow_pickle=False)[label]
    no_config=wnn_cpp.Predictor(plain,half_weights=half).run(raw);np.testing.assert_array_equal(no_config,old)
    for arm in ['memory','current']:
        zero=dict(plain);zero['ctx_projection']=state['ctx_projection'].numpy();zero['head_extra.weight']=np.zeros((128,32),np.float32)
        zero_pred=wnn_cpp.Predictor(zero,half_weights=half,feature_config=metadata(arm)).run(raw);np.testing.assert_array_equal(zero_pred,old)
        ref_state={k:v.clone() for k,v in state.items()}
        if half:
            for k in ref_state:
                if k.startswith('gru.weight_') or k in ['head.0.weight','head_extra.weight']:ref_state[k]=ref_state[k].half().float()
        ref_model=ContextMemoryModel(arm=arm).eval();ref_model.load_state_dict(ref_state)
        with torch.no_grad():expected=ref_model(inputs)[0][0,:,:2].numpy()
        engine=wnn_cpp.Predictor({k:v.numpy() for k,v in state.items()},half_weights=half,feature_config=metadata(arm))
        actual=engine.run(raw);assert np.isfinite(actual).all();np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4)
        engine.reset();stream=np.stack([engine.step(row) for row in raw]);np.testing.assert_array_equal(actual,stream)
        np.testing.assert_array_equal(engine.run(raw[:137]),actual[:137]);changed=raw.copy();changed[137:]=.5;np.testing.assert_array_equal(engine.run(changed)[:137],actual[:137])
        engine.run(raw[:22]);np.testing.assert_array_equal(engine.run(raw),actual)
        outputs[label+'_'+arm]=actual
        checks.append({'precision':label,'arm':arm,'rows':len(raw),'nonzero_extra_native_vs_python_max_abs':float(np.max(np.abs(actual-expected))),'zero_extra_vs_frozen_old_native':'exact','disabled_ctx_vs_frozen_old_native':'exact','stream_reset_prefix_future':'exact','half_policy':'GRU/head0/extra matrices half then float; projection/input/head2/bias remain float' if half else 'float32'})
np.savez(a.output.with_suffix('.npz'),**outputs)
result={'status':'PASS','mode':a.mode,'synthetic_only':True,'generated_models':True,'empirical_rows':0,'checkpoint_reads':0,'WP_queries':0,'native_sha256':hashlib.sha256(Path(wnn_cpp.__file__).read_bytes()).hexdigest(),'checks':checks,'output_sha256':hashlib.sha256(a.output.with_suffix('.npz').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'actor_sources':{n:hashlib.sha256((a.actor/n).read_bytes()).hexdigest() for n in ['model.py','memory.py']}}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

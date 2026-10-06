from pathlib import Path
import json,numpy as np,torch
from linear_baseline import LinearBaseline
from wnn.data import SequenceFile,FEATURE_COLUMNS
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave42';torch.set_num_threads(2)
c=torch.load(R/'artifacts/runs/w13base_direct/model_ep0.pt',weights_only=False,map_location='cpu');m=LinearBaseline(211,128,2,n_out=2).eval();m.load_state_dict(c['model'])
x=SequenceFile(R/'datasets/train.parquet').read(0,targets=False,mask=False).features[:2000];inputs=torch.cat([torch.tensor(x)[None],torch.zeros(1,2000,99)],-1)
with torch.no_grad():p,_=m(inputs)
np.savez_compressed(D/'train-probe.npz',x=x,reference=p[0].numpy(),row_group=0)
(D/'scope.json').write_text(json.dumps({'cause':'Same positional train feature-reader error as wave33; only wave13/precision-report train panel consumed those oldnative traincaches. Wave13 main/scale reports andwave18 mixture reports usevalidonly. GPU baselineFT/composite decisions useorgbase/w13frozen correctly loaded. Distillationtraining usesGPUorgbase caches, notthese nativecachefiles.','affected':['pred_w13native_{fp32,half,int8,weight8,oldhalf}_train.npy','wave13/precision-report.json trainpanel'],'repair':'Explicitnamedfeatures, canonicalSequenceFile equality andTorchmodelparity, preservebadversions andrerun.'},indent=2))

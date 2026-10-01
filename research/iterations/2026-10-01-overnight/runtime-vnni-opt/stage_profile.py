import argparse,json,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--native',required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,a.native);import wnn_cpp
if a.output.exists():raise FileExistsError(a.output)
spec=json.loads((a.package/'members.json').read_text());weights=[dict(np.load(a.package/z['weights']))for z in spec];x=np.ascontiguousarray(np.load(a.raw,mmap_mode='r')[0,:10000]);result={}
for quant in (False,True):
 e=wnn_cpp.Ensemble(weights,[.25]*4,half_weights=True,dynamic_int8=quant);e.run(x[:300]);wnn_cpp.profile_reset();t=time.perf_counter();e.run(x);elapsed=time.perf_counter()-t;totals=wnn_cpp.profile();result['dynamic' if quant else 'fp16']={'total_us':elapsed/len(x)*1e6,'stage_us_per_fullrow':dict(zip(['features','projection','projection_gelu','recurrent_matrices','recurrent_update','head_matrix','head_gelu_final'],[z/len(x)*1e6 for z in totals]))}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

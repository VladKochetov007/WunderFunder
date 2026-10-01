"""Frozen delivered baseline on only critic-generated synthetic fixture."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--fixture',type=Path,required=True);ap.add_argument('--native',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
sys.path.insert(0,str(a.native.resolve()));import wnn_cpp
digest=hashlib.sha256(Path(wnn_cpp.__file__).read_bytes()).hexdigest();assert digest=='dd6e59b2d75518ee06fcbe3ba49f8450b3ee83695fa5692e4b8edaf86d7c8bad'
with np.load(a.fixture,allow_pickle=False) as z:
    raw=z['raw'];weights={k:np.ascontiguousarray(z[k]) for k in z.files if k!='raw'}
assert raw.shape==(701,112)
out={label:wnn_cpp.Predictor(weights,half_weights=half).run(raw) for label,half in [('float',False),('half',True)]}
np.savez(a.output.with_suffix('.npz'),**out)
result={'status':'PASS','native_sha256':digest,'rows':701,'fixture_sha256':hashlib.sha256(a.fixture.read_bytes()).hexdigest(),'generated_inputs_only':True,'empirical_rows':0,'checkpoint_reads':0,'output_sha256':hashlib.sha256(a.output.with_suffix('.npz').read_bytes()).hexdigest()}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

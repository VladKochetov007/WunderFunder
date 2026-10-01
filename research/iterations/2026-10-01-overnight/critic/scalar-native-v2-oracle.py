"""Generated literal checks for the one no-contract/output-restrict derivative."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
P=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--level',choices=['v2','avx2','v3'],required=True);a=ap.parse_args()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('blind',P/'critic/scalar-spec-oracle.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
sys.path.insert(0,str(P/'scalar/native_v2/build'/a.level));import wnn_cpp
D=P/'critic/scalar-independent-forward-fixture'
with np.load(D/'weights.npz',allow_pickle=False) as z:base={k:v.copy() for k,v in z.items()}
raw=np.load(D/'raw.npy');expected=np.load(D/'expected.npy')
observed=wnn_cpp.scalar_features(base,raw)
assert np.array_equal(observed.view(np.uint32),expected.view(np.uint32))
cases=['97_frozen_boundary_rows_bitexact']
indices=list(range(22))+list(range(52,74))
for name in ['contraction_witness','ordered_feature_sum']:
 weights={k:v.copy() for k,v in base.items()};weights['scalar.knots'].fill(0);weights['scalar.table'].fill(0)
 x=np.zeros((1,112),np.float32)
 if name=='contraction_witness':
  weights['scalar.counts'].fill(2);weights['scalar.knots'][:,1:]=1
  witness=oracle.run_generated_checks()['observations']['fma_witness']
  weights['scalar.table'][0,0,:]=witness['a'];weights['scalar.table'][0,1,:]=witness['b'];x[0,0]=witness['fraction']
 else:
  weights['scalar.counts'].fill(1)
  for j,value in enumerate([2**24,1,-2**24,1]+[0]*40):weights['scalar.table'][j,:,:]=value
 target=np.asarray([oracle.encode(x[0,indices].tolist(),weights['scalar.knots'].tolist(),weights['scalar.counts'].tolist(),weights['scalar.table'].tolist())[0]],np.float32)
 got=wnn_cpp.scalar_features(weights,x)
 assert np.array_equal(got.view(np.uint32),target.view(np.uint32)),name
 cases.append(name+'_bitexact')
result={'status':'PASS','level':a.level,'native_sha256':sha(wnn_cpp.__file__),'header_sha256':sha(P/'scalar/native_v2/include/wnn/scalar_embedding.hpp'),'checker_sha256':sha(__file__),'oracle_sha256':sha(P/'critic/scalar-spec-oracle.py'),'checks':cases,'generated_only':True,'empirical_data_reads':0,'GPU_used':False,'timing_claimed':False,'forecast_fits':0,'WP_queries':0}
(P/f'critic/scalar-native-v2-oracle-{a.level}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

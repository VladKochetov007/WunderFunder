"""Bind source-blind literal oracle to generated Torch/native scalar implementations."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

P=Path(__file__).resolve().parent.parent; ROOT=P.parents[2]
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['torch','v2','avx2','v3'],required=True);a=ap.parse_args()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('blind',P/'critic/scalar-spec-oracle.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
D=P/'critic/scalar-independent-forward-fixture';D.mkdir(exist_ok=True)
result={'mode':a.mode,'generated_only':True,'empirical_data_reads':0,'GPU_used':False,'WP_queries':0,'checker_sha256':sha(__file__),'blind_oracle_sha256':sha(P/'critic/scalar-spec-oracle.py')}
if a.mode=='torch':
 import torch
 torch.set_num_threads(1)
 sys.path[:0]=[str(P/'scalar'),str(ROOT/'src')]
 from scalar_model import ScalarRNNModel, RAW_INDICES
 sm=oracle.SMIN;fm=oracle.FMAX
 templates=[[3.]*16,[-2.,0.,1.,4.],[0.,sm,2*sm],[-fm,fm],[-fm,0.],[0.,fm],[1.,oracle.from_bits(0x3f800001),oracle.from_bits(0x3f800002)],[-1.+i/8 for i in range(16)]]
 kc=[oracle.canonical_knots(templates[j%len(templates)]) for j in range(44)]
 knots=np.asarray([v[0] for v in kc],np.float32);counts=np.asarray([v[1] for v in kc],np.int64)
 torch.manual_seed(919)
 model=ScalarRNNModel(211,knots=knots,counts=counts)
 rng=np.random.default_rng(7129);table=rng.uniform(-100,100,size=(44,16,16)).astype(np.float32)
 # Explicit product/add contraction witness is embedded in every nonconstant feature.
 witness=oracle.run_generated_checks()['observations']['fma_witness']
 for j in range(44):
  if counts[j]>1:table[j,0,0]=witness['a'];table[j,1,0]=witness['b']
 with torch.no_grad():model.scalar.table.copy_(torch.from_numpy(table))
 raw=rng.normal(size=(97,112)).astype(np.float32)
 for j,col in enumerate(RAW_INDICES):
  k=knots[j,:counts[j]]
  candidates=[-fm,fm,-sm,sm,0.,-0.]+[float(v) for v in k]
  for v in k:
   candidates.extend([float(np.nextafter(v,np.float32(-np.inf))),float(np.nextafter(v,np.float32(np.inf)))])
  candidates=[v for v in candidates if np.isfinite(v)]
  for row in range(len(raw)):raw[row,col]=candidates[row%len(candidates)]
 expected=np.asarray([oracle.encode(row[list(RAW_INDICES)].tolist(),knots.tolist(),counts.tolist(),table.tolist())[0] for row in raw],np.float32)
 inputs=np.zeros((len(raw),211),np.float32);inputs[:,:112]=raw
 with torch.no_grad():observed=model.scalar.aggregated(torch.from_numpy(inputs)).numpy()
 assert np.array_equal(observed.view(np.uint32),expected.view(np.uint32)),np.max(np.abs(observed-expected))
 np.savez(D/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
 np.save(D/'raw.npy',raw);np.save(D/'expected.npy',expected)
 # Check all grad-enabled / no-grad and frozen-table mode combinations return the same values.
 assert np.array_equal(model.scalar.aggregated(torch.from_numpy(inputs)).detach().numpy().view(np.uint32),expected.view(np.uint32))
 model.scalar.table.requires_grad_(False)
 assert np.array_equal(model.scalar.aggregated(torch.from_numpy(inputs)).detach().numpy().view(np.uint32),expected.view(np.uint32))
 result.update(scalar_source_sha256=sha(P/'scalar/scalar_model.py'),rows=len(raw),bitexact_literal_oracle=True,all_grad_modes_same=True)
else:
 sys.path.insert(0,str(P/'scalar/native/build'/a.mode));import wnn_cpp
 with np.load(D/'weights.npz',allow_pickle=False) as z:weights=dict(z)
 raw=np.load(D/'raw.npy');expected=np.load(D/'expected.npy')
 observed=wnn_cpp.scalar_features(weights,raw)
 assert np.array_equal(observed.view(np.uint32),expected.view(np.uint32)),np.max(np.abs(observed-expected))
 # Nonfinite inputs fail before interpolation.
 for bad in [np.nan,np.inf,-np.inf]:
  broken=raw[:1].copy();broken[0,0]=bad
  try:wnn_cpp.scalar_features(weights,broken)
  except ValueError:pass
  else:raise AssertionError('nonfinite raw accepted')
 result.update(rows=len(raw),bitexact_literal_oracle=True,native_sha256=sha(wnn_cpp.__file__),native_header_sha256=sha(P/'scalar/native/include/wnn/scalar_embedding.hpp'))
result.update(status='PASS',fixture_sha256={p.name:sha(p) for p in sorted(D.iterdir())})
out=P/f'critic/scalar-independent-forward-{a.mode}.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

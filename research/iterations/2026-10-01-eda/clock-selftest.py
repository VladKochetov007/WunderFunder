"""Synthetic correctness checks; never reads competition arrays or fitted artifacts."""
import collections, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('clock_diagnostic',root/'clock-diagnostic.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
k=((0.,),(1.,));u=((8.,),(9.,));union=[k,u]
trans={k:collections.Counter({(0,0):4,(1,0):3,(2,0):3})}
cc=[collections.Counter({(k,(0,0)):4,(k,(1,0)):3}),collections.Counter({(k,(2,0)):3})]
model=dict(trans=trans,chain_counts=cc)
w=np.ones((m.BOOT,2));w[0]=0;w[1]=[3,4];w[2]=[0,2]
b=m.boot_state(model,w,union,{k:0,u:1})
assert not b['quote'][3:,0].any(), 'must choose signed mode0, not aggregate nonzero majority6/10'
assert not b['known'][0].any(), 'zero-total bootstrap edge must be absent'
assert b['ties'][1,0] and not b['quote'][1,0], '12/12 tie must keep first-observed zero mode'
assert b['quote'][2,0], 'signed nonzero modal move must map to quoteTrue'
base=m.state({k:(0,0)},union)
counts=np.array([[[2,5],[1,4]]]);totals=np.array([[25,15,11]])
r=m.metric_arrays(counts,totals,base,b)
assert r['xor_per_row'][0].shape==(m.BOOT,) and r['xor_per_row'][1].shape==(1,)
assert r['xor_per_row'][0][0]==5, 'absent lookup must fall back to observed proxy'
assert r['xor_per_row'][0][2]==7, 'known zero versus known nonzero differs on every row of that edge'
assert m.ci(*r['xor_per_row']) is not None
assert m.ci(np.array([0,0]),np.array([0])) is None
# All-unknown identity, with code events absent from dictionary and observed proxy count retained.
empty=m.state({},union);rr=m.metric_arrays(counts,totals,empty,empty)
assert rr['xor_per_code_event'][0][0]==0 and rr['xor_per_code_event'][1][0]==15
assert rr['xor_per_union_update'][1][0]==11
x=np.zeros((6,112),np.float32);x[1:3,:22]=1;x[3,:22]=2;x[4:,:22]=3
keys=m.codes(x);table={(keys[0],keys[1]):(0,0),(keys[2],keys[3]):(1,0)}
expected=np.array([False,False,False,True,True,False])
assert np.array_equal(m.mask(x,table),expected)
for chunk in (1,2,3,6):assert np.array_equal(m.stream_mask(x,table,chunk),expected)
for cut in range(1,7):assert np.array_equal(m.mask(x[:cut],table),expected[:cut])
# Strictly interior update counting: quote endpoints must be excluded.
x=np.zeros((6,112),np.float32);x[1,22]=1;x[2,22]=2;x[3,22]=3;x[4,22]=4
iv=m.intervals(x,np.array([False,True,False,False,True,False]))
assert iv['closed_intervals']==1 and iv['blocks']['i0_volume_code']['updates']==2
report=dict(status='passed',synthetic_only=True,competition_data_reads=0,checks=['signed-mode-before-binary-vs-binary-majority','deterministic-modal-tie','zero-total-bootstrap-fallback','single-denominator-to1000-bootstrap-broadcast','zero-denominator-null','unknown-dictionary-union-counts','causal-prefix-stream-chunk','strict-interior-interval-counts'],diagnostic_sha256=hashlib.sha256((root/'clock-diagnostic.py').read_bytes()).hexdigest())
(root/'clock-selftest-result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))

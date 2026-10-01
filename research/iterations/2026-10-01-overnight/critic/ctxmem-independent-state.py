"""Independent literal double recurrence and matrix oracle; synthetic inputs only."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--actor',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--native',type=Path);a=ap.parse_args()
assert not a.output.exists();sys.path.insert(0,str(a.actor.resolve()))
import memory
p=memory.fixed_projection()
correction=json.loads((a.actor.parent/'ctxmem-projection-correction.json').read_text())
rows=sorted(range(128),key=lambda i:hashlib.sha256((correction['seed_ascii']+':'+str(i)).encode()).digest())[:16]
scale=np.float32(1/np.sqrt(128))
literal=np.asarray([[scale*(-1 if bin(r&c).count('1')%2 else 1) for c in range(128)] for r in rows],np.float32)
np.testing.assert_array_equal(p,literal)
assert memory.LAMBDA_FAST.hex()=='0x1.ae89f995ad3adp-1'
assert memory.LAMBDA_SLOW.hex()=='0x1.f50765b6e4540p-1'
rng=np.random.default_rng(135679);q=rng.normal(size=(3,8193,16)).astype(np.float32)
q[1]*=np.float32(1e20);q[2]*=np.float32(1e-20)
event=np.zeros((3,8193),bool);event[0,::13]=True;event[1]=True;event[2,[1,2,127,128,129,8192]]=True
# Row0 event is deliberately true; initialization still must not decay q0.
fast=np.zeros((3,16),np.float64);slow=fast.copy();expected=np.empty((3,8193,32),np.float32)
for b in range(3):
    for t in range(8193):
        for k in range(16):
            v=float(q[b,t,k])
            if t==0:fast[b,k]=slow[b,k]=v
            elif event[b,t]:
                fast[b,k]=float(memory.LAMBDA_FAST)*float(fast[b,k])+(1-float(memory.LAMBDA_FAST))*v
                slow[b,k]=float(memory.LAMBDA_SLOW)*float(slow[b,k])+(1-float(memory.LAMBDA_SLOW))*v
        expected[b,t,:16]=fast[b];expected[b,t,16:]=slow[b]
full,state=memory.event_histories(q,event)
np.testing.assert_array_equal(full,expected);np.testing.assert_array_equal(state[0],fast);np.testing.assert_array_equal(state[1],slow)
carry=None;parts=[];start=0
for stop in [0,1,2,2,3,127,128,129,129,1000,4001,8192,8193,8193]:
    out,carry=memory.event_histories(q[:,start:stop],event[:,start:stop],carry);parts.append(out);start=stop
np.testing.assert_array_equal(np.concatenate(parts,1),expected)
for x,y in zip(carry,state):np.testing.assert_array_equal(x,y)
# Different batch initialization states must not reinitialize already-live streams.
old=(np.full((3,16),.125,np.float64),np.full((3,16),-.25,np.float64),np.asarray([True,False,True]))
saved=tuple(v.copy() for v in old);single,single_state=memory.event_histories(q[:,:1],np.zeros((3,1),bool),old)
np.testing.assert_array_equal(single[0,0],np.r_[saved[0][0],saved[1][0]].astype(np.float32));np.testing.assert_array_equal(single[1,0],np.r_[q[1,0],q[1,0]])
for x,y in zip(old,saved):np.testing.assert_array_equal(x,y)
native=None
if a.native:
    sys.path.insert(0,str(a.native.resolve()));import wnn_cpp
    for arm in ['memory','current']:
        for b in range(3):
            eng=wnn_cpp.ContextMemory(memory.metadata(arm));got=np.stack([eng.update(v,bool(e)) for v,e in zip(q[b],event[b])]);ref=expected[b] if arm=='memory' else np.concatenate([q[b],q[b]],axis=-1)
            np.testing.assert_array_equal(got,ref);eng.reset();np.testing.assert_array_equal(eng.update(q[b,0],True),ref[0])
    native={'sha256':hashlib.sha256(Path(wnn_cpp.__file__).read_bytes()).hexdigest(),'supplied_q_rows':3*8193,'arms':['memory','current'],'outputs_and_reset':'bitexact'}
result={'status':'PASS','scope':'synthetic only; literal separate Python double multiplication/addition oracle, no model/target/data read','matrix_sha256':hashlib.sha256(p.tobytes()).hexdigest(),'memory_source_sha256':hashlib.sha256((a.actor/'memory.py').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows_per_stream':8193,'streams':3,'float32_outputs_bitexact':True,'internal_float64_final_states_bitexact':True,'arbitrary_empty_chunk_carry_bitexact':True,'mixed_initialization_and_no_input_state_mutation':True,'native':native}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

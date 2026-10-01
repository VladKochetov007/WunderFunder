"""Fixed generated-only FP32 parity/cost check. Never loads source data/checkpoints."""
from pathlib import Path
import ctypes as C
import hashlib
import json
import os
import resource
import time
import numpy as np
import torch
from torch_reference import MinimalRNN, blocked_scan, streaming_states

P = Path(__file__).resolve().parent
OUT = P / 'generated-01'
FP = C.POINTER(C.c_float)
checks = []
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ptr(a):
    assert a.dtype == np.float32 and a.flags.c_contiguous
    return a.ctypes.data_as(FP)


def close(name, actual, expected, atol=2e-5, rtol=2e-5):
    a, b = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(a).all() and np.isfinite(b).all(), name
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=name)
    checks.append(dict(name=name, max_abs=float(np.abs(a.astype('float64')-b).max()), atol=atol, rtol=rtol))


def pack(model):
    sd = model.state_dict()
    names = ['mean', 'std', 'inp.weight', 'inp.bias', 'head.0.weight', 'head.0.bias', 'head.2.weight', 'head.2.bias']
    for i in range(model.layers):
        names += [f'cells.{i}.{p}.{q}' for p in ['gate', 'candidate'] for q in ['weight', 'bias']]
    return np.concatenate([sd[k].numpy().reshape(-1) for k in names]).astype('float32')


class Native:
    def __init__(self, lib, model):
        self.lib = lib
        self.h, self.l, self.o = model.hidden, model.layers, model.head[2].out_features
        flat = pack(model)
        self.handle = lib.min_create(model.mean.numel(), self.h, self.l, self.o, ptr(flat), flat.size)
        assert self.handle, 'native rejected generated model'

    def reset(self, state=None):
        a = None if state is None else np.ascontiguousarray(state.reshape(self.l, self.h), dtype='float32')
        assert self.lib.min_set_state(self.handle, None if a is None else ptr(a)) == 0

    def state(self):
        a = np.empty((self.l, 1, self.h), dtype='float32')
        assert self.lib.min_get_state(self.handle, ptr(a)) == 0
        return a

    def run(self, x):
        x = np.ascontiguousarray(x, dtype='float32')
        y = np.empty((len(x), self.o), dtype='float32')
        assert self.lib.min_run(self.handle, ptr(x), len(x), ptr(y)) == 0
        return y

    def destroy(self):
        self.lib.min_destroy(self.handle)


def main():
    start = time.perf_counter()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert list(os.sched_getaffinity(0)) == [34]
    cg = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split(':', 2)[2].lstrip('/')
    limits = {k:(cg/k).read_text().strip() for k in ['cpu.max', 'memory.max', 'memory.swap.max']}
    assert limits == {'cpu.max':'200000 100000', 'memory.max':str(4*(1<<30)), 'memory.swap.max':'0'}, limits
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2043)
    lib = C.CDLL(str(P/'libminimal_fp32.so'))
    lib.min_create.argtypes = [C.c_int]*4+[FP,C.c_longlong]; lib.min_create.restype = C.c_void_p
    lib.min_destroy.argtypes = [C.c_void_p]; lib.min_destroy.restype = None
    for name in ['min_set_state', 'min_get_state']:
        getattr(lib,name).argtypes = [C.c_void_p,FP]; getattr(lib,name).restype = C.c_int
    lib.min_run.argtypes=[C.c_void_p,FP,C.c_int,FP]; lib.min_run.restype=C.c_int
    lib.min_recur.argtypes=[FP,FP,FP,C.c_int,C.c_int,FP];lib.min_recur.restype=C.c_int
    model = MinimalRNN(n_out=23).eval()
    model.mean.copy_(.2*torch.randn(211))
    model.std.copy_(.5+torch.rand(211))
    x = torch.randn(1,2000,211)
    h0 = torch.rand(2,1,256)*2+.1
    np.savez(OUT/'generated-inputs.npz', x=x.numpy(), positive_h0=h0.numpy())
    np.savez(OUT/'generated-model23.npz', **{k:v.numpy() for k,v in model.state_dict().items()})
    two = MinimalRNN(n_out=2).eval()
    two.load_state_dict({k:(v[:2].clone() if k.startswith('head.2.') else v) for k,v in model.state_dict().items()})
    np.savez(OUT/'generated-model2.npz', **{k:v.numpy() for k,v in two.state_dict().items()})
    extreme = MinimalRNN(n_out=23).eval();extreme.load_state_dict(model.state_dict())
    levels = torch.tensor([-80.,-20.,0.,20.,80.]).repeat(52)[:256]
    with torch.no_grad():
        for cell in extreme.cells:
            cell.gate.weight.zero_();cell.gate.bias.copy_(levels)
            cell.candidate.weight.zero_();cell.candidate.bias.copy_(levels)
    np.savez(OUT/'generated-extreme23.npz', **{k:v.numpy() for k,v in extreme.state_dict().items()})
    sources = ['minimal_native.cpp','kernels.hpp','activations.hpp','torch_reference.py','check_native.py','plan.json','libminimal_fp32.so']
    inventory={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in [*[P/f for f in sources],*sorted(OUT.glob('generated-*.npz'))]}
    (OUT/'inputs-before-check.json').write_text(json.dumps(inventory,indent=2)+'\n')
    rows=x[0].numpy()
    outputs={}; torch_seconds={}
    with torch.inference_mode():
        for label,m in [('head23',model),('head2',two),('extreme23',extreme)]:
            n=Native(lib,m)
            for init_label,initial in [('zero',torch.zeros_like(h0)),('positive',h0)]:
                tag=label+'_'+init_label
                n.reset(initial.numpy());native=n.run(rows);end=n.state()
                begin=time.perf_counter();step_rows=[];carry=initial.clone()
                for row in x.unbind(1):
                    out,carry=m.step(row,carry);step_rows.append(out)
                torch_seconds[tag]=time.perf_counter()-begin
                streaming=torch.stack(step_rows,1)[0].numpy()
                full,last=m(x,initial)
                tol=2e-4 if label=='extreme23' else 2e-5
                close(tag+'_step_outputs',native,streaming,atol=tol)
                close(tag+'_step_state',end,carry.numpy(),atol=tol)
                close(tag+'_forward_outputs',native,full[0].numpy(),atol=tol)
                close(tag+'_forward_state',end,last.numpy(),atol=tol)
                n.reset(initial.numpy());pieces=[]
                stops=[0,1,63,64,65,511,999,2000]
                for lo,hi in zip(stops[:-1],stops[1:]): pieces.append(n.run(rows[lo:hi]))
                close(tag+'_irregular_chunks',np.concatenate(pieces),native,0,0)
                close(tag+'_irregular_state',n.state(),end,0,0)
                n.reset(initial.numpy());prefix=n.run(rows[:511]);prefix_state=n.state()
                n.reset(initial.numpy());changed=rows.copy();changed[511:]+=10
                future=n.run(changed)
                close(tag+'_future_prefix',future[:511],prefix,0,0)
                _,pt=m(x[:,:511],initial);close(tag+'_prefix_state',prefix_state,pt.numpy(),atol=tol)
                n.reset(initial.numpy());close(tag+'_reset_replay',n.run(rows),native,0,0)
                outputs[tag]=native
            # Actual h0 is copied literally, including exact zero; no g(h0).
            n.reset(h0.numpy());close(label+'_actual_initial',n.state(),h0.numpy(),0,0)
            before=n.state()
            for bad in [np.full_like(before,-1),np.full_like(before,np.nan),np.full_like(before,np.inf)]:
                assert lib.min_set_state(n.handle,ptr(bad)) == -2
                close(label+'_rejected_state_unchanged',n.state(),before,0,0)
            badx=rows[:1].copy();badx[0,0]=np.nan;y=np.empty((1,n.o),dtype='float32')
            assert lib.min_run(n.handle,ptr(badx),1,ptr(y)) == -2
            close(label+'_rejected_input_unchanged',n.state(),before,0,0)
            n.reset();close(label+'_zero_reset',n.state(),np.zeros_like(before),0,0)
            close(label+'_null_reset_outputs',n.run(rows),outputs[label+'_zero'],0,0)
            n.destroy()
        for tag in ['zero','positive']:
            close('export2_matches_23_'+tag,outputs['head2_'+tag],outputs['head23_'+tag][:,:2],0,0)
        k=torch.tensor([-80.,-20.,0.,20.,80.]).repeat(1,2000,1)
        v=torch.tensor([-80.,-8.,0.,8.,80.]).repeat(1,2000,1)
        initial=torch.tensor([[0.,.2,.5,2.,20.]])
        a=np.empty((2000,5),dtype='float32')
        assert lib.min_recur(ptr(k.numpy()),ptr(v.numpy()),ptr(initial.numpy()),2000,5,ptr(a))==0
        literal,_=streaming_states(k,v,initial);scan,_=blocked_scan(k,v,initial);double,_=streaming_states(k.double(),v.double(),initial.double())
        for name,want in [('literal',literal),('scan',scan),('double',double)]:close('extreme_recurrence_'+name,a,want[0].numpy(),atol=2e-4)
    n=Native(lib,two);n.reset();n.run(rows)
    native_times=[];callback_times=[]
    for _ in range(5):
        n.reset();t=time.perf_counter();n.run(rows);native_times.append(time.perf_counter()-t)
        n.reset();t=time.perf_counter()
        for row in rows: n.run(row[None])
        callback_times.append(time.perf_counter()-t)
    n.destroy()
    np.savez(OUT/'generated-native-outputs.npz', **outputs)
    assert sum(p.stat().st_size for p in P.rglob('*') if p.is_file()) <= 200*(1<<20)
    for name,rec in inventory.items():
        p=(P/name) if (P/name).exists() else OUT/name
        assert sha(p)==rec['sha256'], name
    result=dict(status='PASS_GENERATED_NATIVE_FP32_ONLY',checks=checks,seed=2043,sequence_length=2000,
        outputs=[23,2],hidden=256,layers=2,feature_width=211,parameters_dtype='float32',state_dtype='float32',
        copied_reference_sha256=sha(P/'torch_reference.py'),inputs_inventory_sha256=sha(OUT/'inputs-before-check.json'),
        native_library_sha256=sha(P/'libminimal_fp32.so'),torch_version=torch.__version__,CPU_affinity=[34],limits=limits,
        native_loop_2000_seconds=native_times,ctypes_row_loop_2000_seconds=callback_times,torch_row_loop_2000_seconds=torch_seconds,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.perf_counter()-start,
        GPU_used=False,real_data_or_trained_checkpoints_read=False,forecast_fit_or_metric=False,
        qualifications=['FP32 generated model only; no AMP equivalence or training qualification.',
        'Input is 211 engineered float32 features; native applies (x-mean)/std from generated model then exact-erf GELU.',
        'Native timing excludes raw112-to-211 feature extraction, competition callback/reset wrapper, other members and scorer ABI.',
        'One host CPU; generated dense weights/input, descriptive warm-cache cost only; no deadline extrapolation.',
        'Independent copied GEMV/activation source; no active or closed native engine modified.'])
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','seconds','native_loop_2000_seconds','ctypes_row_loop_2000_seconds','peak_RSS_KiB']}),flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        (OUT/'failure.json').write_text(json.dumps({'status':'FAILED_NO_MATH_RETRY','exception':repr(exc),'checks_completed':checks},indent=2)+'\n')
        raise

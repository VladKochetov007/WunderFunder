"""SOURCE-ONLY until root allocation: generated B32/T2000 FP16 interval smoke.

Two sequential arms, two carried chunks each, fixed synthetic ordinary warmstart.
No dataset/checkpoint reads, metrics, scientific fits, or alternative configurations.
External root scope must impose45s+termination grace, CPU200%, RAM4GiB/swap0,
and owned GPU NVML<=7168MiB. This script never treats foreign GPU use as its own.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, signal, sys, time
from datetime import datetime, timezone
from pathlib import Path
START=time.monotonic()
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
MODEL_SHA='7ef61881baf1ad0befe229ea087c958f9aad6aa65240c0b3d8580dd90015d241'
BATCH=32;CHUNK=2000;N_IN=211;N_OUT=23
ATOL=3e-3;RMSE_MAX=3e-4

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--seconds',type=int,default=45);ap.add_argument('--gpu-fraction',type=float,default=.375)
    ap.add_argument('--threads',type=int,default=2);a=ap.parse_args()
    assert a.seconds==45 and a.gpu_fraction==.375 and a.threads==2
    assert os.environ.get('CUDA_VISIBLE_DEVICES','') not in ('','-1')
    assert sha(HERE/'interval_memory.py')==MODEL_SHA
    out=a.output_dir.resolve();assert out.is_relative_to(HERE) and out!=HERE and not out.exists();out.mkdir()
    def timeout(signum,frame):raise TimeoutError('fixed45s generated budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(a.seconds)
    result=dict(status='RUNNING',started_utc=utc(),PID=os.getpid(),argv=sys.argv,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),HERE/'interval_memory.py',ROOT/'src/wnn/nn.py',ROOT/'src/wnn/resources.py')},
        shape=[BATCH,CHUNK,N_IN,N_OUT],arms=['accumulated','instantaneous'],chunks_per_arm=2,
        synthetic_optimizer_steps_per_arm=2,real_data=False,checkpoints_read=False,forecast_fits=0,metrics=0,
        CPU_affinity=sorted(os.sched_getaffinity(0)),limits=dict(seconds=45,threads=2,allocator_fraction=.375,external_owned_GPU_MiB=7168,external_RAM_bytes=4*2**30),
        parity_limits=dict(max_abs=ATOL,RMSE=RMSE_MAX,meaning='fixed initial zero-extra AMP compatibility; not a forecast-quality threshold'))
    write(out/'started.json',result)
    torch=None
    try:
        os.environ['WNN_THREADS']='2';os.environ['WNN_GPU_FRACTION']='0.375'
        sys.path[:0]=[str(HERE),str(ROOT/'src')]
        import torch as imported_torch
        torch=imported_torch
        from wnn.nn import RNNModel
        from wnn.resources import configure_torch
        from interval_memory import IntervalMemoryModel
        configure_torch(gpu_fraction=a.gpu_fraction)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        device=torch.device('cuda:0');props=torch.cuda.get_device_properties(device)
        assert props.total_memory*a.gpu_fraction<=6144*2**20
        result.update(device=dict(name=props.name,total_bytes=props.total_memory),torch_version=torch.__version__,
            cudnn_version=torch.backends.cudnn.version(),cudnn_benchmark=torch.backends.cudnn.benchmark,
            TF32_policy=dict(matmul=torch.backends.cuda.matmul.allow_tf32,cudnn=torch.backends.cudnn.allow_tf32))
        torch.manual_seed(2026100201)
        baseline=RNNModel(N_IN,256,1,0.,n_out=N_OUT)
        plain={k:v.detach().clone() for k,v in baseline.state_dict().items()}
        del baseline
        def fingerprint(state):
            h=hashlib.sha256()
            for k,v in sorted(state.items()):
                t=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
            return h.hexdigest()
        def generated(chunk):
            g=torch.Generator(device='cpu').manual_seed(2026100202+chunk)
            x=torch.randn(BATCH,CHUNK,N_IN,generator=g)*.3
            target=torch.randn(BATCH,CHUNK,N_OUT,generator=g)*.2
            row=torch.arange(CHUNK)[None]+chunk*CHUNK;b=torch.arange(BATCH)[:,None]
            events=(row+3*b)%(11+7*(b%7))==0
            events[0]=False
            if chunk==0:events[:,0]=False
            else:events[1,0]=True  # closes a carried interval on first row of chunk2
            x[...,178]=events.float()
            return x.to(device),target.to(device),events.sum(1).tolist()
        def compare(left,right):
            d=left.float()-right.float();record=dict(max_abs=float(d.abs().max()),RMSE=float(d.square().mean().sqrt()))
            assert record['max_abs']<=ATOL and record['RMSE']<=RMSE_MAX,record
            return record
        def detached(state):return tuple(x.detach() for x in state)
        def finite(state):return all(bool(torch.isfinite(v).all()) for v in state)
        # One ordinary reference forward in training mode. Drop its graph before
        # either arm so the resource test never holds two parent graphs at once.
        torch.cuda.reset_peak_memory_stats()
        ref=RNNModel(N_IN,256,1,0.,n_out=N_OUT);ref.load_state_dict(plain,strict=True);ref=ref.to(device).train()
        x,y,_=generated(0)
        with torch.autocast('cuda',dtype=torch.float16):pred,h=ref(x)
        ref_pred=pred.detach().cpu();ref_hidden=h.detach().cpu()
        del pred,h,ref,x,y;gc.collect();torch.cuda.synchronize()
        result['ordinary_reference_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['ordinary_reference_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        torch.cuda.empty_cache()
        result['arms_results']=[];initial_hash=None;first_pred=None
        for arm in ('accumulated','instantaneous'):
            torch.manual_seed(2026100203)
            model=IntervalMemoryModel(arm);model.load_plain_baseline(plain)
            digest=fingerprint(model.state_dict())
            if initial_hash is None:initial_hash=digest
            else:assert digest==initial_hash
            model=model.to(device).train();ema=__import__('copy').deepcopy(model).eval()
            optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.01,fused=True)
            scaler=torch.amp.GradScaler('cuda')
            carry=ema_carry=None;steps=[];torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
            for chunk in (0,1):
                x,target,event_counts=generated(chunk)
                assert carry is None if chunk==0 else len(carry)==4 and carry[3].dtype==torch.long
                with torch.autocast('cuda',dtype=torch.float16):
                    prediction,carry=model(x,carry)
                carry=detached(carry)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    teacher,ema_carry=ema(x,ema_carry)
                ema_carry=detached(ema_carry)
                assert prediction.shape==(BATCH,CHUNK,N_OUT) and finite((prediction,teacher,*carry,*ema_carry))
                row=dict(chunk=chunk,event_counts=event_counts,output_dtype=str(prediction.dtype),incoming_carried=chunk==1)
                if chunk==0:
                    p=prediction.detach().cpu();row['zero_extra_vs_plain']=compare(p,ref_pred)
                    row['row_hidden_vs_plain']=compare(carry[0].cpu(),ref_hidden)
                    row['student_vs_initial_EMA']=compare(prediction.detach(),teacher)
                    if first_pred is None:first_pred=p
                    else:row['paired_initial_output']=compare(p,first_pred)
                # Full23-output generated target plus own-EMA first2 consistency;
                # this is a resource/finite-gradient smoke, not a scientific loss.
                loss=(prediction.float()-target).square().mean()+(prediction[...,:2].float()-teacher[...,:2].float()).square().mean()
                optimizer.zero_grad(set_to_none=True);scale_before=scaler.get_scale()
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                gradients=[p.grad for p in model.parameters() if p.grad is not None]
                assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);assert bool(torch.isfinite(norm))
                scaler.step(optimizer);scaler.update()
                assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
                step_values={float(s['step']) for s in optimizer.state.values()}
                assert step_values=={float(chunk+1)},step_values
                with torch.no_grad():
                    for ep,p in zip(ema.parameters(),model.parameters()):ep.lerp_(p,.001)
                row.update(loss=float(loss.detach()),gradient_norm_before_clip=float(norm),gradient_tensors=len(gradients),
                    scaler_before=scale_before,scaler_after=scaler.get_scale(),optimizer_step_values=sorted(step_values),
                    carried_sum_dtype=str(carry[2].dtype),carried_count_max=int(carry[3].max()))
                steps.append(row)
                del x,target,prediction,teacher,loss,gradients,norm
            torch.cuda.synchronize()
            assert torch.equal(model.mean.cpu(),plain['mean']) and torch.equal(model.std.cpu(),plain['std'])
            assert bool(torch.count_nonzero(model.extra_weight)>0)
            model.freeze_trunk();trainable=[n for n,p in model.named_parameters() if p.requires_grad]
            assert set(trainable)=={'extra_weight','head.0.weight','head.0.bias','head.2.weight','head.2.bias'}
            rec=dict(arm=arm,seconds=time.monotonic()-begin,initial_state_sha256=digest,steps=steps,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                final_live_allocated_bytes=torch.cuda.memory_allocated(),H2_trainable=trainable,normalization_unchanged=True)
            result['arms_results'].append(rec);print(json.dumps(rec),flush=True)
            del optimizer,scaler,model,ema,carry,ema_carry
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        result.update(status='PASS_GENERATED_INTERVAL_CUDA',matched_synthetic_initial_state_sha256=initial_hash,
            peak_allocated_bytes=max([result['ordinary_reference_peak_allocated_bytes']]+[r['peak_allocated_bytes'] for r in result['arms_results']]),
            peak_reserved_bytes=max([result['ordinary_reference_peak_reserved_bytes']]+[r['peak_reserved_bytes'] for r in result['arms_results']]),
            limitations='Generated resource/AMP smoke only. No actual frozen trainer auxiliaries/loader, no forecast fit, no native latency or quality claim; external controller owns NVML/cgroup limits.')
        assert time.monotonic()-START<a.seconds
        assert sha(HERE/'interval_memory.py')==MODEL_SHA
    except BaseException as exc:
        result.update(status='FAILED_GENERATED_NO_RETRY',failure=repr(exc));raise
    finally:
        signal.alarm(0)
        result.update(seconds=time.monotonic()-START,finished_utc=utc())
        if torch is not None and torch.cuda.is_initialized():
            try:result['terminal_memory']=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
            except Exception as exc:result['terminal_memory_error']=repr(exc)
        write(out/'result.json',result);print(json.dumps(dict(status=result['status'],seconds=result['seconds'],failure=result.get('failure'))),flush=True)
if __name__=='__main__':main()

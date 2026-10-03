from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import logging
import resource
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
R2 = REPO/'research/campaign_2026-10-03/continuation/round2/usage'
sys.path.insert(0,str(R2))
from usage_core import I0, I1, cached_array_prefix, quiet_ages
sys.path.insert(0,str(REPO/'scripts'))
train = importlib.import_module('train_rnn')
ENGINEERED = I0[I0>=112]
CLOCK = np.array([186,187])
FLAGS = np.arange(178,183)
HELD = np.arange(189,211)

def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value,indent=2))

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
    return h.hexdigest()

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))

def parity(a: np.ndarray,b: np.ndarray) -> dict:
    return dict(max_abs=float(np.max(np.abs(a-b))),rms=rms(a-b),passed=bool(np.allclose(a,b,atol=2e-5,rtol=2e-5)))

def choose_donor(raw: np.ndarray,t: int,std: np.ndarray) -> dict:
    start,end=t-288,t-32
    candidates=[]
    for lag in [64,128,256]:
        if start-lag>=0:
            distance=rms((raw[start:end,:52]-raw[start-lag:end-lag,:52])/std[:52])
            candidates.append(dict(lag=lag,rms=distance))
    finite=[c for c in candidates if np.isfinite(c['rms'])]
    if not finite:
        return dict(query=t,start=start,end=end,lag=None,candidates=candidates)
    chosen=min(finite,key=lambda c:(c['rms'],c['lag']))
    return dict(query=t,start=start,end=end,lag=chosen['lag'],raw_displacement=chosen['rms'],candidates=candidates)

def windows(raw: np.ndarray,packed: np.ndarray,donor: dict) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    t,s,e,lag=donor['query'],donor['start'],donor['end'],donor['lag']
    altered=raw[:t+1].copy()
    altered[s:e,:52]=raw[s-lag:e-lag,:52]
    assert np.array_equal(altered[:s],raw[:s])
    assert np.array_equal(altered[e:],raw[e:t+1])
    assert np.array_equal(altered[:,52:],raw[:t+1,52:])
    recomputed=train.build_inputs(torch.from_numpy(altered)[None])[0].numpy()
    assert np.array_equal(recomputed[:s],packed[:s])
    assert np.array_equal(recomputed[:,I1],packed[:t+1,I1])
    assert np.array_equal(recomputed[:,104:112],packed[:t+1,104:112])
    direct=packed[s:t+1].copy()
    direct[:256,I0]=packed[s-lag:e-lag,I0]
    assert np.array_equal(direct[256:,:112],raw[e:t+1])
    assert np.array_equal(direct[:,:112],altered[s:])
    return altered,recomputed[s:],direct

def baseline(model: torch.nn.Module,inputs: torch.Tensor,queries: list[int]) -> tuple[np.ndarray,dict[int,torch.Tensor]]:
    n=inputs.shape[1]
    boundaries=sorted({0,n,*range(0,n,1000),*(t-288 for t in queries)})
    state=torch.zeros(2,1,256)
    states={0:state.clone()}
    pred=np.empty((n,2),dtype=np.float32)
    for s,e in zip(boundaries[:-1],boundaries[1:],strict=True):
        output,state=model(inputs[:,s:e],state)
        assert torch.isfinite(output).all() and torch.isfinite(state).all()
        pred[s:e]=output[0,:,:2].numpy()
        if e in {t-288 for t in queries}:
            states[e]=state.clone()
    return pred,states

def main(index: int) -> None:
    started=time.monotonic()
    frozen=json.loads((ROOT/'samples.json').read_text())
    assert sha(ROOT/'PREREG.json')==frozen['prereg_sha256']
    state=json.loads((ROOT/'STATE.json').read_text())
    assert sha(ROOT/'samples.json')==state['sample_sha256']
    sample=frozen['samples'][index]
    if (ROOT/f'seq_{index:02d}.json').exists():
        raise RuntimeError('Already completed; preserve artifacts, do not overwrite')
    control=frozen['control']
    for path,expected in control['source_sha256'].items():
        assert sha(REPO/path)==expected
    assert sha(REPO/control['checkpoint'])==control['checkpoint_sha256']
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    ckpt=torch.load(REPO/control['checkpoint'],map_location='cpu',weights_only=False)
    model=train.make_model(ckpt['args'],ckpt['n_in'],control['n_out'])
    model.load_state_dict(ckpt['model']); model.eval()
    reference_model=train.make_model(ckpt['args'],ckpt['n_in'],control['n_out'])
    reference_model.load_state_dict(ckpt['model']); reference_model.eval()
    std=np.maximum(model.std.numpy(),1e-6)
    p=pq.ParquetFile(REPO/'datasets/valid.parquet')
    names=p.schema_arrow.names
    features=names[names.index('i0_p0'):names.index('a7')+1]
    table=p.read_row_group(sample['row_group'],columns=features+['seq_ix','step_in_seq','need_prediction','is_scored'],use_threads=False)
    raw=np.column_stack([table[n].to_numpy() for n in features]).astype(np.float32)
    assert np.all(table['seq_ix'].to_numpy()==sample['seq_ix'])
    assert np.array_equal(table['step_in_seq'].to_numpy(),np.arange(20000))
    mask=table['need_prediction'].to_numpy()&table['is_scored'].to_numpy()
    qs=sample['queries']
    assert mask[qs].all() and np.sum(mask[512:])==sample['n_eligible']
    donors=[choose_donor(raw,t,std) for t in qs]
    # Donor choices are serialized before loading any targets/predictions.
    dump(ROOT/f'donors_{index:02d}.json',dict(sample=sample,donors=donors,raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
    target_table=p.read_row_group(sample['row_group'],columns=['t0','t1'],use_threads=False)
    targets=np.column_stack([target_table[n].to_numpy() for n in ['t0','t1']])
    assert np.array_equal(targets,cached_array_prefix('y',sample['cache_position'],20000))
    assert np.array_equal(mask,cached_array_prefix('mask',sample['cache_position'],20000))
    if not qs:
        dump(ROOT/f'seq_{index:02d}.json',dict(sample=sample,empty=True,seconds=time.monotonic()-started))
        return
    records=[]
    replay=np.empty((len(qs),3,289,2),dtype=np.float32)
    predictions=np.empty((len(qs),3,2),dtype=np.float32)
    with torch.inference_mode():
        inputs=train.build_inputs(torch.from_numpy(raw)[None])
        packed=inputs[0].numpy()
        reference=train.predict(reference_model,torch.from_numpy(raw)[None],4000,torch.device('cpu'))[0]
        base,states=baseline(model,inputs,qs)
        cpu_gate=parity(base,reference)
        assert cpu_gate['passed'],cpu_gate
        cached=np.load(REPO/'cache/pred_c20261001_d256_s3_f0_valid.npy',mmap_mode='r')[sample['cache_position']].copy()
        cache_gate=dict(max_abs=float(np.abs(reference-cached).max()),rms=rms(reference-cached),correlations=[float(np.corrcoef(reference[:,j],cached[:,j])[0,1]) for j in range(2)])
        cache_gate['r2_bound_passed']=bool(cache_gate['max_abs']<=.02 and cache_gate['rms']<=.005 and min(cache_gate['correlations'])>=.999)
        canary=None
        for qi,d in enumerate(donors):
            if d['lag'] is None:
                raise RuntimeError('No finite donor; record cannot-replay instead of replacement')
            t,s,e=d['query'],d['start'],d['end']
            altered,recomputed,direct=windows(raw,packed,d)
            identity=packed[s:t+1]
            for arm,w in enumerate([identity,recomputed,direct]):
                output,h=model(torch.from_numpy(w)[None],states[s].clone())
                assert torch.isfinite(output).all() and torch.isfinite(h).all()
                replay[qi,arm]=output[0,:,:2].numpy()
                predictions[qi,arm]=replay[qi,arm,-1]
            identity_gate=parity(replay[qi,0],base[s:t+1])
            assert identity_gate['passed'],identity_gate
            mismatch=(direct-recomputed)
            cur_delta=recomputed[-1]-identity[-1]
            seam_old=rms((raw[s,:52]-raw[s-1,:52])/std[:52])
            seam_new=rms((altered[s,:52]-altered[s-1,:52])/std[:52])
            seam_return=rms((altered[e,:52]-altered[e-1,:52])/std[:52])
            clock_diff=mismatch[:,CLOCK]
            records.append(dict(query=t,lag=d['lag'],raw_displacement=d['raw_displacement'],zero_displacement=d['raw_displacement']==0,identity=identity_gate,packed_inconsistent=bool(np.any(mismatch[:,ENGINEERED]!=0)),packed_engineered_mismatch_rms=rms(mismatch[:,ENGINEERED]/std[ENGINEERED]),packed_flags_mismatch_fraction=float(np.mean(mismatch[:,FLAGS]!=0)),packed_clock_mismatch_fraction=float(np.mean(clock_diff!=0)),packed_clock_max_abs=float(np.abs(clock_diff).max()),packed_held_mismatch_rms=rms(mismatch[:,HELD]/std[HELD]),current_raw_equal=True,current_engineered_changed=bool(np.any(cur_delta[ENGINEERED]!=0)),current_engineered_rms=rms(cur_delta[ENGINEERED]/std[ENGINEERED]),current_clock_max_abs=float(np.abs(cur_delta[CLOCK]).max()),protected_tail_engineered_rms=rms((recomputed[256:,ENGINEERED]-identity[256:,ENGINEERED])/std[ENGINEERED]),raw_recomputed_input_displacement=rms((recomputed-identity)[:,I0]/std[I0]),packed_input_displacement=rms((direct-identity)[:,I0]/std[I0]),seam_original=seam_old,seam_altered=seam_new,seam_return=seam_return))
            if qi==0:
                future=raw.copy()
                future[s:e,:52]=altered[s:e,:52]
                future[t+1:]=0
                changed_inputs=train.build_inputs(torch.from_numpy(future)[None])[0].numpy()
                assert np.array_equal(changed_inputs[:t+1],train.build_inputs(torch.from_numpy(altered)[None])[0].numpy())
                future_out,_=reference_model(torch.from_numpy(changed_inputs[s:t+1])[None],states[s].clone())
                canary=parity(future_out[0,:,:2].numpy(),replay[qi,1])
                assert canary['passed']
    boundaries=sorted(states)
    np.savez(ROOT/f'seq_{index:02d}.npz',raw=raw,packed=packed,baseline=base,reference=reference,cached=cached,targets=targets,mask=mask,queries=np.array(qs),predictions=predictions,replay=replay,quiet_age=quiet_ages(raw)[qs],state_boundaries=np.array(boundaries),original_states=np.stack([states[s].numpy() for s in boundaries]),std=std)
    result=dict(sample=sample,records=records,cpu_reference=cpu_gate,gpu_cache=cache_gate,future_canary=canary,unmatched=0,cannot_replay=0,alignment='IDs/steps/targets/masks exact; every original row and all warmup processed',seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,backend=dict(torch=torch.__version__,device='cpu',dtype='float32',threads=1))
    dump(ROOT/f'seq_{index:02d}.json',result)
    state=json.loads((ROOT/'STATE.json').read_text())
    state['completed_sequences']=sorted(set(state['completed_sequences']+[index]))
    state['phase']='replaying'
    dump(ROOT/'STATE.json',state)
    logging.info('sequence %s: %.2fs RSS %.1fMiB identity %.3g raw_response %.4g packed_response %.4g',index,result['seconds'],result['peak_rss_mib'],max(r['identity']['max_abs'] for r in records),rms(predictions[:,1]-base[qs]),rms(predictions[:,2]-base[qs]))

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    parser=argparse.ArgumentParser()
    parser.add_argument('--sequence',type=int,choices=range(24),required=True)
    arg=parser.parse_args()
    try:
        main(arg.sequence)
    except Exception:
        error=traceback.format_exc()
        (ROOT/f'failure_{arg.sequence:02d}.txt').write_text(error)
        state=json.loads((ROOT/'STATE.json').read_text()); state['phase']='failure';state['failure']=error
        dump(ROOT/'STATE.json',state)
        raise

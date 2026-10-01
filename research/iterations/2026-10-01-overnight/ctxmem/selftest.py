"""Synthetic arrays/models only. No checkpoint, dataset, cache, or forecast metric I/O."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from memory import (HADAMARD_ROWS, LAMBDA_FAST, LAMBDA_SLOW, PROJECTION_SHA256,
                    event_histories, fixed_projection, metadata, projection_sha, selected_features)


def scalar_oracle(q, event):
    result = np.empty((*q.shape[:2], 32), np.float32)
    for b in range(len(q)):
        fast = q[b, 0].astype(np.float64).copy()
        slow = fast.copy()
        for t in range(q.shape[1]):
            if t and event[b, t]:
                fast = (1-LAMBDA_FAST)*q[b,t].astype(np.float64) + LAMBDA_FAST*fast
                slow = (1-LAMBDA_SLOW)*q[b,t].astype(np.float64) + LAMBDA_SLOW*slow
            result[b,t] = np.r_[fast,slow]
    return result


def recurrence_checks():
    rng = np.random.default_rng(20261003)
    p = fixed_projection()
    assert projection_sha(p) == PROJECTION_SHA256
    assert len(np.unique(p,axis=1).T) == 128
    np.testing.assert_allclose(p.astype(np.float64) @ p.astype(np.float64).T, np.eye(16), atol=4e-8, rtol=0)
    q = rng.normal(size=(3,401,16)).astype(np.float32)
    event = rng.random((3,401)) < .13
    event[0] = False; event[1] = True
    got, state = event_histories(q,event)
    np.testing.assert_array_equal(got, scalar_oracle(q,event))
    for cut in [1,2,3,17,99,199,400,401]:
        np.testing.assert_array_equal(event_histories(q[:,:cut],event[:,:cut])[0],got[:,:cut])
    carried = None; chunks=[]; start=0
    for stop in [0,1,1,8,19,200,200,201,398,401]:
        out, carried = event_histories(q[:,start:stop],event[:,start:stop],carried)
        chunks.append(out); start=stop
    np.testing.assert_array_equal(np.concatenate(chunks,1),got)
    for a,b in zip(state,carried): np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(event_histories(q,event,None)[0],got)
    np.testing.assert_array_equal(got[0,:,:16], np.broadcast_to(q[0,0],got[0,:,:16].shape))
    np.testing.assert_array_equal(selected_features(q,got,'current'),np.concatenate([q,q],-1))
    # Two histories can differ for identical current q; current control cannot.
    assert np.max(np.abs(got[2,:,:16]-q[2])) > .1
    return q,event,got


def torch_checks(device):
    import torch
    from wnn.nn import RNNModel
    from model import ContextMemoryModel
    torch.set_num_threads(2)
    torch.manual_seed(4731)
    baseline = RNNModel(211,128,2,0,n_out=23)
    after_base = torch.get_rng_state().clone()
    torch.manual_seed(4731)
    memory = ContextMemoryModel(arm='memory')
    np.testing.assert_array_equal(torch.get_rng_state().numpy(),after_base.numpy())
    memory.load_state_dict(baseline.state_dict())
    control = ContextMemoryModel(arm='current'); control.load_state_dict(baseline.state_dict())
    for model in (baseline,memory,control):
        model.inp.requires_grad_(False); model.gru.requires_grad_(False); model.to(device)
    torch.manual_seed(5731)
    x = torch.randn(3,137,211,device=device)*.2
    x[...,178]=(torch.arange(137,device=device)%7==0).float()
    x[:,0,178]=0
    with torch.no_grad():
        ref,_=baseline(x)
        mem,_=memory(x);ctl,_=control(x)
    assert torch.equal(ref,mem) and torch.equal(ref,ctl), 'Zero extra map changes baseline outputs'
    # Exact original-head gradient equality at the zero-extension starting point.
    for model in (baseline,memory,control):
        model.zero_grad(set_to_none=True)
        out,_=model(x)
        out.square().mean().backward()
    for key,param in baseline.named_parameters():
        if key.startswith('head.'):
            assert torch.equal(param.grad,dict(memory.named_parameters())[key].grad)
            assert torch.equal(param.grad,dict(control.named_parameters())[key].grad)
    for model in (memory,control):
        assert all(p.grad is None for k,p in model.named_parameters() if k.startswith(('inp.','gru.')))
        assert model.head_extra.weight.grad is not None and model.head_extra.weight.grad.abs().max()>0
    # Duplicated q columns start symmetrically and receive identical gradients.
    assert torch.equal(control.head_extra.weight.grad[:,:16],control.head_extra.weight.grad[:,16:])
    with torch.no_grad():
        torch.manual_seed(6731)
        memory.head_extra.weight.copy_(torch.randn_like(memory.head_extra.weight)*.03)
        control.head_extra.weight.copy_(memory.head_extra.weight)
        full,full_state=memory(x)
        carried=None;pieces=[]
        for lo,hi in zip([0,1,17,50,136],[1,17,50,136,137]):
            out,carried=memory(x[:,lo:hi],carried);pieces.append(out)
        difference=float((torch.cat(pieces,1)-full).abs().max())
        torch.testing.assert_close(torch.cat(pieces,1),full,atol=2e-6,rtol=2e-6)
        repeat,_=memory(x,None);assert torch.equal(repeat,full)
        for cut in [1,17,50,136]:
            partial,_=memory(x[:,:cut]);torch.testing.assert_close(partial,full[:,:cut],atol=2e-6,rtol=2e-6)
        # Algebraic foldability of the equal-parameter current-q control.
        folded=RNNModel(211,128,2,0,n_out=23).to(device)
        folded.load_state_dict(baseline.state_dict())
        effective=(control.head_extra.weight[:,:16]+control.head_extra.weight[:,16:]) @ control.ctx_projection
        folded.head[0].weight[:,:128].add_(effective)
        a,_=folded(x);b,_=control(x)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
        fold_difference=float((a-b).abs().max())
    return {'zero_extra_outputs_bitexact':True,'zero_extra_original_head_gradients_bitexact':True,
            'constructor_rng_parity':True,'frozen_backbone_no_gradients':True,
            'current_control_duplicate_gradients_equal':True,'full_model_chunk_max_abs':difference,
            'current_control_fold_max_abs':fold_difference}


def native_checks(path,q,event,expected):
    sys.path.insert(0,str(path))
    import wnn_cpp
    outputs=[]
    for b in range(len(q)):
        engine=wnn_cpp.ContextMemory(metadata('memory'))
        got=np.stack([engine.update(row,bool(flag)) for row,flag in zip(q[b],event[b])])
        np.testing.assert_array_equal(got,expected[b]);engine.reset()
        repeat=np.stack([engine.update(row,bool(flag)) for row,flag in zip(q[b],event[b])])
        np.testing.assert_array_equal(repeat,expected[b]);outputs.append(got)
        ctl=wnn_cpp.ContextMemory(metadata('current'))
        current=np.stack([ctl.update(row,bool(flag)) for row,flag in zip(q[b],event[b])])
        np.testing.assert_array_equal(current,np.concatenate([q[b],q[b]],-1))
    return {'native_memory_identical_supplied_q_bitexact':True,'native_reset_repeat':True,
            'native_current_q_bitexact':True,'rows':int(np.prod(q.shape[:2]))}


def native_model_checks(path):
    import torch
    sys.path.insert(0,str(path))
    import wnn_cpp
    from model import ContextMemoryModel
    root=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(root/'scripts'))
    import train_rnn
    from wnn.nn import RNNModel
    torch.set_num_threads(2);torch.manual_seed(20261007)
    base=RNNModel(211,128,2,0,n_out=23).eval()
    rng=np.random.default_rng(20261008)
    raw=rng.normal(0,.3,(257,112)).astype(np.float32)
    raw[1:70]=raw[0];raw[91:143]=raw[90];raw[180:224]=raw[179]
    built=train_rnn.build_inputs(torch.from_numpy(raw[None]))
    base_weights={k:v.detach().numpy() for k,v in base.state_dict().items()}
    base_native=wnn_cpp.Predictor(base_weights,fast_activations=False,half_weights=False)
    base_output=base_native.run(raw)
    results=[]
    for arm in ('current','memory'):
        model=ContextMemoryModel(arm=arm).eval();model.load_state_dict(base.state_dict())
        weights={k:v.detach().numpy() for k,v in model.state_dict().items()}
        engine=wnn_cpp.Predictor(weights,fast_activations=False,half_weights=False,feature_config=metadata(arm))
        zero_output=engine.run(raw);np.testing.assert_array_equal(zero_output,base_output)
        with torch.no_grad():model.head_extra.weight.copy_(torch.from_numpy(rng.normal(0,.03,(128,32)).astype(np.float32)))
        weights={k:v.detach().numpy() for k,v in model.state_dict().items()}
        engine=wnn_cpp.Predictor(weights,fast_activations=False,half_weights=False,feature_config=metadata(arm))
        native=engine.run(raw)
        with torch.no_grad():reference=model(built)[0][0,:,:2].numpy()
        np.testing.assert_allclose(native,reference,atol=2e-5,rtol=2e-5)
        assert np.max(np.abs(native-base_output))>1e-5
        np.testing.assert_array_equal(engine.run(raw),native)
        for cut in (1,17,99,256):np.testing.assert_array_equal(engine.run(raw[:cut]),native[:cut])
        results.append({'arm':arm,'zero_extra_native_baseline_bitexact':True,
                        'nonzero_extra_native_vs_Python_FP32_max_abs':float(np.max(np.abs(native-reference))),
                        'prefix_reset_repeat_bitexact':True,'rows':len(raw)})
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--torch',action='store_true');parser.add_argument('--native',type=Path)
    args=parser.parse_args();start=time.monotonic();q,event,expected=recurrence_checks()
    result={'status':'PASS','synthetic_only':True,'projection_sha256':PROJECTION_SHA256,
            'ordered_rows':list(HADAMARD_ROWS),'recurrence_scalar_oracle_bitexact':True,
            'all_prefix_and_empty_chunk_carry_bitexact':True,'first_row_initialization_and_reset':True}
    if args.torch: result['torch']=torch_checks('cpu')
    if args.native:
        result['native']=native_checks(args.native,q,event,expected)
        if args.torch:result['native_full_models']=native_model_checks(args.native)
    result['seconds']=time.monotonic()-start
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()

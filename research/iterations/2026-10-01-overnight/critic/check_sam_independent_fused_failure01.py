"""Independent synthetic SAM recurrence/AMP/oracle audit; no competition inputs."""
from pathlib import Path
import argparse,ast,copy,hashlib,json,sys
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];SAM=HERE.parent/'sam'
sys.path[:0]=[str(ROOT/'src'),str(SAM)]
from sam_step import sam_backward,skip_evaluation_rng
from wnn.nn import RNNModel

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def assert_nested(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_nested(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested(x,y)
    else:assert a==b

def static_checks():
    source=(SAM/'train_sam.py').read_text();tree=ast.parse(source)
    loss=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='chunk_loss')
    baseline=ast.parse((ROOT/'scripts/train_rnn.py').read_text())
    # The copied closure's exact scientific statements equal the original contiguous loss statements.
    main=next(n for n in baseline.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    chunk=next(n for n in ast.walk(main) if isinstance(n,ast.For) and ast.unparse(n.target)=='s' and 'args.chunk' in ast.unparse(n.iter))
    start=next(i for i,n in enumerate(chunk.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='moments is not None')
    end=next(i for i,n in enumerate(chunk.body) if isinstance(n,ast.Expr) and ast.unparse(n)=='timer.mark(\'loss\')')
    expected=chunk.body[start:end]
    assert [ast.dump(n,include_attributes=False) for n in loss.body[1:-1]]==[ast.dump(n,include_attributes=False) for n in expected]
    closure=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='sam_closure')
    assert 'forward_chunk(model, inputs, sl, h_in)' in ast.unparse(closure)
    assert 'perturbed_out, _' in ast.unparse(closure)
    assert source.index('sam_backward(loss, sam_closure')<source.index('scaler.unscale_(opt)')<source.index('scaler.step(opt)')<source.index('ema_model.update_parameters(model)',source.index('sam_backward(loss, sam_closure'))
    assert 'res = evaluate(' not in source and 'weighted_pearson"] > best' not in source
    return ['exact AST loss body matches original','second pass uses same h_in and discards hidden output','restore helper precedes one unscale/AdamW/scaler/EMA path','no validation evaluate or metric-selected checkpoints']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    device=torch.device(a.device);torch.set_num_threads(2);torch.manual_seed(119037)
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(.10)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    checks=static_checks()
    original=RNNModel(7,hidden=16,layers=2,n_out=5).to(device)
    xs=[torch.randn(3,n,7,device=device) for n in [17,21,13]]
    ys=[torch.randn(3,x.shape[1],5,device=device) for x in xs]
    h0=torch.randn(2,3,16,device=device)
    def trajectory(mode,rho):
        m=copy.deepcopy(original);teacher=copy.deepcopy(original)
        for p in teacher.parameters():p.requires_grad_(False)
        opt=torch.optim.AdamW(m.parameters(),lr=7e-4,weight_decay=.01,fused=device.type=='cuda')
        scaler=torch.amp.GradScaler(device.type,init_scale=32.,growth_interval=2,enabled=True)
        state=h0.clone();last=[]
        for x,y in zip(xs,ys):
            incoming=state
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                value,state_plain=m(x,incoming)
            with torch.no_grad(),torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                frozen_teacher=teacher(x,incoming)[0]
            lossfn=lambda v:(v.float()-y).square().mean()+.3*(v.float()-frozen_teacher.float()).square().mean()
            loss=lossfn(value);opt.zero_grad(set_to_none=True)
            before={k:v.detach().clone() for k,v in m.state_dict().items()}
            def closure():
                assert torch.equal(incoming,state)
                with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    z,_=m(x,incoming)
                return lossfn(z)
            if mode=='baseline':scaler.scale(loss).backward()
            else:sam_backward(loss,closure,m.parameters(),scaler,rho,force_two_pass=mode=='forced')
            assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items()), 'perturbed weights leaked to optimizer'
            scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            scaler.step(opt);scaler.update()
            with torch.no_grad():
                for tp,mp in zip(teacher.parameters(),m.parameters()):tp.copy_(.9*tp+.1*mp)
            state=state_plain.detach();last.append(state.clone())
        return {'model':m.state_dict(),'teacher':teacher.state_dict(),'states':last,'optimizer':opt.state_dict(),'scaler':scaler.state_dict()}
    expected=trajectory('baseline',0.)
    for mode in ['sam','forced']:assert_nested(expected,trajectory(mode,0.))
    nonzero=trajectory('sam',.05)
    assert any(not torch.equal(expected['model'][k],v) for k,v in nonzero['model'].items())
    checks+=['rho0 exact single-pass and forced-two-pass model/AdamW/EMA/scaler/hidden trajectory','nonzero rho changes update and restores every tensor before optimizer']
    # Coupled quadratic has a closed-form gradient across two separate parameter tensors.
    p=torch.nn.Parameter(torch.tensor([.6,-.8],device=device));q=torch.nn.Parameter(torch.tensor([1.2],device=device))
    oldp=p.detach().clone();oldq=q.detach().clone()
    matrix=torch.tensor([[3.,.7,.1],[.7,2.,-.4],[.1,-.4,1.5]],device=device)
    target=torch.tensor([.2,-.3,.7],device=device)
    vec=torch.cat([oldp,oldq]);g=matrix@vec-target
    epsilon=(g.double()*(.05/g.double().norm())).float();expected_gradient=matrix@(vec+epsilon)-target
    def value():
        v=torch.cat([p,q]);return .5*(v@(matrix@v))-target@v
    scaler=torch.amp.GradScaler(device.type,init_scale=128.,enabled=True)
    sam_backward(value(),value,[p,q],scaler,.05)
    actual=torch.cat([p.grad,q.grad])/128.;error=float((actual-expected_gradient).abs().max().detach())
    assert error<5e-7 and torch.equal(p,oldp) and torch.equal(q,oldq)
    checks.append('globally normalized coupled quadratic gradient agrees with analytic oracle despite nonunit AMP scale')
    # Check first/second pass infinities and explicit exception with a real enabled GradScaler.
    for stage in ['first','second','exception']:
        v=torch.nn.Parameter(torch.tensor([.3,-.7],device=device));old=v.detach().clone();opt=torch.optim.AdamW([v],lr=.1,fused=device.type=='cuda')
        gs=torch.amp.GradScaler(device.type,init_scale=16.,enabled=True);called=[]
        def closure():
            called.append(True)
            if stage=='exception':raise RuntimeError('planted')
            return (v*float('inf')).sum()
        if stage=='exception':
            try:sam_backward(v.square().sum(),closure,[v],gs,.05)
            except RuntimeError as exc:assert str(exc)=='planted'
            else:raise AssertionError('exception disappeared')
        else:
            first=(v*float('inf')).sum() if stage=='first' else v.square().sum()
            result=sam_backward(first,closure,[v],gs,.05)
            assert bool(called)==(stage=='second')
            gs.unscale_(opt);torch.nn.utils.clip_grad_norm_([v],1.);gs.step(opt);gs.update()
            assert not opt.state and gs.get_scale()==8.
        assert torch.equal(v,old)
    checks.append('first/second AMP overflow skip without stale perturbation; exception restores exactly')
    result={'status':'PASS','device':a.device,'checks':checks,'analytic_gradient_error':error,'torch':torch.__version__,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,'source_sha256':{name:digest(SAM/name) for name in ['train_sam.py','sam_step.py']},'script_sha256':digest(__file__),'competition_data_reads':0,'forecasting_fits':0,'WP_queries':0}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

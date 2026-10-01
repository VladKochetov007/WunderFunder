"""Independent CPU-only checks of the exact actor loss AST and skipped-eval RNG."""
from pathlib import Path
from types import SimpleNamespace
import ast, copy, hashlib, json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
from wnn.nn import RNNModel, weighted_pooled_corr_loss
from wnn.outcomes import N_SP_AUX
ACTOR = ROOT/'research/iterations/2026-10-01-overnight/training/train_aux.py'
BASE = ROOT/'scripts/train_rnn.py'

def loss_blocks(path):
    tree=ast.parse(path.read_text())
    blocks=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.If) and isinstance(n.test,ast.Compare):
            if ast.unparse(n.test) in ('args.aux_struct > 0','args.aux_next > 0'):
                if any(isinstance(k,ast.Name) and k.id=='loss' for k in ast.walk(n)):
                    blocks.append(n)
    assert len(blocks)==2
    return compile(ast.Module(body=blocks,type_ignores=[]),str(path),'exec')

def calculation(code, initial, args):
    out=initial.detach().clone().requires_grad_()
    B,T,_=out.shape
    g=torch.Generator().manual_seed(620031)
    y=torch.randn(B,T,2,generator=g)
    mask=torch.rand(B,T,generator=g)>.2
    main=weighted_pooled_corr_loss(out[...,:2],y,mask)
    shared={"loss":main,"out":out,"args":args,"sl":slice(None),"sw":mask.float(),"nw":mask.float(),"F":F,
            "struct_at":2,"N_SP_AUX":N_SP_AUX,"next_at":2+N_SP_AUX+7,
            "sp_lab":torch.randint(N_SP_AUX,(B,T),generator=g),
            "br_lab":torch.randint(3,(B,T),generator=g),"cl_lab":torch.randint(4,(B,T),generator=g),
            "nx_rows":torch.randn(B,T,generator=g),"nx_up":torch.randint(2,(B,T),generator=g).float(),"kth":[],
            "main_valid":mask.float()}
    if code is not None:exec(code,shared)
    shared['loss'].backward()
    return shared['loss'].detach(),out.grad

def main():
    torch.set_num_threads(2)
    start=time.monotonic()
    torch.manual_seed(197)
    out=torch.randn(3,29,2+N_SP_AUX+9)
    kwargs={'aux_struct':.3,'aux_next':.3}
    a=calculation(loss_blocks(BASE),out,SimpleNamespace(**kwargs))
    b=calculation(loss_blocks(ACTOR),out,SimpleNamespace(aux_loss_scale=1.,**kwargs))
    z=calculation(loss_blocks(ACTOR),out,SimpleNamespace(aux_loss_scale=0.,**kwargs))
    m=calculation(None,out,SimpleNamespace(**kwargs))
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
    assert torch.equal(z[0],m[0]) and torch.equal(z[1],m[1])
    assert z[1] is not None and torch.count_nonzero(z[1][...,2:])==0
    assert torch.count_nonzero(z[1][...,:2])>0
    # Frozen trunk + private auxiliary rows retain tensors after a real backward/AdamW step.
    net=RNNModel(211,hidden=8,layers=2,n_out=out.shape[-1])
    for name,p in net.named_parameters():
        if name.startswith(('inp','gru')):p.requires_grad_(False)
    before={k:v.detach().clone() for k,v in net.state_dict().items()}
    pred,_=net(torch.randn(2,17,211))
    loss=weighted_pooled_corr_loss(pred[...,:2],torch.randn(2,17,2),torch.ones(2,17,dtype=torch.bool))+.0*pred[...,2:].square().mean()
    loss.backward()
    for name,p in net.named_parameters():
        if name.startswith(('inp','gru')):assert p.grad is None
        else:assert p.grad is not None
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=5e-4,weight_decay=.01)
    opt.step()
    assert all(torch.equal(before[k],v) for k,v in net.state_dict().items() if k.startswith(('inp','gru')))
    assert not torch.equal(before['head.0.weight'],net.state_dict()['head.0.weight'])
    # DataLoader's sequential iterator consumes one CPU base seed, including actual 2-worker iteration.
    loader=DataLoader(TensorDataset(torch.arange(23)),batch_size=16,num_workers=2,prefetch_factor=1)
    state=torch.get_rng_state()
    actual=[b[0].tolist() for b in loader]
    after=torch.get_rng_state()
    torch.set_rng_state(state)
    torch.empty((),dtype=torch.int64).random_(generator=loader.generator)
    emulated=torch.get_rng_state()
    assert torch.equal(after,emulated)
    assert actual==[list(range(16)),list(range(16,23))]
    result={'status':'PASS','actor_sha256':hashlib.sha256(ACTOR.read_bytes()).hexdigest(),'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
            'checks':['exact AST scale1 loss and output-gradient equality','scale0 exact main-only loss and gradient equality','private aux output gradients retained as zero tensors','frozen trunk unchanged after optimizer step','head gradients present and head updated','actual sequential 2-worker evaluation iterator RNG equals one explicit base-seed draw'],
            'scope':'synthetic correctness only; no empirical fit, dataset read, candidate WP, or GPU work','elapsed_seconds':time.monotonic()-start,'python':sys.version.split()[0],'torch':torch.__version__}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

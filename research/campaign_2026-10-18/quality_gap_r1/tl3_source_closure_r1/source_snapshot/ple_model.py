"""Direct bounded FP32 ramp basis before original input GELU; no shared rank factor."""
import torch
from torch import nn
from torch.nn import functional as F
from compress_model import CompressModel

SCHEMA_VERSION=1
PLE_MODE="clipped_full_basis"
PLE_KEYS=frozenset(('ple_weight','ple_feature','ple_left','ple_right','ple_support_count','ple_knot_count'))
KNOT_KEYS=('feature','left','right','support_count','knot_count')

def validate_knots(knots,n_in):
    """Validate CPU metadata before registration; actual support provenance belongs to extractor."""
    dtype={'feature':torch.int64,'left':torch.float32,'right':torch.float32,'support_count':torch.int64,'knot_count':torch.int64}
    result={}
    for key in KNOT_KEYS:
        v=torch.as_tensor(knots[key]).detach().cpu()
        assert v.dtype==dtype[key] and v.ndim==1,(key,v.dtype,v.shape)
        result[key]=v.contiguous().clone()
    f,l,r,s,k=(result[key] for key in KNOT_KEYS)
    assert len(s)==len(k)==n_in and len(f)==len(l)==len(r)
    assert bool(((s>=1)&(s<=3)).all()) and bool(((k>=1)&(k<=9)).all())
    assert bool(torch.isfinite(l).all()&torch.isfinite(r).all())
    gap=r-l
    assert bool((torch.isfinite(gap)&(gap>0)).all())
    expected=[]
    for i in range(n_in):
        if s[i]>=3 and k[i]>=3:expected.extend([i]*(int(k[i])-1))
    assert torch.equal(f,torch.tensor(expected,dtype=torch.int64)), 'Intervals must be sorted, complete, and eligible by BOTH support and knots'
    for i in range(n_in):
        ix=torch.nonzero(f==i).flatten()
        if len(ix):assert torch.equal(l[ix[1:]],r[ix[:-1]]),'Non-contiguous feature knots'
    return result

def knots_from_state(state):
    assert {k for k in state if k.startswith('ple_')}==PLE_KEYS
    return {k:state['ple_'+k] for k in KNOT_KEYS}

class FullPLEModel(CompressModel):
    def install_ple(self,knots):
        """After loading ancestor, before optimizer; trained restore installs shapes then strict-loads."""
        assert not any(k.startswith('ple_') for k in self.state_dict()),'PLE already installed'
        d=validate_knots(knots,self.inp.in_features)
        device=self.inp.weight.device
        for k,v in d.items():self.register_buffer('ple_'+k,v.to(device))
        # torch.zeros consumes no RNG; independent output coefficient for every interval.
        self.ple_weight=nn.Parameter(torch.zeros(self.inp.out_features,len(d['feature']),device=device,dtype=torch.float32))
        return self

    def ple_basis(self,v):
        assert hasattr(self,'ple_weight'),'Explicit PLE knot installation required'
        assert v.shape[-1]==self.inp.in_features
        with torch.autocast(v.device.type,enabled=False):
            x=v.float()
            torch._assert_async(torch.isfinite(x).all(),'Nonfinite normalized PLE input')
            raw=(x.index_select(-1,self.ple_feature)-self.ple_left)/(self.ple_right-self.ple_left)
            # No overflow/sentinel replacement. Tiny intervals are accepted only if finite arithmetic.
            torch._assert_async(torch.isfinite(raw).all(),'Nonfinite FP32 ramp arithmetic')
            basis=raw.clamp(0.,1.)
            torch._assert_async(torch.isfinite(basis).all(),'Nonfinite PLE basis')
        return basis

    def encode(self,v):
        base=self.inp(v)
        basis=self.ple_basis(v)
        residual=F.linear(basis,self.ple_weight).to(base.dtype)
        return F.gelu(base+residual)

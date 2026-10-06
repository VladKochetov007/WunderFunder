import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from wnn.nn import RNNModel

class CompressModel(RNNModel):
    """Exact paired-code residual; no labels, future state, or sequence identity in lookup."""
    def __init__(self,n_in,hidden,layers,dropout,n_out,codebook=''):
        super().__init__(n_in,hidden,layers,dropout,n_out)
        self.family_head=nn.Linear(hidden,8)
        self.has_codebook=bool(codebook)
        if codebook:
            table=np.load(codebook)
            with torch.random.fork_rng(devices=[]):
                self.code_emb=nn.ModuleList()
                for i,name in enumerate(('i0','i1')):
                    keys=np.asarray(table[name],dtype=np.int64)
                    assert len(keys)>0 and np.all(keys[1:]>keys[:-1])
                    self.register_buffer(f'code_keys{i}',torch.from_numpy(keys.copy()))
                    self.code_emb.append(nn.Embedding(len(keys)+1,16,padding_idx=0))
                self.code_proj=nn.Linear(32,hidden,bias=False)
                nn.init.zeros_(self.code_proj.weight)
    @staticmethod
    def pair_key(x,a,b):
        hi=x[...,a].float().contiguous().view(torch.int32).long()
        lo=x[...,b].float().contiguous().view(torch.int32).long() & 0xffffffff
        return (hi<<32)|lo
    def code_features(self,x):
        vectors=[]
        for i,(a,b) in enumerate(((0,11),(52,63))):
            key=self.pair_key(x,a,b);keys=getattr(self,f'code_keys{i}')
            pos=torch.searchsorted(keys,key).clamp_max(len(keys)-1)
            ids=torch.where(keys[pos]==key,pos+1,0)
            vectors.append(self.code_emb[i](ids))
        return torch.cat(vectors,-1)
    def forward(self,x,h=None):
        u=self.inp((x-self.mean)/self.std)
        if self.has_codebook:u=u+self.code_proj(self.code_features(x))
        z=F.gelu(u);o,h=self.gru(z,h)
        f=self.head[1](self.head[0](torch.cat((o,z),-1)))
        self.family_output=self.family_head(f)
        return self.head[2](f),h

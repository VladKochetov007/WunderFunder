import torch
from torch import nn
from wnn.nn import RNNModel

class CompressModel(RNNModel):
    def __init__(self,n_in,hidden,layers,dropout,n_out,enc="linear"):
        super().__init__(n_in,hidden,layers,dropout,n_out,enc=enc)
        self.family_head=nn.Linear(hidden,8)
    def forward(self,x,h=None):
        z=self.encode((x-self.mean)/self.std);o,h=self.gru(z,h)
        f=self.head[1](self.head[0](torch.cat([o,z],-1)))
        self.family_output=self.family_head(f)
        return self.head[2](f),h

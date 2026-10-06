import torch
from wnn.nn import RNNModel

class LinearBaseline(RNNModel):
    def forward(self,x,h=None):
        z=self.inp((x-self.mean)/self.std)
        o,h=self.gru(z,h)
        return self.head[2](self.head[0](torch.cat([o,z],-1))),h

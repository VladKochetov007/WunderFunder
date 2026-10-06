import torch
from torch import nn
from wnn.nn import RNNModel
SSL_COLS=(0,11,22,33,44,48,52,63,74,85,96,100,104,105,106,108)

class SSLModel(RNNModel):
    def __init__(self,n_in,hidden,layers,dropout,n_out,stage):
        super().__init__(n_in,hidden,layers,dropout,n_out)
        self.stage=stage;self.ssl_head=nn.Linear(hidden,len(SSL_COLS))
        nn.init.zeros_(self.ssl_head.weight);nn.init.zeros_(self.ssl_head.bias)
    def forward(self,x,h=None):
        z=self.encode((x-self.mean)/self.std);o,h=self.gru(z,h)
        f=self.head[1](self.head[0](torch.cat([o,z],-1)))
        # Control/FT fits the predictive feature head without changing supervised representations.
        self.ssl_output=self.ssl_head(f.detach() if self.stage in ('control','ft') else f)
        return self.head[2](f),h

def ssl_loss(model,inputs,start,length):
    end=min(start+length,inputs.shape[1]-1)
    if end<=start:return model.ssl_output.float().sum()*0
    current=inputs[:,start:end,SSL_COLS].float()
    nxt=inputs[:,start+1:end+1,SSL_COLS].float()
    target=((nxt-current)/model.std[list(SSL_COLS)]).clamp(-3,3)
    return (model.ssl_output[:,:end-start].float()-target).square().mean()

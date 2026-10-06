import torch
from wnn.nn import DualClockModel

class AdaptiveModel(DualClockModel):
    """Cheap state every row; heavy state at current price events or every eighth row."""
    def __init__(self,n_in,hidden,layers,event_hidden,dropout,n_out,schedule):
        super().__init__(n_in,hidden,layers,event_hidden,dropout,n_out)
        self.schedule=schedule
    def forward(self,x,events,state=None):
        count=0 if state is None else int(state[-1].item())
        base_state=None if state is None else state[:3]
        rows=torch.arange(count,count+x.shape[1],device=x.device)
        if self.schedule=='all':events=torch.ones_like(events)
        else:events=events|((rows%8)==0)[None]
        self.last_update_fraction=events.float().mean().detach()
        out,base_state=super().forward(x,events,base_state)
        return out,(*base_state,torch.tensor(count+x.shape[1],device=x.device))

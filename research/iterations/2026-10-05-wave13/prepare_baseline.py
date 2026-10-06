from pathlib import Path
import json
import numpy as np
import torch
import onnx
from onnx import numpy_helper
from linear_baseline import LinearBaseline
from wnn.data import SequenceFile
from wnn.features import event_features_torch
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave13'

def main():
    torch.set_num_threads(2)
    m=onnx.load(R/'baseline/baseline.onnx');init={i.name:numpy_helper.to_array(i).astype('float32') for i in m.graph.initializer}
    def order(a):z,r,n=np.split(a,3,axis=0);return np.concatenate([r,z,n],axis=0)
    model=LinearBaseline(211,128,2,n_out=2);sd={k:np.zeros(v.shape,dtype='float32') for k,v in model.state_dict().items()}
    sd['std'][:]=1;sd['inp.weight'][np.arange(112),np.arange(112)]=1
    for layer,node in enumerate(n for n in m.graph.node if n.op_type=='GRU'):
        W,rec,B=(init[node.input[j]][0] for j in (1,2,3))
        wi=order(W)
        if layer==0:
            padded=np.zeros((384,128),dtype='float32');padded[:,:112]=wi;wi=padded
        sd[f'gru.weight_ih_l{layer}']=wi;sd[f'gru.weight_hh_l{layer}']=order(rec)
        sd[f'gru.bias_ih_l{layer}']=order(B[:384]);sd[f'gru.bias_hh_l{layer}']=order(B[384:])
    sd['head.0.weight'][np.arange(128),np.arange(128)]=1
    sd['head.2.weight']=init[[n for n in m.graph.node if n.op_type=='MatMul'][0].input[1]].T
    sd['head.2.bias']=init['reg_head.bias']
    model.load_state_dict({k:torch.tensor(v) for k,v in sd.items()});model.eval()
    out=R/'artifacts/runs/w13base_direct';assert not out.exists();out.mkdir()
    args={'hidden':128,'layers':2,'dropout':0.,'arch':'gru','raw_only':True,'linear_baseline':True}
    torch.save({'model':model.state_dict(),'args':args,'n_in':211,'epoch':0},out/'model_ep0.pt')
    np.savez_compressed(D/'baseline-linear.npz',**sd,**{'meta.linear_activations':np.array([1,1],dtype='float32')})
    import onnxruntime as ort
    for v in list(m.graph.input)+list(m.graph.output):
        if v.name in ('features','prediction'):v.type.tensor_type.shape.dim[1].dim_param='T'
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(m.SerializeToString(),options,providers=['CPUExecutionProvider'])
    sf=SequenceFile(R/'datasets/valid.parquet');groups=np.load(R/'cache/valid_targets.npz')['row_group'];cached=np.load(R/'cache/pred_orgbase_valid.npy',mmap_mode='r');reports=[]
    for group in (0,5,12):
        seq=sf.read(group);xx=seq.features[:2000];raw=torch.tensor(xx)[None];x=torch.cat([raw,torch.zeros((1,len(xx),99))],-1)
        with torch.no_grad():p,_=model(x)
        h=np.zeros((1,1,128),dtype='float32');ref=session.run(None,{'features':xx[None],'hidden_0':h,'hidden_1':h})[0][0]
        err=float(abs(p[0].numpy()-ref).max());assert err<2e-5,err
        cache=cache_diff=float(abs(cached[np.flatnonzero(groups==group)[0],:2000]-ref).max())
        reports.append({'group':group,'torch_onnx_max_abs':err,'cached_orgbase_onnx_max_abs':cache_diff})
        if group==0:np.savez_compressed(D/'baseline-probe.npz',x=xx,reference=ref,y=seq.targets[:len(xx)],mask=seq.scored[:len(xx)])
    (D/'baseline-parity.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports),flush=True)
if __name__=='__main__':main()

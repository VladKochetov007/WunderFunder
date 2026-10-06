"""Export --raw-only / --zero-groups checkpoints for the native engine: zero the dropped input columns,
fold their constant normalized value ((0-mean)/std) into inp.bias."""
import sys,numpy as np,torch
sys.path.insert(0,"/home/grinchenkoeg/WunderFunder-creative/scripts")
N_RAW=112
def dropped_cols(args,n_in):
    cols=set()
    if args.get("raw_only"):cols|=set(range(N_RAW,n_in))
    if args.get("zero_groups"):
        import train_rnn;cols|=set(train_rnn.zero_group_cols(args["zero_groups"]))
    return sorted(cols)
def exp_raw(path,out):
    ck=torch.load(path,map_location="cpu",weights_only=False);args=ck["args"]
    a={k:v.detach().cpu().float().numpy() for k,v in ck["model"].items()}
    W=a["inp.weight"];cols=dropped_cols(args,W.shape[1]);assert cols,path
    z=(0.0-a["mean"][cols])/a["std"][cols]
    a["inp.bias"]=a["inp.bias"]+W[:,cols]@z;W[:,cols]=0;a["inp.weight"]=W
    a["head.2.weight"]=a["head.2.weight"][:2].copy();a["head.2.bias"]=a["head.2.bias"][:2].copy();np.savez(out,**a);return args

import sys,pickle,numpy as np
from multiprocessing import Pool
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"src");sys.path.insert(0,S+"scripts")
from wnn.data import SequenceFile,CORRUPT_TRAIN_ROW_GROUPS
from i1swap_aug import labels_with_mask,swap_instruments
NOSWAP=len(sys.argv)>1 and sys.argv[1]=="noswap";TAG="i0rb" if NOSWAP else "i1swap"
moves,spreads=pickle.load(open(S+"cache/class3_moves_full.pkl","rb"))
cls=np.load(S+"research/artifacts/market-classes-v1.npz")["train"]
rgs=np.array([g for g in range(10607) if g not in CORRUPT_TRAIN_ROW_GROUPS and cls[g]==3])
sf=None
def work(g):
    global sf
    if sf is None:sf=SequenceFile(S+"datasets/train.parquet")
    x=sf.read(int(g)).features;xs=x if NOSWAP else swap_instruments(x);y,ok=labels_with_mask(xs[:,0:22],moves,spreads);return y,ok
if __name__=="__main__":
    Y=np.lib.format.open_memmap(S+f"cache/{TAG}_train_y.npy",mode="w+",dtype=np.float32,shape=(len(rgs),20000,2))
    OK=np.lib.format.open_memmap(S+f"cache/{TAG}_train_ok.npy",mode="w+",dtype=np.bool_,shape=(len(rgs),20000))
    with Pool(40) as pool:
        for j,(y,ok) in enumerate(pool.imap(work,rgs,chunksize=4)):
            Y[j]=y;OK[j]=ok
    Y.flush();OK.flush();np.save(S+f"cache/{TAG}_train_rg.npy",rgs)
    print("seqs",len(rgs),"reliable share",float(np.asarray(OK).mean()))

"""Frozen native fixed midpoint comparison paths and input checks; no model/score execution."""
import hashlib, json, os
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[3]
Q4=ROOT/'research/iterations/2026-10-01-continuation/student/native-quality-01'
NAMES=('oct02_wm_s3h2','on_ctl128s3','on_ctl128s4','on_ctlc500')
METRIC_SHA='d05791f5f8c6ea64d9863d4f4b45430eb46b9d0b29ccbdac1a4ff5b36a2573c0'
NATIVE_SHA='b865301e5ff0c9dce78c29c258d991a7b5542e7a63d44eeafd32c3b526166867'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def identity(path):
    p=Path(path);s=p.stat();return {'resolved':str(p.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def guards(cpu):
    assert sorted(os.sched_getaffinity(0))==[cpu]
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[2].lstrip('/')
    limits={k:(cg/k).read_text().strip() for k in ['cpu.max','memory.max','memory.swap.max']}
    assert limits=={'cpu.max':'100000 100000','memory.max':str(4*(1<<30)),'memory.swap.max':'0'}
    allocation=json.loads((P/'allocation.json').read_text());assert allocation['authorized'] is True and allocation['OOF_passes']==2 and allocation['metric_readouts']==1
    assert allocation['members']==list(NAMES) and allocation['coefficients']==[.25]*4 and allocation['epoch']==1
    assert allocation['lanes']=={'0':30,'1':35} and allocation['prediction_wall_seconds_each']==1800
    assert allocation['native_sha256']==NATIVE_SHA and cpu in (30,35)
    for path,digest in allocation['source_sha256'].items():assert sha(ROOT/path)==digest,path
    return allocation,limits
def rowmaps():
    import numpy as np
    with np.load(ROOT/'cache/valid_targets.npz',allow_pickle=False) as z:groups=z['row_group']
    with np.load(ROOT/'cache/chain_folds.npz',allow_pickle=False) as z:cf={k:z[k] for k in ['split','row_group','fold','chain','pure']}
    valid=cf['split']=='valid';info={int(g):(int(f),int(c),bool(p)) for g,f,c,p in zip(cf['row_group'][valid],cf['fold'][valid],cf['chain'][valid],cf['pure'][valid],strict=True)}
    assert len(groups)==1873 and len(np.unique(groups))==1873 and set(info)==set(groups.tolist())
    train=set(cf['chain'][cf['split']=='train'].tolist())
    for c in np.unique(cf['chain'][valid]):
        ix=valid&(cf['chain']==c);assert len(np.unique(cf['fold'][ix]))==1
        if cf['pure'][ix].any():assert int(c) not in train
    return groups,info,hashlib.sha256(groups.tobytes()).hexdigest()

from __future__ import annotations
import json
import numpy as np
import torch
from probe import ROOT, choose_donor, windows, train, rms, CLOCK

def main() -> None:
    torch.set_num_threads(1)
    rng=np.random.default_rng(614204)
    raw=np.repeat(rng.normal(size=(40,112)).astype(np.float32),37,axis=0)
    packed=train.build_inputs(torch.from_numpy(raw)[None])[0].numpy()
    std=np.ones(211,dtype=np.float32)
    planted,sham,mismatch=[],[],[]
    for t in [512,768,1024]:
        d=choose_donor(raw,t,std)
        altered,recomputed,direct=windows(raw,packed,d)
        original=raw[t-80,0]
        planted.append([0,float(altered[t-80,0]-original),float(direct[208,0]-original)])
        sham.append([0,float(altered[t,52]-raw[t,52]),float(direct[-1,52]-raw[t,52])])
        mismatch.append(float(np.mean(recomputed[:,CLOCK]!=direct[:,CLOCK])))
    planted=np.array(planted);sham=np.array(sham)
    assert np.array_equal(planted[:,1],planted[:,2])
    assert rms(planted[:,1])>.01 and rms(sham)==0 and rms(planted[:,0])==0
    assert max(mismatch)>0
    result=dict(passed=True,seed=614204,planted_history_response_rms=rms(planted[:,1]),sham_history_response_rms=rms(sham),identity_response_rms=rms(planted[:,0]),packed_clock_mismatch_fractions=mismatch,scope='known synthetic mechanism validates raw/packed donor, planted history and sham; not model predictive value')
    (ROOT/'controls.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':
    main()

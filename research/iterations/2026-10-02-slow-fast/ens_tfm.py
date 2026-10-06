import numpy as np
exec(open("ens_mix.py").read().split("cands={")[0])
cands={"muon pair (h128)":[("muon6",1),("muon6s4",1)],"muon6 + tfmmu":[("muon6",1),("tfmmu",1)],"muon6 + tfmad":[("muon6",1),("tfmad",1)],
 "d256mu6+muon6+muon6s4":[("d256mu6",1),("muon6",1),("muon6s4",1)],"d256mu6+muon6+tfmmu":[("d256mu6",1),("muon6",1),("tfmmu",1)]}
for n,s in cands.items():
    P=ens(s);print(n.ljust(26)," ".join(f"{k}={wp(P,g&sel):.4f}" for k,g in groups.items() if k!="c3active"))
import numpy as np
a=L("muon6");b=L("tfmmu");c=L("muon6s4");m=M&sel[:,None]
print("corr muon6-tfmmu",np.corrcoef(a[m][:,0],b[m][:,0])[0,1].round(4),"corr muon6-muon6s4",np.corrcoef(a[m][:,0],c[m][:,0])[0,1].round(4))

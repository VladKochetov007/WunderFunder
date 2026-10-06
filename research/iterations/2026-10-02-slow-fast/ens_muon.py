import numpy as np
exec(open("ens_mix.py").read().split("cands={")[0])
cands={"adam pair ref+refs4":[("ref",1),("refs4",1)],"muon2 pair":[("muon2",1),("muon2s4",1)],
 "mix muon2+refs4":[("muon2",1),("refs4",1)],"mix muon2s4+ref":[("muon2s4",1),("ref",1)],
 "adam+muon 4":[("ref",1),("refs4",1),("muon2",1),("muon2s4",1)],"d256+adam pair":[("d256",2),("ref",1),("refs4",1)],"d256+muon pair":[("d256",2),("muon2",1),("muon2s4",1)]}
for n,s in cands.items():
    P=ens(s);print(n.ljust(24)," ".join(f"{k}={wp(P,g&sel):.4f}" for k,g in groups.items() if k!="c3active"))

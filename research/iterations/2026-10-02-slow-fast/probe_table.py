import numpy as np
exec(open("mix.py").read().split("for run in sys.argv")[0].replace("import numpy as np,sys","import numpy as np,sys"))
P,sel=load("d256");vm=np.bincount(cls[sel],minlength=4)/sel.sum()
Z3=P.copy();Z3[cls==3]=0;O3=P.copy();O3[cls!=3]=0
print("share3  full   c3zero c3only  (rest in valid proportions)")
for s3 in (0.30,0.37,0.45,0.55,0.65,0.75):
    rest=vm[:3]/vm[:3].sum()*(1-s3);mix=np.append(rest,s3);sw=(mix/vm)[cls]
    print(f"{s3:.2f}   {wp(P,sel,sw):.4f} {wp(Z3,sel,sw):.4f} {wp(O3,sel,sw):.4f}")
for name,mix in (("trainmix",np.array([.123,.119,.329,.430])),):
    sw=(mix/vm)[cls];print(name,f"{wp(P,sel,sw):.4f} {wp(Z3,sel,sw):.4f} {wp(O3,sel,sw):.4f}")

exec(open("build_kf.py").read().split("R=H+")[0])
C=H+"/WunderFunder-creative/artifacts/runs"
pkg("mix8",[f"{C}/{p}_s{s}h2/model_ep1.pt" for p in ("fmu128","fmuQ","fmuK","fmuA") for s in (3,4)])

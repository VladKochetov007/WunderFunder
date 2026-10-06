exec(open("build_kf.py").read().split("R=H+")[0])
C=H+"/WunderFunder-creative/artifacts/runs"
pkg("pair_onorm",[f"{C}/fmuON_s3h2/model_ep1.pt",f"{C}/fmuON_s4h2/model_ep1.pt"])
pkg("par_onorm",[f"{C}/fmuON_s3h2/model_ep1.pt"])

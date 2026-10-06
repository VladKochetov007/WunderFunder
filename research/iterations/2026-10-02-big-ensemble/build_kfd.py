exec(open("build_kf.py").read().split("R=H+")[0])
O=H+"/WunderFunder/artifacts/runs";C=H+"/WunderFunder-creative/artifacts/runs"
pkg("kf_d256",[f"{O}/ft256d_f0h/model.pt",f"{O}/ft256d_f1h/model.pt",f"{C}/ft256d_f2a/model.pt",f"{C}/ft256d_f3a/model.pt"])
pkg("full_d256",[f"{O}/find256_s3h/model.pt"])

exec(open("build_kf.py").read().split("R=H+")[0])
C=H+"/WunderFunder-creative/artifacts/runs"
pkg("pair_ffA_lr2e4",[f"{C}/ffA_s3/model_ep1.pt",f"{C}/ffA_s4/model_ep1.pt"])
pkg("pair_ffB_4ep",[f"{C}/ffB_s3/model_ep3.pt",f"{C}/ffB_s4/model_ep3.pt"])
pkg("pair_ffD_muon",[f"{C}/ffD_s3/model_ep1.pt",f"{C}/ffD_s4/model_ep1.pt"])

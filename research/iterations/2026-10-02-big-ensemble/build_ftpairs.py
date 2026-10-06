exec(open("build_kf.py").read().split("R=H+")[0])
C=H+"/WunderFunder-creative/artifacts/runs"
pkg("pair_fullft",[f"{C}/ffull_s3/model_ep1.pt",f"{C}/ffull_s4/model_ep1.pt"])
pkg("pair_ft4",[f"{C}/fft4_s3/model_ep3.pt",f"{C}/fft4_s4/model_ep3.pt"])
pkg("pair_v2",[f"{C}/fmuV2_s3h2/model_ep1.pt",f"{C}/fmuV2_s4h2/model_ep1.pt"])

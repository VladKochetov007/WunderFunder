exec(open("build_kf.py").read().split("R=H+")[0])
R=H+"/WunderFunder/artifacts/runs"
pkg("kf_h256",[f"{R}/ft256_f{k}h/model.pt" for k in range(4)])
pkg("full_h256",[f"{R}/fin256_s3h/model.pt"])

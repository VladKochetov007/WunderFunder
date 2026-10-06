exec(open("build_kf.py").read().split("R=H+")[0])
R=H+"/WunderFunder-creative/artifacts/runs"
pkg("pair_wise05",[f"{R}/wise05full_s3/model.pt",f"{R}/wise05full_s4/model.pt"])

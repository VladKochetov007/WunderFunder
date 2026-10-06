exec(open("build_kf.py").read().split("R=H+")[0])
C=H+"/WunderFunder-creative/artifacts/runs"
pkg("pair_boot_h2",[f"{C}/fmuB256_s3h2/model_ep1.pt",f"{C}/fmuB256_s4h2/model_ep1.pt"])
pkg("pair_boot_fullft",[f"{C}/fbB_full_s3/model_ep1.pt",f"{C}/fbB_full_s4/model_ep1.pt"])
pkg("pair_boot_combo",[f"{C}/fmuB256_s3/model_ep3.pt",f"{C}/fmuB256_s3h2/model_ep1.pt",f"{C}/fmuB256_s4/model_ep3.pt",f"{C}/fmuB256_s4h2/model_ep1.pt"])

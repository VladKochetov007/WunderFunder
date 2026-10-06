exec(open("build_kf.py").read().split("R=H+")[0])
C=H+"/WunderFunder-creative/artifacts/runs"
for tag,pre in (("pair_prop1","fmuP1"),("pair_since0","fmuS0"),("pair_vtm0","fmuV0")):
    pkg(tag,[f"{C}/{pre}_s3h2/model_ep1.pt",f"{C}/{pre}_s4h2/model_ep1.pt"])

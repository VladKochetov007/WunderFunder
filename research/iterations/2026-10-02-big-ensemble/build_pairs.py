exec(open("build_kf.py").read().split("R=H+")[0])
C=H+"/WunderFunder-creative/artifacts/runs"
for tag,pre in (("pair_base","fmu128"),("pair_quiet","fmuQ"),("pair_classmix","fmuK"),("pair_advval","fmuA")):
    pkg(tag,[f"{C}/{pre}_s3h2/model_ep1.pt",f"{C}/{pre}_s4h2/model_ep1.pt"])

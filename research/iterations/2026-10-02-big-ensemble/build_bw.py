exec(open("build_vlad.py").read().split("pkgN(\"vlad_br\"")[0])
RAWW=lambda c:[("raw",f"{R}/frawW_s3ft/model_ep1.pt",c/2),("raw",f"{R}/frawW_s4ft/model_ep1.pt",c/2)]
pkgN("vlad_bw",[("vlad",VL,0.5)]+RAWW(0.25)+[BASE])
pkgN("pair_fullft_bw",PAIR(0.5)+RAWW(0.25)+[BASE])

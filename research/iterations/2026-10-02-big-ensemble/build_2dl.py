exec(open("build_2d.py").read().split("D2=lambda")[0])
pkgN("blend_2d_lite",[("vlad",VL,0.35),("run",f"{R}/fbc1kft_s5/model_ep1.pt",0.2),("raw",f"{R}/frawW_s3ft/model_ep1.pt",0.15),BASE]);neutral("blend_2d_lite")

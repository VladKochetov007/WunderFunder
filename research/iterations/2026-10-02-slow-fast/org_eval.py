import numpy as np
exec(open("pair_blend2.py").read().split("bases = ")[0])
O=lambda n:[n]
rep("org pair alone",[(1,pr("org"),1)])
rep("fullft + org pair .25",[(0.75,pr("fullft"),1),(0.25,pr("org"),1)])
rep("fullft + base",[(0.75,pr("fullft"),1),(0.25,["orgbase"],0.25)])
rep("fullft + base + org pair .2",[(0.55,pr("fullft"),1),(0.25,["orgbase"],0.25),(0.2,pr("org"),1)])
rep("fullft + orgall s3 .25",[(0.75,pr("fullft"),1),(0.25,["cmp_orgall_s3"],1)])
rep("fullft + orgw s3 .25",[(0.75,pr("fullft"),1),(0.25,["cmp_orgw_s3"],1)])
rep("fullft + base + orgall .2",[(0.55,pr("fullft"),1),(0.25,["orgbase"],0.25),(0.2,["cmp_orgall_s3"],1)])
rep("wise05 + base + org pair .2",[(0.55,pr("wise05"),1),(0.25,["orgbase"],0.25),(0.2,pr("org"),1)])

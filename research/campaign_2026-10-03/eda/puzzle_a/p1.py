import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from util import *
df = load("train", 0)
order = [19,16,20,10,5,18,13,2,12,1,17,4,7,6,9,15,11,8,21,14,0,3]
fig, ax = plt.subplots(6,1, figsize=(18,20), sharex=True)
x = np.arange(len(df))
for k,i in enumerate(order):
    ax[0].plot(x, df[f"i0_p{i}"], lw=.5, color=plt.cm.coolwarm(k/21))
    ax[1].plot(x, df[f"i1_p{i}"], lw=.5, color=plt.cm.coolwarm(k/21))
ax[0].set_title("i0 prices (cols sorted by mean)"); ax[1].set_title("i1 prices")
for i in range(4):
    ax[2].plot(x, df[f"i0_dp{i}"], lw=.5, label=f"dp{i}")
ax[2].legend(); ax[2].set_title("i0 dp")
ax[3].plot(x, df.t0, lw=.6, label="t0"); ax[3].plot(x, df.t1, lw=.6, label="t1"); ax[3].legend()
for i in [2,3,5,6]:
    ax[4].plot(x, df[f"a{i}"], lw=.6, label=f"a{i}")
ax[4].legend()
for i in [0,1,4]:
    ax[5].plot(x, df[f"a{i}"], lw=.3, label=f"a{i}")
ax[5].legend()
plt.tight_layout(); plt.savefig("fig01_seq0_overview.png", dpi=60)
fig, ax = plt.subplots(5,1, figsize=(18,16), sharex=True)
sl = slice(5000,5600); x = np.arange(5000,5600)
for k,i in enumerate(order):
    ax[0].step(x, df[f"i0_p{i}"].values[sl], lw=.8, color=plt.cm.coolwarm(k/21), where="post")
    ax[1].step(x, df[f"i1_p{i}"].values[sl], lw=.8, color=plt.cm.coolwarm(k/21), where="post")
for i in range(4):
    ax[2].step(x, df[f"i0_dp{i}"].values[sl], lw=.8, label=f"dp{i}", where="post")
    ax[2].step(x, df[f"i0_dv{i}"].values[sl], lw=.8, ls=":", label=f"dv{i}", where="post")
ax[2].legend(ncol=8)
ax[3].step(x, df.t0.values[sl], label="t0", where="post"); ax[3].step(x, df.t1.values[sl], label="t1", where="post"); ax[3].legend()
for i in range(8):
    ax[4].plot(x, df[f"a{i}"].values[sl], lw=.8, label=f"a{i}")
ax[4].legend(ncol=8)
plt.tight_layout(); plt.savefig("fig02_seq0_zoom.png", dpi=60)

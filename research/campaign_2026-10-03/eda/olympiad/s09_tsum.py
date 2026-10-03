import pandas as pd, numpy as np
pd.set_option("display.width",250)
for s in ["train","valid"]:
    d = pd.read_parquet(f"research/campaign_2026-10-03/eda/olympiad/targets_{s}.parquet")
    print("=====", s, len(d))
    print("div counts", d["div"].value_counts().to_dict(), "max resid", d.resid.max())
    qv = d.q.round(5).value_counts()
    print("n distinct q", len(qv)); print(qv.head(30).to_string())
    print(d[["k0abs","k1abs","z0","z1","runs0","runs1","corr","sum_abs","sum0","std0","std1","mean0","mean1"]].describe().T.to_string())
    if s=="valid": print(d[["nscored","first_scored"]].describe().T)
    print("seq_ix sorted?", (np.diff(d.seq.values)>0).all(), "unique", d.seq.nunique())

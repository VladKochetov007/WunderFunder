import numpy as np, pandas as pd
from util import *
df = load("train", 0)
cols = [c for c in df.columns if c not in ("seq_ix","step_in_seq","need_prediction")]
codes = {c: np.unique(df[c].values, return_inverse=True)[1] for c in cols}
nu = {c: codes[c].max()+1 for c in cols}
def determines(a, b):
    # does a -> b functionally?
    pair = codes[a].astype(np.int64) * (nu[b]) + codes[b]
    return len(np.unique(pair)) == nu[a]
pairs = []
for a in cols:
    for b in cols:
        if a != b and determines(a, b):
            pairs.append((a, b))
bij = [(a,b) for a,b in pairs if (b,a) in pairs and a < b]
print("bijections:", bij)
print("one-way:", [(a,b) for a,b in pairs if (b,a) not in pairs])

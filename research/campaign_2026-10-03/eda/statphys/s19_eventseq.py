import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
import pandas as pd
pf = pq.ParquetFile(TRAIN)
rg = int(sys.argv[1]); a, b = int(sys.argv[2]), int(sys.argv[3])
X, Y, u, _ = load(pf, rg)
pch = np.r_[True, np.any(np.diff(X[:, :22], axis=0) != 0, axis=1)]
idx = np.where(pch)[0]
st = pd.factorize(pd.util.hash_pandas_object(pd.DataFrame(X[:, :22]), index=False).values)[0]
T = np.round(Y / u).astype(int)
df = pd.DataFrame({'row': idx, 'dur': np.diff(np.r_[idx, 20000]), 'state': st[idx], 'T0': T[idx, 0], 'T1': T[idx, 1]})
for j in [0, 3, 10, 11, 14, 21]: df[f'p{j}'] = np.round(X[idx, j], 3)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(df.iloc[a:b].to_string(index=False))

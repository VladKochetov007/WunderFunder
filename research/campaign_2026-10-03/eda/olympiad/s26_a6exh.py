# Exhaustive: rows where a6 == min sentinel  <=>  rows where (t0,t1) changes (excluding row 0). Also: a6 strictly increasing inside runs.
import pyarrow.parquet as pq, numpy as np, sys, collections
split = sys.argv[1]
pf = pq.ParquetFile(f"datasets/{split}.parquet"); c = collections.Counter(); bad = []
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=["a6","t0","t1"])
    a6 = t.column("a6").to_numpy(); t0 = t.column("t0").to_numpy(); t1 = t.column("t1").to_numpy()
    s = (a6 <= a6.min()) & (a6 < -5.19)
    T = np.r_[False, (t0[1:]!=t0[:-1])|(t1[1:]!=t1[:-1])]
    s[0] = False
    c["rows"] += len(a6); c["resets"] += int(s.sum()); c["T"] += int(T.sum())
    c["T&~reset"] += int((T&~s).sum()); c["reset&~T"] += int((s&~T).sum())
    m = ~s[1:]
    c["inrun_pairs"] += int(m.sum()); c["inrun_decrease"] += int(np.sum(a6[1:][m] < a6[:-1][m])); c["inrun_equal"] += int(np.sum(a6[1:][m] == a6[:-1][m]))
    if ((T&~s).sum() or (s&~T).sum()) and len(bad) < 20: bad.append((rg, int((T&~s).sum()), int((s&~T).sum())))
    c["seqs_bad"] += int(bool((T&~s).sum() or (s&~T).sum()))
print(split, dict(c)); print("examples", bad)

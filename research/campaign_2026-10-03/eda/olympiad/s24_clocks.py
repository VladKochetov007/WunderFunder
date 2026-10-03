# Are a5/a6/a7 "time since last event X" counters? reset rows vs block-change rows; strict monotonicity between resets.
import pyarrow.parquet as pq, numpy as np, pandas as pd
pf = pq.ParquetFile("datasets/train.parquet")
for rg in [0, 500, 3000, 7000]:
    df = pf.read_row_group(rg).to_pandas()
    ch = lambda c: np.r_[True, df[c].values[1:]!=df[c].values[:-1]]
    ev = {"i0P": ch("i0_p0")|ch("i0_p11"), "i0DP": ch("i0_dp0"), "i0DV": ch("i0_dv0"), "i0V": ch("i0_v0")|ch("i0_v11"),
          "i1P": ch("i1_p0")|ch("i1_p11"), "i1DP": ch("i1_dp0"), "i1DV": ch("i1_dv0"), "i1V": ch("i1_v0")|ch("i1_v11")}
    print("== rg", rg)
    for a in ["a5","a6","a7"]:
        x = df[a].values; s = x < -5.19
        res = {k: (round((s & v).sum()/max(s.sum(),1),3), round((s & v).sum()/max(v.sum(),1),3)) for k,v in ev.items()}
        # strict increase between resets: x[t] > x[t-1] whenever not s[t] and not s[t-1]
        m = ~s[1:] & ~s[:-1]
        inc = np.mean(x[1:][m] > x[:-1][m]); eq = np.mean(x[1:][m] == x[:-1][m])
        print(a, "nreset", s.sum(), "P(ev|reset),P(reset|ev):", res, "between resets: frac strictly increasing", round(inc,4), "equal", round(eq,4))

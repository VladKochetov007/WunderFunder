import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 120)
from util import *
df = load("train", 0)
print(df.shape, df.seq_ix.unique(), df.step_in_seq.min(), df.step_in_seq.max())
print(df.need_prediction.mean(), df.need_prediction.values[:105].astype(int))
d = df.describe().T
print(d.to_string())

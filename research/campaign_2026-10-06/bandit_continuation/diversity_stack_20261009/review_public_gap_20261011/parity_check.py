"""REVIEW: fp32 torch (CPU) vs shipped Q8Vnni ZIP predictions on the 10 gate sequences of fold 1.

usage: parity_check.py RUN EPOCH GATE_NPZ   (run from repo root; PYTHONPATH = trainer source scripts+src)
Scores both against the labels with the official accumulator (these 10 sequences only) and prints correlation q8 vs fp32.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import train_rnn
from train_rnn import Sequences, make_model, predict
from wnn.gmetric import GlobalAccumulator
from wnn.sentinel import SentinelPreprocessor

run, epoch, gate = sys.argv[1], int(sys.argv[2]), sys.argv[3]
z = np.load(gate)
groups = [int(g) for g in z["groups"]]
ck = torch.load(ROOT / "artifacts/runs" / run / f"model_ep{epoch}.pt", map_location="cpu", weights_only=False)
torch.set_num_threads(1)
device = torch.device("cpu")
model = make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0]).to(device)
model.load_state_dict(ck["model"]); model._inference_fp32 = True; model.eval()
train_rnn.SENTINEL_PREPROCESSOR = SentinelPreprocessor(ck["sentinel_preprocessing"]) if ck.get("sentinel_preprocessing") else None
train_rnn.RAW_ONLY = bool(ck["args"].get("raw_only")); train_rnn.ONLINE_NORM = bool(ck["args"].get("online_norm"))
train_rnn.HEURISTICS = bool(ck["args"].get("heuristics")); train_rnn.INPUT_LEAD = int(ck["args"].get("input_lead", 0))
train_rnn.RESEARCH_FEATURES = ck["args"].get("research_features", "none")
ds = Sequences("valid", np.array(groups))
acc_fp, acc_q8, acc_mix = GlobalAccumulator(), GlobalAccumulator(), GlobalAccumulator()
corrs, rows = [], []
for i, g in enumerate(groups):
    item = ds[i]
    x, y, m = item[0], item[1], item[2]
    p_fp = np.asarray(predict(model, x[None].to(device), 4000, device))[0, :, :2].astype(np.float32)
    p_q8 = z[f"pred_{g}"]
    yy = y.numpy() if hasattr(y, "numpy") else np.asarray(y); mm = (m.numpy() if hasattr(m, "numpy") else np.asarray(m)).astype(bool)
    mm = mm & (np.arange(len(mm)) >= 99)
    acc_fp.add(yy[None], p_fp[None], mm[None]); acc_q8.add(yy[None], p_q8[None], mm[None])
    sel = mm
    c = np.corrcoef(p_fp[sel].ravel(), p_q8[sel].ravel())[0, 1]
    corrs.append(c); rows.append((g, float(np.abs(p_fp[sel]-p_q8[sel]).mean()), float(np.abs(p_fp[sel]).mean())))
    print(g, "corr(fp32,q8)=%.5f  mean|diff|=%.4f mean|fp32|=%.4f" % (c, rows[-1][1], rows[-1][2]), flush=True)
print("WP fp32 %.5f   WP q8 %.5f   (10 sequences only)" % (acc_fp.result()["wp"], acc_q8.result()["wp"]))

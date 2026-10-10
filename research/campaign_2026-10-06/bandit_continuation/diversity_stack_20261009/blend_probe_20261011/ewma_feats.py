"""Per-pair causal EWMA features at scored rows, computed over ALL rows of each sequence (state starts at row 0, init at first value).
Usage: ewma_feats.py FOLD STEM_A STEM_B ; writes flat/ewma_f{fold}_{A}__{B}.npz with arrays dE,pE of shape (n,2,4) for hl 50,100,200,500."""
import sys, json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from crossfold_blend import load_fold
from lib import S
HLS = [50, 100, 200, 500]
fold, a, b = int(sys.argv[1]), sys.argv[2], sys.argv[3]
BANKS = {0: "BANK_FOLD0_FULL.jsonl", 1: "BANK_FOLD1.jsonl"}
split = ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
bank = ROOT / "research/campaign_2026-10-06/bandit_continuation/prediction_bank" / BANKS[fold]
y, mask, comp, models = load_fold(bank, fold, split, ROOT / "cache/valid_targets.npz")
def full(stem):
    _, arr, o = models[stem]; return np.clip(np.asarray(arr[o], dtype=np.float32), -2, 2)
pa, pb = full(a), full(b)
d = np.abs(pa - pb); pm = np.abs(0.5 * (pa + pb)); del pa, pb
def ew(x, hl):
    al = 1 - 0.5 ** (1.0 / hl)
    zi = (1 - al) * x[:, :1, :]
    return lfilter([al], [1, -(1 - al)], x, axis=1, zi=zi)[0]
dE = np.stack([ew(d, h)[mask] for h in HLS], -1); pE = np.stack([ew(pm, h)[mask] for h in HLS], -1)
np.savez(S / f"ewma_f{fold}_{a}__{b}.npz", dE=dE.astype(np.float32), pE=pE.astype(np.float32))
print(dE.shape)

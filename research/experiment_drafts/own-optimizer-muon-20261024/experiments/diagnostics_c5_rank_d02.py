"""C5 zero-fit diagnostic: spectral stats of matched D02 AdamW vs Muon checkpoints.

The D02 quartet (350-seq matched screen, 220 updates/arm, same seed/init) is the
repo's only matched optimizer comparison on competition data. This script loads
the four frozen checkpoints and computes per-2D-parameter singular-value
statistics (effective rank, stable rank, 95%-energy rank) to test the
Muon-raises-effective-rank mechanism on real fitted weights. Read-only.

Bounded: CPU, 2 threads, ~1 minute, <2 GB.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

torch.set_num_threads(2)
# Own frozen campaign artifacts (sha256-bound in ROOT_QUARTET_BINDINGS_R2.json);
# allowlist the metadata global they contain instead of disabling weights_only.
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

BASE = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    "/cloud_delivery"
)
ARMS = {
    "adamw_wmse": BASE / "gpu_loss_pair_r5_export/terminal_collect_r1/attempt_001/outputs/out/arms/wmse/model/model.pt",
    "adamw_huber_d1": BASE / "gpu_loss_pair_r5_export/terminal_collect_r1/attempt_001/outputs/out/arms/huber_d1/model/model.pt",
    "muon_wmse": BASE / "d02_r5/terminal_collect_r1/attempt_001/outputs/out/arms/wmse/model/model.pt",
    "muon_huber_d1": BASE / "d02_r5/terminal_collect_r1/attempt_001/outputs/out/arms/huber_d1/model/model.pt",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_of(obj):
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "model_state", "module"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    return obj


def spectral_stats(m: torch.Tensor) -> dict:
    m = m.detach().float()
    s = torch.linalg.svdvals(m)
    ps = s / s.sum().clamp_min(1e-12)
    entropy = -(ps * torch.log(ps.clamp_min(1e-12))).sum().item()
    erank = math.exp(entropy) / min(m.shape)
    fro2 = (s**2).sum().item()
    stable = fro2 / max((s[0] ** 2).item(), 1e-12) / min(m.shape)
    cum = torch.cumsum(s, 0) / s.sum().clamp_min(1e-12)
    e95 = int((cum < 0.95).sum().item()) + 1
    return {
        "shape": list(m.shape),
        "erank_norm": round(erank, 4),
        "stable_rank_norm": round(stable, 4),
        "energy95_rank": e95,
        "min_dim": int(min(m.shape)),
    }


def main() -> None:
    out = {"arms": {}, "note": "zero-fit spectral diagnostic on frozen D02 checkpoints"}
    for arm, path in ARMS.items():
        if not path.exists():
            out["arms"][arm] = {"error": "missing", "path": str(path)}
            continue
        sd = state_dict_of(torch.load(path, map_location="cpu", weights_only=True))
        stats = {}
        for name, t in sd.items():
            if t.ndim == 2 and t.numel() >= 64 and t.dtype.is_floating_point:
                stats[name] = spectral_stats(t)
        out["arms"][arm] = {
            "checkpoint_sha256": sha256(path),
            "params": stats,
        }
    # matched WMSE comparison table
    keys = ["gru.weight_ih_l0", "gru.weight_hh_l0"]
    print(f"{'param':<18}{'stat':<18}{'adamw_wmse':<12}{'muon_wmse':<12}{'delta':<10}")
    for k in keys:
        a = out["arms"].get("adamw_wmse", {}).get("params", {}).get(k)
        m = out["arms"].get("muon_wmse", {}).get("params", {}).get(k)
        if a and m:
            for stat in ("erank_norm", "stable_rank_norm", "energy95_rank"):
                print(f"{k:<18}{stat:<18}{a[stat]:<12}{m[stat]:<12}{m[stat]-a[stat]:<10.4f}")
    dest = Path(
        "research/experiment_drafts/own-optimizer-muon-20261024/experiments/results/C5_RANK_D02_R1.json"
    )
    dest.write_text(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()

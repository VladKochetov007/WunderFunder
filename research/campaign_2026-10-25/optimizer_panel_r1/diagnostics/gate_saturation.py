"""Zero-fit E0: GRU gate saturation and memory timescales, matched AdamW vs Muon checkpoints.

Replays each checkpoint's exact recurrence (linear encoder; PyTorch gate order r, z, n;
h' = (1 - z) * n + z * h) on held-out F0 evaluation sequences and reports the fraction of
saturated gates, the distribution of update-gate memory timescales 1/(1 - z), and the update
gate's preactivation contributions from input vs recurrence. Usage:
gate_saturation.py OUT_JSON N_SEQ RUN [RUN ...]
"""

import json
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "src")
from wnn.features import event_features  # noqa: E402

torch.set_num_threads(2)
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
NON_FEATURES = {"seq_ix", "step_in_seq", "need_prediction", "t0", "t1", "is_scored"}


def load(run: str) -> dict:
    ep = run.split("@")[1] if "@" in run else "2"
    run = run.split("@")[0]
    state = torch.load(f"artifacts/runs/{run}/model_ep{ep}.pt", map_location="cpu", weights_only=False)
    return state.get("model", state)


def main() -> None:
    out, n_seq, runs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    split = json.load(open("research/campaign_2026-09-29/artifacts/splits/fold0.json"))
    groups = [int(g) for g in split["evaluation"]["valid"]][:n_seq]
    pf = pq.ParquetFile("datasets/valid.parquet")
    names = [n for n in pf.schema_arrow.names if n not in NON_FEATURES]
    raw = []
    for g in groups:
        t = pf.read_row_group(g, columns=names)
        raw.append(np.column_stack([t[n].to_numpy() for n in names]).astype(np.float32))
    report = {"n_seq": n_seq, "groups": groups, "runs": {}}
    for run in runs:
        w = load(run)
        width = w["mean"].shape[0]
        hidden = w["gru.weight_hh_l0"].shape[1]
        stats = {k: [] for k in ("z_sat", "r_sat", "z_mean", "tau_q50", "tau_q90", "tau_q99", "z_in_rms", "z_rec_rms")}
        for x in raw:
            ev = event_features(x)
            feats = np.concatenate([x, ev, np.zeros((len(x), width - x.shape[1] - ev.shape[1]), np.float32)], 1)
            xn = (torch.from_numpy(feats) - w["mean"]) / w["std"]
            u = xn @ w["inp.weight"].T + w["inp.bias"]
            gi = u @ w["gru.weight_ih_l0"].T + w["gru.bias_ih_l0"]
            whh, bhh = w["gru.weight_hh_l0"], w["gru.bias_hh_l0"]
            h = torch.zeros(hidden)
            zs, rs, zin, zrec = [], [], [], []
            for step in range(99, len(x), 1):
                gh = whh @ h + bhh
                r = torch.sigmoid(gi[step, :hidden] + gh[:hidden])
                z = torch.sigmoid(gi[step, hidden:2 * hidden] + gh[hidden:2 * hidden])
                n = torch.tanh(gi[step, 2 * hidden:] + r * gh[2 * hidden:])
                h = (1 - z) * n + z * h
                if step % 10 == 0:
                    zs.append(z)
                    rs.append(r)
                    zin.append(gi[step, hidden:2 * hidden])
                    zrec.append(gh[hidden:2 * hidden])
            z, r = torch.stack(zs), torch.stack(rs)
            tau = 1.0 / (1.0 - z).clamp_min(1e-6)
            stats["z_sat"].append(float(((z < 0.02) | (z > 0.98)).float().mean()))
            stats["r_sat"].append(float(((r < 0.02) | (r > 0.98)).float().mean()))
            stats["z_mean"].append(float(z.mean()))
            q = torch.quantile(tau.flatten()[::7], torch.tensor([0.5, 0.9, 0.99]))
            stats["tau_q50"].append(float(q[0]))
            stats["tau_q90"].append(float(q[1]))
            stats["tau_q99"].append(float(q[2]))
            stats["z_in_rms"].append(float(torch.stack(zin).pow(2).mean().sqrt()))
            stats["z_rec_rms"].append(float(torch.stack(zrec).pow(2).mean().sqrt()))
        report["runs"][run] = {k: float(np.mean(v)) for k, v in stats.items()}
        print(run, {k: round(v, 4) for k, v in report["runs"][run].items()}, flush=True)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()

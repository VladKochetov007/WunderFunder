"""Zero-fit optimizer-state and weight-spectrum diagnostics for matched AdamW vs Muon fast-recipe fits."""
import json
import sys

import torch

torch.set_num_threads(2)
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
RUNS = sys.argv[1:-1]
OUT = sys.argv[-1]


def stats(x: torch.Tensor) -> dict:
    x = x.flatten().double()
    q = torch.quantile(x, torch.tensor([0.05, 0.5, 0.95], dtype=torch.double))
    return {"q05": float(q[0]), "q50": float(q[1]), "q95": float(q[2]), "mean": float(x.mean())}


report = {}
for run in RUNS:
    s = torch.load(f"artifacts/runs/{run}/training_state.pt", map_location="cpu", weights_only=False)
    model = s["model"]
    names = [k for k in model if k not in ("mean", "std")]
    opt = s["optimizer"]
    group_params = [i for g in opt["param_groups"] for i in g["params"]]
    beta1, beta2 = opt["param_groups"][0].get("betas", (0.9, 0.999))
    r = {"params": {}}
    for idx, name in zip(group_params, names):
        st = opt["state"].get(idx, {})
        w = model[name].float()
        entry = {"shape": list(w.shape), "weight_rms": float(w.pow(2).mean().sqrt())}
        m = st.get("exp_avg", st.get("momentum_buffer", st.get("m")))
        v = st.get("exp_avg_sq", st.get("v"))
        if m is not None:
            entry["m_rms"] = float(m.float().pow(2).mean().sqrt())
        if m is not None and v is not None:
            snr = m.float().abs() / (v.float().sqrt() + 1e-12)
            entry["adam_snr_|m|/sqrt(v)"] = stats(snr)
            entry["v_sqrt"] = stats(v.float().sqrt())
        if w.ndim == 2:
            rows = w.shape[0]
            blocks = 3 if name.startswith("gru.weight") else 1
            sv_list = []
            for b in range(blocks):
                blk = w[b * rows // blocks:(b + 1) * rows // blocks]
                sv = torch.linalg.svdvals(blk.double())
                sr = float((sv.pow(2).sum() / sv[0].pow(2)))
                sv_list.append({"sv_max": float(sv[0]), "sv_min": float(sv[-1]), "stable_rank": sr,
                                "sv_q50": float(sv.median()), "cond": float(sv[0] / sv[-1].clamp_min(1e-12))})
            entry["blocks"] = sv_list
            if name == "gru.weight_hh_l0":
                hh = w[2 * rows // 3:].double() if False else None
        r["params"][name] = entry
    report[run] = r
json.dump(report, open(OUT, "w"), indent=1)
for run, r in report.items():
    print("==", run)
    for n, e in r["params"].items():
        line = f" {n:22s} wrms {e['weight_rms']:.3f}"
        if "adam_snr_|m|/sqrt(v)" in e:
            q = e["adam_snr_|m|/sqrt(v)"]; line += f" snr q50 {q['q50']:.3f} q95 {q['q95']:.3f}"
        if "m_rms" in e:
            line += f" m_rms {e['m_rms']:.2e}"
        if "blocks" in e:
            line += " | " + " ".join(f"[sv {b['sv_max']:.2f}/{b['sv_min']:.3f} sr {b['stable_rank']:.1f}]" for b in e["blocks"])
        print(line)

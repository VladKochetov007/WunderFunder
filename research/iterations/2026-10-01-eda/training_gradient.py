"""TDIAG2: fixed saved-checkpoint geometry, no parameter update or validation score."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from wnn.data import SequenceFile
from wnn.features import event_features_torch
from wnn.nn import RNNModel, weighted_pooled_corr_loss
from wnn.outcomes import sequence_class, structure_labels
from train_rnn import next_change_labels

WINDOWS = ((0, 2000), (8000, 10000), (18000, 20000))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(path):
    s = Path(path).stat()
    return [s.st_size, s.st_mtime_ns, s.st_ino]


def write(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def names(seed, fold):
    return (f"ref{'s4' if seed == 4 else ''}_f{fold}/model_ep3.pt",
            f"ft128{'s4' if seed == 4 else ''}_f{fold}h/model.pt")


def gradient_summary(grads, shared, hidden):
    gm, gs, gn = grads
    ga = gs + gn
    bias = shared + 23 * hidden
    def norm(g):
        return float(g.norm())
    def cos(a, b):
        den = norm(a) * norm(b)
        return float(torch.dot(a, b) / den) if den else None
    return {
        "shared_cosine_main_struct": cos(gm[:shared], gs[:shared]),
        "shared_cosine_main_next": cos(gm[:shared], gn[:shared]),
        "shared_cosine_main_aux": cos(gm[:shared], ga[:shared]),
        "norms_all": [norm(g) for g in grads],
        "norms_shared": [norm(g[:shared]) for g in grads],
        "norms_main_private": [norm(torch.cat([g[shared:shared+2*hidden], g[bias:bias+2]])) for g in grads],
        "norms_aux_private": [norm(torch.cat([g[shared+2*hidden:bias], g[bias+2:]])) for g in grads],
        "clip_control": min(1., 1. / (norm(gm+ga)+1e-6)),
        "clip_main_only": min(1., 1. / (norm(gm)+1e-6)),
    }


def examine(head, features, y, mask, labels, chosen, window):
    a, b = window
    sp, br, cl, nr, nu, nok = labels
    p = head(features[:, a:b])
    ce = sum(F.cross_entropy(p[..., lo:hi].reshape(-1, hi-lo), lab[:, a:b].reshape(-1),
                              reduction="none").reshape(len(y), b-a)
             for lo, hi, lab in ((2, 14, sp), (14, 17, br), (17, 21, cl)))
    ne = (p[..., 21]-(nr[:, a:b]-3)/2).square() + F.binary_cross_entropy_with_logits(
        p[..., 22], nu[:, a:b], reduction="none")
    params = list(head.parameters())
    shared = params[0].numel() + params[1].numel()
    m0 = mask[:, a:b].float()
    nw0 = (mask & nok)[:, a:b].float()
    def gradients(m, nw):
        losses = [weighted_pooled_corr_loss(p[..., :2], y[:, a:b], m),
                  .3*(ce*m).sum()/m.sum().clamp_min(1e-8),
                  .3*(ne*nw).sum()/nw.sum().clamp_min(1e-8)]
        g = [torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss, params, retain_graph=True)])
             for loss in losses]
        if not all(torch.isfinite(v).all() for v in g):
            raise ValueError("nonfinite gradients")
        return gradient_summary(g, shared, head[2].in_features)
    result = gradients(m0, nw0)
    w = y[:, a:b].clamp(-2, 2).abs() * m0[..., None]
    mass = w.sum((1, 2)).detach().numpy()
    result.update(rows=list(window), scored_count=int(m0.sum()), next_label_count=int(nw0.sum()),
                  chain_metric_mass=mass.tolist(), chain_aux_count=m0.sum(1).tolist(),
                  class_counts=np.bincount([s["class"] for s in chosen], minlength=4).tolist(),
                  class_metric_mass=[float(sum(mass[i] for i, s in enumerate(chosen) if s["class"] == c)) for c in range(4)],
                  metric_mass_concentration_count=float(mass.sum()**2 / np.square(mass).sum()) if mass.sum() else None,
                  fraction_abs_p_gt2=float(((p[..., :2].detach().abs()>2)*m0[..., None]).sum()/(2*m0.sum()).clamp_min(1)),
                  weight_fraction_abs_p_gt2=float((w*(p[..., :2].detach().abs()>2)).sum()/w.sum().clamp_min(1e-8)))
    loo = []
    for i, selected in enumerate(chosen):
        m, nw = m0.clone(), nw0.clone()
        m[i] = 0; nw[i] = 0
        r = gradients(m, nw)
        loo.append({"omitted_chain": selected["chain"], "omitted_metric_mass_share": float(mass[i]/mass.sum()) if mass.sum() else None,
                    "shared_cosine_main_aux": r["shared_cosine_main_aux"],
                    "clip_control": r["clip_control"], "clip_main_only": r["clip_main_only"]})
    result["leave_one_chain_out"] = loo
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out, inv = args.out, args.out.with_suffix(".inputs.json")
    if out.exists() or inv.exists():
        raise FileExistsError("immutable output already exists")
    start = time.monotonic()
    result = {"status": "started", "cases": [], "paired_stage": [], "optimizer_updates": 0,
              "WP_queries": 0, "precision": "CPU float32, no AMP or historical Adam replay"}
    def alarm(signum, frame):
        raise TimeoutError(f"signal {signum}")
    signal.signal(signal.SIGTERM, alarm); signal.signal(signal.SIGALRM, alarm); signal.alarm(870)
    try:
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        root = args.source_root
        cfpath, clpath = root/"cache/chain_folds.npz", root/"research/artifacts/market-classes-v1.npz"
        data = root/"datasets/valid.parquet"
        paths = [cfpath, clpath, data]
        checkpoints = [root/"artifacts/runs"/n for f in (0, 1) for s in (3, 4) for n in names(s, f)]
        before = {str(p): fingerprint(p) for p in paths+checkpoints}
        cf, cls = np.load(cfpath), np.load(clpath)["valid"]
        selected = {}
        for fold in (0, 1):
            valid = (cf["split"] == "valid") & cf["pure"] & (cf["fold"] != fold)
            reps = {}
            for ch, rg, f in zip(cf["chain"][valid], cf["row_group"][valid], cf["fold"][valid]):
                key = hashlib.sha256(f"TDIAG2:row:{int(rg)}".encode()).hexdigest()
                if int(ch) not in reps or key < reps[int(ch)][2]:
                    reps[int(ch)] = (int(rg), int(f), key)
            chains = sorted(reps, key=lambda ch: (hashlib.sha256(f"TDIAG2:{ch}".encode()).hexdigest(), ch))[:16]
            if len(chains) != 16:
                raise ValueError("insufficient eligible chains")
            selected[fold] = [{"chain": ch, "row_group": reps[ch][0], "fold": reps[ch][1], "class": int(cls[reps[ch][0]])} for ch in chains]
        result["selected"] = selected
        result["overlap_chains_across_model_folds"] = sorted({r["chain"] for r in selected[0]} & {r["chain"] for r in selected[1]})
        ds = SequenceFile(data)
        raw, row_hashes = {}, {}
        for rg in sorted({r["row_group"] for rows in selected.values() for r in rows}):
            seq = ds.read(rg)
            if seq.features.shape != (20000, 112) or seq.targets.shape != (20000, 2) or seq.scored is None:
                raise ValueError("unexpected sequence shapes")
            raw[rg] = (seq.features.copy(), seq.targets.copy(), seq.scored.copy())
            h = hashlib.sha256()
            for a in raw[rg]:
                h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
            row_hashes[rg] = h.hexdigest()
        code = [Path(__file__), Path(__file__).with_name("training-gradient-plan.json"),
                REPO/"src/wnn/nn.py", REPO/"src/wnn/features.py", REPO/"src/wnn/outcomes.py", REPO/"scripts/train_rnn.py"]
        write(inv, {"selected": selected, "used_array_sha256": row_hashes, "fingerprints": before,
                    "hashes": {str(p): sha(p) for p in checkpoints+[cfpath, clpath]+code},
                    "data_role": "FT-training valid rows excluding held fold and impure chains; class is descriptive only"})
        result["inventory_sha256"] = sha(inv)
        for fold in (0, 1):
            rows = selected[fold]
            y = torch.from_numpy(np.stack([raw[r["row_group"]][1] for r in rows]))
            valid = torch.arange(20000)[None].expand(16, -1) >= 99
            mask = valid & torch.from_numpy(np.stack([raw[r["row_group"]][2] for r in rows])).bool()
            sp, br, cl = structure_labels(y, sequence_class(y))
            nr, nu, nok = next_change_labels(y, valid)
            labels = sp, br, cl, nr, nu, nok
            for seed in (3, 4):
                parent_path, head_path = [root/"artifacts/runs"/n for n in names(seed, fold)]
                parent, child = [torch.load(p, map_location="cpu", weights_only=False) for p in (parent_path, head_path)]
                if parent["n_in"] != child["n_in"] or parent["n_in"] != 211:
                    raise ValueError("input shape mismatch")
                trunk = [k for k in parent["model"] if not k.startswith("head.")]
                if set(parent["model"]) != set(child["model"]) or any(not torch.equal(parent["model"][k], child["model"][k]) for k in trunk):
                    raise ValueError("parent/head trunk mismatch; cannot reuse features")
                for ck in (parent, child):
                    cfg = ck["args"]
                    if cfg.get("seed") != seed or cfg.get("hidden") != 128 or cfg.get("dropout", 0) != 0:
                        raise ValueError("parent identity or dropout mismatch")
                    if cfg.get("aux_struct") != .3 or cfg.get("aux_next") != .3:
                        raise ValueError("auxiliary recipe mismatch")
                    if any(cfg.get(k) for k in ("online_norm", "heuristics", "raw_only", "quant_feats", "quant_i1", "class_heads", "lookahead", "level_aug", "level_drop")):
                        raise ValueError("unsupported model/preprocessing recipe")
                cfg = child["args"]
                if cfg.get("freeze") != "inp,gru" or cfg.get("loss") != "gcorr" or not cfg.get("mask_weight"):
                    raise ValueError("unexpected head FT recipe")
                if any(cfg.get(k) for k in ("propensity", "since_weight", "episode_weight", "consistency")):
                    raise ValueError("unsupported head weighting")
                model = RNNModel(211, 128, parent["args"]["layers"], 0, n_out=23)
                model.load_state_dict(parent["model"], strict=True); model.eval()
                feats = []
                with torch.no_grad():
                    for j in range(0, 16, 4):
                        x = torch.from_numpy(np.stack([raw[r["row_group"]][0] for r in rows[j:j+4]]))
                        inp = torch.cat([x, event_features_torch(x)], -1)
                        z = model.encode((inp-model.mean)/model.std)
                        recurrent, _ = model.gru(z)
                        feats.append(torch.cat([recurrent, z], -1))
                features = torch.cat(feats); del feats
                stage_records = []
                for stage, ck in (("parent_ep3", parent), ("saved_head1", child)):
                    model.load_state_dict(ck["model"], strict=True)
                    rec = {"fold": fold, "seed": seed, "stage": stage, "trunk_equal_asserted": True,
                           "eval_train_equivalent": "dropout0, linear encoder/head; no stochastic augmentation",
                           "windows": [examine(model.head, features, y, mask, labels, rows, win) for win in WINDOWS]}
                    result["cases"].append(rec); stage_records.append(rec)
                result["paired_stage"].append({"fold": fold, "seed": seed, "deltas_head1_minus_parent": [
                    {"rows": list(win), "shared_cosine_main_aux": b["shared_cosine_main_aux"]-a["shared_cosine_main_aux"]
                     if a["shared_cosine_main_aux"] is not None and b["shared_cosine_main_aux"] is not None else None,
                     "clip_control": b["clip_control"]-a["clip_control"],
                     "weight_fraction_abs_p_gt2": b["weight_fraction_abs_p_gt2"]-a["weight_fraction_abs_p_gt2"]}
                    for win, a, b in zip(WINDOWS, stage_records[0]["windows"], stage_records[1]["windows"])]})
                del features, model, parent, child
        if any(fingerprint(p) != fp for p, fp in before.items()):
            raise RuntimeError("input changed during diagnostic")
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "timeout" if isinstance(exc, TimeoutError) else "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        result["wall_seconds"] = time.monotonic()-start
        write(out, result)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

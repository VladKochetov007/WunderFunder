"""Stream fixed F0 endpoint and H512 member diagnostics with NumPy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

BINDING = Path(__file__).with_name("SOURCE_BINDING_R1.json")
BINDING_SHA = "1988dba4f0672e1d3cd275e6e801e7ed02ba694e280b6f581b2aae21eb173698"
PLAN_SHA = "9066b72851ec401b2f0a4dc67de998f132874e8479f5f8ea04345e2b5e642ebf"
NAMES = ("B", "R", "S", "H", "Hswa")
BINS = ((99, 500), (500, 2000), (2000, 10000), (10000, 20000))


def need(ok: bool, message: str) -> None:
    """Fail closed when a frozen input differs."""
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def resolve(root: Path, value: str | Path) -> Path:
    """Resolve an absolute or repository-relative frozen path."""
    path = Path(value)
    return path.resolve(strict=True) if path.is_absolute() else (root / path).resolve(strict=True)


def frozen_paths(
    root: Path, bind: dict[str, Any], plan_path: Path
) -> tuple[dict[str, Path], dict[str, str]]:
    """Collect the exact source, receipt, and prediction paths."""
    paths = {"plan": plan_path, "binding": BINDING.resolve(strict=True)}
    expected = {"plan": PLAN_SHA, "binding": BINDING_SHA}
    for section in ("reference_files", "source_pins"):
        for name, rec in bind[section].items():
            paths[name], expected[name] = resolve(root, rec["path"]), rec["sha256"]
    paths["target_cache"] = resolve(root, bind["target_cache"]["path"])
    expected["target_cache"] = bind["target_cache"]["sha256"]
    preds = {
        "B": bind["baseline"],
        "R": bind["native_endpoints"]["cp5"],
        "S": bind["native_endpoints"]["swa345"],
        "H": bind["h512_member_predictions"]["cp5"],
        "Hswa": bind["h512_member_predictions"]["swa345"],
    }
    for name, rec in preds.items():
        for kind, value, digest in (
            ("prediction", rec["path"], rec["sha256"]),
            ("sidecar", rec["sidecar"], rec["sidecar_sha256"]),
        ):
            key = f"{name}_{kind}"
            paths[key], expected[key] = resolve(root, value), digest
    paths["script"] = Path(__file__).resolve(strict=True)
    return paths, expected


def pair(state: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Merge one sequence's weighted bivariate centered moments."""
    keep = w > 0
    if not keep.any():
        return state
    x, y, w = (np.asarray(v[keep], dtype=np.float64) for v in (x, y, w))
    wb = float(w.sum())
    mx = float(x[0] + np.sum(w * (x - x[0])) / wb)
    my = float(y[0] + np.sum(w * (y - y[0])) / wb)
    dx, dy = x - mx, y - my
    batch = np.array([wb, mx, my, np.sum(w * dx**2), np.sum(w * dy**2), np.sum(w * dx * dy)])
    if state[0] == 0:
        return batch
    total = state[0] + wb
    dx, dy = mx - state[1], my - state[2]
    cross = state[0] * wb / total
    return np.array(
        [
            total,
            state[1] + dx * wb / total,
            state[2] + dy * wb / total,
            state[3] + batch[3] + dx**2 * cross,
            state[4] + batch[4] + dy**2 * cross,
            state[5] + batch[5] + dx * dy * cross,
        ]
    )


def summary(s: np.ndarray) -> dict[str, float]:
    """Convert weighted centered moments to interpretable statistics."""
    w, mx, my, xx, yy, xy = map(float, s)
    need(w > 0, "weighted moments have no positive target weight")
    vx, vy, cov = xx / w, yy / w, xy / w
    corr = cov / math.sqrt(vx * vy) if vx > 1e-16 and vy > 1e-16 else 0.0
    return {
        "weight_sum": w,
        "target_mean": mx,
        "prediction_mean": my,
        "target_std": math.sqrt(max(vx, 0)),
        "prediction_std": math.sqrt(max(vy, 0)),
        "covariance": cov,
        "correlation": float(np.clip(corr, -1, 1)),
    }


def raw_update(s: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Merge unweighted per-target raw-output moments for one sequence."""
    n = len(values)
    if not n:
        return s
    v = values.astype(np.float64)
    mean, m2 = v.mean(axis=0), np.square(v - v.mean(axis=0)).sum(axis=0)
    if s[0] == 0:
        return np.array([n, *mean, *m2])
    total, delta = s[0] + n, mean - s[1:3]
    return np.array(
        [total, *(s[1:3] + delta * n / total), *(s[3:5] + m2 + delta**2 * s[0] * n / total)]
    )


def raw_summary(state: np.ndarray) -> dict[str, list[float]]:
    """Read stored means and population variances from a merged raw state."""
    return {
        "mean": state[1:3].tolist(),
        "std": np.sqrt(state[3:5] / state[0]).tolist(),
    }


def load_prediction(
    path: Path, sidecar_path: Path, rec: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate one NPY header and sidecar, then map its payload."""
    side = read_json(sidecar_path)
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, order, dtype = reader(stream)
        header = stream.tell()
    dtype = np.dtype(dtype)
    need(
        not order and list(shape) == rec["shape"] and dtype == np.float32,
        f"array layout differs: {path}",
    )
    need(
        path.stat().st_size == header + math.prod(shape) * dtype.itemsize,
        f"NPY payload size differs: {path}",
    )
    need(
        sha(path) == rec["sha256"] and side.get("prediction_sha256") == rec["sha256"],
        f"prediction hash differs: {path}",
    )
    return np.load(path, mmap_mode="r", allow_pickle=False), side


def main() -> int:
    """Run the frozen, create-only, sequence-streamed F0 diagnosis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    need(sha(BINDING) == BINDING_SHA, "source binding hash changed")
    bind = read_json(BINDING)
    root = (args.root or Path(bind["root"])).resolve(strict=True)
    need(str(root) == bind["root"], "project root differs from source binding")
    plan_path = resolve(root, bind["frozen_plan"]["path"])
    need(sha(plan_path) == PLAN_SHA, "frozen plan hash changed")
    plan = read_json(plan_path)
    output = args.output or Path(bind["default_output"])
    need(not output.exists(), f"refusing to overwrite {output}")
    paths, expected = frozen_paths(root, bind, plan_path)
    before = {name: sha(path) for name, path in paths.items()}
    need(
        all(before[name] == digest for name, digest in expected.items()),
        "frozen input digest mismatch",
    )

    grade, endpoint_plan = (
        read_json(paths["native_grade"]),
        read_json(paths["endpoint_screen_plan"]),
    )
    need(grade.get("official_metric") == "global_wp_clip2_v1", "native metric changed")
    need(grade.get("plan_sha256") == expected["endpoint_screen_plan"], "grade plan pin differs")
    need(
        grade.get("input_helper_sha256") == expected["endpoint_score_inputs"],
        "grade helper pin differs",
    )
    need(
        endpoint_plan.get("schema") == "wnn.tl3-h512-endpoint-screen.v1",
        "endpoint plan schema changed",
    )
    packet = (
        root / "research/campaign_2026-10-20/submission_delivery_r1/tl3_h512_endpoint_screen_r1"
    )
    sys.path[:0] = [str(packet), str(root / "src")]
    from endpoint_score_inputs import load_campaign_split, load_targets

    from wnn.gmetric import GlobalAccumulator

    source = load_campaign_split(root)
    ys, masks, components, groups_array, identity = load_targets(root, source)
    groups = list(map(int, groups_array))
    need(len(groups) == 468 and len(np.unique(components)) == 141, "F0 panel identity differs")
    need(ys.shape == (468, 20_000, 2) and masks.shape == (468, 20_000), "target shape differs")
    records = {
        "B": bind["baseline"],
        "R": bind["native_endpoints"]["cp5"],
        "S": bind["native_endpoints"]["swa345"],
        "H": bind["h512_member_predictions"]["cp5"],
        "Hswa": bind["h512_member_predictions"]["swa345"],
    }
    arrays, sides = {}, {}
    for name, rec in records.items():
        arrays[name], sides[name] = load_prediction(
            paths[f"{name}_prediction"], paths[f"{name}_sidecar"], rec
        )
    seqs = list(map(int, identity["sequence_ids"]))
    for name in ("B", "R", "S"):
        need(
            list(map(int, sides[name].get("row_groups", []))) == groups, f"{name} row order differs"
        )
        need(
            list(map(int, sides[name].get("sequence_ids", []))) == seqs,
            f"{name} sequence order differs",
        )
    full_valid = set(map(int, source.split.valid_groups)) | set(groups)
    h_rows = {}
    for name in ("H", "Hswa"):
        side, rec = sides[name], records[name]
        rows = list(map(int, side.get("row_groups", [])))
        need(
            side.get("checkpoint_sha256") == rec["checkpoint_sha256"], f"{name} checkpoint differs"
        )
        need(
            side.get("split_manifest_sha256") == source.split.sha256
            and len(rows) == 1873
            and set(rows) == full_valid,
            f"{name} full-valid coverage differs",
        )
        h_rows[name] = {group: i for i, group in enumerate(rows)}
    retained = {}
    for name, endpoint in (("R", "cp5"), ("S", "swa345")):
        package = grade["endpoint_sources"][endpoint]["package_binding"]
        retained[endpoint] = [
            (m.get("name"), m.get("weights"), m.get("coef")) for m in package["members"][1:]
        ]
        need(
            sides[name].get("zip_sha256") == package["zip_sha256"],
            f"{name} package identity differs",
        )

    scores = {n: GlobalAccumulator() for n in NAMES}
    bins = {(a, b): {n: GlobalAccumulator() for n in NAMES} for a, b in BINS}
    moments = {n: [np.zeros(6) for _ in range(2)] for n in NAMES}
    cross = {n: [np.zeros(6) for _ in range(2)] for n in ("H", "Hswa")}
    raw = {n: np.zeros(5) for n in NAMES}
    clipped = {n: np.zeros(2, dtype=np.int64) for n in NAMES}
    target_raw, target_clipped = np.zeros(5), np.zeros(2, dtype=np.int64)
    parity_sum, parity_max, parity_count = (
        {"raw": 0.0, "clipped": 0.0},
        {"raw": 0.0, "clipped": 0.0},
        0,
    )
    pos = np.arange(20_000)
    for i, group in enumerate(groups):
        y, mask = ys[i], masks[i]
        pred = {
            "B": arrays["B"][i],
            "R": arrays["R"][i],
            "S": arrays["S"][i],
            "H": arrays["H"][h_rows["H"][group]],
            "Hswa": arrays["Hswa"][h_rows["Hswa"][group]],
        }
        need(mask.any() and np.isfinite(y[mask]).all(), f"invalid target rows in {group}")
        yc = np.clip(y, -2, 2)
        target_raw = raw_update(target_raw, y[mask])
        target_clipped += np.count_nonzero(np.abs(y[mask]) > 2, axis=0)
        pc = {}
        for name, p in pred.items():
            need(np.isfinite(p).all(), f"nonfinite {name} predictions in {group}")
            scores[name].add(y[None], p[None], mask[None])
            pc[name] = np.clip(p, -2, 2)
            raw[name] = raw_update(raw[name], p[mask])
            clipped[name] += np.count_nonzero(np.abs(p[mask]) > 2, axis=0)
            for k in range(2):
                moments[name][k] = pair(
                    moments[name][k], yc[mask, k], pc[name][mask, k], np.abs(yc[mask, k])
                )
            for a, b in BINS:
                bm = mask & (pos >= a) & (pos < b)
                bins[a, b][name].add(y[None], p[None], bm[None])
        for name in ("H", "Hswa"):
            for k in range(2):
                cross[name][k] = pair(
                    cross[name][k], pc["B"][mask, k], pc[name][mask, k], np.abs(yc[mask, k])
                )
        delta = pred["R"].astype(np.float64) - pred["S"].astype(np.float64)
        for label, hp, hsp in (("raw", pred["H"], pred["Hswa"]), ("clipped", pc["H"], pc["Hswa"])):
            resid = delta - 0.34 * (hp.astype(np.float64) - hsp.astype(np.float64))
            # Native caches contain zeros where the callback returns no prediction.
            resid = resid[99:]
            parity_sum[label] += float(np.square(resid).sum(dtype=np.float64))
            parity_max[label] = max(parity_max[label], float(np.abs(resid).max(initial=0)))
            if label == "raw":
                parity_count += resid.size

    result = {n: s.result() for n, s in scores.items()}
    reproduced = {}
    for n, key in (
        ("B", "original_tl3"),
        ("R", "h512_cp5_replacement"),
        ("S", "h512_swa345_replacement"),
    ):
        want = grade["endpoint_scores"][key]
        diff = {k: result[n][k] - want[k] for k in ("wp", "t0", "t1")}
        need(
            result[n]["selected_rows"] == want["selected_rows"]
            and max(map(abs, diff.values())) <= 1e-7,
            f"{n} official score reproduction differs",
        )
        reproduced[n] = {"expected": want, "recomputed": result[n], "difference": diff}
    weighted = {n: {f"t{k}": summary(moments[n][k]) for k in range(2)} for n in NAMES}
    bin_report = {}
    for (a, b), accs in bins.items():
        vals = {n: acc.result() for n, acc in accs.items()}
        bin_report[f"{a}_{b}"] = {
            "scores": vals,
            "deltas_vs_B": {
                n: {k: vals[n][k] - vals["B"][k] for k in ("wp", "t0", "t1")}
                for n in NAMES
                if n != "B"
            },
        }
    ratios, partial = {}, {}
    for k in range(2):
        base = weighted["B"][f"t{k}"]
        ratios[f"t{k}"], partial[f"t{k}"] = {}, {}
        for n in ("R", "S"):
            c = weighted[n][f"t{k}"]
            cr = c["covariance"] / base["covariance"] if base["covariance"] else None
            scale = base["prediction_std"] / c["prediction_std"] if c["prediction_std"] else None
            ratios[f"t{k}"][n] = {
                "covariance_ratio": cr,
                "sqrt_var_B_over_var_candidate": scale,
                "decomposition_product": cr * scale
                if cr is not None and scale is not None
                else None,
                "correlation_ratio": c["correlation"] / base["correlation"]
                if base["correlation"]
                else None,
            }
        vy, vb = base["target_std"] ** 2, base["prediction_std"] ** 2
        for n in ("H", "Hswa"):
            h, bh = weighted[n][f"t{k}"], summary(cross[n][k])
            ry = max(vy - base["covariance"] ** 2 / vb, 0) if vb else 0
            rh = max(h["prediction_std"] ** 2 - bh["covariance"] ** 2 / vb, 0) if vb else 0
            cov = h["covariance"] - base["covariance"] * bh["covariance"] / vb if vb else 0
            partial[f"t{k}"][n] = cov / math.sqrt(ry * rh) if ry > 0 and rh > 0 else None

    after = {name: sha(path) for name, path in paths.items()}
    need(after == before, "a pinned input changed during diagnosis")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "wnn.h512-tl3-replacement-diagnosis.v1",
        "experiment_id": plan["experiment_id"],
        "status": "POSTHOC_EXPOSED_FITTED_DIAGNOSTIC",
        "exposure": plan["exposure"],
        "metric_id": "global_wp_clip2_v1",
        "selected_rows": result["B"]["selected_rows"],
        "input_sha256_before": before,
        "input_sha256_after": after,
        "target_identity": identity,
        "valid_parquet_identity": source.data_identity,
        "official_scores": result,
        "native_score_reproduction": reproduced,
        "full_fold_deltas_vs_B": {
            n: {k: result[n][k] - result["B"][k] for k in ("wp", "t0", "t1")}
            for n in NAMES
            if n != "B"
        },
        "position_bins": bin_report,
        "clipped_weighted_moments": weighted,
        "raw_selected_stats": {
            "target": raw_summary(target_raw),
            **{n: raw_summary(raw[n]) for n in NAMES},
        },
        "clipping_fraction": {
            "target": (target_clipped / target_raw[0]).tolist(),
            **{n: (clipped[n] / raw[n][0]).tolist() for n in NAMES},
        },
        "replacement_correlation_ratio_decomposition": ratios,
        "H_partial_target_correlation_conditional_on_B": partial,
        "retained_package_members_match": retained["cp5"] == retained["swa345"],
        "retained_members": retained,
        "parity": {
            "formula": "(R-S)-0.34*(H-Hswa)",
            "rms": {k: math.sqrt(v / parity_count) for k, v in parity_sum.items()},
            "max_abs": parity_max,
            "cells": parity_count,
            "rows": "All need_prediction rows; first 99 callback warmup rows excluded.",
            "interpretation": "Arithmetic diagnostic only; infer no retained-member identity unless package identities match.",
        },
        "limits": "F0 is exposed; TL3 ancestry is unknown. Descriptive fitted diagnostics do not establish causality or public transfer.",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"output": str(output), "status": report["status"], "scores": result}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

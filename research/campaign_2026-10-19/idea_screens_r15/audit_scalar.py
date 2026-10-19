#!/usr/bin/env python3
"""One-shot R15 scalar audit. Reads only compact JSON receipts; never opens NPZs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MF = "ad26d2589ed34101b4eed1cd4655f7924c11349f5cb5ea3fd4a345189cd8f798"
SAMPLE = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBER = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
SPLIT = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
SERVICE, TOL = "computer-vision-homework-eda19-r15-v1.service", 1e-4

def ck(ok, msg):
    if not ok:
        raise ValueError(msg)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_bytes())

def wp(s):
    """W,WY,WP,WYY,WPP,WYP; canonical roundoff handling for raw six sums."""
    ck(len(s) == 2 and all(len(r) == 6 for r in s), "not a 2x6 WP sum")
    out, eps, tol = [], 1e-8, 8 * 2.220446049250313e-16
    for row in s:
        w, wy, p, yy, pp, yp = map(float, row)
        ck(all(math.isfinite(x) for x in (w, wy, p, yy, pp, yp)) and w >= 0, "invalid WP sums")
        if w < eps:
            out.append(0.0)
            continue
        my, mp, sy2, sp2 = wy/w, p/w, yy/w, pp/w
        vy, vp = sy2-my*my, sp2-mp*mp
        vy = vy if vy > tol*max(abs(sy2), my*my) else 0.0
        vp = vp if vp > tol*max(abs(sp2), mp*mp) else 0.0
        sy, sp, cov = math.sqrt(max(vy,0)), math.sqrt(max(vp,0)), yp/w-my*mp
        out.append(0.0 if min(sy,sp) <= eps else max(-1.0,min(1.0,cov/(sy*sp))))
    return {"t0":out[0], "t1":out[1], "wp":math.fsum(out)/2}

def pool(g, omit=None):
    keys = sorted(k for k in g if k != omit)
    ck(keys, "empty pool")
    return [[math.fsum(float(g[k][t][j]) for k in keys) for j in range(6)] for t in range(2)]


def err(a, b):
    return max(abs(float(a[k]) - b[k]) for k in ("t0", "t1", "wp"))


def distribution(values):
    ordered = sorted(float(v) for v in values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "count": len(ordered),
        "positive": sum(v > 0 for v in ordered),
        "negative": sum(v < 0 for v in ordered),
        "zero": sum(v == 0 for v in ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": math.fsum(ordered) / len(ordered),
        "median": median,
    }

def find(folder, role, choices):
    for n in choices:
        p = folder / n
        if p.is_file():
            return p
    return None

def run(folder, manifest_path, extras):
    choices={"result":("REMOTE_RESULT.json","RESULT.json"),"source":("REMOTE_SOURCE.json","SOURCE.json"),
      "done":("REMOTE_DONE.json","DONE.json"),"start":("REMOTE_START.json","START.json"),
      "validation":("ROOT_CAPTURE_VALIDATION.json","VALIDATION.json"),"stop":("STOP_RECEIPT.json",),
      "admission":("ROOT_ADMISSION.json",),"launch":("ROOT_LAUNCH.json",)}
    paths = {r: find(folder, r, n) for r, n in choices.items()}
    for item in extras:
        role, sep, path = item.partition("=")
        ck(sep and role and path, "--receipt format ROLE=PATH")
        paths[role] = Path(path)
    ck(all(paths.get(r) and paths[r].is_file() for r in choices), "missing required execution receipt")
    d = {k: read(p) for k, p in paths.items()}
    result, source, done, start = (d[k] for k in ("result", "source", "done", "start"))
    m = read(manifest_path)
    msha = sha(manifest_path)
    ck(msha == MF and len(m["files"]) == 31 and len(m["runtime_py"]) == 25, "reviewed R15 manifest mismatch")
    ck(source.get("source_pins") == m["files"] and set(source.get("runtime_imports", {})) == set(m["runtime_py"]), "runtime source pins/import inventory mismatch")
    for name,key in m["runtime_py"].items():
        suffix = key.split(":", 1)[-1] if key.startswith("canonical:") else key
        ck(str(source["runtime_imports"][name]).replace("\\", "/").endswith(suffix), f"runtime path mismatch: {name}")
    rs, ss = sha(paths["result"]), sha(paths["source"])
    ck(result.get("schema") == "nf19.r15.result.v1" and done.get("schema") == "nf19.r15.done.v1", "result schema mismatch")
    ck(result.get("source_sha256") == done.get("source_sha256") == ss and done.get("result_sha256") == rs and done.get("result_bytes") == paths["result"].stat().st_size, "result/source/done binding mismatch")
    ck(result.get("preparation_manifest_sha256") == msha, "result manifest binding mismatch")
    for r in (result, source, start):
        ck((r.get("sample_sha256"), r.get("membership_sha256"), r.get("split_sha256")) == (SAMPLE, MEMBER, SPLIT), "sample/membership/split binding mismatch")
    counts = (source.get("sample_records"), source.get("mapped_npy_file_count"), source.get("fit_components"), source.get("fit_rows"), source.get("eval_components"), source.get("eval_scored_rows"))
    ck(counts == (96, 288, 64, 31808, 32, 59436), "panel/mapping counts mismatch")
    ck(source.get("torch_threads") == source.get("torch_interop_threads") == 1, "Torch thread limits mismatch")
    ck(not source.get("cuda_initialized") and not source.get("autocast_cuda_enabled") and not source.get("autocast_cpu_enabled"), "CPU-only runtime flags mismatch")
    ck(not source.get("tf32_matmul_enabled") and not source.get("tf32_cudnn_enabled"), "TF32 runtime flags mismatch")

    ev = result["evaluation"]
    arms = ev.get("arms", {})
    ck(set(arms) == {"dense", "sparse"} and ev.get("metric_id") == "global_wp_clip2_v1", "evaluation arms/metric mismatch")
    summaries, official, rows = {}, {}, {}
    maxc = maxd = 0.0
    for arm,a in arms.items():
        groups = a["official_clipped_sufficient_statistics_w_wy_wp_wyy_wpp_wyp"]
        keys = set(groups)
        rows[arm] = sum(map(int, a["component_rows"].values()))
        ck(len(keys) == 32 and set(a["component_rows"]) == keys and rows[arm] == 59436, "component coverage mismatch")
        ck(set(a["component_scores"]) == keys and set(a["component_drop_one_scores"]) == keys, "component/drop-one metrics incomplete")
        ce = max(err(a["component_scores"][k], wp(groups[k])) for k in keys)
        pooled = wp(pool(groups))
        pe = err(a["official_metric"], pooled)
        de = max(err(a["component_drop_one_scores"][k], wp(pool(groups, k))) for k in keys)
        maxc, maxd = max(maxc, ce, pe), max(maxd, de)
        official[arm] = a["official_metric"]
        summaries[arm] = {"official": official[arm], "six_sum_pool": pooled, "components": 32, "rows": rows[arm], "max_component_or_pool_error": max(ce, pe), "max_drop_one_error": de}
    ck(rows["dense"] == rows["sparse"] and max(maxc, maxd) <= TOL, "paired masks/six-sum reconstruction mismatch")
    primary = official["sparse"]["wp"] - official["dense"]["wp"]
    targets = [official["sparse"][f"t{i}"] - official["dense"][f"t{i}"] for i in range(2)]
    ck(math.isclose(primary, result["primary_sparse_minus_dense_wp"], abs_tol=1e-12) and all(math.isclose(targets[i], result["target_deltas_sparse_minus_dense"][i], abs_tol=1e-12) for i in range(2)), "paired endpoint deltas mismatch")
    gate = result["prospective_gate"]
    passed = primary >= 0.005 and all(t > 0 for t in targets)
    ck(gate.get("minimum_pooled_delta") == 0.005 and gate.get("passed") == passed, "prospective gate mismatch")
    paired = {}
    for scope in ("component_scores", "component_drop_one_scores"):
        common = set(arms["dense"][scope]) & set(arms["sparse"][scope])
        ck(len(common) == 32 and set(arms["dense"][scope]) == set(arms["sparse"][scope]), f"paired {scope} keys differ")
        paired[scope] = {
            metric: distribution([arms["sparse"][scope][k][metric] - arms["dense"][scope][k][metric] for k in common])
            for metric in ("t0", "t1", "wp")
        }

    fit = result["fit"]
    dic, pred = fit["dictionaries"], fit["predictors"]
    ck(result.get("fit_count") == 4 and result.get("fit_total_updates") == 3024 and result.get("nonfinite_count") == 0, "fit totals mismatch")
    exposures = 0
    for stage in (dic,pred):
        for _arm,r in stage.items():
            ck(r.get("attempted_updates") == r.get("successful_updates") == 756 and r.get("nonfinite_count") == 0 and r.get("row_exposures") == 381696, "fit counters/exposures mismatch")
            epochs = r.get("epochs", [])
            ck(len(epochs) == 12 and [e.get("epoch") for e in epochs] == list(range(1, 13)) and all(e.get("successful_updates") == 63 * (i + 1) for i, e in enumerate(epochs)), "fit epoch records mismatch")
            exposures += r["row_exposures"]
    initial, frozen, final = (result[k] for k in ("dictionary_initial_sha256", "dictionary_frozen_sha256", "dictionary_final_sha256"))
    ck(initial["dense"] == initial["sparse"] and frozen == final and all(initial[a] != frozen[a] and dic[a]["initial_sha256"] == initial[a] and dic[a]["final_sha256"] == frozen[a] for a in ("dense", "sparse")), "dictionary initial/frozen/final fingerprints mismatch")
    pi, per_arm = result["predictor_initialization_sha256"], {}
    for arm in ("dense","sparse"):
        x, y = dic[arm], pred[arm]
        residual, norms = float(y.get("residual_identity_max_abs_all_fit_sequence_rows", math.inf)), x.get("final_decoder_column_norms", [])
        ck(y.get("initialization_sha256") == pi and y.get("unused_variance_head_unchanged") is True, "predictor init/unused head mismatch")
        ck(x.get("fit_stride_row_count") == y.get("fit_rows") == 31808 and y.get("feature_width") == 256 and y.get("normalizer_floor") == 1e-6 and float(y.get("normalizer_scale_min", 0)) >= 1e-6, "fit width/normalizer mismatch")
        ck(len(y.get("code_active_rows_by_fit_component_all_history", {})) == 64 and residual <= 1e-5 and len(norms) == 16 and max(abs(float(v) - 1) for v in norms) <= 2e-6, "full-history/residual/unit-column check mismatch")
        last = y["epochs"][-1]
        per_arm[arm] = {"fit_rows": y["fit_rows"], "feature_width": y["feature_width"], "scale_floor": y["normalizer_floor"], "min_scale": y["normalizer_scale_min"], "max_residual_identity_error": residual, "unused_variance_head_unchanged": y["unused_variance_head_unchanged"], "final_fit_wmse": last["fit_wmse"], "final_fit_wp": last["fit_wp"], "final_fit_t0": last["fit_t0_wp"], "final_fit_t1": last["fit_t1_wp"]}
    shapes = {k: v.get("shape") for k, v in result["model_archive"].get("arrays", {}).items()}
    ck(shapes.get("raw_input_mean") == shapes.get("raw_input_scale") == [112], "raw normalizer width mismatch")
    for a in ("dense", "sparse"):
        ck(shapes.get(f"{a}_encoder_weight") == [16, 112] and shapes.get(f"{a}_decoder_weight") == [112, 16] and shapes.get(f"{a}_downstream_mean") == shapes.get(f"{a}_downstream_scale") == [256], f"{a} array metadata shapes mismatch")

    validation, admission, launch, stop = (d[k] for k in ("validation", "admission", "launch", "stop"))
    caps = admission.get("caps", {})
    ck(admission.get("unit") == launch.get("unit") == SERVICE, "admission/launch service mismatch")
    ck(caps.get("cpu_cores") == 0.5 and caps.get("memory_high_bytes") == 1_400_000_000 and caps.get("memory_max_bytes") == 1_500_000_000, "admitted CPU/memory caps mismatch")
    ck(caps.get("memory_swap_max_bytes") == 0 and caps.get("runtime_max_s") == 600 and caps.get("blas_threads") == 1, "admitted swap/runtime/BLAS caps mismatch")
    ck(launch.get("systemd_run_returncode") == 0 and launch.get("invocation_id") == validation.get("invocation_id"), "launch/capture invocation mismatch")
    ck(validation.get("status") == "PASS_BEFORE_EXACT_STOP" and validation.get("result_sha256") == rs and validation.get("source_sha256") == ss, "capture validation binding/status mismatch")
    remote_stop = stop.get("remote", {})
    before, after = remote_stop.get("before", {}), remote_stop.get("after", {})
    ck(stop.get("returncode") == 0 and remote_stop.get("unit") == SERVICE, "stop receipt service/return code mismatch")
    ck(before.get("Id") == SERVICE and before.get("ActiveState") == "active" and before.get("SubState") == "exited" and before.get("Result") == "success" and str(before.get("ExecMainStatus")) == "0", "pre-stop service state mismatch")
    ck(before.get("MemoryHigh") == "1400000000" and before.get("MemoryMax") == "1500000000" and before.get("MemorySwapMax") == "0", "observed memory caps mismatch")
    ck(before.get("CPUQuotaPerSecUSec") == "500ms" and before.get("RuntimeMaxUSec") == "10min", "observed CPU/runtime caps mismatch")
    ck(after.get("Id") == SERVICE and after.get("ActiveState") == "inactive" and after.get("SubState") == "dead" and after.get("LoadState") == "not-found", "post-stop service state mismatch")
    ck(after.get("Result") == "success" and str(after.get("ExecMainStatus")) == "0", "post-stop result mismatch")
    rec = {
        "validation": {"file": paths["validation"].name, "sha256": sha(paths["validation"]), "schema": validation.get("schema"), "status": validation.get("status"), "invocation_id": validation.get("invocation_id")},
        "admission": {"file": paths["admission"].name, "sha256": sha(paths["admission"]), "unit": admission.get("unit"), "caps": caps},
        "launch": {"file": paths["launch"].name, "sha256": sha(paths["launch"]), "unit": launch.get("unit"), "invocation_id": launch.get("invocation_id"), "systemd_run_returncode": launch.get("systemd_run_returncode")},
        "stop": {"file": paths["stop"].name, "sha256": sha(paths["stop"]), "returncode": stop.get("returncode"), "unit": remote_stop.get("unit"), "before": {k: before.get(k) for k in ("ActiveState", "SubState", "Result", "ExecMainStatus", "MemoryHigh", "MemoryMax", "MemorySwapMax", "CPUQuotaPerSecUSec", "RuntimeMaxUSec")}, "after": {k: after.get(k) for k in ("ActiveState", "SubState", "LoadState", "Result", "ExecMainStatus")}},
    }
    return {"schema":"nf19.r15_scalar_audit.v1","status":"PASS_SCALAR_AUDIT","scope":"JSON receipts and six scalar sums only; no NPY/NPZ/tensor/prediction payload opened; no fit or scorer replay",
      "binding_sha256":{k:sha(paths[k]) for k in ("result","source","done","start")},
      "source_panel":{"manifest_sha256":msha,"source_pins":31,"runtime_modules":25,"fit_components":64,"fit_rows":31808,"eval_components":32,"eval_rows":59436,"mapped_npy":288},
      "scores":{"arms":summaries,"primary_sparse_minus_dense_wp":primary,"target_deltas_sparse_minus_dense":targets,"gate":{**gate,"recomputed_passed":passed},"max_component_or_pool_error":maxc,"max_drop_one_error":maxd,"tolerance":TOL,"note":"centered official scores compared with exposed clipped additive W/WY/WP/WYY/WPP/WYP sums"},
      "paired_component_and_drop_one_contrasts":paired,
      "training":{"fit_count":4,"updates_each":756,"total_updates":3024,"epochs_each":12,"row_exposures_each":381696,"row_exposures_total":exposures,"nonfinite_count":0,"dictionary_initial":initial,"dictionary_frozen":frozen,"dictionary_final":final,"predictor_shared_initialization":pi,"per_arm":per_arm,"model_archive_shapes_only":shapes,"npz_payload_opened":False},
      "execution_receipts":rec,"expected_caps":{"service":SERVICE,"cpu_cores":.5,"memory_high_bytes":1400000000,"memory_max_bytes":1500000000,"swap_bytes":0,"runtime_seconds":600}}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=HERE / "PREPARATION_MANIFEST.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receipt", action="append", default=[], metavar="ROLE=PATH")
    a = p.parse_args()
    report = run(a.capture_dir.resolve(), a.manifest.resolve(), a.receipt)
    body = json.dumps(report, indent=2, allow_nan=False).encode() + b"\n"
    with a.output.open("xb") as f:
        f.write(body)
        f.flush()
    print(json.dumps({"status": report["status"], "audit_sha256": hashlib.sha256(body).hexdigest(), "output": str(a.output)}))


if __name__ == "__main__":
    main()

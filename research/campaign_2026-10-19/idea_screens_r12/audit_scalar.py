"""Recheck captured R12 metadata and official WP from JSON six-sums only.

Uses only the local JSON/text receipts and source files. It never opens arrays,
the model archive, or checkpoint payloads.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAMPAIGN = HERE.parent
REC = HERE / "records" / "terminal_capture"
EPS, F64 = 1e-8, 2.220446049250313e-16


def load(name: str) -> dict:
    return json.loads((REC / name).read_text())


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wp(sums: list[list[float]]) -> dict[str, float]:
    rho = []
    for w, wy, wp_, wyy, wpp, wyp in sums:
        if w < EPS:
            rho.append(0.0)
            continue
        my, mp = wy / w, wp_ / w
        yy, pp = wyy / w, wpp / w
        vy, vp = yy - my * my, pp - mp * mp
        if vy <= 8 * F64 * max(abs(yy), my * my):
            vy = 0.0
        if vp <= 8 * F64 * max(abs(pp), mp * mp):
            vp = 0.0
        sy, sp = math.sqrt(max(vy, 0.0)), math.sqrt(max(vp, 0.0))
        cov = wyp / w - my * mp
        rho.append(max(-1.0, min(1.0, cov / (sy * sp))) if sy > EPS and sp > EPS else 0.0)
    return {"t0": rho[0], "t1": rho[1], "wp": math.fsum(rho) / 2}


def sum_components(stats: dict, ids: list[str]) -> list[list[float]]:
    return [[math.fsum(stats[i][t][k] for i in ids) for k in range(6)] for t in range(2)]


def err(a: dict, b: dict) -> float:
    return max(abs(a[k] - b[k]) for k in ("t0", "t1", "wp"))


def main() -> None:
    mfbytes = (HERE / "PREPARATION_MANIFEST.json").read_bytes()
    mf = json.loads(mfbytes)
    Rb, Sb, Db = [(REC / n).read_bytes() for n in ("REMOTE_RESULT.json", "REMOTE_SOURCE.json", "REMOTE_DONE.json")]
    R, S, D = [json.loads(b) for b in (Rb, Sb, Db)]
    M = sha(mfbytes)
    assert len(mf["files"]) == len(S["source_file_pins"]) == 32 and S["source_file_pins"] == mf["files"]
    assert M == S["preparation_manifest_sha256"] == R["preparation_manifest_sha256"] == D.get("preparation_manifest_sha256", M)
    assert len(mf["runtime_py"]) == len(S["runtime_import_inventory"]) == 26
    assert set(mf["runtime_py"]) == set(S["runtime_import_inventory"])
    for name, digest in mf["files"].items():
        path = (REPO / "src" / name.removeprefix("canonical:")) if name.startswith("canonical:") else CAMPAIGN / name
        assert path.is_file() and sha(path.read_bytes()) == digest, name

    V, C = load("ROOT_CAPTURE_VALIDATION.json"), json.loads((REC / "capture.stdout").read_text())
    start, smoke, launch = load("REMOTE_START.json"), load("REMOTE_ROOT_IMPORT_SMOKE.json"), load("REMOTE_ROOT_LAUNCH.json")
    for x in (start, S, R):
        assert x["design_sha256"] == S["design_sha256"] and x["sample_sha256"] == S["sample_sha256"]
        assert x["membership_sha256"] == S["membership_sha256"] and x["source_revision"] == "a266b036"
    assert sha(Rb) == D["result_sha256"] == V["files"]["RESULT.json"]["sha256"]
    assert len(Rb) == D["result_bytes"] == V["files"]["RESULT.json"]["bytes"]
    assert sha(Sb) == V["files"]["SOURCE.json"]["sha256"] and sha(Db) == V["files"]["DONE.json"]["sha256"]
    assert sha((REC / "capture.stdout").read_bytes()) == V["capture_stdout_sha256"]
    assert V["status"] == "PASS_RAW_BYTES_DURABLE_BEFORE_STOP" and V["invocation_id"] == launch["invocation_id"]
    assert C["unit"]["InvocationID"] == launch["invocation_id"]
    assert smoke["status"] == "PASS" and smoke["reader_sentinel_calls"] == []
    assert smoke["array_reads"] == smoke["parquet_reads"] == smoke["checkpoint_reads"] == 0

    # Captured allowlist hashes bind all transferred files; the NPZ is metadata only.
    names = {"ROOT_ADMISSION.json":"REMOTE_ROOT_ADMISSION.json", "ROOT_LAUNCH.json":"REMOTE_ROOT_LAUNCH.json",
             "ROOT_IMPORT_SMOKE.json":"REMOTE_ROOT_IMPORT_SMOKE.json", "PROBE_LOG.txt":"REMOTE_PROBE_LOG.txt",
             "START.json":"REMOTE_START.json", "SOURCE.json":"REMOTE_SOURCE.json", "RESULT.json":"REMOTE_RESULT.json",
             "DONE.json":"REMOTE_DONE.json", "PROGRESS.jsonl":"REMOTE_PROGRESS.jsonl"}
    assert set(names) == set(V["files"])
    assert set(C["files"]) == set(names)
    for remote, local in names.items():
        b = (REC / local).read_bytes()
        assert sha(b) == V["files"][remote]["sha256"] and len(b) == V["files"][remote]["bytes"]

    assert S["sample_metadata"]["records"] == 96
    assert S["panel"]["fit_components"] == 64 and S["panel"]["eval_components"] == 32
    assert R["fit_rows"] == 31_808 and R["evaluation_scored_rows"] == 59_436
    assert R["updates_per_arm"] == 128 and R["row_exposures_per_arm"] == 31_808 * 128 == 4_071_424
    assert R["total_row_exposures"] == 8_142_848
    assert R["initialization_sha256"]["initial"] == R["initialization_sha256"]["wmse_initial"] == R["initialization_sha256"]["pooled_corr_initial"]
    assert S["fit_normalizer"]["fit_rows_only"] and S["gamma_prepass"]["fit_rows_only"]
    assert S["gamma_prepass"]["model_state_unchanged"]
    assert S["gamma_prepass"]["optimizer_decay_excluded"]
    assert S["gamma_prepass"]["variance_head_excluded"]
    assert S["gamma_prepass"]["parameter_grad_fields_none"]
    gamma = R["initial_gradient_norms"]["wmse"] / R["initial_gradient_norms"]["pooled_corr"]
    assert math.isclose(R["gamma"], gamma, rel_tol=1e-13)
    assert S["gamma_prepass"]["gamma"] == R["gamma"]
    for arm, summary in R["fit_arm_summaries"].items():
        assert summary["attempted_updates"] == summary["successful_updates"] == 128
        assert summary["row_exposures"] == 4_071_424 and summary["nonfinite_count"] == 0
        assert summary["variance_head_unchanged"] and summary["variance_head_grad_fields_none"]
    assert D["attempted_updates"] == D["successful_updates"] == 256 and D["nonfinite_count"] == 0

    E, recon, errors, deltas, drops = R["final_evaluation"], {}, [], {}, {}
    assert E["metric_id"] == "global_wp_clip2_v1" and E["components"] == 32
    for arm, output in E["arms"].items():
        stats = output["official_clipped_prediction_sufficient_statistics_w_wy_wp_wyy_wpp_wyp"]
        ids = sorted(stats, key=int)
        assert len(ids) == 32 and sum(output["component_rows"].values()) == E["rows"]
        total = wp(sum_components(stats, ids))
        errors.append(err(total, output["official_metric"]))
        ce, de = [], []
        for i in ids:
            cs = wp(stats[i]); ds = wp(sum_components(stats, [j for j in ids if j != i]))
            ce.append(err(cs, output["component_scores"][i]))
            de.append(err(ds, output["component_drop_one_scores"][i]))
        errors.extend((max(ce), max(de)))
        recon[arm] = {"official": output["official_metric"], "six_sum_reconstruction": total,
                      "component_max_error": max(ce), "drop_one_max_error": max(de)}
    assert max(errors) < 1e-12
    delta = recon["pooled_corr"]["six_sum_reconstruction"]["wp"] - recon["wmse"]["six_sum_reconstruction"]["wp"]
    assert math.isclose(delta, E["primary_delta_pooled_corr_minus_wmse_wp"], rel_tol=1e-13, abs_tol=1e-13)
    for i in E["arms"]["wmse"]["component_scores"]:
        deltas[i] = E["arms"]["pooled_corr"]["component_scores"][i]["wp"] - E["arms"]["wmse"]["component_scores"][i]["wp"]
        drops[i] = E["arms"]["pooled_corr"]["component_drop_one_scores"][i]["wp"] - E["arms"]["wmse"]["component_drop_one_scores"][i]["wp"]

    unit = C["unit"]
    assert unit["Result"] == "success" and unit["ExecMainStatus"] == "0"
    assert unit["CPUQuotaPerSecUSec"] == "500ms" and unit["MemoryHigh"] == "1400000000"
    assert unit["MemoryMax"] == "1500000000" and unit["MemorySwapMax"] == "0" and unit["RuntimeMaxUSec"] == "10min"
    stop_raw = (REC / "stop.stdout").read_bytes(); stop = json.loads(stop_raw)
    stop_receipt = load("STOP_RECEIPT.json")
    assert sha(stop_raw) == stop_receipt["stdout_sha256"] and stop_receipt["returncode"] == 0
    assert stop["after"]["LoadState"] == "not-found" and stop["after"]["ActiveState"] == "inactive" and stop["after"]["SubState"] == "dead"
    assert C["own_cgroup_empty_terminal"] and C["own_cgroup"] is None
    assert V["model_archive_payload_transferred"] is False
    assert D["archive_sha256"] == R["archive"]["sha256"] and D["archive_bytes"] == R["archive"]["bytes"]
    assert C["output_inventory"]["FINAL_MODELS_AND_NORMALIZER.npz"] == R["archive"]["bytes"]

    def signs(values: dict[str, float]) -> dict:
        xs = list(values.values())
        return {"positive": sum(x > 0 for x in xs), "negative": sum(x < 0 for x in xs),
                "min": min(xs), "max": max(xs)}

    audit = {
        "schema": "nf19.r12_independent_scalar_audit.v1", "status": "PASS",
        "bindings": {"manifest_sha256": M, "result_sha256": sha(Rb), "source_sha256": sha(Sb),
                     "done_sha256": sha(Db), "sample_sha256": S["sample_sha256"],
                     "membership_sha256": S["membership_sha256"], "source_revision": S["source_revision"],
                     "capture_sha256": sha((REC / "capture.stdout").read_bytes()),
                     "capture_validation_sha256": sha((REC / "ROOT_CAPTURE_VALIDATION.json").read_bytes()),
                     "stop_stdout_sha256": sha(stop_raw), "stop_envelope_sha256": sha((REC / "STOP_RECEIPT.json").read_bytes()),
                     "invocation_id": launch["invocation_id"], "source_pins": 32, "runtime_modules": 26},
        "training": {"fit_rows": R["fit_rows"], "fit_components": 64, "eval_rows": E["rows"], "eval_components": 32,
                     "updates_per_arm": 128, "row_exposures_per_arm": R["row_exposures_per_arm"],
                     "gamma": R["gamma"], "fit_only_norm_ratio": gamma, "initialization_identical": True,
                     "initial_gradient_norms": R["initial_gradient_norms"],
                     "fit_only_gamma_prepass_flags": {k: S["gamma_prepass"][k] for k in
                         ("fit_rows_only", "model_state_unchanged", "optimizer_decay_excluded", "variance_head_excluded", "parameter_grad_fields_none")},
                     "objective_control": R["objective_control"], "objective_candidate": R["objective_candidate"],
                     "fit_mean_clip_outside_rows": {arm: [v["outside_clip_rows"] for v in R["fit_prediction_clipping_diagnostics"][arm]] for arm in ("wmse", "pooled_corr")},
                     "nonfinite_updates": 0, "variance_head_unchanged": True},
        "evaluation": {"reconstruction": recon, "max_absolute_six_sum_error": max(errors),
                       "pooled_corr_minus_wmse_wp": delta,
                       "component_delta_signs": signs(deltas), "drop_one_delta_signs": signs(drops)},
        "runtime_stop": {"systemd_result": unit["Result"], "exec_main_status_string": unit["ExecMainStatus"],
                         "cpu_quota_string": unit["CPUQuotaPerSecUSec"], "memory_high_string": unit["MemoryHigh"],
                         "memory_max_string": unit["MemoryMax"], "memory_swap_max_string": unit["MemorySwapMax"],
                         "runtime_max_string": unit["RuntimeMaxUSec"], "systemd_memory_peak_bytes_string": unit["MemoryPeak"],
                         "own_cgroup_memory": "unavailable: terminal cgroup empty", "stop_returncode": stop_receipt["returncode"],
                         "post_stop": {k: stop["after"][k] for k in ("LoadState", "ActiveState", "SubState")},
                         "done_elapsed_wall_s": D["elapsed_wall_s"], "done_elapsed_cpu_s": D["elapsed_cpu_s"],
                         "archive_metadata_only": {"sha256": D["archive_sha256"], "bytes": D["archive_bytes"],
                                                   "payload_transferred": False, "payload_opened": False}},
        "limits": ["One fixed 128-update objective pair on the exposed 32-component panel.",
                   "Component and leave-one-component summaries are sensitivities, not independent replications.",
                   "Equal rows and optimizer updates do not imply equal FLOPs or wall time; this does not establish general objective superiority or full-campaign improvement."],
    }
    (HERE / "RESULT_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "max_six_sum_error": max(errors), "delta_wp": delta,
                      "audit": str(HERE / "RESULT_AUDIT.json")}, sort_keys=True))


if __name__ == "__main__":
    main()

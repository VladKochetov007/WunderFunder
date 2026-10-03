"""Independent stored-array and feature audit; no model inference imports."""

from __future__ import annotations

import ast
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ACTOR = HERE.parent / "matched_usage"
sys.path.insert(0, str(ROOT))
from utils import weighted_pearson  # noqa: E402
from wnn.features import event_features_torch  # noqa: E402

I0 = np.r_[0:52, 112:156, 178:183, 186:188, 189:211]
I1 = np.r_[52:104, 156:178, 183:186, 188:189]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            sha.update(piece)
    return sha.hexdigest()


def pearson(moment: np.ndarray) -> np.ndarray:
    """Independent centered reduction of six reconstructed additive moments."""
    weight, y, p, yy, pp, yp = np.moveaxis(moment, -1, 0)
    covariance = yp - y * p / weight
    denominator = np.sqrt(np.maximum(yy - y * y / weight, 0) * np.maximum(pp - p * p / weight, 0))
    return np.clip(covariance / denominator, -1, 1)


def packed_inputs(raw: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(raw)[None]
    # Exact active n_in211 build_inputs: raw112 + causal event99, no optional columns.
    return torch.cat([tensor, event_features_torch(tensor)], -1)[0].numpy()


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(value.astype(np.float64) ** 2)))


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    samples = json.loads((ACTOR / "samples.json").read_text())
    actor_results = json.loads((ACTOR / "results.json").read_text())
    registry = json.loads((ACTOR / "artifact_provenance.json").read_text())["artifacts"]
    before = {name: digest(ACTOR / name) for name in registry}
    assert all(before[name] == entry["sha256"] for name, entry in registry.items())
    control = samples["control"]
    assert (
        control["n_in"] == 211
        and not control["args"]["online_norm"]
        and not control["args"]["raw_only"]
    )
    assert digest(ROOT / control["checkpoint"]) == control["checkpoint_sha256"]
    cache_path = ROOT / "cache/pred_c20261001_d256_s3_f0_valid.npy"
    cache_sidecar = json.loads(cache_path.with_suffix(".json").read_text())
    assert digest(cache_path) == cache_sidecar["prediction_sha256"]
    assert cache_sidecar["checkpoint_sha256"] == control["checkpoint_sha256"]
    cached_values = np.load(cache_path, mmap_mode="r")
    assert all(digest(ROOT / name) == value for name, value in control["source_sha256"].items())
    provenance = json.loads(
        (ROOT / "artifacts/runs/c20261001_d256_s3_f0/provenance.json").read_text()
    )
    # Direct source inspection of selection boundary, retained as concrete evidence.
    tree = ast.parse((ACTOR / "probe.py").read_text())
    donor_function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "choose_donor"
    )
    donor_names = sorted({n.id for n in ast.walk(donor_function) if isinstance(n, ast.Name)})
    assert [a.arg for a in donor_function.args.args] == ["raw", "t", "std"]
    assert not {"targets", "mask", "predictions", "y", "is_scored"} & set(donor_names)
    parquet = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    rng = np.random.default_rng(614203)
    moments = np.zeros((24, 3, 2, 6))
    all_y, all_pred, all_base, repeats, designs, current_equal = [], [], [], [], [], []
    diagnostics, query_records = [], []
    suffix_equal = 0
    raw_current_equal = 0
    packed_inconsistent = 0
    donor_count = 0
    mask_equal = True
    all_finite = True
    cpu_pass = True
    max_cpu_error = 0.0
    max_identity_error = 0.0
    for index, sample in enumerate(samples["samples"]):
        print(f"checking stored sequence {index:02d}", flush=True)
        donors = json.loads((ACTOR / f"donors_{index:02d}.json").read_text())["donors"]
        meta = json.loads((ACTOR / f"seq_{index:02d}.json").read_text())
        with np.load(ACTOR / f"seq_{index:02d}.npz") as z:
            raw, packed, std = z["raw"], z["packed"], z["std"]
            target, mask, q = z["targets"], z["mask"], z["queries"]
            pred, base, reference, cache = (
                z["predictions"],
                z["baseline"],
                z["reference"],
                z["cached"],
            )
            assert np.array_equal(cache, cached_values[sample["cache_position"]])
            replay = z["replay"]
            all_finite &= all(np.isfinite(z[name]).all() for name in z.files)
            metadata = parquet.read_row_group(
                sample["row_group"],
                columns=["seq_ix", "step_in_seq", "need_prediction", "is_scored"],
                use_threads=False,
            )
            need = metadata["need_prediction"].to_numpy()
            assert np.array_equal(need, np.arange(20000) >= 99)
            assert np.all(metadata["seq_ix"].to_numpy() == sample["seq_ix"])
            assert np.array_equal(metadata["step_in_seq"].to_numpy(), np.arange(20000))
            mask_equal &= np.array_equal(mask, need & metadata["is_scored"].to_numpy())
            eligible = np.flatnonzero(mask & (np.arange(20000) >= 512))
            assert len(eligible) == sample["n_eligible"]
            assert sample["m_selected"] == 12 and sample["query_pi"] == 12 / len(eligible)
            chosen = np.sort(rng.choice(eligible, 12, replace=False))
            assert np.array_equal(q, chosen) and q.tolist() == sample["queries"]
            assert np.array_equal(packed_inputs(raw), packed)
            y = np.clip(target[q], -2, 2).astype(np.float64)
            p = np.clip(pred, -2, 2).astype(np.float64)
            design = np.full(12, len(eligible) / 12)
            weight = design[:, None] * np.abs(y)
            for arm in range(3):
                moments[index, arm] = np.stack(
                    [
                        weight.sum(0),
                        (weight * y).sum(0),
                        (weight * p[:, arm]).sum(0),
                        (weight * y * y).sum(0),
                        (weight * p[:, arm] * p[:, arm]).sum(0),
                        (weight * y * p[:, arm]).sum(0),
                    ],
                    -1,
                )
            all_y.append(target[q])
            all_pred.append(pred)
            all_base.append(base[q])
            repeats.extend([len(eligible)] * 12)
            designs.extend(design.tolist())
            cpu_pass &= np.allclose(base, reference, atol=2e-5, rtol=2e-5)
            max_cpu_error = max(max_cpu_error, float(np.abs(base - reference).max()))
            identity_max = max(
                float(np.abs(replay[j, 0] - base[t - 288 : t + 1]).max()) for j, t in enumerate(q)
            )
            max_identity_error = max(max_identity_error, identity_max)
            difference = reference - cache
            correlations = [
                float(np.corrcoef(reference[:, k], cache[:, k])[0, 1]) for k in range(2)
            ]
            drift_pass = (
                float(np.abs(difference).max()) <= 0.02
                and rms(difference) <= 0.005
                and min(correlations) >= 0.999
            )
            diagnostics.append(
                {
                    "sequence": index,
                    "row_group": sample["row_group"],
                    "cache_max_abs": float(np.abs(difference).max()),
                    "cache_rms": rms(difference),
                    "correlations": correlations,
                    "cache_bound_pass": drift_pass,
                    "identity_max_abs": identity_max,
                    "actor_cache_gate_matches": drift_pass == meta["gpu_cache"]["r2_bound_passed"],
                }
            )
            for j, t in enumerate(q):
                start, end = int(t) - 288, int(t) - 32
                candidates = []
                for lag in (64, 128, 256):
                    if start >= lag:
                        candidates.append(
                            (
                                rms(
                                    (raw[start:end, :52] - raw[start - lag : end - lag, :52])
                                    / std[:52]
                                ),
                                lag,
                            )
                        )
                amplitude, lag = min(candidates)
                assert (
                    donors[j]["lag"] == lag
                    and abs(amplitude - donors[j]["raw_displacement"]) < 1e-12
                )
                donor_count += 1
                altered = raw[: int(t) + 1].copy()
                altered[start:end, :52] = raw[start - lag : end - lag, :52]
                reconstructed = packed_inputs(altered)
                assert np.array_equal(reconstructed[:start], packed[:start])
                assert np.array_equal(reconstructed[:, I1], packed[: int(t) + 1, I1])
                assert np.array_equal(reconstructed[:, 104:112], packed[: int(t) + 1, 104:112])
                raw_equal = np.array_equal(altered[end:], raw[end : int(t) + 1])
                assert raw_equal
                now_equal = np.array_equal(reconstructed[-1], packed[t])
                full_suffix_equal = np.array_equal(reconstructed[end:], packed[end : int(t) + 1])
                direct = packed[start : int(t) + 1].copy()
                direct[:256, I0] = packed[start - lag : end - lag, I0]
                inconsistent = not np.array_equal(direct, reconstructed[start:])
                current_equal.append(now_equal)
                raw_current_equal += np.array_equal(altered[-1], raw[t])
                suffix_equal += full_suffix_equal
                packed_inconsistent += inconsistent
                assert now_equal == (not meta["records"][j]["current_engineered_changed"])
                assert full_suffix_equal == (
                    meta["records"][j]["protected_tail_engineered_rms"] == 0
                )
                query_records.append(
                    {
                        "sequence": index,
                        "query": int(t),
                        "lag": lag,
                        "current_input_equal": bool(now_equal),
                        "protected33_inputs_equal": bool(full_suffix_equal),
                        "packed_inconsistent": bool(inconsistent),
                    }
                )
    y, pred, base = np.concatenate(all_y), np.concatenate(all_pred), np.concatenate(all_base)
    score = pearson(moments.sum(0)).mean(-1)
    repeat_y = np.repeat(y, repeats, axis=0)
    helper_scores = []
    for arm in range(3):
        repeated_pred = np.repeat(pred[:, arm], repeats, axis=0)
        helper_scores.append(
            np.mean([weighted_pearson(repeat_y[:, k], repeated_pred[:, k]) for k in range(2)])
        )
    with np.load(ACTOR / "panel.npz") as panel:
        moment_error = float(np.abs(moments - panel["moments"]).max())
        np.testing.assert_allclose(moments, panel["moments"], rtol=1e-12, atol=1e-12)
        draws = panel["bootstrap_component_indices"]
        assert np.array_equal(draws, np.random.default_rng(614205).integers(0, 24, (1000, 24)))
        assert np.array_equal(panel["predictions"], pred)
        assert np.array_equal(panel["targets"], y)
        np.testing.assert_allclose(panel["query_design_weight"], designs, rtol=1e-15)
    boot = pearson(moments[draws].sum(1)).mean(-1)
    cis = np.quantile(boot - boot[:, :1], [0.025, 0.975], axis=0).T
    quoted_score = np.array([a["scoped_sampled_wp"] for a in actor_results["arms"]])
    quoted_ci = np.array([a["wp_difference_component_ci"] for a in actor_results["arms"]])
    np.testing.assert_allclose(score, quoted_score, atol=1e-12, rtol=0)
    np.testing.assert_allclose(score, helper_scores, atol=1e-12, rtol=0)
    np.testing.assert_allclose(cis, quoted_ci, atol=1e-12, rtol=0)
    equal = np.array(current_equal)
    raw_delta = pred[:, 1] - base
    failed = [d["sequence"] for d in diagnostics if not d["cache_bound_pass"]]
    assert len(failed) == actor_results["gates"]["gpu_cache_r2_bound_failures"] == 7
    assert (
        raw_current_equal == 288
        and equal.sum() == 172
        and suffix_equal == 0
        and packed_inconsistent == 288
    )
    assert mask_equal and all_finite and cpu_pass
    after = {name: digest(ACTOR / name) for name in registry}
    assert before == after
    result = {
        "verdict": "PASS scoped reconstruction; FAIL full-WP/fresh-confirmation/clean32-input-retention interpretation",
        "queries": len(y),
        "eligible_domain": int(sum(designs)),
        "moments_max_abs_error": moment_error,
        "scores": score.tolist(),
        "standalone_helper_scores": np.asarray(helper_scores).tolist(),
        "helper_repeated_rows": len(repeat_y),
        "wp_change_component_intervals": cis.tolist(),
        "current_raw_equal": int(raw_current_equal),
        "current_all_input_equal": int(equal.sum()),
        "protected33_all_input_equal": int(suffix_equal),
        "packed_inconsistent": int(packed_inconsistent),
        "raw_response_current_input_equal": rms(raw_delta[equal]),
        "raw_response_current_input_changed": rms(raw_delta[~equal]),
        "donors_recomputed": donor_count,
        "donor_function_names": donor_names,
        "cpu_reference_pass": bool(cpu_pass),
        "cpu_reference_max_abs": max_cpu_error,
        "identity_max_abs": max_identity_error,
        "cache_failed_sequences": failed,
        "drift_by_sequence": diagnostics,
        "all_stored_arrays_finite": bool(all_finite),
        "masks_raw_metadata_equal": bool(mask_equal),
        "actor_hashes_verified_unchanged": len(before),
        "actual_model_forwards": 0,
        "training_provenance_keys": sorted(provenance),
        "cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time(),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "query_checks.json").write_text(json.dumps(query_records, indent=2) + "\n")
    (HERE / "input_hashes.json").write_text(
        json.dumps(
            {
                "actor": before,
                "standalone_grader": digest(ROOT / "utils.py"),
                "feature_source": digest(ROOT / "src/wnn/features.py"),
                "trainer": digest(ROOT / "scripts/train_rnn.py"),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

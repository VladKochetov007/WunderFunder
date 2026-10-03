from __future__ import annotations

import hashlib
import json
import logging
import os
import resource
import time
from pathlib import Path

import numpy as np

OWN = Path(__file__).resolve().parent
REPO = OWN.parents[4]
CONT = OWN.parent.parent
SOURCE = CONT / "round3" / "matched_usage"
PANEL = CONT / "round4" / "peer_innovation" / "samples.json"
LOG = logging.getLogger(__name__)


def save(name: str, data: object) -> None:
    (OWN / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1048576):
            digest.update(block)
    return digest.hexdigest()


def clocks(peer: np.ndarray) -> dict[str, np.ndarray]:
    """Three distinct observable clocks, with original row order preserved."""
    changed = np.zeros(peer.shape, dtype=bool)
    changed[1:] = peer[1:] != peer[:-1]
    events = {
        "scalar": changed[:, 0],
        "top_pair": changed[:, 0] | changed[:, 11],
        "any22": changed.any(axis=1),
    }
    rows = np.arange(len(peer))
    result = {}
    for name, event in events.items():
        result[f"{name}_event"] = event
        result[f"{name}_age"] = rows - np.maximum.accumulate(np.where(event, rows, 0))
    return result


def counts(c: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, int]:
    scalar, top, any22 = (c[f"{name}_event"][rows] for name in ("scalar", "top_pair", "any22"))
    assert not (scalar & ~top).any() and not (top & ~any22).any()
    return {
        "rows": len(rows),
        "scalar_events": int(scalar.sum()),
        "top_pair_events": int(top.sum()),
        "any22_events": int(any22.sum()),
        "any_fresh_scalar_idle": int((any22 & ~scalar).sum()),
        "any_fresh_top_pair_idle": int((any22 & ~top).sum()),
        "top_pair_fresh_scalar_idle": int((top & ~scalar).sum()),
        "scalar_vs_any_age_disagreements": int(
            (c["scalar_age"][rows] != c["any22_age"][rows]).sum()
        ),
        "top_pair_vs_any_age_disagreements": int(
            (c["top_pair_age"][rows] != c["any22_age"][rows]).sum()
        ),
    }


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2))) if len(values) else 0.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    start, cpu = time.monotonic(), time.process_time()
    panel = json.loads(PANEL.read_text())
    paths = [
        PANEL,
        SOURCE / "samples.json",
        REPO / "src/wnn/features.py",
        OWN / "PREREG.json",
        OWN / "audit.py",
    ]
    paths += [SOURCE / f"seq_{i:02d}.npz" for i in range(24)]
    before = {str(p): sha(p) for p in paths}
    original = json.loads((SOURCE / "samples.json").read_text())
    assert original["samples"] == panel["samples"]
    provenance = json.loads((SOURCE / "artifact_provenance.json").read_text())
    toy = np.zeros((4, 22))
    toy[1:, 2] = 1
    toy[2:, 11] = 1
    toy[3:, 0] = 1
    tc = clocks(toy)
    assert np.array_equal(tc["any22_event"], [False, True, True, True])
    assert np.array_equal(tc["top_pair_event"], [False, False, True, True])
    assert np.array_equal(tc["scalar_event"], [False, False, False, True])
    sequences, query_records = [], []
    global_std = None
    for i, sample in enumerate(panel["samples"]):
        path = SOURCE / f"seq_{i:02d}.npz"
        assert before[str(path)] == provenance["artifacts"][path.name]["sha256"]
        with np.load(path) as arrays:
            raw, packed, std = arrays["raw"], arrays["packed"], arrays["std"]
        assert raw.shape[1] == 112 and packed.shape[1] == 211
        assert np.isfinite(raw).all() and np.isfinite(packed).all() and np.isfinite(std).all()
        if global_std is None:
            global_std = std.copy()
        else:
            assert np.array_equal(std, global_std)
        peer = raw[:, 52:74].astype(np.float64)
        c = clocks(peer)
        assert np.array_equal(c["top_pair_event"].astype(np.float32), packed[:, 183])
        expected_age = np.log1p(np.minimum(c["top_pair_age"], 10000)).astype(np.float32)
        clock_error = float(np.abs(expected_age - packed[:, 188]).max())
        assert clock_error <= 2e-6
        qs = np.array(sample["queries"])
        rows = np.arange(512, len(raw))
        delta = np.diff(peer, axis=0)
        event_rows = np.any(delta != 0, axis=1)
        common = delta.mean(axis=1)
        residual = delta - common[:, None]
        exact_common = event_rows & (delta == delta[:, 0, None]).all(axis=1)
        energy = float((delta**2).sum())
        common_energy = float(22 * (common**2).sum())
        projection_fraction = common_energy / energy if energy else 0.0
        scalar_step = np.r_[0.0, delta[:, 0]] / float(std[52])
        positive = scalar_step[scalar_step != 0]
        prefix_steps = scalar_step[1:512]
        post_steps = scalar_step[512:]
        # Float64 algebraic gauge controls preserve the supplied float32 values.
        static = peer + 8
        dynamic_shift = 0.125 * (np.arange(len(peer)) // 32)
        dynamic = peer + dynamic_shift[:, None]
        contrasts = peer - peer[:, 0, None]
        static_error = float(np.abs(static - static[:, 0, None] - contrasts).max())
        dynamic_error = float(np.abs(dynamic - dynamic[:, 0, None] - contrasts).max())
        static_delta_error = float(np.abs(np.diff(static, axis=0) - delta).max())
        assert max(static_error, dynamic_error, static_delta_error) <= 1e-11
        static_clocks = clocks(static)
        assert all(np.array_equal(static_clocks[name], c[name]) for name in c)
        dynamic_delta = np.diff(dynamic[:, 0]) - delta[:, 0]
        shifted_rows = np.diff(dynamic_shift) != 0
        assert np.allclose(dynamic_delta, np.diff(dynamic_shift), atol=1e-11, rtol=0)
        assert np.any(np.abs(dynamic_delta) > 0)
        dyn_clocks = clocks(dynamic)
        for t in qs:
            query_records.append(
                {
                    "component": sample["component"],
                    "seq_ix": sample["seq_ix"],
                    "query": int(t),
                    "stratum": "fit" if sample["component"] in panel["fit_components"] else "test",
                    **{name: int(values[t]) for name, values in c.items()},
                }
            )
        sequence = {
            "index": i,
            "component": sample["component"],
            "seq_ix": sample["seq_ix"],
            "row_group": sample["row_group"],
            "stratum": "fit" if sample["component"] in panel["fit_components"] else "test",
            "time_order_preserved": True,
            "query_counts": counts(c, qs),
            "all_after511_counts": counts(c, rows),
            "packed_peer_clock_max_abs": clock_error,
            "packed_top_pair_event_exact": True,
            "units": {
                "frozen_std52": float(std[52]),
                "scalar_step_all_RMS_normalized": rms(post_steps),
                "scalar_nonzero_step_RMS_normalized": rms(positive),
                "nonzero_step_abs_quantiles": np.quantile(
                    np.abs(positive), [0.1, 0.5, 0.9]
                ).tolist()
                if len(positive)
                else [],
                "prefix512_step_RMS_normalized": rms(prefix_steps),
                "later_step_RMS_normalized": rms(post_steps),
            },
            "coordinate_motion": {
                "any22_change_rows": int(event_rows.sum()),
                "exact_uniform_increment_rows": int(exact_common.sum()),
                "common_projection_energy_fraction": projection_fraction,
                "shape_residual_RMS": rms(residual.ravel()),
            },
            "gauge_controls": {
                "static_relative_shape_max_abs": static_error,
                "dynamic_relative_shape_max_abs": dynamic_error,
                "static_delta_max_abs": static_delta_error,
                "static_all_clock_identity": True,
                "dynamic_shift_boundary_rows": int(shifted_rows.sum()),
                "dynamic_scalar_delta_changed_rows": int((np.abs(dynamic_delta) > 1e-11).sum()),
                "dynamic_scalar_clock_changed_rows": int(
                    (dyn_clocks["scalar_age"] != c["scalar_age"]).sum()
                ),
                "dynamic_top_pair_clock_changed_rows": int(
                    (dyn_clocks["top_pair_age"] != c["top_pair_age"]).sum()
                ),
                "dynamic_normalized_scalar_delta_difference_RMS": rms(
                    dynamic_delta / float(std[52])
                ),
            },
        }
        sequences.append(sequence)
        LOG.info("sequence%d query freshness %s", i, sequence["query_counts"])
    summaries = {}
    for label in ("all", "fit", "test"):
        subset = [s for s in sequences if label == "all" or s["stratum"] == label]
        summaries[label] = {}
        for domain in ("query_counts", "all_after511_counts"):
            summaries[label][domain] = {
                key: sum(s[domain][key] for s in subset) for key in subset[0][domain]
            }
        scales = [s["units"]["scalar_nonzero_step_RMS_normalized"] for s in subset]
        summaries[label]["sequence_nonzero_step_RMS_range"] = [min(scales), max(scales)]
        summaries[label]["sequence_equal_median_nonzero_step_RMS"] = float(np.median(scales))
        summaries[label]["common_projection_fraction_range"] = [
            min(s["coordinate_motion"]["common_projection_energy_fraction"] for s in subset),
            max(s["coordinate_motion"]["common_projection_energy_fraction"] for s in subset),
        ]
    assert len(query_records) == 288
    assert before == {str(p): sha(p) for p in paths}
    results = {
        "tests": 2,
        "controls": {
            "ordered_clock_toy_passed": True,
            "static_and_dynamic_common_translation_controls_passed_components": 24,
        },
        "summaries": summaries,
        "sequences": sequences,
        "all288_query_clock_IDs": query_records,
        "frozen_std_identical_all24": True,
        "input_sha256": before,
        "cost": {
            "wall_seconds": time.monotonic() - start,
            "cpu_seconds": time.process_time() - cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "forbidden_operations": {
            "new_raw_parquet_reads": 0,
            "Y_mask_values_loaded": False,
            "fits": 0,
            "model_forwards": 0,
            "GPU": False,
        },
    }
    save("results.json", results)
    save(
        "STATE.json",
        {
            "phase": "audit_scored",
            "tests": 2,
            "raw_parquet_reads": 0,
            "Y_or_mask_values_loaded": False,
            "fits": 0,
            "GRU_forwards": 0,
            "sources_preserved": True,
        },
    )
    LOG.info("SUMMARY %s", json.dumps(summaries))
    LOG.info("COST %s", results["cost"])


if __name__ == "__main__":
    main()

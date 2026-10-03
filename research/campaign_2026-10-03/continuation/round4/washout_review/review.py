from __future__ import annotations

import hashlib
import json
import logging
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from wnn.features import event_features_torch
from wnn.nn import RNNModel

OWN = Path(__file__).resolve().parent
REPO = OWN.parents[4]
ACTOR = OWN.parent.parent / "round3" / "washout_usage"
PRIOR = ACTOR.parent / "matched_usage"
LOG = logging.getLogger(__name__)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1048576):
            h.update(block)
    return h.hexdigest()


def save(name: str, data: object) -> None:
    (OWN / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def pack(raw: np.ndarray) -> np.ndarray:
    """Independent wrapper of the frozen production causal feature primitive."""
    x = torch.from_numpy(raw)[None]
    return torch.cat((x, event_features_torch(x)), dim=-1)[0].numpy()


def rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.square(np.asarray(array, dtype=np.float64)).mean()))


def parity(actual: np.ndarray, expected: np.ndarray) -> dict:
    passed = np.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert passed
    return {
        "max_abs": float(np.max(np.abs(actual - expected))),
        "rms": rms(actual - expected),
        "passed": bool(passed),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    begin, cpu = time.monotonic(), time.process_time()
    panel = read(ACTOR / "samples.json")
    control = panel["control"]
    gates = [read(ACTOR / f"gates_{i:02d}.json") for i in range(24)]
    selected = {i: next(r["query"] for r in gates[i]["records"] if r["accepted"]) for i in range(4)}
    save(
        "selected_forward_IDs.json",
        {
            str(i): {
                "query": t,
                "component": panel["samples"][i]["component"],
                "seq_ix": panel["samples"][i]["seq_ix"],
            }
            for i, t in selected.items()
        },
    )
    paths = [
        ACTOR / n
        for n in (
            "REPORT.md",
            "PREREG.json",
            "samples.json",
            "probe.py",
            "common.py",
            "summarize.py",
            "controls.py",
            "verify.py",
            "controls.json",
            "artifact_provenance.json",
        )
    ]
    paths += [REPO / p for p in control["source_sha256"]]
    paths += [
        REPO / control["checkpoint"],
        REPO / control["split_manifest"],
        OWN / "PREREG.json",
        OWN / "review.py",
        ACTOR.parent / "observable_simulator" / "REPORT.txt",
    ]
    for i in range(24):
        paths += [
            ACTOR / f"gates_{i:02d}.json",
            ACTOR / f"seq_{i:02d}.json",
            ACTOR / f"seq_{i:02d}.npz",
            PRIOR / f"seq_{i:02d}.npz",
        ]
    before = {str(p): sha(p) for p in paths}
    for p, expected in control["source_sha256"].items():
        assert before[str(REPO / p)] == expected
    assert before[str(REPO / control["checkpoint"])] == control["checkpoint_sha256"]
    assert before[str(REPO / control["split_manifest"])] == control["split_sha256"]
    checkpoint = torch.load(REPO / control["checkpoint"], map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    assert args.get("arch", "gru") == "gru" and args.get("enc", "linear") == "linear"
    model = RNNModel(
        checkpoint["n_in"],
        args["hidden"],
        args["layers"],
        args["dropout"],
        n_out=control["n_out"],
        cell=args.get("cell", "gru"),
        enc=args.get("enc", "linear"),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    split = read(REPO / control["split_manifest"])
    components, records, forwards = [], [], []
    all_delta, all_identity, all_hidden = [], [], []
    source_checks = 0
    for i, sample in enumerate(panel["samples"]):
        assert sample["row_group"] in split["evaluation"]["valid"]
        assert abs(sample["query_pi"] - 12 / sample["n_eligible"]) < 1e-15
        meta = read(ACTOR / f"seq_{i:02d}.json")
        assert (
            sample["queries"]
            == [r["query"] for r in gates[i]["records"]]
            == [r["query"] for r in meta["records"]]
        )
        assert before[str(PRIOR / f"seq_{i:02d}.npz")] == gates[i]["prior_array_sha256"]
        with np.load(PRIOR / f"seq_{i:02d}.npz") as old:
            raw, packed, std = old["raw"], old["packed"], old["std"]
            reference = old["reference"] if i in selected else None
        assert raw.shape[1] == 112 and packed.shape[1] == 211
        assert np.array_equal(pack(raw), packed)
        source_checks += 1
        z = np.load(ACTOR / f"seq_{i:02d}.npz")
        query_to_index = {int(t): j for j, t in enumerate(z["queries"])}
        responses = z["predictions"].astype(float) - z["identity_predictions"].astype(float)
        identity_error = z["identity_predictions"].astype(float) - z[
            "original_reference_query"
        ].astype(float)
        accepted_count = 0
        for r in gates[i]["records"]:
            t, start, end = r["query"], r["query"] - 352, r["query"] - 96
            choices = []
            for lag in (64, 128, 256):
                if start >= lag:
                    displacement = (raw[start:end, :52] - raw[start - lag : end - lag, :52]) / std[
                        :52
                    ]
                    choices.append((rms(displacement), lag))
            amplitude, lag = min(choices)
            assert lag == r["lag"] and abs(amplitude - r["amplitude"]) < 1e-12
            altered = raw[: t + 1].copy()
            altered[start:end, :52] = raw[start - lag : end - lag, :52]
            assert np.array_equal(altered[:start], raw[:start])
            assert np.array_equal(altered[end:], raw[end : t + 1])
            assert np.array_equal(altered[:, 52:], raw[: t + 1, 52:])
            rebuilt = pack(altered)
            assert np.array_equal(rebuilt[:start], packed[:start])
            difference = rebuilt[t - 32 : t + 1] - packed[t - 32 : t + 1]
            finite = bool(np.isfinite(rebuilt).all())
            accepted = finite and np.array_equal(rebuilt[t - 32 : t + 1], packed[t - 32 : t + 1])
            clocks = int(np.count_nonzero(difference[:, [186, 187]]))
            held = int(np.count_nonzero(difference[:, 189:211]))
            assert (
                accepted == r["accepted"]
                and clocks == r["suffix_clock_unequal_entries"]
                and held == r["suffix_held_unequal_entries"]
            )
            record = {
                "component": sample["component"],
                "seq_ix": sample["seq_ix"],
                "query": t,
                "accepted": bool(accepted),
                "clock_mismatch_entries": clocks,
                "held_mismatch_entries": held,
                "suffix_max_abs": float(np.abs(difference).max()),
                "weight": 1 / sample["query_pi"],
                "lag": lag,
                "finite": finite,
            }
            records.append(record)
            if accepted:
                accepted_count += 1
                j = query_to_index[t]
                assert np.array_equal(rebuilt[start:], z["recomputed_windows"][j])
                assert np.array_equal(rebuilt[-33:], packed[t - 32 : t + 1])
                if selected.get(i) == t:
                    with torch.inference_mode():
                        h = torch.zeros(args["layers"], 1, args["hidden"])
                        prefix_parity = []
                        for offset in range(0, start, 1000):
                            last = min(offset + 1000, start)
                            pred, h = model(torch.from_numpy(packed[offset:last])[None], h)
                            prefix_parity.append(
                                parity(pred[0, :, :2].numpy(), reference[offset:last])
                            )
                        boundary = parity(h.numpy(), z["original_states"][j])
                        old_pred, old_h = model(
                            torch.from_numpy(packed[start : t + 1])[None], h.clone()
                        )
                        new_pred, new_h = model(torch.from_numpy(rebuilt[start:])[None], h.clone())
                        identity = parity(old_pred[0, :, :2].numpy(), reference[start : t + 1])
                        stored_old = parity(old_pred[0, :, :2].numpy(), z["replay"][j, 0])
                        stored_new = parity(new_pred[0, :, :2].numpy(), z["replay"][j, 1])
                        delta = new_pred[0, -1, :2].numpy() - old_pred[0, -1, :2].numpy()
                        response_match = parity(delta, responses[j])
                        hidden = torch.sqrt(((new_h - old_h) ** 2).mean(dim=(1, 2))).numpy()
                        hidden_match = parity(hidden, z["hidden_rms"][j])
                        canary = raw.copy()
                        canary[start:end, :52] = altered[start:end, :52]
                        canary[t + 1 :] = 0
                        canary_features = pack(canary)
                        assert np.array_equal(canary_features[start : t + 1], rebuilt[start:])
                    forwards.append(
                        {
                            "component_index": i,
                            "component": sample["component"],
                            "seq_ix": sample["seq_ix"],
                            "query": t,
                            "boundary": start,
                            "warmup_included": 99,
                            "prefix_rows_forwarded": start,
                            "prefix_max_abs": max(p["max_abs"] for p in prefix_parity),
                            "boundary_parity": boundary,
                            "identity_parity": identity,
                            "stored_original": stored_old,
                            "stored_altered": stored_new,
                            "response_match": response_match,
                            "response_rms": rms(delta),
                            "hidden_match": hidden_match,
                            "future_feature_canary": "zero all rows strictly after query; complete recomputed prefix identical; actual same deterministic window input therefore same forward",
                        }
                    )
        assert accepted_count == len(z["queries"]) == meta["accepted"]
        w = 1 / sample["query_pi"]
        components.append(
            {
                "component": sample["component"],
                "accepted": accepted_count,
                "rejected": 12 - accepted_count,
                "accepted_mass": w * accepted_count,
                "squared_response_total": float(w * np.square(responses).mean(axis=1).sum()),
                "response_rms": rms(responses),
            }
        )
        all_delta.append(responses)
        all_identity.append(identity_error)
        all_hidden.append(z["hidden_rms"])
        z.close()
        LOG.info(
            "independent component%d accepted%d rejected%d", i, accepted_count, 12 - accepted_count
        )
    accepted = [r for r in records if r["accepted"]]
    rejected = [r for r in records if not r["accepted"]]
    cw = np.array([c["accepted_mass"] for c in components])
    cs = np.array([c["squared_response_total"] for c in components])
    draws = np.random.default_rng(614305).integers(0, 24, size=(1000, 24))
    boot = np.sqrt(cs[draws].sum(axis=1) / cw[draws].sum(axis=1))
    response = np.concatenate(all_delta)
    errors = np.concatenate(all_identity)
    weights = np.array([r["weight"] for r in accepted])
    statistics = {
        "selected": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "all_components": len(components),
        "accepted_mass": float(cw.sum()),
        "rejected_mass": sum(r["weight"] for r in rejected),
        "domain_mass": sum(s["n_eligible"] for s in panel["samples"]),
        "raw_response_rms": rms(response),
        "per_channel_rms": np.sqrt((response**2).mean(axis=0)).tolist(),
        "inverse_probability_rms": float(np.sqrt(cs.sum() / cw.sum())),
        "component_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "identity_rms": rms(errors),
        "weighted_hidden_layer_rms": np.sqrt(
            np.average(np.concatenate(all_hidden) ** 2, axis=0, weights=weights)
        ).tolist(),
        "rejected_clock_cases": sum(r["clock_mismatch_entries"] > 0 for r in rejected),
        "rejected_held_cases": sum(r["held_mismatch_entries"] > 0 for r in rejected),
        "nonfinite": sum(not r["finite"] for r in records),
        "accepted_per_component_range": [
            min(c["accepted"] for c in components),
            max(c["accepted"] for c in components),
        ],
        "donor_lags": {
            str(lag_value): sum(r["lag"] == lag_value for r in records)
            for lag_value in (64, 128, 256)
        },
    }
    claimed = read(ACTOR / "results.json")
    assert (
        abs(
            statistics["inverse_probability_rms"] - claimed["response"]["design_weighted_query_rms"]
        )
        < 1e-12
    )
    assert np.allclose(
        statistics["component_bootstrap95"],
        claimed["response"]["conditional_component95_interval"],
        atol=1e-12,
    )
    after = {str(p): sha(p) for p in paths}
    assert before == after
    result = {
        "verdict": "SUPPORTED_NARROW_RETENTION_ON_ACCEPTED_SUFFIX_MATCHED_PANEL",
        "statistics": statistics,
        "component_results": components,
        "all288_records": records,
        "independent_forward_pairs": forwards,
        "original_packed_raw_identity_components": source_checks,
        "checkpoint_source_split_hashes": {
            p: h
            for p, h in before.items()
            if p in [str(REPO / control["checkpoint"]), str(REPO / control["split_manifest"])]
            or p in [str(REPO / q) for q in control["source_sha256"]]
        },
        "assumptions": [
            "original eligibility-mask census and query sampling seed inherited, not independently resampled",
            "main split row-group exclusion checked; complete connected-component assignment and teacher lineage inherited, not independently reaudited",
            "production causal feature primitive shared, actor replay helpers not imported",
            "future canary tests prefix computation only, not training-label lineage or market support",
            "no targets loaded",
        ],
        "input_sha256": before,
        "cost": {
            "wall_seconds": time.monotonic() - begin,
            "cpu_seconds": time.process_time() - cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "GPU": False,
        },
    }
    save("results.json", result)
    save(
        "STATE.json",
        {
            "phase": "review_scored",
            "verdict": result["verdict"],
            "actor_sources_imported": False,
            "target_arrays_loaded": False,
            "cost": result["cost"],
        },
    )
    LOG.info("SUMMARY %s", json.dumps(statistics))
    LOG.info("COST %s", result["cost"])


if __name__ == "__main__":
    main()

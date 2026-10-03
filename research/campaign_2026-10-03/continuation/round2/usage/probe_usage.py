from __future__ import annotations

import argparse
import importlib
import json
import logging
import resource
import sys
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
import torch
from inspect_control import digest
from usage_core import (
    LAGS,
    N_ROWS,
    NAMES,
    QUERIES,
    REPO,
    ROOT,
    cached_array_prefix,
    intervention_window,
    quiet_ages,
    update_state,
)

sys.path.insert(0, str(REPO / "scripts"))
train_rnn = importlib.import_module("train_rnn")


def parity(left: np.ndarray, right: np.ndarray) -> dict[str, float | bool]:
    """Report float32 CPU equivalence using the frozen absolute/relative bound."""
    difference = left.astype(float) - right.astype(float)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(difference**2))),
        "passed": bool(np.allclose(left, right, atol=2e-5, rtol=2e-5)),
    }


def read_prefix(row_group: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read one validation group, retaining only the declared prefix."""
    parquet = pq.ParquetFile(REPO / "datasets/valid.parquet")
    names = parquet.schema_arrow.names
    features = names[names.index("i0_p0") : names.index("a7") + 1]
    table = parquet.read_row_group(
        row_group,
        columns=features + ["t0", "t1", "is_scored", "need_prediction", "seq_ix", "step_in_seq"],
        use_threads=False,
    )
    raw = np.column_stack([table[name].to_numpy()[:N_ROWS] for name in features])
    target = np.column_stack([table[name].to_numpy()[:N_ROWS] for name in ["t0", "t1"]])
    mask = table["is_scored"].to_numpy()[:N_ROWS].copy()
    need = table["need_prediction"].to_numpy()[:N_ROWS]
    ids = table["seq_ix"].to_numpy()[:N_ROWS]
    if not np.array_equal(table["step_in_seq"].to_numpy()[:N_ROWS], np.arange(N_ROWS)):
        raise AssertionError("Parquet prefix steps are not contiguous")
    if not np.all(ids == ids[0]) or not np.array_equal(need, np.arange(N_ROWS) >= 99):
        raise AssertionError("Sequence ID or warmup mismatch")
    return raw, target, mask & need, int(ids[0])


def baseline_replay(
    model: torch.nn.Module, inputs: torch.Tensor
) -> tuple[np.ndarray, dict[int, torch.Tensor]]:
    """Propagate the full original prefix and retain exact boundary states."""
    boundaries = {0, N_ROWS}
    for query in QUERIES:
        boundaries.update(
            [int(query - 256), int(query - 32), int(query), int(query + 1), int(query + 33)]
        )
    ordered = sorted(boundaries)
    state = torch.zeros(2, 1, 256)
    states = {0: state.clone()}
    predictions = np.empty((N_ROWS, 2), dtype=np.float32)
    for start, end in zip(ordered[:-1], ordered[1:], strict=True):
        output, state = model(inputs[:, start:end], state)
        predictions[start:end] = output[0, :, :2].numpy()
        states[end] = state.clone()
    return predictions, states


def main(index: int) -> None:
    """Measure all frozen interventions for one independently replayed sequence."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.monotonic()
    sample = json.loads((ROOT / "samples.json").read_text())["samples"][index]
    update_state(f"inference_sequence_{index}_running", {"active_sample": sample})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not json.loads((ROOT / "synthetic_results.json").read_text())["passed"]:
        raise AssertionError("Synthetic instrument control has not passed")
    control = json.loads((ROOT / "control_identity.json").read_text())
    checkpoint_path = REPO / control["checkpoint"]
    if digest(checkpoint_path) != control["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after preregistration")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = train_rnn.make_model(checkpoint["args"], checkpoint["n_in"], control["n_out"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    reference_model = train_rnn.make_model(checkpoint["args"], checkpoint["n_in"], control["n_out"])
    reference_model.load_state_dict(checkpoint["model"])
    reference_model.eval()
    raw, targets, mask, seq_id = read_prefix(sample["row_group"])
    if seq_id != sample["seq_ix"]:
        raise AssertionError("Raw sequence ID differs from frozen metadata")
    cached_targets = cached_array_prefix("y", sample["cache_position"], N_ROWS)
    cached_mask = cached_array_prefix("mask", sample["cache_position"], N_ROWS)
    if not np.array_equal(targets, cached_targets) or not np.array_equal(mask, cached_mask):
        raise AssertionError("Cached targets/masks differ from raw parquet prefix")
    raw_tensor = torch.from_numpy(raw)[None]
    with torch.inference_mode():
        reference_started = time.monotonic()
        reference = train_rnn.predict(reference_model, raw_tensor, 4000, torch.device("cpu"))[0]
        reference_seconds = time.monotonic() - reference_started
        inputs = train_rnn.build_inputs(raw_tensor)
        packed = inputs[0].numpy()
        if not np.array_equal(
            train_rnn.build_inputs(raw_tensor[:, :1000]).numpy(), inputs[:, :1000].numpy()
        ):
            raise AssertionError("Original feature pipeline is not prefix invariant")
        baseline, states = baseline_replay(model, inputs)
        baseline_gate = parity(baseline, reference)
        if not baseline_gate["passed"]:
            raise AssertionError(f"Original CPU reference parity failed: {baseline_gate}")
        cached = np.load(REPO / "cache/pred_c20261001_d256_s3_f0_valid.npy", mmap_mode="r")[
            sample["cache_position"], :N_ROWS
        ].copy()
        cache_difference = reference.astype(float) - cached.astype(float)
        cache_correlation = [
            float(np.corrcoef(reference[:, target], cached[:, target])[0, 1]) for target in range(2)
        ]
        cache_gate = {
            "max_abs": float(np.abs(cache_difference).max()),
            "rms": float(np.sqrt(np.mean(cache_difference**2))),
            "correlations": cache_correlation,
        }
        cache_gate["passed"] = (
            cache_gate["max_abs"] <= 0.02
            and cache_gate["rms"] <= 0.005
            and min(cache_correlation) >= 0.999
        )
        if not cache_gate["passed"]:
            raise AssertionError(f"Cached GPU/CPU alignment bound failed: {cache_gate}")
        responses = np.empty((len(QUERIES), len(NAMES), len(LAGS), 2), dtype=np.float32)
        predictions = np.empty_like(responses)
        hidden_query = np.empty((len(QUERIES), len(NAMES), 2))
        hidden_end = np.empty_like(hidden_query)
        windows = np.full((len(QUERIES), len(NAMES), 289, 2), np.nan, dtype=np.float32)
        starts = np.empty((len(QUERIES), len(NAMES)), dtype=int)
        lengths = np.empty_like(starts)
        identity_max = identity_squares = identity_count = 0
        for query_index, query in enumerate(QUERIES):
            for name_index, name in enumerate(NAMES):
                start, window = intervention_window(packed, int(query), name)
                query_position = int(query - start)
                view = torch.from_numpy(window)[None]
                output, altered_state = model(view[:, : query_position + 1], states[start].clone())
                hidden_query[query_index, name_index] = torch.sqrt(
                    torch.mean((altered_state - states[int(query + 1)]) ** 2, dim=(1, 2))
                ).numpy()
                future, altered_state = model(view[:, query_position + 1 :], altered_state)
                hidden_end[query_index, name_index] = torch.sqrt(
                    torch.mean((altered_state - states[int(query + 33)]) ** 2, dim=(1, 2))
                ).numpy()
                complete = torch.cat([output, future], dim=1)[0, :, :2].numpy()
                starts[query_index, name_index], lengths[query_index, name_index] = (
                    start,
                    len(complete),
                )
                windows[query_index, name_index, : len(complete)] = complete
                predictions[query_index, name_index] = complete[query_position + LAGS]
                responses[query_index, name_index] = (
                    predictions[query_index, name_index] - baseline[query + LAGS]
                )
                if name == "identity":
                    gate = parity(complete, baseline[start : int(query + 33)])
                    if not gate["passed"]:
                        raise AssertionError(f"Identity replay failed: {gate}")
                    identity_max = max(identity_max, gate["max_abs"])
                    identity_squares += gate["rms"] ** 2 * complete.size
                    identity_count += complete.size
            if (query_index + 1) % 16 == 0:
                logging.info("sequence%s queries %s/%s", index, query_index + 1, len(QUERIES))
    age = quiet_ages(raw)
    organizer_event = packed[:, 178] > 0.5
    rows = np.arange(N_ROWS)
    organizer_age = rows - np.maximum.accumulate(np.where(organizer_event, rows, 0))
    state_boundaries = sorted(states)
    np.savez(
        ROOT / f"sequence_{index:02d}.npz",
        raw=raw,
        packed=packed,
        baseline=baseline,
        reference=reference,
        cached=cached,
        targets=targets,
        mask=mask,
        queries=QUERIES,
        lags=LAGS,
        predictions=predictions,
        responses=responses,
        quiet_age=age[QUERIES],
        organizer_age=organizer_age[QUERIES],
        hidden_query=hidden_query,
        hidden_end=hidden_end,
        replay_predictions=windows,
        replay_starts=starts,
        replay_lengths=lengths,
        state_boundaries=np.array(state_boundaries),
        original_states=np.stack([states[key].numpy() for key in state_boundaries]),
    )
    summary = {
        "sample": sample,
        "baseline_cpu_reference_gate": baseline_gate,
        "cached_gpu_vs_cpu_gate": cache_gate,
        "identity_replay_gate": {
            "passed": True,
            "max_abs": identity_max,
            "rms": float(np.sqrt(identity_squares / identity_count)),
        },
        "alignment": {
            "seq_ix": True,
            "row_group": True,
            "targets": True,
            "mask": True,
            "all_warmup_rows": True,
            "prefix_pipeline": True,
        },
        "query_count": len(QUERIES),
        "scored_query_count": int(mask[QUERIES].sum()),
        "reference_seconds": reference_seconds,
        "seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "backend": {"torch": torch.__version__, "device": "cpu", "dtype": "float32", "threads": 1},
    }
    (ROOT / f"sequence_{index:02d}.json").write_text(json.dumps(summary, indent=2))
    logging.info(json.dumps(summary, indent=2))
    state = json.loads((ROOT / "STATE.json").read_text())
    completed = sorted(set(state.get("completed_sequences", []) + [index]))
    update_state(
        f"inference_sequence_{index}_complete",
        {
            "completed_sequences": completed,
            "last_failure": None,
            "last_runtime_seconds": summary["seconds"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, choices=range(6), required=True)
    arguments = parser.parse_args()
    try:
        main(arguments.sequence)
    except Exception:
        error = traceback.format_exc()
        (ROOT / f"failure_sequence_{arguments.sequence:02d}.txt").write_text(error)
        update_state("failure_no_automatic_same_condition_retry", {"last_failure": error})
        raise

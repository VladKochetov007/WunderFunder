from __future__ import annotations

import copy
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[5]
OWN = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    """Hash a bounded source, checkpoint or own artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maxdiff(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    """Compare aligned finite CPU arrays without reducing validity to clipping."""
    av = a.numpy() if isinstance(a, torch.Tensor) else a
    bv = b.numpy() if isinstance(b, torch.Tensor) else b
    assert av.shape == bv.shape and np.isfinite(av).all() and np.isfinite(bv).all()
    return float(np.max(np.abs(av - bv)))


def main() -> None:
    import train_rnn

    from wnn.data import FEATURE_COLUMNS
    from wnn.features import StreamingEventFeatures
    from wnn.splits import load_campaign_split

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protected = ["PREMORTEM.md", "TESTS.json", "checks.py", "results.json", "STATE.json"]
    original_hashes = {name: digest(OWN / name) for name in protected}
    source_paths = [
        "scripts/predict_run.py",
        "scripts/train_rnn.py",
        "src/wnn/nn.py",
        "src/wnn/features.py",
        "src/wnn/data.py",
        "src/wnn/splits.py",
        "src/wnn/event_content.py",
        "src/wnn/resources.py",
    ]
    source_hashes = {name: digest(ROOT / name) for name in source_paths}
    checkpoint = ROOT / "artifacts/runs/c20261001_d256_s3_f0/model_ep3.pt"
    prereg = json.loads((OWN / "stream_equivalence_prereg.json").read_text())
    assert digest(checkpoint) == prereg["expected_checkpoint_sha256"]
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ck["args"]
    assert ck["epoch"] == 3 and ck["n_in"] == 211
    assert args["hidden"] == 256 and args["layers"] == 2
    assert args["arch"] == "gru" and args["enc"] == "linear"
    assert not any(args.get(k) for k in ("online_norm", "heuristics", "prop_input", "raw_only"))
    assert ck.get("readout") is None
    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split = load_campaign_split(split_path, ROOT)
    assert (
        split.sha256 == args["split_manifest_sha256"] == ck["provenance"]["split_manifest_sha256"]
    )
    manifest = json.loads(split_path.read_text())
    components = {
        (s, n["row_group"]): n["component"]
        for s, nodes in manifest["sequences"].items()
        for n in nodes
    }
    fitted = ck["provenance"]["fit_row_groups"]
    split.assert_fit_groups(fitted["train"], fitted["valid"])
    fit_components = {components[(s, g)] for s, groups in fitted.items() for g in groups}
    selected = []
    seen = set()
    for group in sorted(split.evaluation_groups):
        component = components[("valid", group)]
        if component not in seen:
            assert component not in fit_components
            selected.append(group)
            seen.add(component)
        if len(selected) == 2:
            break
    assert len(selected) == 2
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    metadata = []
    raw_arrays = []
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS]
    for group in selected:
        batch = next(
            pf.iter_batches(batch_size=256, row_groups=[group], columns=columns, use_threads=False)
        )
        assert batch.num_rows == 256
        raw = np.ascontiguousarray(
            np.column_stack(
                [batch.column(batch.schema.get_field_index(c)).to_numpy() for c in FEATURE_COLUMNS]
            ),
            dtype=np.float32,
        )
        steps = batch.column(batch.schema.get_field_index("step_in_seq")).to_numpy()
        seq = batch.column(batch.schema.get_field_index("seq_ix")).to_numpy()
        need = batch.column(batch.schema.get_field_index("need_prediction")).to_numpy(
            zero_copy_only=False
        )
        assert np.array_equal(steps, np.arange(256)) and np.unique(seq).size == 1
        raw_arrays.append(raw)
        metadata.append(
            {
                "row_group": group,
                "seq_ix": int(seq[0]),
                "component": components[("valid", group)],
                "prefix_steps": [0, 255],
                "need_prediction_count": int(need.sum()),
                "raw_float32_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            }
        )
    raw_np = np.stack(raw_arrays)
    prefix_path = OWN / "stream_equivalence_prefixes.npz"
    if prefix_path.exists():
        with np.load(prefix_path) as frozen:
            assert np.array_equal(frozen["raw"], raw_np)
    else:
        np.savez(prefix_path, raw=raw_np)
    (OWN / "stream_equivalence_sample.json").write_text(
        json.dumps(
            {
                "sample": metadata,
                "split_manifest_sha256": split.sha256,
                "fit_component_disjoint": True,
                "exposure": prereg["exposure"],
                "sources_sha256": source_hashes,
                "protected_artifacts_sha256": original_hashes,
                "checkpoint_sha256": digest(checkpoint),
                "checkpoint_provenance": ck["provenance"],
                "checkpoint_tensor_dtypes": sorted({str(t.dtype) for t in ck["model"].values()}),
            },
            indent=2,
        )
        + "\n"
    )
    model = train_rnn.make_model(args, ck["n_in"], ck["model"]["head.2.weight"].shape[0])
    model.load_state_dict(ck["model"])
    model.eval()
    train_rnn.RAW_ONLY = train_rnn.ONLINE_NORM = train_rnn.HEURISTICS = False
    train_rnn.INPUT_PROPENSITY = train_rnn.READOUT = None
    raw_t = torch.from_numpy(raw_np)
    results: dict[str, object] = {}
    with torch.inference_mode():
        inputs = train_rnn.build_inputs(raw_t)
        assert inputs.shape == (2, 256, 211) and torch.isfinite(inputs).all()
        feature_hash = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()

        def partition(chunk: int, data: torch.Tensor = inputs) -> tuple[torch.Tensor, torch.Tensor]:
            state = None
            outputs = []
            for start in range(0, 256, chunk):
                out, state = train_rnn.forward_chunk(
                    model, data, slice(start, start + chunk), state
                )
                assert torch.isfinite(out).all() and torch.isfinite(state).all()
                outputs.append(out)
            return torch.cat(outputs, dim=1), state

        reference, final_state = partition(256)
        repeated, _ = partition(256)
        repeat_error = maxdiff(reference, repeated)
        tolerance = float(
            max(
                8 * repeat_error,
                64 * np.finfo(np.float32).eps * max(1.0, float(reference.abs().max())),
            )
        )
        results["identity_repeat_max_abs"] = repeat_error
        results["model_atol"] = tolerance
        results["partition_max_abs"] = {}
        results["final_state_partition_max_abs"] = {}
        for chunk in prereg["chunk_sizes"]:
            output, state = partition(chunk)
            error, state_error = maxdiff(reference, output), maxdiff(final_state, state)
            results["partition_max_abs"][str(chunk)] = error
            results["final_state_partition_max_abs"][str(chunk)] = state_error
            assert error <= tolerance and state_error <= tolerance
        reversed_output, _ = partition(256, inputs.flip(0))
        results["batch_reorder_max_abs"] = maxdiff(reference, reversed_output.flip(0))
        assert results["batch_reorder_max_abs"] <= tolerance
        # Pause after two 64-row chunks; copy actual recurrent state, not reset.
        state = None
        for start in (0, 64):
            _, state = train_rnn.forward_chunk(model, inputs, slice(start, start + 64), state)
        saved = state.clone()
        tails = []
        for resumed_state in (state, saved.clone()):
            outs = []
            for start in (128, 192):
                out, resumed_state = train_rnn.forward_chunk(
                    model, inputs, slice(start, start + 64), resumed_state
                )
                outs.append(out)
            tails.append(torch.cat(outs, 1))
        results["pause_replay_max_abs"] = maxdiff(*tails)
        assert results["pause_replay_max_abs"] <= tolerance
        # Corrupt only the unseen suffix; earlier raw/features/output must agree.
        changed_raw = raw_t.clone()
        changed_raw[:, 128:] = raw_t.flip(0)[:, 128:]
        changed_inputs = train_rnn.build_inputs(changed_raw)
        results["suffix_feature_prefix_max_abs"] = maxdiff(inputs[:, :128], changed_inputs[:, :128])
        changed_out, _ = partition(256, changed_inputs)
        results["suffix_prediction_prefix_max_abs"] = maxdiff(
            reference[:, :128], changed_out[:, :128]
        )
        assert results["suffix_feature_prefix_max_abs"] == 0
        assert results["suffix_prediction_prefix_max_abs"] <= tolerance
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            helper = {
                str(chunk): train_rnn.predict(model, raw_t, chunk, torch.device("cpu"))
                for chunk in prereg["chunk_sizes"]
            }
            ordered = [
                train_rnn.predict(model, raw_t[i : i + 1], 256, torch.device("cpu")) for i in (0, 1)
            ]
            reordered = {
                i: train_rnn.predict(model, raw_t[i : i + 1], 256, torch.device("cpu"))
                for i in (1, 0)
            }
        results["predict_helper_cpu_warnings"] = sorted({str(w.message) for w in caught})
        results["predict_helper_vs_precomputed_max_abs"] = {
            k: maxdiff(reference[..., :2], p) for k, p in helper.items()
        }
        results["predict_helper_call_reorder_max_abs"] = max(
            maxdiff(ordered[i], reordered[i]) for i in (0, 1)
        )
        assert max(results["predict_helper_vs_precomputed_max_abs"].values()) <= tolerance
        assert results["predict_helper_call_reorder_max_abs"] <= tolerance
        # Negative control: deliberately carry A's state into B; not a real-helper bug.
        _, state_a = partition(256, inputs[:1])
        bad, _ = train_rnn.forward_chunk(model, inputs[1:2], slice(0, 256), state_a)
        results["missing_model_reset_negative_control_max_abs"] = maxdiff(reference[1:2], bad)
        results["negative_control_detected"] = (
            results["missing_model_reset_negative_control_max_abs"] > tolerance
        )
    streaming = StreamingEventFeatures()
    streamed = []
    repeat_streamed = []
    replay_errors = []
    missing_reset = 0.0
    for i, rows in enumerate(raw_np):
        if i:
            missing_reset = float(
                np.max(np.abs(streaming.update(rows[0]).copy() - inputs[i, 0, 112:].numpy()))
            )
        streaming.reset()
        out = []
        for row in rows[:128]:
            out.append(streaming.update(row).copy())
        snapshot = copy.deepcopy(streaming)
        tail = np.stack([streaming.update(row).copy() for row in rows[128:]])
        replay = np.stack([snapshot.update(row).copy() for row in rows[128:]])
        replay_errors.append(maxdiff(tail, replay))
        streamed.append(np.concatenate([np.stack(out), tail]))
        fresh = StreamingEventFeatures()
        repeat_streamed.append(np.stack([fresh.update(row).copy() for row in rows]))
    stream_np = np.stack(streamed)
    stream_repeat = maxdiff(stream_np, np.stack(repeat_streamed))
    feature_tolerance = max(8 * stream_repeat, 8 * np.finfo(np.float32).eps)
    feature_error = maxdiff(stream_np, inputs[..., 112:].numpy())
    # Only log1p recency columns may differ through NumPy/Torch rounding.
    exact_columns = list(range(74)) + list(range(77, 99))
    assert (
        maxdiff(stream_np[..., exact_columns], inputs[..., 112:].numpy()[..., exact_columns]) == 0
    )
    assert feature_error <= feature_tolerance and max(replay_errors) == 0
    results["streaming_feature_max_abs"] = feature_error
    results["streaming_feature_identity_max_abs"] = stream_repeat
    results["streaming_feature_atol"] = float(feature_tolerance)
    results["streaming_pause_replay_max_abs"] = max(replay_errors)
    results["missing_feature_reset_negative_control_max_abs"] = missing_reset
    suffix_streamed = []
    for rows in changed_raw.numpy():
        primitive = StreamingEventFeatures()
        suffix_streamed.append(np.stack([primitive.update(row).copy() for row in rows]))
    results["streaming_suffix_prefix_max_abs"] = maxdiff(
        stream_np[:, :128], np.stack(suffix_streamed)[:, :128]
    )
    assert results["streaming_suffix_prefix_max_abs"] == 0
    for name, expected in source_hashes.items():
        assert digest(ROOT / name) == expected
    assert digest(checkpoint) == prereg["expected_checkpoint_sha256"]
    assert {name: digest(OWN / name) for name in protected} == original_hashes
    result = {
        "status": "passed narrow CPU callable checks",
        "torch_version": torch.__version__,
        "sample": metadata,
        "checkpoint_sha256": digest(checkpoint),
        "source_sha256": source_hashes,
        "precomputed_feature_sha256": feature_hash,
        "dtype": "CPU float32; predict CUDA autocast disabled on CPU",
        "checks": results,
        "protected_artifacts_unchanged": True,
        "excluded": "GPU/FP16/native/compiled/deployed callback, full sequences, WP and predictive information",
    }
    (OWN / "stream_equivalence_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

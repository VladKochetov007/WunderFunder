"""Allocated-only prefix extraction for SELRISK01. No metrics or training."""
from pathlib import Path
import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
import numpy as np
from common import ROOT, HERE, PANELS, BINS, load_allocation, verify_inputs, select_records, repo, sha


def extract_sequence(model, build_inputs, forward_chunk, raw, positions, chunk=2000):
    """Gather selected rows after one complete causal forward; raw inputs only."""
    import torch
    assert raw.ndim == 3 and raw.shape[0] == 1 and raw.shape[2] == 112
    positions = np.asarray(positions, dtype=np.int64)
    assert positions.ndim == 2 and positions.shape[0] == 3
    assert np.all((positions >= 0) & (positions < raw.shape[1]))
    result = np.full((*positions.shape, 341), np.nan, dtype=np.float32)
    captured = []
    hook = model.head[2].register_forward_pre_hook(lambda module, args_: captured.append(args_[0].detach()))
    try:
        with torch.no_grad():
            inputs = build_inputs(raw)
            assert inputs.shape == (1, raw.shape[1], 211) and torch.isfinite(inputs).all()
            state = None
            for begin in range(0, raw.shape[1], chunk):
                captured.clear()
                out, state = forward_chunk(model, inputs, slice(begin, begin + chunk), state)
                assert len(captured) == 1
                selected = (positions >= begin) & (positions < begin + chunk)
                target_indices = np.argwhere(selected)
                unique = np.unique(positions[selected])
                if not len(unique):
                    continue
                ix = torch.as_tensor(unique - begin, device=raw.device)
                absolute_ix = torch.as_tensor(unique, device=raw.device)
                normalized = (inputs[0, absolute_ix] - model.mean) / model.std
                values = torch.cat([normalized, captured[0][0, ix], out[0, ix, :2]], dim=-1).cpu().numpy()
                assert values.shape == (len(unique), 341) and np.isfinite(values).all()
                for panel, row_index in target_indices:
                    result[panel, row_index] = values[np.searchsorted(unique, positions[panel, row_index])]
    finally:
        hook.remove()
    assert np.isfinite(result).all()
    return result


def main():
    start = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--allocation", type=Path, required=True)
    ap.add_argument("--allocation-sha256", required=True)
    args = ap.parse_args()
    allocation, bindings, output = load_allocation(args.allocation, args.allocation_sha256, "extract")
    gpu = int(allocation["gpu_index"])
    assert gpu not in (6, 7), "Foreign GPU6 and live student GPU7 are forbidden"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu)
    assert not output.exists(), "No retries/overwrites"
    output.mkdir()
    (output / "started.json").write_text(json.dumps({"pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat(),
        "allocation_sha256": args.allocation_sha256, "gpu_index": gpu, "bindings_sha256": allocation["bindings_sha256"]}, indent=2))
    verify_inputs(bindings)
    cf = np.load(repo("cache/chain_folds.npz"), allow_pickle=False)
    classes = np.load(repo("research/artifacts/market-classes-v1.npz"), allow_pickle=False)["valid"]
    records = select_records(cf, classes)
    # The exact metadata manifest is closed before any checkpoint or feature/target read.
    manifest = {"records": records, "panels": list(PANELS), "bin_lengths": np.diff(BINS).tolist(),
                "bindings_sha256": allocation["bindings_sha256"], "metadata_only_created_utc": datetime.now(timezone.utc).isoformat()}
    (output / "selected-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    driver = {"peak_MiB": 0, "samples": 0, "errors": []}
    done = threading.Event()

    def monitor():
        while not done.is_set():
            if time.monotonic() - start > 598:
                (output / "resource-failure.json").write_text(json.dumps({"reason": "extract wall cap", **driver}))
                os._exit(70)
            try:
                result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_gpu_memory",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=3)
                used = [float(line.split(",")[1]) for line in result.stdout.splitlines()
                        if int(line.split(",")[0]) == os.getpid()]
                if used:
                    driver["peak_MiB"] = max(driver["peak_MiB"], sum(used))
                    driver["samples"] += 1
                    if sum(used) > 5632:
                        (output / "resource-failure.json").write_text(json.dumps({"reason": "own GPU usage above5632MiB", **driver}))
                        os._exit(70)
            except Exception as exc:
                driver["errors"].append(str(exc))
                (output / "resource-failure.json").write_text(json.dumps({"reason": "GPU monitor unavailable", **driver}))
                os._exit(70)
            done.wait(1)

    watch = threading.Thread(target=monitor, daemon=True)
    watch.start()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.device_count() == 1
    free, total = torch.cuda.mem_get_info()
    assert free >= 6200 * 2**20, "Insufficient free memory before context/model"
    torch.cuda.set_per_process_memory_fraction(min(.30, 5400 * 2**20 / total))
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
    import train_rnn
    from wnn.data import SequenceFile
    reader = SequenceFile(repo("datasets/valid.parquet"))
    X = np.lib.format.open_memmap(output / "X.npy", mode="w+", dtype=np.float32, shape=(128, 3, 256, 341))
    Y = np.lib.format.open_memmap(output / "Y.npy", mode="w+", dtype=np.float32, shape=(128, 3, 256, 2))
    M = np.lib.format.open_memmap(output / "M.npy", mode="w+", dtype=np.bool_, shape=(128, 3, 256))
    X[:] = np.nan
    Y[:] = np.nan
    resets = rows = 0
    with torch.no_grad(), (output / "progress.jsonl").open("x") as log:
        for fold in (0, 1):
            checkpoint = repo(f"artifacts/runs/on_ctl128s3_f{fold}/model_ep1.pt")
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            a = ck["args"]
            assert int(ck["epoch"]) == 1 and int(a["valid_fold"]) == fold
            assert ck["n_in"] == 211 and a["hidden"] == 128 and a["layers"] == 2 and a["dropout"] == 0
            assert a.get("arch", "gru") == "gru" and a.get("enc", "linear") == "linear" and a.get("cell", "gru") == "gru"
            assert not any(a.get(k) for k in ("lookahead", "class_heads", "atom_head", "raw_only", "online_norm", "quant_feats", "quant_i1", "state_reset", "prop_input", "heuristics"))
            model = train_rnn.make_model(a, ck["n_in"], ck["model"]["head.2.weight"].shape[0])
            model.load_state_dict(ck["model"], strict=True)
            model = model.to("cuda", dtype=torch.float32).eval()
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            del ck
            for index, record in enumerate(records):
                if record["fold"] != fold:
                    continue
                sequence = reader.read(record["row_group"], targets=True, mask=True)
                assert sequence.targets is not None and sequence.scored is not None
                raw = torch.from_numpy(sequence.features).unsqueeze(0).to("cuda")
                positions = np.asarray([record["panels"][panel]["positions"] for panel in PANELS])
                Y[index] = sequence.targets[positions]
                M[index] = sequence.scored[positions]
                X[index] = extract_sequence(model, train_rnn.build_inputs, train_rnn.forward_chunk, raw, positions)
                resets += 1
                rows += 20000
                assert np.isfinite(Y[index]).all()
                del raw, sequence
                log.write(json.dumps({"index": index, "fold": fold, "chain": record["chain"], "rows_total": rows,
                                      "seconds": time.monotonic() - start, "driver_peak_MiB": driver["peak_MiB"]}) + "\n")
                log.flush()
            del model
    for arr in (X, Y, M):
        arr.flush()
    del X, Y, M
    assert resets == 128 and rows == 2560000
    verify_inputs(bindings)
    receipt = {"status": "COMPLETE", "allocation_sha256": args.allocation_sha256,
               "manifest_sha256": sha(output / "selected-manifest.json"),
               "files": {name: sha(output / name) for name in ("X.npy", "Y.npy", "M.npy")},
               "resets": resets, "prefix_rows": rows, "sampled_records": 98304,
               "seconds": time.monotonic() - start, "driver": driver,
               "max_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
               "max_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20,
               "maxRSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "precision": "FP32; autocast off; TF32 off", "forecast_fits": 0, "WP_queries": 0}
    assert receipt["seconds"] < 600
    done.set()
    watch.join(4)
    assert driver["samples"] > 0 and not driver["errors"]
    (output / "extraction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETE", "output": str(output), "seconds": receipt["seconds"]}))


if __name__ == "__main__":
    main()

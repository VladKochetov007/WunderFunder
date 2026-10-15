"""Check a pinned GRU growth on one complete fitting sequence, without fitting."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from wnn.features import event_features_torch
from wnn.gru_growth import grow_gru_state
from wnn.nn import RNNModel
from wnn.splits import load_campaign_split

PARENT = "c20261003_combo_s3_f0"
CHECKPOINT_SHA = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
PROVENANCE_SHA = "9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f"
FIXTURE_SHA = "941b656d8408be6f5c8f5816c1e32f3c8b24e59040bb4ed84bb895078a9f9f05"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"


def sha256(path: Path) -> str:
    """Hash one explicitly bounded input or emitted artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Compare all output/state coordinates and save a derived initializer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initializer-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.initializer_dir.exists():
        raise FileExistsError("refusing to overwrite a diagnostic or initializer")
    started = time.perf_counter()
    checkpoint_path = args.root / f"artifacts/runs/{PARENT}/model_ep3.pt"
    provenance_path = checkpoint_path.with_name("provenance.json")
    fixture_path = (
        args.root / "research/campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_TIMING_FIXTURE_R1.json"
    )
    for path, digest, bound in (
        (checkpoint_path, CHECKPOINT_SHA, 8_000_000),
        (provenance_path, PROVENANCE_SHA, 2_000_000),
        (fixture_path, FIXTURE_SHA, 100_000),
    ):
        if path.stat().st_size > bound or sha256(path) != digest:
            raise ValueError(f"input size/hash mismatch: {path}")
    provenance = json.loads(provenance_path.read_text())
    split = load_campaign_split(
        args.root / "research/campaign_2026-09-29/artifacts/splits/fold0.json", args.root
    )
    if split.sha256 != SPLIT_SHA or provenance["split_manifest_sha256"] != SPLIT_SHA:
        raise ValueError("parent split differs")
    fit = provenance["fit_row_groups"]
    split.assert_fit_groups(set(map(int, fit["train"])), set(map(int, fit["valid"])))
    fixture = json.loads(fixture_path.read_text())
    entry = fixture["sequences"][0]
    if entry["source"] != "train" or entry["train_row_group"] not in fit["train"]:
        raise ValueError("fixture is not in the declared parent fitting membership")
    spec = fixture["arrays"]["features"]
    raw_path = args.root / fixture["payload_root"] / spec["file"]
    if (
        spec["shape"] != [8, 20000, 112]
        or spec["dtype"] != "<f4"
        or spec["bytes"] != 71_680_000
        or raw_path.stat().st_size != spec["bytes"]
        or sha256(raw_path) != spec["sha256"]
    ):
        raise ValueError("fixture payload differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if any(key in checkpoint for key in ("optimizer", "scheduler", "scaler", "rng")):
        raise ValueError("expected an inference checkpoint, not a resumable training state")
    embedded = checkpoint.get("provenance", {})
    if "split_manifest_sha256" in embedded and embedded["split_manifest_sha256"] != SPLIT_SHA:
        raise ValueError("embedded checkpoint split differs from pinned provenance")
    if "fit_row_groups" in embedded:
        for source in ("train", "valid"):
            if set(map(int, embedded["fit_row_groups"][source])) != set(map(int, fit[source])):
                raise ValueError("embedded checkpoint fitting membership differs")
    options = checkpoint["args"]
    defaults = {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "readout_activation": "gelu",
        "numeric_map": "none",
        "head_layers": 1,
        "research_features": "none",
        "privileged_mode": "none",
        "online_norm": False,
        "raw_only": False,
        "heuristics": False,
        "prop_input": "",
        "sentinel_base_width": 0,
    }
    for key, value in defaults.items():
        if options.get(key, value) != value:
            raise ValueError(f"unsupported parent semantics: {key}")
    for key in ("encoder_width", "recurrent_width", "readout_width"):
        if options.get(key) not in (None, 0, 256):
            raise ValueError(f"unexpected parent width: {key}")
    if checkpoint.get("quantization") or checkpoint.get("factorization"):
        raise ValueError("only dense float parents are supported")
    state = checkpoint["model"]
    if state["mean"].shape != (211,):
        raise ValueError("expected the unchanged 211-feature parent")
    torch.set_num_threads(1)
    torch.manual_seed(734)
    n_out = state["head.2.weight"].shape[0]
    parent = RNNModel(211, hidden=256, layers=2, n_out=n_out).eval()
    parent.load_state_dict(state, strict=True)
    child = RNNModel(211, hidden=384, layers=2, n_out=n_out).eval()
    child.load_state_dict(grow_gru_state(state, child.state_dict()), strict=True)
    raw_map = np.memmap(raw_path, mode="r", dtype="<f4", shape=(8, 20000, 112))
    raw = torch.from_numpy(raw_map[0].copy()).unsqueeze(0)
    if not torch.isfinite(raw).all():
        raise ValueError("nonfinite input fixture")
    inputs = torch.cat((raw, event_features_torch(raw)), dim=-1)
    max_output = max_hidden = square_output = 0.0
    count = 0
    first_parent = first_child = None
    hp = hc = None
    with torch.inference_mode():
        for lo in range(0, 20000, 256):
            chunk = inputs[:, lo : lo + 256]
            yp, hp = parent(chunk, hp)
            yc, hc = child(chunk, hc)
            difference = (yc - yp).double()
            max_output = max(max_output, difference.abs().max().item())
            max_hidden = max(max_hidden, (hc[..., :256] - hp).abs().max().item())
            square_output += difference.square().sum().item()
            count += difference.numel()
            if lo == 0:
                first_parent, first_child = yp.clone(), yc.clone()
            if not (
                torch.isfinite(yp).all()
                and torch.isfinite(yc).all()
                and torch.isfinite(hp).all()
                and torch.isfinite(hc).all()
            ):
                raise FloatingPointError("nonfinite grown prediction/state")
        replay_parent, _ = parent(inputs[:, :256])
        replay_child, _ = child(inputs[:, :256])
    if first_parent is None or first_child is None:
        raise AssertionError("missing first chunk")
    torch.testing.assert_close(replay_parent, first_parent, atol=0, rtol=0)
    torch.testing.assert_close(replay_child, first_child, atol=0, rtol=0)
    if max_output > 1e-4 or max_hidden > 1e-4:
        raise AssertionError(f"FP32 parity tolerance exceeded: {max_output}, {max_hidden}")
    transform = {
        "kind": "function_preserving_gru_growth",
        "seed": 734,
        "parent_run": PARENT,
        "parent_checkpoint_sha256": CHECKPOINT_SHA,
        "parent_provenance_sha256": PROVENANCE_SHA,
        "widths": [256, 384],
        "new_training_updates": 0,
        "optimizer_state_carried": False,
        "epoch_semantics": "retained source-parent epoch, not a new fitted epoch",
    }
    derived = copy.copy(checkpoint)
    derived["model"] = child.state_dict()
    derived["args"] = options | {
        "hidden": 384,
        "encoder_width": 384,
        "recurrent_width": 384,
        "readout_width": 384,
    }
    derived["derived_initializer"] = transform
    derived["provenance"] = dict(checkpoint.get("provenance", {})) | {
        "derived_initializer": transform
    }
    args.initializer_dir.mkdir(parents=True)
    initializer = args.initializer_dir / "model_ep3.pt"
    torch.save(derived, initializer)
    derived_provenance = provenance | {
        "kind": "derived_initializer",
        "derived_initializer": transform,
        "checkpoint_sha256": {"3": sha256(initializer)},
    }
    (args.initializer_dir / "provenance.json").write_text(
        json.dumps(derived_provenance, indent=2) + "\n"
    )
    group = next(
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "PASS_COMPLETE_SEQUENCE_FP32_GROWTH",
        "scope": "One fitting sequence; no labels, masks, fitting, native export or quality score.",
        "parent": transform,
        "fixture_manifest_sha256": FIXTURE_SHA,
        "raw_payload_sha256": spec["sha256"],
        "sequence": entry,
        "rows": 20000,
        "chunk_rows": 256,
        "output_coordinates": n_out,
        "output_max_abs": max_output,
        "output_rms": (square_output / count) ** 0.5,
        "old_hidden_max_abs": max_hidden,
        "reset_replay_exact": True,
        "embedded_split_present": "split_manifest_sha256" in embedded,
        "embedded_fit_present": "fit_row_groups" in embedded,
        "resource": {
            name: (cgroup / name).read_text().strip()
            for name in ("memory.peak", "memory.max", "memory.swap.max", "memory.events", "cpu.max")
        },
        "elapsed_seconds": time.perf_counter() - started,
        "initializer_path": str(initializer),
        "initializer_sha256": sha256(initializer),
        "source_sha256": {
            str(path): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).parents[1] / "src/wnn/gru_growth.py",
                Path(__file__).parents[1] / "src/wnn/nn.py",
                Path(__file__).parents[1] / "src/wnn/features.py",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

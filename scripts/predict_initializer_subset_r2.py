"""Export one initializer arm over the exact complete F0 valid cache row order."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

PROFILE_CONTRACT_SHA256 = "08b943d63913e5ceebd845f1419feed65af36b375589304ed85fb50ed03fa3e7"
SUBSET_SHA256 = "05f9e671783ad741de4514ad4a566d988280380f3d171e6baee56ceff337893c"
PARENT_CHECKPOINT_SHA256 = "1baa583573089da3eec02bee9256c32e3caeae59e15f77c7475eea0717f006e6"
HOST_MEMORY_LIMIT_BYTES = 6_000_000_000
CUDA_ALLOCATOR_CAP_BYTES = 3_500_000_000
OBSERVED_VRAM_LIMIT_BYTES = 5_000_000_000
ROWS = 20_000
CHUNK_ROWS = 2_000
MAX_JSON_BYTES = 8_000_000


def sha256_file(path: Path) -> str:
    """Hash a file in fixed-size reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_json(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read a small JSON object after checking its physical size."""
    size = path.stat().st_size
    if size <= 0 or size > MAX_JSON_BYTES:
        raise ValueError(f"JSON input exceeds its {MAX_JSON_BYTES}-byte bound: {path}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"JSON input changed while being read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, raw


def load_module(name: str, path: Path) -> Any:
    """Import one pinned module by absolute path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_cgroup_limit() -> int:
    """Read this process's nested cgroup-v2 memory limit safely."""
    memberships = [
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
        if line.startswith("0::")
    ]
    if len(memberships) != 1:
        raise RuntimeError("could not identify prediction worker cgroup")
    root = Path("/sys/fs/cgroup").resolve()
    cgroup = (root / memberships[0].lstrip("/")).resolve()
    if not cgroup.is_relative_to(root):
        raise RuntimeError("prediction worker cgroup escaped its mount")
    text = (cgroup / "memory.max").read_text(encoding="ascii").strip()
    if not text.isdigit():
        raise RuntimeError("prediction requires a finite cgroup memory cap")
    limit = int(text)
    if limit > HOST_MEMORY_LIMIT_BYTES or HOST_MEMORY_LIMIT_BYTES - limit >= 4096:
        raise RuntimeError("prediction cgroup does not match the declared 6 GB reservation")
    return limit


def observed_vram_bytes() -> int:
    """Read total GPU memory use from the independent device monitor."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not values or any(not value.isascii() or not value.isdigit() for value in values):
        raise RuntimeError("could not parse the observed GPU memory counter")
    return sum((int(value) + 1) * 1024**2 for value in values)


class FeatureSequences(Dataset[torch.Tensor]):
    """Feature-only dataset preserving exact valid-target row order."""

    def __init__(self, groups: list[int], seq_ix_by_group: dict[int, int], path: Path) -> None:
        from wnn.data import SequenceFile

        self.groups = groups
        self.seq_ix_by_group = seq_ix_by_group
        self._file = SequenceFile(path)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> torch.Tensor:
        group = self.groups[index]
        sequence = self._file.read(group, targets=False, mask=False)
        if sequence.seq_ix != self.seq_ix_by_group[group] or sequence.features.shape != (ROWS, 112):
            raise ValueError(f"valid Parquet source identity changed for row group {group}")
        return torch.from_numpy(np.asarray(sequence.features).copy())

    def batch_tensor_bytes(self, batch_size: int) -> int:
        """Return the actual raw feature payload for a batch."""
        return batch_size * ROWS * 112 * np.dtype(np.float32).itemsize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--profile-contract", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    if os.environ.get("WNN_GPU_MEMORY_BYTES") != str(CUDA_ALLOCATOR_CAP_BYTES):
        raise RuntimeError("prediction requires the explicit 3.5 GB allocator cap")
    host_limit = read_cgroup_limit()
    if not torch.cuda.is_available():
        raise RuntimeError("initializer prediction was admitted without CUDA")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    contract, contract_raw = bounded_json(args.profile_contract)
    if hashlib.sha256(contract_raw).hexdigest() != PROFILE_CONTRACT_SHA256:
        raise ValueError("initializer profile contract differs from the reviewed R3 packet")
    if sha256_file(args.subset) != SUBSET_SHA256:
        raise ValueError("initializer component subset differs from its frozen manifest")
    subset, _ = bounded_json(args.subset, 5_000_000)
    match = re.fullmatch(
        r"student_init_subf0_s3_(constant|first_row|prefix)_r([1-9][0-9]?)", args.run
    )
    if match is None:
        parser.error("run name must be a frozen seed-3 initializer arm with attempt number 1-99")
    data_root = Path(contract["data_root"]).resolve(strict=True)
    profile_root = Path(contract["source_root"]).resolve(strict=True)
    trainer_root = Path(contract["training_source"]["source_root"]).resolve(strict=True)
    for relative, digest in contract["source_sha256"].items():
        if sha256_file(profile_root / relative) != digest:
            raise ValueError(f"reviewed initializer source changed: {relative}")
    for relative, digest in contract["training_source"]["source_sha256"].items():
        if sha256_file(trainer_root / relative) != digest:
            raise ValueError(f"reviewed trainer source changed: {relative}")
    sys.path.insert(0, str(trainer_root / "src"))
    sys.path.insert(0, str(profile_root / "src"))
    trainer = load_module("train_rnn", trainer_root / "scripts/train_rnn.py")
    trainer.ROOT = data_root
    from wnn.splits import load_campaign_split
    from wnn.state_initialization import InitializedGRU, export_native_initializer

    split_path = data_root / contract["split_path"]
    if sha256_file(split_path) != contract["split_sha256"]:
        raise ValueError("frozen source-qualified F0 split changed")
    split = load_campaign_split(split_path, data_root)
    checkpoint_path = data_root / "artifacts/runs" / args.run / "model_ep0.pt"
    provenance_path = data_root / "artifacts/runs" / args.run / "provenance.json"
    fit_receipt_path = data_root / "artifacts/runs" / args.run / "fit_receipt.json"
    provenance, _ = bounded_json(provenance_path)
    fit_receipt, _ = bounded_json(fit_receipt_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        fit_receipt.get("result") != "PASS"
        or fit_receipt.get("run") != args.run
        or fit_receipt.get("quality_claim") is not None
        or fit_receipt.get("mode") != match.group(1)
        or fit_receipt.get("subset_manifest_sha256") != SUBSET_SHA256
        or fit_receipt.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or fit_receipt.get("updates_attempted") != 258
        or fit_receipt.get("updates_completed") != 258
        or fit_receipt.get("parameter_update_count") != 258
        or not isinstance(fit_receipt.get("nonzero_initializer_gradient_updates"), int)
        or fit_receipt.get("nonzero_initializer_gradient_updates", 0) <= 0
        or fit_receipt.get("nonfinite_losses") != 0
        or fit_receipt.get("nonfinite_gradients") != 0
        or fit_receipt.get("backbone_unchanged") is not True
        or fit_receipt.get("checkpoint_reload_prediction_exact") is not True
        or fit_receipt.get("native_export_after_reload_exact") is not True
        or fit_receipt.get("eval_groups_used_during_fit") != 0
        or fit_receipt.get("model_checkpoint_sha256") != checkpoint_sha256
        or fit_receipt.get("fit_order_sha256") != subset["fit_order_sha256"]
        or fit_receipt.get("normalization_sha256") != contract["normalization"]["hashes"]
        or provenance.get("fit_subset_manifest_sha256") != SUBSET_SHA256
        or provenance.get("split_manifest_sha256") != split.sha256
        or provenance.get("native_profile_gate_result") != "PASS"
    ):
        raise ValueError("initializer fit receipt/provenance failed its fixed-count gate")
    if provenance.get("fit_row_groups") != subset["selected_groups"]:
        raise ValueError(
            "fitted source-qualified membership differs from the frozen component subset"
        )
    split.assert_fit_groups(
        provenance["fit_row_groups"]["train"], provenance["fit_row_groups"]["valid"]
    )
    evaluated_components = {
        component
        for source, group, component in split.components
        if source == "valid" and group in split.evaluation_groups
    }
    fitted_components = {
        component
        for source, group, component in split.components
        if group in set(provenance["fit_row_groups"][source])
    }
    if fitted_components & evaluated_components:
        raise ValueError("initializer fit membership overlaps an evaluated component")

    checkpoint_size = checkpoint_path.stat().st_size
    if checkpoint_size > 80_000_000:
        raise ValueError("initializer run model checkpoint exceeds the 80 MB read bound")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("format") != "wnn.initializer-only-subset-model"
        or checkpoint.get("epoch") != 0
        or checkpoint.get("args", {}).get("name") != args.run
        or checkpoint.get("args", {}).get("split_manifest_sha256") != split.sha256
    ):
        raise ValueError("initializer model checkpoint is not the frozen epoch-0 endpoint")
    initializer_record = checkpoint.get("initializer")
    if (
        not isinstance(initializer_record, dict)
        or initializer_record.get("mode") != fit_receipt["mode"]
    ):
        raise ValueError("initializer model checkpoint lacks its declared learned state")
    run_root = checkpoint_path.parent
    initializer_path = run_root / initializer_record["checkpoint"]
    native_path = run_root / initializer_record["native_archive"]
    if (
        sha256_file(initializer_path) != fit_receipt["initializer_checkpoint_sha256"]
        or sha256_file(native_path) != fit_receipt["native_archive_sha256"]
        or checkpoint.get("initializer", {}).get("checkpoint_sha256")
        != fit_receipt["initializer_checkpoint_sha256"]
        or checkpoint.get("initializer", {}).get("native_archive_sha256")
        != fit_receipt["native_archive_sha256"]
    ):
        raise ValueError("initializer learned checkpoint or native archive changed")

    model = trainer.make_model(
        checkpoint["args"], checkpoint["n_in"], checkpoint["model"]["head.2.weight"].shape[0]
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    wrapper = InitializedGRU(
        model,
        initializer_record["mode"],
        residual_bound=float(initializer_record["residual_bound"]),
    )
    state_checkpoint = torch.load(initializer_path, map_location="cpu", weights_only=False)
    if (
        state_checkpoint.get("format") != "wnn.initializer-only-subset-checkpoint"
        or state_checkpoint.get("mode") != initializer_record["mode"]
        or state_checkpoint.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or state_checkpoint.get("subset_manifest_sha256") != SUBSET_SHA256
        or state_checkpoint.get("fit_order_sha256") != subset["fit_order_sha256"]
    ):
        raise ValueError("initializer-only checkpoint ancestry or order hash changed")
    state = wrapper.state_dict()
    learned = state_checkpoint.get("initializer_state_dict")
    if not isinstance(learned, dict) or not learned:
        raise ValueError("initializer-only checkpoint lacks learned parameters")
    for name, value in learned.items():
        if name not in state or name.startswith("backbone."):
            raise ValueError(f"unsupported learned initializer parameter: {name}")
        state[name] = value
    wrapper.load_state_dict(state, strict=True)
    device = torch.device("cuda")
    wrapper = wrapper.to(device)
    wrapper.eval()
    wrapper._inference_fp32 = True
    trainer.HEURISTICS = False
    trainer.RAW_ONLY = False
    trainer.ONLINE_NORM = False
    trainer.RESEARCH_FEATURES = "none"
    trainer.RESEARCH_FEATURE_CONTROL = "none"
    trainer.PRIVILEGED_MODE = "none"
    trainer.INPUT_PROPENSITY = None
    trainer.INPUT_LEAD = 0
    trainer.SENTINEL_PREPROCESSOR = None
    trainer.SNAP_CODEBOOK = None
    trainer.RESEARCH_FEATURE_PRIOR = None
    trainer.READOUT = (
        [value.to(device) for value in checkpoint["readout"]]
        if checkpoint.get("readout") is not None
        else None
    )
    trainer.READOUT_TILTED = bool(checkpoint.get("readout_tilted", False))
    if any(not torch.isfinite(value).all() for value in wrapper.state_dict().values()):
        raise FloatingPointError("final subset checkpoint contains a nonfinite model tensor")
    with np.load(native_path, allow_pickle=False) as archive:
        native_config = {
            "mode": initializer_record["mode"],
            "residual_bound": float(initializer_record["residual_bound"]),
        }
        for name in archive.files:
            value = np.asarray(archive[name])
            if (
                value.dtype != np.float32
                or not value.flags.c_contiguous
                or not np.isfinite(value).all()
            ):
                raise FloatingPointError(f"native archive field is invalid: {name}")
            native_config[name] = value
    exported_config = export_native_initializer(wrapper)
    if set(native_config) != set(exported_config) or any(
        isinstance(value, np.ndarray) and not np.array_equal(value, native_config[name])
        for name, value in exported_config.items()
    ):
        raise ValueError("native payload differs from the final learned checkpoint")

    # Check the fitted, concrete NPZ through the serving C++ callback before a panel export.
    from wnn.data import SequenceFile

    sample = contract["sample"]
    profile_sequence = SequenceFile(data_root / "datasets" / f"{sample['source']}.parquet").read(
        int(sample["row_group"]), targets=False, mask=False
    )
    if (
        profile_sequence.seq_ix != sample["seq_ix"]
        or profile_sequence.features.shape != (ROWS, 112)
        or checkpoint.get("readout") is not None
    ):
        raise ValueError(
            "post-fit native parity source or output path differs from the pinned profile"
        )
    python_profile = trainer.predict(
        wrapper,
        torch.from_numpy(np.asarray(profile_sequence.features).copy())[None],
        CHUNK_ROWS,
        device,
    )
    native_weights = {
        name: np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        for name, tensor in wrapper.backbone.state_dict().items()
    }
    native = __import__("wnn_native")
    native_model = native.EnsemblePredictor(
        [native_weights],
        [1.0],
        fast_activations=False,
        precision="f32",
        state_initializers=[native_config],
    )

    def native_predict(seq_ix: int) -> np.ndarray:
        output = np.empty((ROWS, 2), dtype=np.float32)
        for row_index, feature_row in enumerate(profile_sequence.features):
            value = native_model.predict(
                SimpleNamespace(
                    seq_ix=seq_ix,
                    state=feature_row,
                    need_prediction=True,
                )
            )
            if value is None:
                raise RuntimeError("native initializer callback omitted a prediction")
            output[row_index] = np.asarray(value, dtype=np.float32)
        return output

    native_started = time.perf_counter()
    native_profile = native_predict(int(profile_sequence.seq_ix))
    native_replay = native_predict(int(profile_sequence.seq_ix) + 1)
    native_seconds = time.perf_counter() - native_started
    expected_profile = python_profile[0]
    np.testing.assert_allclose(native_profile, expected_profile, rtol=3e-4, atol=3e-5)
    np.testing.assert_array_equal(native_replay, native_profile)
    native_delta = native_profile - expected_profile
    if not np.isfinite(native_model.hidden_states[0]).all():
        raise FloatingPointError("native fitted initializer ended with a nonfinite hidden state")

    with np.load(data_root / "cache/valid_targets.npz", allow_pickle=False) as targets:
        groups = np.asarray(targets["row_group"], dtype=np.int64).tolist()
    if len(groups) != 1873 or len(set(groups)) != len(groups):
        raise ValueError("full valid target cache row order is not the fixed 1,873-group F0 index")
    all_valid = set(split.valid_groups) | set(split.evaluation_groups)
    if set(groups) != all_valid:
        raise ValueError("full valid prediction membership differs from the frozen F0 split")
    seq_ix_by_group = {}
    split_manifest, _ = bounded_json(split_path, 2_000_000)
    for item in split_manifest["sequences"]["valid"]:
        seq_ix_by_group[int(item["row_group"])] = int(item["seq_ix"])
    dataset = FeatureSequences(groups, seq_ix_by_group, data_root / "datasets/valid.parquet")

    output_path = data_root / "cache" / f"pred_{args.run}_valid.npy"
    sidecar_path = output_path.with_suffix(".json")
    temporary_path = output_path.with_name(output_path.name + ".partial")
    if output_path.exists() or sidecar_path.exists() or temporary_path.exists():
        raise FileExistsError(f"preserving existing initializer prediction attempt: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = np.lib.format.open_memmap(
        temporary_path, mode="w+", dtype=np.float32, shape=(len(groups), ROWS, 2)
    )
    row = 0
    with torch.inference_mode():
        for index in range(len(dataset)):
            raw = dataset[index][None]
            prediction = trainer.predict(wrapper, raw, CHUNK_ROWS, device)
            if prediction.shape != (1, ROWS, 2) or not np.isfinite(prediction).all():
                raise FloatingPointError(
                    f"invalid full-sequence prediction at row group {groups[index]}"
                )
            predictions[row] = prediction[0]
            row += 1
            del raw, prediction
    predictions.flush()
    del predictions
    if row != len(groups):
        raise RuntimeError(f"wrote {row} of {len(groups)} full valid sequences")
    cuda_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    cuda_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    observed_vram = observed_vram_bytes()
    if cuda_peak_reserved > CUDA_ALLOCATOR_CAP_BYTES:
        raise RuntimeError("subset prediction exceeded the explicit 3.5 GB allocator cap")
    if observed_vram >= OBSERVED_VRAM_LIMIT_BYTES:
        raise RuntimeError("subset prediction reached the fixed observed-VRAM ceiling")
    with temporary_path.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary_path.replace(output_path)
    prediction_sha256 = sha256_file(output_path)
    sidecar = {
        "run": args.run,
        "split": "valid",
        "epoch": 0,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_sha256": prediction_sha256,
        "split_manifest_sha256": split.sha256,
        "row_groups": groups,
        "inference_precision": "FP32 Torch InitializedGRU; C2000; autocast and TF32 disabled",
        "initializer_mode": initializer_record["mode"],
        "initializer_checkpoint_sha256": fit_receipt["initializer_checkpoint_sha256"],
        "native_archive_sha256": fit_receipt["native_archive_sha256"],
        "native_profile_gate_sha256": provenance["native_profile_gate_sha256"],
        "subset_manifest_sha256": SUBSET_SHA256,
        "fit_order_sha256": subset["fit_order_sha256"],
        "quality_claim": None,
    }
    raw_sidecar = (json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with sidecar_path.open("xb") as stream:
        stream.write(raw_sidecar)
        stream.flush()
        os.fsync(stream.fileno())
    result = {
        "format": "wnn.initializer-only-full-valid-prediction",
        "version": 1,
        "run": args.run,
        "result": "PASS",
        "quality_claim": None,
        "prediction_path": str(output_path),
        "prediction_sha256": prediction_sha256,
        "prediction_sidecar_sha256": hashlib.sha256(raw_sidecar).hexdigest(),
        "row_groups": len(groups),
        "sequence_rows": ROWS,
        "split_manifest_sha256": split.sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "initializer_checkpoint_sha256": fit_receipt["initializer_checkpoint_sha256"],
        "native_archive_sha256": fit_receipt["native_archive_sha256"],
        "native_profile_sequence": {
            "source": sample["source"],
            "row_group": sample["row_group"],
            "seq_ix": sample["seq_ix"],
            "rows": ROWS,
            "parity_atol": 3e-5,
            "parity_rtol": 3e-4,
            "max_abs_prediction_difference": float(np.abs(native_delta).max()),
            "rms_prediction_difference": float(
                np.sqrt(np.mean(native_delta.astype(np.float64) ** 2))
            ),
            "reset_replay_exact": True,
            "elapsed_seconds_two_passes": native_seconds,
        },
        "inference_precision": sidecar["inference_precision"],
        "batch_size": 1,
        "chunk_rows": CHUNK_ROWS,
        "host_memory_limit_bytes": host_limit,
        "cuda_peak_allocated_bytes": cuda_peak_allocated,
        "cuda_peak_reserved_bytes": cuda_peak_reserved,
        "observed_vram_bytes_at_completion": observed_vram,
        "elapsed_seconds": time.perf_counter() - started,
        "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    receipt_path = run_root / "prediction_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"preserving existing prediction receipt: {receipt_path}")
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run": args.run, "result": "PASS", "prediction_sha256": prediction_sha256}))


if __name__ == "__main__":
    main()

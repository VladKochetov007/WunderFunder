"""Fit one learned causal initializer on the frozen complete-component subset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROFILE_CONTRACT_SHA256 = "08b943d63913e5ceebd845f1419feed65af36b375589304ed85fb50ed03fa3e7"
PROFILE_RESULT_SHA256 = "42e8ad8aa838f0570d2a26ba2dabb6b8d93fb00f5c321a6c4d6f28b327e74711"
PROFILE_ATTEMPT_ID = "student-initializer-profile-f0s3-h256l2-b1-c2000-fp32-r3"
PROFILE_ATTEMPT_SHA256 = "97d69a1da3435f5cdbb749ddd3729171143465180320fa4e6a6035a7c3548a52"
PROFILE_ADMISSION_SHA256 = "ae54dea5e8cdb728db29932897e58f730205036ade82831928955df826e75dc6"
PROFILE_SOURCE_COMMIT = "2800d26d200c70d858a8bbe50d9fdabd90190d5f"
SUBSET_SHA256 = "05f9e671783ad741de4514ad4a566d988280380f3d171e6baee56ceff337893c"
PARENT_CHECKPOINT_SHA256 = "1baa583573089da3eec02bee9256c32e3caeae59e15f77c7475eea0717f006e6"
PARENT_VALIDATION_SHA256 = "1042e31a73456edd88f40a8bd03b11c91ff346907ace16c50abc10b58fecdf43"
EXPECTED_MODES = ("constant", "first_row", "prefix")
ROWS = 20_000
CHUNK_ROWS = 2_000
WARMUP_ROWS = 99
CUDA_ALLOCATOR_CAP_BYTES = 3_500_000_000
OBSERVED_VRAM_LIMIT_BYTES = 5_000_000_000
HOST_MEMORY_LIMIT_BYTES = 5_000_000_000
MAX_SMALL_JSON_BYTES = 8_000_000
MAX_SUBSET_BYTES = 5_000_000


def sha256_file(path: Path) -> str:
    """Hash a file in fixed-size reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_json(path: Path, maximum: int = MAX_SMALL_JSON_BYTES) -> tuple[dict[str, Any], bytes]:
    """Read a JSON object only after enforcing its on-disk bound."""
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise ValueError(f"JSON input exceeds its {maximum}-byte bound: {path}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"JSON input changed while being read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, raw


def load_pinned_module(name: str, path: Path) -> Any:
    """Load the immutable trainer source selected by the reviewed contract."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load trainer module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def current_cgroup_bytes(counter: str) -> int:
    """Read one finite cgroup-v2 memory counter for this worker."""
    memberships = [
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
        if line.startswith("0::")
    ]
    if len(memberships) != 1:
        raise RuntimeError("could not identify the worker's cgroup-v2 memory limit")
    root = Path("/sys/fs/cgroup").resolve()
    cgroup = (root / memberships[0].lstrip("/")).resolve()
    if not cgroup.is_relative_to(root):
        raise RuntimeError("worker cgroup path escaped the cgroup-v2 mount")
    value = (cgroup / counter).read_text(encoding="ascii").strip()
    if not value.isdigit():
        raise RuntimeError(f"worker cgroup counter is not finite: {counter}")
    return int(value)


def check_worker_envelope() -> None:
    """Reject an unbounded or differently capped execution environment."""
    if os.environ.get("WNN_GPU_MEMORY_BYTES") != str(CUDA_ALLOCATOR_CAP_BYTES):
        raise RuntimeError("fit requires the explicit 3.5 GB CUDA allocator cap")
    limit = current_cgroup_bytes("memory.max")
    if limit > HOST_MEMORY_LIMIT_BYTES or HOST_MEMORY_LIMIT_BYTES - limit >= 4096 or limit <= 0:
        raise RuntimeError("initializer fit cgroup differs from its 5 GB admitted host cap")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU fit was invoked without CUDA")


def observed_vram_bytes() -> int:
    """Read observed device memory independently from PyTorch allocator counters."""
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


def check_source_pins(profile_contract: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Check profile, trainer and model sources before loading executable code."""
    profile_root = Path(profile_contract["source_root"]).resolve(strict=True)
    trainer_root = Path(profile_contract["training_source"]["source_root"]).resolve(strict=True)
    data_root = Path(profile_contract["data_root"]).resolve(strict=True)
    for relative, expected in profile_contract["source_sha256"].items():
        if sha256_file(profile_root / relative) != expected:
            raise ValueError(f"reviewed initializer-profile source changed: {relative}")
    for relative, expected in profile_contract["training_source"]["source_sha256"].items():
        if sha256_file(trainer_root / relative) != expected:
            raise ValueError(f"reviewed teacher/student source changed: {relative}")
    cache_packet = profile_contract["cache_packet"]
    cache_source_root = Path(cache_packet["source_root"]).resolve(strict=True)
    for relative, expected in cache_packet["source_manifest"].items():
        if sha256_file(cache_source_root / relative) != expected:
            raise ValueError(f"reviewed cache/order helper source changed: {relative}")
    return profile_root, trainer_root, data_root


def check_native_gate(
    gate_path: Path, profile_contract: dict[str, Any], profile_result_path: Path
) -> tuple[dict[str, Any], str]:
    """Require the reviewed concrete-profile C++ gate before fitting parameters."""
    gate, raw = bounded_json(gate_path)
    profile_result, profile_raw = bounded_json(profile_result_path)
    if (
        gate.get("format") != "wnn.initializer-native-profile-gate"
        or gate.get("version") != 2
        or gate.get("result") != "PASS"
        or gate.get("quality_claim") is not None
        or gate.get("profile_contract_sha256") != PROFILE_CONTRACT_SHA256
        or gate.get("profile_result_sha256") != hashlib.sha256(profile_raw).hexdigest()
        or gate.get("profile_result_sha256") != PROFILE_RESULT_SHA256
        or gate.get("profile_attempt_id") != PROFILE_ATTEMPT_ID
        or gate.get("profile_attempt_receipt_sha256") != PROFILE_ATTEMPT_SHA256
        or gate.get("profile_admission_receipt_sha256") != PROFILE_ADMISSION_SHA256
        or gate.get("profile_source_commit") != PROFILE_SOURCE_COMMIT
        or gate.get("control_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or profile_result.get("contract_sha256") != PROFILE_CONTRACT_SHA256
    ):
        raise ValueError("actual R3 profile has not passed its source-bound native C++ gate")
    if gate.get("attempt_id") != PROFILE_ATTEMPT_ID:
        raise ValueError("native gate is not bound to the root-admitted R3 profile attempt")
    expected_sequence = profile_contract["sample"]
    if gate.get("sequence") != {
        "source": expected_sequence["source"],
        "row_group": expected_sequence["row_group"],
        "seq_ix": expected_sequence["seq_ix"],
        "rows": ROWS,
    }:
        raise ValueError("native C++ gate sequence differs from the actual profile member")
    modes = gate.get("arms")
    if not isinstance(modes, list) or [item.get("mode") for item in modes] != list(EXPECTED_MODES):
        raise ValueError("native C++ gate did not pass all three learned initializer modes")
    for item in modes:
        if (
            item.get("sequence_rows") != ROWS
            or item.get("native_prediction_rows") != ROWS
            or item.get("reset_replay_exact") is not True
            or item.get("native_hidden_finite") is not True
            or item.get("python_native_atol") != 3e-5
            or item.get("python_native_rtol") != 3e-4
            or item.get("quality_claim") is not None
        ):
            raise ValueError(f"native C++ gate is incomplete for {item.get('mode')}")
    return gate, hashlib.sha256(raw).hexdigest()


def load_sources_and_subset(
    profile_contract_path: Path,
    profile_result_path: Path,
    gate_path: Path,
    subset_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any, Path, Path, Path, dict[str, Any]]:
    """Bind the subset, parent, clean cache, propensity and source-qualified order."""
    profile_contract, contract_raw = bounded_json(profile_contract_path)
    if hashlib.sha256(contract_raw).hexdigest() != PROFILE_CONTRACT_SHA256:
        raise ValueError("initializer profile contract hash differs from the reviewed R3 packet")
    if sha256_file(subset_path) != SUBSET_SHA256:
        raise ValueError("complete-component subset differs from the frozen seed-3 manifest")
    subset, _ = bounded_json(subset_path, MAX_SUBSET_BYTES)
    profile_result, profile_result_raw = bounded_json(profile_result_path)
    if (
        profile_result.get("contract_sha256") != PROFILE_CONTRACT_SHA256
        or hashlib.sha256(profile_result_raw).hexdigest() != PROFILE_RESULT_SHA256
        or profile_result.get("quality_claim") is not None
        or profile_result.get("execution_device", {}).get("resolved") != "cuda:0"
    ):
        raise ValueError("resource profile result is absent, stale, mislabelled or not R3")
    gate, gate_sha256 = check_native_gate(gate_path, profile_contract, profile_result_path)
    profile_root, trainer_root, data_root = check_source_pins(profile_contract)

    sys.path.insert(0, str(trainer_root / "src"))
    sys.path.insert(0, str(profile_root / "src"))
    trainer = load_pinned_module("train_rnn", trainer_root / "scripts/train_rnn.py")
    trainer.ROOT = data_root
    from wnn.distillation_cache import validate_distillation_cache
    from wnn.splits import load_campaign_split

    split_path = data_root / profile_contract["split_path"]
    if sha256_file(split_path) != profile_contract["split_sha256"]:
        raise ValueError("source-qualified F0 split changed")
    split = load_campaign_split(split_path, data_root)
    if split.fold != 0 or len(split.evaluation_groups) != 468:
        raise ValueError("expected the fixed 468-group F0 evaluation panel")
    fit_subset = trainer.load_component_fit_subset(
        subset_path, split, expected_experiment_id=subset["experiment_id"]
    )

    split_manifest, _ = bounded_json(split_path, 2_000_000)
    split_sequence_identities = {
        (source, int(item["row_group"])): (int(item["seq_ix"]), int(item["component"]))
        for source in ("train", "valid")
        for item in split_manifest["sequences"][source]
    }
    identities = {
        (source, int(group)): (int(seq_ix), int(component))
        for source, group, seq_ix, component in subset["selected_source_group_seq_ix_component"]
    }
    expected_identities = {key: split_sequence_identities[key] for key in identities}
    if identities != expected_identities or len(identities) != len(subset["fit_order"]):
        raise ValueError("subset source/group/seq_ix/component identities differ from the split")
    fit_order = [tuple(item) for item in subset["fit_order"]]
    if (
        len(fit_order) != len(set(fit_order))
        or set(fit_order) != set(identities)
        or hashlib.sha256(
            json.dumps(subset["fit_order"], separators=(",", ":"), ensure_ascii=True).encode(
                "ascii"
            )
        ).hexdigest()
        != subset["fit_order_sha256"]
    ):
        raise ValueError(
            "subset update order is not the exact unique frozen source-qualified order"
        )

    cache_root = data_root / profile_contract["cache"]["root"]
    cache_validation = validate_distillation_cache(
        cache_root, data_root, split, "homogeneous", require_families=False
    )
    if (
        cache_validation["metadata_sha256"] != profile_contract["cache"]["metadata_sha256"]
        or cache_validation["verified_files"] != profile_contract["cache"]["target_files"]
    ):
        raise ValueError("strict clean homogeneous cache bytes differ from the reviewed contract")

    # The unchanged producer validator checks target payloads; bind its teacher/propensity inputs too.
    metadata = cache_validation["metadata"]
    if metadata.get("target_contract") != profile_contract["cache"]["target_contract"]:
        raise ValueError("clean target blend formula differs from the initializer recipe")
    actual_sources = {item.get("run"): item for item in metadata.get("sources", [])}
    expected_sources = {
        item["run"]: item for item in profile_contract["cache_ancestry"]["teacher_sources"]
    }
    if set(actual_sources) != set(expected_sources):
        raise ValueError("clean target cache teacher roster changed")
    for run, expected in expected_sources.items():
        actual = actual_sources[run]
        if (
            actual.get("checkpoint_sha256") != expected["checkpoint_sha256"]
            or actual.get("provenance_sha256") != expected["provenance_sha256"]
            or actual.get("epoch") != expected["epoch"]
            or actual.get("family") != expected["family"]
            or float(actual.get("coefficient", -1)) != float(expected["coefficient"])
        ):
            raise ValueError(f"clean teacher cache ancestry changed: {run}")
        if (
            sha256_file(data_root / expected["checkpoint"]) != expected["checkpoint_sha256"]
            or sha256_file(data_root / expected["provenance"]) != expected["provenance_sha256"]
        ):
            raise ValueError(f"clean teacher checkpoint/provenance bytes changed: {run}")

    propensity = profile_contract["cache_ancestry"]["propensity"]
    propensity_path = data_root / f"artifacts/runs/{propensity['run']}/model.pt"
    propensity_provenance = data_root / f"artifacts/runs/{propensity['run']}/provenance.json"
    if (
        sha256_file(propensity_path) != propensity["checkpoint_sha256"]
        or sha256_file(propensity_provenance) != propensity["provenance_sha256"]
    ):
        raise ValueError("pinned F0 propensity checkpoint or provenance changed")
    for source, expected in propensity["caches"].items():
        prop_path = data_root / expected["path"]
        sidecar_path = prop_path.with_suffix(".json")
        sidecar, sidecar_raw = bounded_json(sidecar_path, 64_000)
        if (
            sha256_file(prop_path) != expected["sha256"]
            or prop_path.stat().st_size != expected["bytes"]
            or hashlib.sha256(sidecar_raw).hexdigest() != expected["sidecar_sha256"]
            or sidecar.get("model") != propensity["run"]
            or sidecar.get("model_sha256") != propensity["checkpoint_sha256"]
            or sidecar.get("split") != source
        ):
            raise ValueError(f"pinned propensity cache or sidecar changed: {source}")

    normalization_source = profile_contract["normalization"]["source"]
    norm_run = data_root / "artifacts/runs" / normalization_source["run"]
    if (
        sha256_file(norm_run / normalization_source["checkpoint_file"])
        != normalization_source["checkpoint_sha256"]
        or sha256_file(norm_run / "provenance.json") != normalization_source["provenance_sha256"]
        or normalization_source.get("learned_weights_loaded") is not False
        or normalization_source.get("loaded_fields_only") != ["mean", "std"]
    ):
        raise ValueError("common 211-feature normalization ancestry changed")

    control = profile_contract["control"]
    parent_path = data_root / control["checkpoint"]
    validation_path = data_root / control["fit_validation"]
    if (
        sha256_file(parent_path) != PARENT_CHECKPOINT_SHA256
        or sha256_file(validation_path) != PARENT_VALIDATION_SHA256
    ):
        raise ValueError("H256×2 seed-3 CP0 or its strict fit-validation receipt changed")
    if (
        profile_result.get("observed_vram_bytes_at_completion", OBSERVED_VRAM_LIMIT_BYTES)
        >= OBSERVED_VRAM_LIMIT_BYTES
    ):
        raise ValueError("initializer B1 resource profile reached the observed-VRAM ceiling")
    return (
        profile_contract,
        profile_result,
        gate,
        trainer,
        split,
        profile_root,
        trainer_root,
        data_root,
        {
            "subset": subset,
            "fit_subset": fit_subset,
            "cache_validation": cache_validation,
            "parent_path": parent_path,
            "gate_sha256": gate_sha256,
            "profile_result_sha256": hashlib.sha256(profile_result_raw).hexdigest(),
        },
    )


def _finite_state(model: torch.nn.Module) -> bool:
    """Return whether every state tensor is finite."""
    return all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically without replacing a prior terminal receipt."""
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"preserving an existing initializer output: {path}")
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def fit_one_mode(
    mode: str,
    profile_contract: dict[str, Any],
    profile_result: dict[str, Any],
    gate: dict[str, Any],
    trainer: Any,
    split: Any,
    profile_root: Path,
    trainer_root: Path,
    data_root: Path,
    binding: dict[str, Any],
    attempt_number: int,
) -> dict[str, Any]:
    """Train one learned initializer with 258 matched complete-sequence updates."""
    check_worker_envelope()
    if mode not in EXPECTED_MODES:
        raise ValueError(f"unsupported initializer mode: {mode}")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    device = torch.device("cuda")
    trainer.HEURISTICS = False
    trainer.RAW_ONLY = False
    trainer.ONLINE_NORM = False
    trainer.RESEARCH_FEATURES = "none"
    trainer.PRIVILEGED_MODE = "none"
    trainer.INPUT_PROPENSITY = None

    from wnn.data import SequenceFile
    from wnn.initializer_training import initializer_sequence_loss
    from wnn.propensity import load_propensity
    from wnn.state_initialization import InitializedGRU, export_native_initializer

    subset = binding["subset"]
    fit_order: list[list[str | int]] = subset["fit_order"]
    cache_metadata = binding["cache_validation"]["metadata"]
    cache_root = data_root / profile_contract["cache"]["root"]
    target_arrays: dict[str, np.ndarray] = {}
    target_positions: dict[str, dict[int, int]] = {}
    for source in ("train", "valid"):
        key = f"main_homogeneous_{source}"
        file_record = cache_metadata["files"][key]
        array = np.load(cache_root / file_record["path"], mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or array.shape[1:] != (ROWS, 2):
            raise ValueError(f"clean target array schema changed: {key}")
        target_arrays[source] = array
        target_positions[source] = {
            int(group): index for index, group in enumerate(file_record["row_groups"])
        }

    propensity_record = profile_contract["cache_ancestry"]["propensity"]
    propensity_model = load_propensity(
        data_root / f"artifacts/runs/{propensity_record['run']}/model.pt", device
    )
    propensity_arrays = {
        source: np.load(
            data_root / propensity_record["caches"][source]["path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        for source in ("train", "valid")
    }
    sequence_files = {
        source: SequenceFile(data_root / "datasets" / f"{source}.parquet")
        for source in ("train", "valid")
    }
    sequences = {
        (source, int(group)): (int(seq_ix), int(component))
        for source, group, seq_ix, component in subset["selected_source_group_seq_ix_component"]
    }
    parent = torch.load(binding["parent_path"], map_location="cpu", weights_only=False)
    if parent.get("epoch") != 0 or not isinstance(parent.get("model"), dict):
        raise ValueError("initializer parent is not the fixed H256×2 epoch-0 checkpoint")
    if parent.get("readout") is not None:
        raise ValueError("initializer-only recipe expects no external readout tables")
    model_args = {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "readout_activation": "gelu",
        "numeric_map": "none",
        "ple_intervals": 8,
    }
    backbone = trainer.make_model(model_args, 211, 2)
    backbone.load_state_dict(parent["model"], strict=True)
    base_state = {
        name: value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        for name, value in backbone.state_dict().items()
    }
    for name, expected in profile_contract["normalization"]["hashes"].items():
        digest = hashlib.sha256(
            base_state[name].numpy().astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest()
        if digest != expected:
            raise ValueError(f"common parent normalization buffer differs: {name}")
    del backbone

    backbone = trainer.make_model(model_args, 211, 2).to(device)
    backbone.load_state_dict(parent["model"], strict=True)
    backbone.eval()
    backbone.requires_grad_(False)
    model = InitializedGRU(backbone, mode, residual_bound=1.0).to(device)
    model.train()
    model.backbone.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise ValueError("initializer-only arm did not freeze the full recurrent backbone")
    before_backbone = {
        name: value.detach().cpu().clone() for name, value in backbone.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        fused=False,
    )
    args = argparse.Namespace(
        mask_weight=False,
        prop_power=2.0,
        prop_eps=0.03,
        class_weight="",
        valid_true_mask=0.03,
        sync_free=True,
        since_weight=0.0,
        episode_weight=0.0,
        episode_floor=0.5,
        since_floor=0.25,
        prop_cache=True,
    )
    started = time.perf_counter()
    history: list[dict[str, float | int | bool]] = []
    terminal_inputs: torch.Tensor | None = None
    files = {}
    for source in ("train", "valid"):
        files[source] = {
            int(row_group): (int(seq_ix), int(component))
            for row_group, seq_ix, component in (
                (item[1], item[2], item[3])
                for item in subset["selected_source_group_seq_ix_component"]
                if item[0] == source
            )
        }
    run_name = f"student_init_subf0_s3_{mode}_r{attempt_number}"
    run_root = data_root / "artifacts/runs" / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    progress_path = run_root / "fit_progress.jsonl"
    residual_bound = float(profile_contract["residual_bound"])
    if residual_bound != 1.0:
        raise ValueError("initializer subset recipe requires the profiled residual bound of 1.0")
    if observed_vram_bytes() >= OBSERVED_VRAM_LIMIT_BYTES:
        raise RuntimeError("observed VRAM reached the fixed 5 GB ceiling before fitting")
    torch.cuda.reset_peak_memory_stats(device)

    for update, pair in enumerate(fit_order, start=1):
        source, raw_group = str(pair[0]), int(pair[1])
        expected_seq_ix, component = sequences[(source, raw_group)]
        sequence = sequence_files[source].read(raw_group)
        if (
            int(sequence.seq_ix) != expected_seq_ix
            or sequence.features.shape != (ROWS, 112)
            or sequence.targets.shape != (ROWS, 2)
            or sequence.scored.shape != (ROWS,)
            or files[source][raw_group] != (expected_seq_ix, component)
        ):
            raise ValueError(f"source-qualified complete sequence changed: {(source, raw_group)}")
        cache_position = target_positions[source].get(raw_group)
        if cache_position is None:
            raise ValueError(
                f"clean teacher target lacks source-qualified member: {(source, raw_group)}"
            )
        raw = torch.from_numpy(np.asarray(sequence.features).copy())[None].to(device)
        labels = torch.from_numpy(np.asarray(sequence.targets).copy())[None].to(device)
        teacher = torch.from_numpy(
            np.asarray(target_arrays[source][cache_position], dtype=np.float32).copy()
        )[None].to(device)
        scored = torch.from_numpy(np.asarray(sequence.scored).copy())[None].to(device)
        prop = propensity_arrays[source][raw_group]
        if prop.shape != (ROWS,) or not np.isfinite(prop).all():
            raise ValueError(f"propensity cache row is malformed: {(source, raw_group)}")
        prop_cached = torch.from_numpy(np.asarray(prop, dtype=np.float32).copy())[None].to(device)
        inputs = trainer.build_inputs(raw)
        if tuple(inputs.shape) != (1, ROWS, 211):
            raise ValueError("production feature builder did not return 211 causal features")
        target = trainer.blended_target(labels, teacher, None, slice(0, ROWS), 0.9, 0, 0.0)
        valid = torch.arange(ROWS, device=device)[None] >= WARMUP_ROWS
        with torch.no_grad():
            row_weight = trainer.main_row_weights(
                inputs, labels, valid, scored, propensity_model, prop_cached, args, raw
            )
            weights = labels.float().clamp(-2.0, 2.0).abs() * row_weight[..., None].float()
            weights[:, :WARMUP_ROWS] = 0
        if (
            not torch.isfinite(inputs).all()
            or not torch.isfinite(target).all()
            or not torch.isfinite(weights).all()
            or torch.any(weights < 0)
            or not float(weights.sum()) > 0
        ):
            raise FloatingPointError(f"invalid full-sequence supervision at update {update}")

        optimizer.zero_grad(set_to_none=True)
        loss = initializer_sequence_loss(
            model,
            inputs,
            target,
            weights,
            chunk_rows=CHUNK_ROWS,
            checkpoint_chunks=True,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite complete-sequence loss at update {update}")
        loss.backward()
        if any(parameter.grad is None for parameter in trainable):
            raise FloatingPointError(f"initializer gradient is absent at update {update}")
        gradients_finite = all(
            bool(torch.isfinite(parameter.grad).all()) for parameter in trainable
        )
        if not gradients_finite:
            raise FloatingPointError(f"nonfinite initializer gradient at update {update}")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"nonfinite clipped-gradient norm at update {update}")
        nonzero_gradient = any(bool(torch.count_nonzero(parameter.grad)) for parameter in trainable)
        parameters_before = [parameter.detach().clone() for parameter in trainable]
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in trainable):
            raise FloatingPointError(f"nonfinite model tensor after update {update}")
        parameter_update_observed = any(
            not torch.equal(before, parameter.detach())
            for before, parameter in zip(parameters_before, trainable, strict=True)
        )
        record: dict[str, float | int | bool] = {
            "update": update,
            "source": source,
            "row_group": raw_group,
            "seq_ix": expected_seq_ix,
            "component": component,
            "loss": float(loss.detach().cpu()),
            "global_weight_mass": float(weights.sum().detach().cpu()),
            "loss_finite": True,
            "gradient_finite": True,
            "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
            "nonzero_initializer_gradient": nonzero_gradient,
            "parameter_update_observed": parameter_update_observed,
            "optimizer_update_completed": True,
        }
        history.append(record)
        if update == len(fit_order):
            terminal_inputs = inputs.detach().clone()
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        del raw, labels, teacher, scored, prop_cached, inputs, target, weights, row_weight, loss
        if update % 25 == 0 and observed_vram_bytes() >= OBSERVED_VRAM_LIMIT_BYTES:
            raise RuntimeError("observed VRAM reached the fixed 5 GB ceiling during fitting")

    backbone_unchanged = all(
        torch.equal(before_backbone[name], value.detach().cpu())
        for name, value in backbone.state_dict().items()
    )
    if not backbone_unchanged:
        raise RuntimeError("initializer optimizer modified the frozen H256×2 backbone")
    if not _finite_state(model):
        raise FloatingPointError("final initializer or frozen backbone has a nonfinite tensor")
    if len(history) != len(fit_order) or any(
        item["optimizer_update_completed"] is not True for item in history
    ):
        raise RuntimeError("initializer update count differs from the fixed sample order")
    if not all(bool(item["parameter_update_observed"]) for item in history):
        raise RuntimeError("at least one complete-sequence update changed no initializer parameter")
    nonzero_gradient_updates = sum(bool(item["nonzero_initializer_gradient"]) for item in history)
    if nonzero_gradient_updates == 0:
        raise RuntimeError("no complete-sequence update produced a nonzero initializer gradient")
    if terminal_inputs is None:
        raise RuntimeError("initializer fit ended without a source-qualified reload sequence")

    native_config = export_native_initializer(model)
    native_arrays = {
        name: value for name, value in native_config.items() if isinstance(value, np.ndarray)
    }
    if any(
        value.dtype != np.float32 or not value.flags.c_contiguous or not np.isfinite(value).all()
        for value in native_arrays.values()
    ):
        raise FloatingPointError("native initializer export is not finite contiguous FP32")
    checkpoint_path = run_root / "initializer_ep0.pt"
    torch.save(
        {
            "format": "wnn.initializer-only-subset-checkpoint",
            "version": 1,
            "mode": mode,
            "epoch": 0,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "subset_manifest_sha256": SUBSET_SHA256,
            "fit_order_sha256": subset["fit_order_sha256"],
            "normalization_sha256": profile_contract["normalization"]["hashes"],
            "initializer_state_dict": {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
                if not name.startswith("backbone.")
            },
        },
        checkpoint_path,
    )
    native_path = run_root / "initializer_native_ep0.npz"
    np.savez(native_path, **native_arrays)
    reloaded_backbone = trainer.make_model(model_args, 211, 2).to(device)
    reloaded_backbone.load_state_dict(parent["model"], strict=True)
    reloaded_backbone.requires_grad_(False)
    reloaded = InitializedGRU(reloaded_backbone, mode, residual_bound=residual_bound).to(device)
    saved_initializer = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded_state = reloaded.state_dict()
    for name, value in saved_initializer["initializer_state_dict"].items():
        if name not in reloaded_state or name.startswith("backbone."):
            raise ValueError(f"unsupported initializer checkpoint field: {name}")
        reloaded_state[name] = value
    reloaded.load_state_dict(reloaded_state, strict=True)
    reloaded.eval()
    with torch.inference_mode():
        original_prediction, _ = model(terminal_inputs, None)
        reloaded_prediction, _ = reloaded(terminal_inputs, None)
    torch.testing.assert_close(original_prediction, reloaded_prediction, rtol=0, atol=0)
    native_after_reload = export_native_initializer(reloaded)
    if set(native_after_reload) != set(native_config) or any(
        isinstance(value, np.ndarray) and not np.array_equal(value, native_config[name])
        for name, value in native_after_reload.items()
    ):
        raise RuntimeError("native initializer export changed after terminal checkpoint reload")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    native_sha256 = sha256_file(native_path)
    fit_groups = subset["selected_groups"]
    run_args = {
        "name": run_name,
        "seed": 3,
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "readout_activation": "gelu",
        "numeric_map": "none",
        "ple_intervals": 8,
        "valid_fold": 0,
        "split_manifest_sha256": split.sha256,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 0,
        "no_eval_during_fit": True,
        "fit_subset_manifest_sha256": SUBSET_SHA256,
        "initializer_only": True,
        "initializer_mode": mode,
    }
    fit_seconds = time.perf_counter() - started
    cuda_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    cuda_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    observed_vram = observed_vram_bytes()
    if cuda_peak_reserved > CUDA_ALLOCATOR_CAP_BYTES:
        raise RuntimeError("PyTorch CUDA reservation exceeded the explicit 3.5 GB cap")
    if observed_vram >= OBSERVED_VRAM_LIMIT_BYTES:
        raise RuntimeError("observed VRAM reached the fixed 5 GB ceiling at fit completion")
    source_sha256 = {
        "scripts/fit_initializer_subset_r2.py": sha256_file(Path(__file__).resolve()),
        "scripts/train_rnn.py": sha256_file(trainer_root / "scripts/train_rnn.py"),
        "src/wnn/initializer_training.py": sha256_file(
            profile_root / "src/wnn/initializer_training.py"
        ),
        "src/wnn/state_initialization.py": sha256_file(
            profile_root / "src/wnn/state_initialization.py"
        ),
    }
    provenance = {
        "args": run_args,
        "fit_row_groups": fit_groups,
        "fit_component_ids": subset["selected_components"],
        "fit_subset_manifest_sha256": SUBSET_SHA256,
        "fit_order_sha256": subset["fit_order_sha256"],
        "selected_source_group_seq_ix_component_sha256": subset[
            "selected_source_group_seq_ix_component_sha256"
        ],
        "split_manifest_sha256": split.sha256,
        "parent_run": profile_contract["control"]["run"],
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_fit_validation_sha256": PARENT_VALIDATION_SHA256,
        "profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "profile_result_sha256": binding["profile_result_sha256"],
        "profile_attempt_id": PROFILE_ATTEMPT_ID,
        "profile_attempt_receipt_sha256": gate["profile_attempt_receipt_sha256"],
        "profile_admission_receipt_sha256": gate["profile_admission_receipt_sha256"],
        "profile_source_commit": PROFILE_SOURCE_COMMIT,
        "native_profile_gate_sha256": binding["gate_sha256"],
        "native_profile_gate_result": gate["result"],
        "clean_target_cache": {
            "metadata_sha256": binding["cache_validation"]["metadata_sha256"],
            "verified_target_files": binding["cache_validation"]["verified_files"],
            "variant": "homogeneous",
            "target_contract": profile_contract["cache"]["target_contract"],
            "teacher_sources": profile_contract["cache_ancestry"]["teacher_sources"],
        },
        "propensity_ancestry": profile_contract["cache_ancestry"]["propensity"],
        "normalization_source": profile_contract["normalization"]["source"],
        "normalization_sha256": profile_contract["normalization"]["hashes"],
        "initializer_checkpoint_sha256": checkpoint_sha256,
        "initializer_native_archive_sha256": native_sha256,
        "initializer_mode": mode,
        "initializer_training": {
            "updated_parameters_only": "initializer parameters; frozen H256×2 backbone",
            "objective": "one global complete-sequence weighted MSE per sequence, ten checkpointed C2000 graph chunks, no detach",
            "target": "0.1*clip(label,-2,2)+0.9*clip(homogeneous teacher,-2,2)",
            "weight": "abs(clip(label,-2,2))*production main_row_weights; p^2+0.03 and true-mask epsilon 0.03; first 99 rows zero",
            "optimizer": "AdamW lr=1e-4 betas=(0.9,0.999) eps=1e-8 wd=0.01 fused=false, global grad clip=1, no scheduler",
            "precision": "FP32; AMP and TF32 disabled",
            "updates_attempted": len(fit_order),
            "updates_completed": sum(bool(item["optimizer_update_completed"]) for item in history),
            "parameter_update_count": sum(
                bool(item["parameter_update_observed"]) for item in history
            ),
            "nonfinite_losses": sum(not bool(item["loss_finite"]) for item in history),
            "nonfinite_gradients": sum(not bool(item["gradient_finite"]) for item in history),
            "nonzero_gradient_updates": sum(
                bool(item["nonzero_initializer_gradient"]) for item in history
            ),
            "order": subset["fit_order"],
            "progress_jsonl_sha256": sha256_file(progress_path),
            "subset_source_group_seq_ix_component_sha256": subset[
                "selected_source_group_seq_ix_component_sha256"
            ],
        },
        "source_sha256": source_sha256,
    }
    # The standard scorer consumes one fixed checkpoint path; the recurrent backbone is unchanged.
    model_checkpoint = {
        "format": "wnn.initializer-only-subset-model",
        "version": 1,
        "epoch": 0,
        "n_in": 211,
        "args": run_args,
        "model": base_state,
        "provenance": provenance,
        "initializer": {
            "mode": mode,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha256,
            "native_archive": native_path.name,
            "native_archive_sha256": native_sha256,
            "residual_bound": residual_bound,
        },
    }
    model_path = run_root / "model_ep0.pt"
    torch.save(model_checkpoint, model_path)
    _write_json(run_root / "provenance.json", provenance)
    receipt = {
        "format": "wnn.initializer-only-subset-fit",
        "version": 1,
        "run": run_name,
        "mode": mode,
        "result": "PASS",
        "quality_claim": None,
        "split_manifest_sha256": split.sha256,
        "subset_manifest_sha256": SUBSET_SHA256,
        "fit_order_sha256": subset["fit_order_sha256"],
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "initializer_checkpoint_sha256": checkpoint_sha256,
        "native_archive_sha256": native_sha256,
        "model_checkpoint_sha256": sha256_file(model_path),
        "updates_attempted": len(fit_order),
        "updates_completed": sum(bool(item["optimizer_update_completed"]) for item in history),
        "parameter_update_count": sum(bool(item["parameter_update_observed"]) for item in history),
        "nonzero_initializer_gradient_updates": nonzero_gradient_updates,
        "nonfinite_losses": sum(not bool(item["loss_finite"]) for item in history),
        "nonfinite_gradients": sum(not bool(item["gradient_finite"]) for item in history),
        "backbone_unchanged": backbone_unchanged,
        "checkpoint_reload_prediction_exact": True,
        "native_export_after_reload_exact": True,
        "normalization_sha256": profile_contract["normalization"]["hashes"],
        "fit_seconds": fit_seconds,
        "cuda_peak_allocated_bytes": cuda_peak_allocated,
        "cuda_peak_reserved_bytes": cuda_peak_reserved,
        "observed_vram_bytes_at_completion": observed_vram,
        "host_memory_peak_bytes": current_cgroup_bytes("memory.peak"),
        "fit_groups": fit_groups,
        "eval_groups_used_during_fit": 0,
        "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "quality_interpretation": "subset-screen checkpoint only; no predictions or quality grade made during fitting",
    }
    _write_json(run_root / "fit_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=EXPECTED_MODES, required=True)
    parser.add_argument("--profile-contract", type=Path, required=True)
    parser.add_argument("--profile-result", type=Path, required=True)
    parser.add_argument("--native-gate", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--attempt-number", type=int, choices=range(1, 100), default=1)
    args = parser.parse_args()
    check_worker_envelope()
    prepared = load_sources_and_subset(
        args.profile_contract, args.profile_result, args.native_gate, args.subset
    )
    receipt = fit_one_mode(args.mode, *prepared, args.attempt_number)
    print(json.dumps({"run": receipt["run"], "result": receipt["result"]}, sort_keys=True))


if __name__ == "__main__":
    main()

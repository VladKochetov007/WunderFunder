"""Independently score the sealed E480 two-seed activation/schedule fits."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
E480_DIR = HERE
E476_DIR = ROOT / "research/campaign_2026-10-05/activation_schedule"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
SEALED_FIT_SPEC = E480_DIR / "e480_activation_schedule_replication_twentyfour_fit_v1.sealed.json"
DEFAULT_RUN = E480_DIR / "runs/twentyfour_v1"
DEFAULT_OUTPUT = E480_DIR / "scores/twentyfour_v1"
DEFAULT_ATTEMPT_RECORD = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e480-activation-schedule-replication-twentyfour-fit-v1.json"
)
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
DATA_PATH = ROOT / "datasets/valid.parquet"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))

import fit_activation_optimizer as e470  # noqa: E402

from wnn.campaign import RUNTIME_FILES, runtime_identity  # noqa: E402
from wnn.campaign_spec import digest as campaign_spec_digest  # noqa: E402
from wnn.campaign_spec import validate_spec  # noqa: E402
from wnn.evaluation import paired_component_moments  # noqa: E402
from wnn.gmetric import (  # noqa: E402
    METRIC_CLIP,
    METRIC_ID,
    _merge_moments,
    _wp_from_moments,
    global_wp,
)
from wnn.nn import RNNModel  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

SEEDS = (271_829, 271_830)
BOOTSTRAP_COUNT = 2_000
BOOTSTRAP_SEED = 271_828
UPDATES = 474
CHUNK_ROWS = 256
EXPECTED_PARAMETERS = 61_058
SAMPLE_STEPS = (0, 46, 236, 473)
EXPECTED_SPEC_NAME = "e480_activation_schedule_replication_twentyfour_fit_v1"
EXPECTED_ATTEMPT_ID = "e480-activation-schedule-replication-twentyfour-fit-v1"
EXPECTED_RUN_ARTIFACTS = (
    "research/campaign_2026-10-05/activation_schedule_replication/runs/twentyfour_v1/fit_manifest.json",
    "research/campaign_2026-10-05/activation_schedule_replication/runs/twentyfour_v1/lr_traces.json",
)
EXPECTED_DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
EXPECTED_COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
EXPECTED_E476_TRACE_RECEIPT_SHA256 = (
    "046f38f67792af73450bbbb958403834d948495e4ba64b263a2a9a2ff935e4ff"
)
EXPECTED_MANIFEST_SOURCES = {
    "research/campaign_2026-10-05/activation_schedule_replication/fit_activation_schedule_replication.py",
    "research/campaign_2026-10-05/activation_schedule_replication/tests.py",
    "research/campaign_2026-10-05/activation_schedule_replication/E480_DESIGN.md",
    "research/campaign_2026-10-05/activation_schedule/fit_activation_schedule.py",
    "research/campaign_2026-10-05/activation_schedule/preflight_cpu_v5/lr_traces.json",
    "research/campaign_2026-10-05/activation_placement/fit_activation_placement.py",
    "research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py",
    "src/wnn/optim.py",
    "src/wnn/nn.py",
}
EXPECTED_MANIFEST_SOURCE_SHA256 = {
    "research/campaign_2026-10-05/activation_schedule_replication/fit_activation_schedule_replication.py": "555a99927f94bd62b08b2f8af06b4a4ba5783d43d10e513377b954661377b4b0",
    "research/campaign_2026-10-05/activation_schedule_replication/tests.py": "511a688f40f782a4b4021d89ca0f1df6b7582155aac87a04fda1cdf627d8d1de",
    "research/campaign_2026-10-05/activation_schedule_replication/E480_DESIGN.md": "3ad2ab9d5721b7e342b2549af81e528fab0a9fa1d2a9fe841d43d4fb978d8811",
    "research/campaign_2026-10-05/activation_schedule/fit_activation_schedule.py": "5acc0efa8295338b39ffc0a0c9412484daec277967d8dc2e161affee1d1786e6",
    "research/campaign_2026-10-05/activation_schedule/preflight_cpu_v5/lr_traces.json": EXPECTED_E476_TRACE_RECEIPT_SHA256,
    "research/campaign_2026-10-05/activation_placement/fit_activation_placement.py": "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068",
    "research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py": "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9",
    "src/wnn/optim.py": "620569a752fb611adf69976f1dbed1b7882e641672c58175005f5d3f819d2991",
    "src/wnn/nn.py": "388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3",
}
ACTIVATIONS = ("base", "sine_embed")
OPTIMIZERS = ("adamw", "muon_adamw")
SCHEDULES = ("constant_peak", "warmcos_47_427", "constant_mean_warmcos")
CELLS = tuple(
    (seed, activation, optimizer, schedule)
    for seed in SEEDS
    for activation in ACTIVATIONS
    for optimizer in OPTIMIZERS
    for schedule in SCHEDULES
)
PAIR_CONTRASTS = (
    tuple(
        (
            f"sine_embed__{optimizer}__{schedule}",
            f"base__{optimizer}__{schedule}",
            "sine_embed_minus_base",
        )
        for optimizer in OPTIMIZERS
        for schedule in SCHEDULES
    )
    + tuple(
        (
            f"{activation}__muon_adamw__{schedule}",
            f"{activation}__adamw__{schedule}",
            "muon_adamw_minus_adamw",
        )
        for activation in ACTIVATIONS
        for schedule in SCHEDULES
    )
    + tuple(
        (
            f"{activation}__{optimizer}__warmcos_47_427",
            f"{activation}__{optimizer}__constant_mean_warmcos",
            "warmcos_minus_constant_mean",
        )
        for activation in ACTIVATIONS
        for optimizer in OPTIMIZERS
    )
    + tuple(
        (
            f"{activation}__{optimizer}__constant_peak",
            f"{activation}__{optimizer}__constant_mean_warmcos",
            "constant_peak_minus_constant_mean",
        )
        for activation in ACTIVATIONS
        for optimizer in OPTIMIZERS
    )
)


class SineEmbeddingRNN(RNNModel):
    """Independent E476-equivalent GELU+sine embedding with unchanged state keys."""

    def encode(self, xn: Tensor) -> Tensor:
        projected = self.inp(xn)
        return F.gelu(projected, approximate="none") + 0.1 * torch.sin(projected)


def sha256_file(path: Path) -> str:
    """Hash a file in bounded-size blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_digest(values: np.ndarray) -> str:
    """Hash values in the registered little-endian float64 representation."""
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def expected_lr_traces() -> dict[str, np.ndarray]:
    """Reconstruct all schedule values independently from the E480 actor."""
    base_lr, peak_lr = 0.0001, 0.001
    warm = np.linspace(base_lr, peak_lr, 47, dtype=np.float64)
    phase = np.arange(427, dtype=np.float64) / 426
    cosine = base_lr + (peak_lr - base_lr) * 0.5 * (1.0 + np.cos(np.pi * phase))
    warmcos = np.concatenate((warm, cosine))
    mean_lr = math.fsum(warmcos.tolist()) / UPDATES
    return {
        "constant_peak": np.full(UPDATES, peak_lr, dtype=np.float64),
        "warmcos_47_427": warmcos,
        "constant_mean_warmcos": np.full(UPDATES, mean_lr, dtype=np.float64),
    }


def validate_lr_traces(run: Path) -> dict[str, np.ndarray]:
    """Bind actual E480 receipt to the frozen E476 receipt and every exact trace."""
    e476_receipt_path = E476_DIR / "preflight_cpu_v5/lr_traces.json"
    if sha256_file(e476_receipt_path) != EXPECTED_E476_TRACE_RECEIPT_SHA256:
        raise ValueError("local frozen E476 LR receipt hash changed")
    e476_receipt = json.loads(e476_receipt_path.read_text())
    receipt_path = run / "lr_traces.json"
    if not receipt_path.is_file():
        raise ValueError("E480 LR trace receipt missing")
    receipt = json.loads(receipt_path.read_text())
    if receipt != e476_receipt:
        raise ValueError("E480 full LR trace receipt differs from frozen E476 receipt")
    if receipt.get("updates") != UPDATES or tuple(receipt.get("protocols", {})) != SCHEDULES:
        raise ValueError("E480 trace receipt identity or protocol order changed")
    if receipt.get("protocols") != e476_receipt.get("protocols"):
        raise ValueError("E480 LR protocol receipt differs from frozen E476 values")
    expected = expected_lr_traces()
    for name, values in expected.items():
        reference = np.asarray(e476_receipt["protocols"][name]["values"], dtype=np.float64)
        saved_meta = receipt["protocols"][name]
        saved = np.asarray(saved_meta.get("values"), dtype=np.float64)
        digest = trace_digest(values)
        if not np.array_equal(reference, values) or not np.array_equal(saved, values):
            raise ValueError(f"registered LR values differ from E476 for {name}")
        if (
            saved_meta.get("sha256") != digest
            or e476_receipt["protocols"][name].get("sha256") != digest
        ):
            raise ValueError(f"registered LR trace hash mismatch for {name}")
    if receipt.get("e474_source_reference_trace") != e476_receipt.get(
        "e474_source_reference_trace"
    ):
        raise ValueError("E474 source reference trace changed")
    reference_trace = np.asarray(receipt["e474_source_reference_trace"], dtype=np.float64)
    if receipt.get("e474_source_reference_trace_sha256") != e476_receipt.get(
        "e474_source_reference_trace_sha256"
    ) or receipt.get("e474_source_reference_trace_sha256") != trace_digest(reference_trace):
        raise ValueError("E474 reference LR trace hash changed")
    return expected


def validate_sealed_spec_core(spec: dict) -> dict:
    """Validate E480 fit spec identity and exact fit artifacts."""
    checked = validate_spec(spec)
    if checked.get("name") != EXPECTED_SPEC_NAME or checked.get("kind") != "gpu":
        raise ValueError("unexpected E480 fit specification identity")
    if tuple(checked.get("artifacts", ())) != EXPECTED_RUN_ARTIFACTS:
        raise ValueError("sealed E480 artifact map differs from fit manifest/LR receipt")
    return checked


def validate_attempt_record(attempt_path: Path, spec: dict, run: Path) -> dict:
    """Require successful runner proof, immutable runtime and exact outputs."""
    checked_spec = validate_sealed_spec_core(spec)
    record = json.loads(attempt_path.read_text())
    if record.get("id") != EXPECTED_ATTEMPT_ID or record.get("kind") != "gpu":
        raise ValueError("runner attempt identity or kind differs from E480")
    if record.get("spec") != checked_spec or record.get("spec_sha256") != campaign_spec_digest(
        checked_spec
    ):
        raise ValueError("runner attempt does not carry the exact sealed E480 spec")
    actual_argv = record.get("argv")
    if (
        not isinstance(actual_argv, list)
        or actual_argv[1:] != checked_spec["argv"][1:]
        or not actual_argv
        or not isinstance(actual_argv[0], str)
        or not Path(actual_argv[0]).is_absolute()
    ):
        raise ValueError("runner command differs from the sealed E480 argv")
    runtime = runtime_identity(ROOT)
    if (
        set(record.get("runtime_sha256", {})) != set(RUNTIME_FILES)
        or record["runtime_sha256"] != runtime
    ):
        raise ValueError("runner runtime file identity differs from the receipt")
    if (
        record.get("status") != "finished"
        or record.get("exit_code") != 0
        or record.get("signals") != []
        or record.get("teardown_verified") is not True
        or any(
            record.get(key) is not None for key in ("failure", "failure_scope", "failure_category")
        )
    ):
        raise ValueError("runner attempt did not finish cleanly with verified teardown")
    unit = record.get("unit_properties", {})
    if unit.get("Result") != "success" or unit.get("ExecMainStatus") not in (0, "0"):
        raise ValueError("runner systemd service reports failure")
    for initial_name, final_name in (
        ("initial_memory_events", "final_memory_events"),
        ("initial_parent_memory_events", "final_parent_memory_events"),
    ):
        initial, final = record.get(initial_name), record.get(final_name)
        if not isinstance(initial, dict) or not isinstance(final, dict):
            raise ValueError(f"missing runner OOM snapshots: {initial_name}")
        for key in ("oom", "oom_kill", "oom_group_kill"):
            before, after = initial.get(key), final.get(key)
            if type(before) is not int or type(after) is not int or before < 0 or after < 0:
                raise ValueError(f"invalid OOM counter: {initial_name}/{key}")
            if before != after:
                raise ValueError(f"OOM counter increased: {initial_name}/{key}")
    expected_artifacts = []
    for relative in checked_spec["artifacts"]:
        artifact = ROOT / relative
        if not artifact.is_file():
            raise ValueError(f"sealed E480 output artifact missing: {relative}")
        expected_artifacts.append(
            {"path": relative, "sha256": sha256_file(artifact), "size": artifact.stat().st_size}
        )
    if record.get("artifacts") != expected_artifacts:
        raise ValueError("runner artifact inventory differs from sealed E480 outputs")
    expected_run = ROOT / Path(checked_spec["artifacts"][0]).parent
    if run.resolve() != expected_run.resolve():
        raise ValueError("requested score run directory differs from runner artifacts")
    return {
        "attempt_id": record["id"],
        "attempt_record_path": str(attempt_path.resolve()),
        "attempt_sha256": sha256_file(attempt_path),
        "spec_sha256": record["spec_sha256"],
        "runtime_sha256": runtime,
        "artifacts": expected_artifacts,
        "worker_oom_counters_unchanged": True,
        "parent_oom_counters_unchanged": True,
        "teardown_verified": True,
    }


def validate_sealed_fit(run: Path, report: dict, spec: dict) -> dict:
    """Validate report, data, sources and trace artifact against the future seal."""
    checked = validate_sealed_spec_core(spec)
    expected_run = ROOT / Path(checked["artifacts"][0]).parent
    if run.resolve() != expected_run.resolve():
        raise ValueError("requested E480 run differs from sealed artifact path")
    if report.get("status") != "all_24_complete_test_unread":
        raise ValueError("E480 fit manifest is incomplete or test rows were loaded in training")
    if report.get("fit_only") is not True or report.get("exposed_test_groups_loaded") is not False:
        raise ValueError("E480 report does not certify fit-only training")
    if (
        report.get("seeds") != list(SEEDS)
        or report.get("parameters_per_cell") != EXPECTED_PARAMETERS
    ):
        raise ValueError("E480 seed or parameter identity differs from preregistration")
    if (
        report.get("cohort_sha256") != EXPECTED_COHORT_SHA256
        or sha256_file(COHORT_PATH) != EXPECTED_COHORT_SHA256
    ):
        raise ValueError("E480 fit cohort hash changed")
    if (
        report.get("dataset_sha256") != EXPECTED_DATASET_SHA256
        or sha256_file(DATA_PATH) != EXPECTED_DATASET_SHA256
    ):
        raise ValueError("E480 dataset hash changed")
    sealed_inputs = {entry["path"]: entry for entry in checked["inputs"]}
    for relative, expected_hash in (
        ("research/campaign_2026-10-05/target_mechanics/cohort.json", EXPECTED_COHORT_SHA256),
        ("datasets/valid.parquet", EXPECTED_DATASET_SHA256),
    ):
        entry = sealed_inputs.get(relative)
        local = ROOT / relative
        if entry is None or entry.get("sha256") != expected_hash or not local.is_file():
            raise ValueError(f"sealed input identity missing/changed: {relative}")
        if local.stat().st_size != entry.get("size"):
            raise ValueError(f"sealed input size changed: {relative}")
    fixed = {
        "experiment": "E480 two-seed activation x optimizer x learning-rate replication",
        "cohort_rows": 24,
        "activations": list(ACTIVATIONS),
        "optimizers": list(OPTIMIZERS),
        "schedules": list(SCHEDULES),
        "batch": 4,
        "chunk_rows": CHUNK_ROWS,
        "updates_per_fit": UPDATES,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "gradient_clip": 1.0,
        "sampling_steps_zero_based": list(SAMPLE_STEPS),
    }
    for key, value in fixed.items():
        if report.get(key) != value:
            raise ValueError(f"E480 manifest field differs from design: {key}")
    expected_optimizer_details = {
        "adamw": "all parameters; betas=(0.9,0.999), wd=0.01",
        "muon_adamw": "Muon only on gru.weight_ih_l0, weight_hh_l0, weight_ih_l1, weight_hh_l1; 3 blocks each; AdamW on input/head/biases",
    }
    if report.get("optimizer_details") != expected_optimizer_details:
        raise ValueError("E480 optimizer partition differs from the registered recipe")
    precision = report.get("precision", {})
    if (
        precision.get("device_math") != "FP32 IEEE"
        or precision.get("autocast") is not False
        or precision.get("torch.backends.fp32_precision") != "ieee"
        or precision.get("torch.backends.cuda.matmul.fp32_precision") != "ieee"
        or precision.get("torch.backends.cudnn.fp32_precision") != "ieee"
        or precision.get("torch.backends.cudnn.rnn.fp32_precision") != "ieee"
    ):
        raise ValueError("E480 full-fit precision policy differs from FP32 IEEE")
    if set(report.get("source_sha256", {})) != EXPECTED_MANIFEST_SOURCES:
        raise ValueError("E480 fit manifest source file set changed")
    sealed_content = {
        entry["path"]: entry["sha256"] for entry in checked["code"] + checked["inputs"]
    }
    for relative, expected_hash in EXPECTED_MANIFEST_SOURCE_SHA256.items():
        if report["source_sha256"].get(relative) != expected_hash:
            raise ValueError(
                f"E480 manifest source differs from the preregistered hash: {relative}"
            )
        if sealed_content.get(relative) != expected_hash:
            raise ValueError(f"sealed spec does not bind the exact fit source: {relative}")
        if sha256_file(ROOT / relative) != expected_hash:
            raise ValueError(f"local fit source differs from seal: {relative}")
    if len(report.get("completed_cells", [])) != len(CELLS):
        raise ValueError("E480 manifest does not contain all 24 fixed cells")
    trace_receipt_path = run / "lr_traces.json"
    if not trace_receipt_path.is_file() or EXPECTED_RUN_ARTIFACTS[1] not in checked["artifacts"]:
        raise ValueError("sealed E480 LR receipt is missing from outputs")
    traces = validate_lr_traces(run)
    return {
        "sealed_spec_name": checked["name"],
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "fit_source_sha256": report["source_sha256"],
        "lr_trace_receipt_sha256": sha256_file(trace_receipt_path),
        "validated_schedules": {name: trace_digest(values) for name, values in traces.items()},
    }


def make_model(activation: str) -> RNNModel:
    """Construct the frozen E476 H64 two-layer architecture independently."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"unregistered activation: {activation}")
    model_type = RNNModel if activation == "base" else SineEmbeddingRNN
    model = model_type(42, hidden=64, layers=2, dropout=0.0, n_out=2, cell="gru", enc="linear")
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise ValueError("independent scorer model parameter count differs from 61,058")
    return model


def validate_identity_buffers(state_dict: dict[str, Tensor]) -> None:
    """Require checkpoints to keep model-side normalization buffers at identity."""
    if "mean" not in state_dict or "std" not in state_dict:
        raise ValueError("checkpoint misses E474 external-normalization identity buffers")
    if not torch.equal(state_dict["mean"].cpu(), torch.zeros(42)):
        raise ValueError("checkpoint mean buffer is not identity zero")
    if not torch.equal(state_dict["std"].cpu(), torch.ones(42)):
        raise ValueError("checkpoint std buffer is not identity one")


def validate_cells(
    run: Path,
    report: dict,
    nodes: list[dict],
    traces: dict[str, np.ndarray],
) -> dict[tuple[int, str, str, str], dict]:
    """Check exact role, traversal, 474 steps, telemetry and checkpoint hashes."""
    if (
        len(nodes) != 40
        or sum(node["role"] == "fit" for node in nodes) != 24
        or sum(node["role"] == "test" for node in nodes) != 16
    ):
        raise ValueError("cohort must contain exactly 24 fit and 16 test sequences")
    fit_nodes = [node for node in nodes if node["role"] == "fit"]
    test_nodes = [node for node in nodes if node["role"] == "test"]
    fit_ids = {node["component"] for node in fit_nodes}
    test_ids = {node["component"] for node in test_nodes}
    fit_groups = {node["row_group"] for node in fit_nodes}
    test_groups = {node["row_group"] for node in test_nodes}
    if len(fit_ids) != 24 or len(test_ids) != 16 or fit_ids & test_ids:
        raise ValueError("fit/test components overlap or are not unique")
    if len(fit_groups | test_groups) != 40 or fit_groups & test_groups:
        raise ValueError("fit/test row groups overlap or are not unique")
    observed: dict[tuple[int, str, str, str], dict] = {}
    expected_keys = set(CELLS)
    for cell in report["completed_cells"]:
        key = (
            cell.get("seed"),
            cell.get("activation"),
            cell.get("optimizer"),
            cell.get("schedule"),
        )
        if key not in expected_keys or key in observed:
            raise ValueError(f"unknown or duplicate E480 cell: {key}")
        seed, activation, optimizer, schedule = key
        order = list(range(24))
        np.random.default_rng(seed).shuffle(order)
        expected_order = [fit_nodes[index]["row_group"] for index in order]
        if cell.get("fit_row_groups") != expected_order:
            raise ValueError(f"fit traversal differs for {key}")
        if len(set(expected_order)) != 24 or set(expected_order) & test_groups:
            raise ValueError(f"invalid or exposed fit row groups for {key}")
        if (
            cell.get("updates") != UPDATES
            or cell.get("parameters") != EXPECTED_PARAMETERS
            or cell.get("parameters_changed") is not True
            or cell.get("changed_parameter_tensors", 0) <= 0
        ):
            raise ValueError(f"cell update/parameter receipt invalid: {key}")
        if cell.get("lr_trace_sha256") != trace_digest(traces[schedule]):
            raise ValueError(f"cell trace hash mismatch: {key}")
        telemetry = cell.get("telemetry", [])
        if [entry.get("step_zero_based") for entry in telemetry] != list(SAMPLE_STEPS):
            raise ValueError(f"cell telemetry steps differ: {key}")
        for entry in telemetry:
            step = int(entry["step_zero_based"])
            if entry.get("lr") != float(traces[schedule][step]):
                raise ValueError(f"telemetry LR differs from frozen trace: {key}/{step}")
            for field in ("gradient_norm_preclip", "fit_loss"):
                value = float(entry.get(field, float("nan")))
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid telemetry {field}: {key}/{step}")
            families = entry.get("parameter_update_rms", {})
            if set(families) != {
                "input_projection",
                "recurrent_core",
                "hidden_head",
                "output_head",
            }:
                raise ValueError(f"telemetry family set differs: {key}/{step}")
            if any(
                not math.isfinite(float(value)) or float(value) < 0 for value in families.values()
            ):
                raise ValueError(f"invalid update RMS: {key}/{step}")
        expected_name = f"seed{seed}__{activation}__{optimizer}__{schedule}.pt"
        if cell.get("checkpoint") != expected_name:
            raise ValueError(f"checkpoint naming differs: {key}")
        checkpoint = run / expected_name
        if not checkpoint.is_file() or sha256_file(checkpoint) != cell.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint missing or hash mismatch: {key}")
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if any(not torch.isfinite(value).all() for value in state_dict.values()):
            raise ValueError(f"checkpoint has a nonfinite tensor: {key}")
        validate_identity_buffers(state_dict)
        if set(state_dict) != set(make_model(activation).state_dict()):
            raise ValueError(f"checkpoint tensor names differ from standalone model: {key}")
        observed[key] = cell
    if set(observed) != expected_keys:
        raise ValueError("completed cell identities differ from fixed 24-cell matrix")
    return observed


def validate_external_normalizer(
    fit_manifest: dict,
    fit_x: np.ndarray,
    fit_nodes: list[dict],
    all_manifest: dict,
    all_x: np.ndarray,
    all_nodes: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Assert the same fit-only external normalization feeds fit and test inputs."""
    fit_norm, all_norm = fit_manifest.get("normalization"), all_manifest.get("normalization")
    if not isinstance(fit_norm, dict) or fit_norm != all_norm:
        raise ValueError("fit/test feature loader normalizers differ")
    mean = np.asarray(fit_norm.get("fit_mean"), dtype=np.float32)
    scale = np.asarray(fit_norm.get("fit_scale"), dtype=np.float32)
    if (
        mean.shape != (42,)
        or scale.shape != (42,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
    ):
        raise ValueError("invalid fit-only external normalizer metadata")
    fit_index = {node["row_group"]: index for index, node in enumerate(fit_nodes)}
    all_index = {node["row_group"]: index for index, node in enumerate(all_nodes)}
    if set(fit_index) != {node["row_group"] for node in fit_nodes}:
        raise ValueError("duplicate fit row groups in fit-only normalization cohort")
    for row_group, index in fit_index.items():
        if row_group not in all_index or not np.array_equal(
            fit_x[index], all_x[all_index[row_group]]
        ):
            raise ValueError("test-inclusive loader changed fit-normalized input features")
    if not np.isfinite(all_x).all():
        raise ValueError("normalized test cohort contains nonfinite features")
    return mean, scale


def normalizer_from_fit_features(fit_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen E470 float64 population-statistic arithmetic."""
    fit_features = np.asarray(fit_features, dtype=np.float32)
    if fit_features.ndim != 3 or fit_features.shape[0] != 24 or fit_features.shape[2] != 42:
        raise ValueError("fit-only causal features must have shape (24, rows, 42)")
    mean = fit_features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(fit_features.std(axis=(0, 1), dtype=np.float64), 0.001).astype(np.float32)
    if mean.shape != (42,) or scale.shape != (42,) or not np.isfinite(mean).all():
        raise ValueError("independently derived fit normalizer is invalid")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("independently derived fit scale is invalid")
    return mean, scale


def derive_fit_normalizer_independently(fit_nodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Recompute frozen E470 fit-only feature statistics from raw fit rows."""
    if len(fit_nodes) != 24 or any(node.get("role") != "fit" for node in fit_nodes):
        raise ValueError("normalizer derivation requires exactly the 24 registered fit nodes")
    parquet = pq.ParquetFile(DATA_PATH)
    causal = []
    for node in fit_nodes:
        table = parquet.read_row_group(node["row_group"], columns=e470.FEATURES, use_threads=False)
        raw = np.column_stack([table[column].to_numpy() for column in e470.FEATURES])
        causal.append(e470.causal_features(raw))
    return normalizer_from_fit_features(np.stack(causal))


def validate_raw_target_component(
    raw_targets: np.ndarray,
    mask: np.ndarray,
    clipped_targets: np.ndarray,
    row_group: int,
) -> np.ndarray:
    """Reject raw selected nonfinite labels before clipping and verify clipping."""
    raw = np.asarray(raw_targets, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    clipped = np.asarray(clipped_targets, dtype=np.float32)
    if (
        raw.ndim != 2
        or raw.shape[-1] != 2
        or selected.shape != raw.shape[:1]
        or clipped.shape != raw.shape
    ):
        raise ValueError(f"malformed labels or mask for row group {row_group}")
    if not selected.any() or not np.isfinite(raw[selected]).all():
        raise ValueError(f"selected raw test labels are empty/nonfinite for {row_group}")
    if not np.isfinite(raw).all():
        raise ValueError(f"raw test labels contain a nonfinite value for {row_group}")
    expected = np.clip(raw, -METRIC_CLIP, METRIC_CLIP).astype(np.float32)
    if not np.array_equal(expected, clipped):
        raise ValueError(f"clipped labels differ from source for row group {row_group}")
    return raw


def read_raw_test_targets(
    test_nodes: list[dict],
    clipped_targets: np.ndarray,
    selected_mask: np.ndarray,
) -> np.ndarray:
    """Read raw test labels and reject selected nonfinite values before clipping."""
    parquet = pq.ParquetFile(DATA_PATH)
    raw_targets = np.empty((len(test_nodes), 20_000, 2), dtype=np.float64)
    if clipped_targets.shape != raw_targets.shape or selected_mask.shape != raw_targets.shape[:2]:
        raise ValueError("test labels/masks do not match the fixed 16×20,000 cohort")
    for index, node in enumerate(test_nodes):
        table = parquet.read_row_group(
            node["row_group"],
            columns=["t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        raw = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")]).astype(np.float64)
        source_mask = table["need_prediction"].to_numpy().astype(bool)
        source_mask &= table["is_scored"].to_numpy().astype(bool)
        if not np.array_equal(source_mask, selected_mask[index]):
            raise ValueError(f"official selected mask differs for row group {node['row_group']}")
        raw_targets[index] = validate_raw_target_component(
            raw, source_mask, clipped_targets[index], node["row_group"]
        )
    return raw_targets


def predict_test_sequences(
    checkpoint: Path,
    activation: str,
    x_test: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Stream full test sequences with carried state and finite checks."""
    model = make_model(activation).to(device).float()
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    if any(not torch.isfinite(value).all() for value in state_dict.values()):
        raise ValueError(f"checkpoint includes nonfinite values: {checkpoint.name}")
    validate_identity_buffers(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    predictions = np.empty((len(x_test), x_test.shape[1], 2), dtype=np.float32)
    precision = contextlib.nullcontext()
    if device.type == "cuda":
        from fit_activation_placement import full_ieee_float32  # local import after sys.path

        precision = full_ieee_float32()
    with precision, torch.no_grad():
        for component_index, sequence in enumerate(x_test):
            state = None
            parts = []
            for start in range(0, sequence.shape[0], CHUNK_ROWS):
                chunk = torch.from_numpy(sequence[start : start + CHUNK_ROWS][None]).to(
                    device=device, dtype=torch.float32
                )
                output, state = model(chunk, state)
                if not torch.isfinite(output).all() or not torch.isfinite(state).all():
                    raise ValueError(f"nonfinite sequence output/state for {checkpoint.name}")
                parts.append(output[0].cpu().numpy())
            predictions[component_index] = np.concatenate(parts, axis=0)
    if not np.isfinite(predictions).all():
        raise ValueError(f"nonfinite full sequence prediction before metric: {checkpoint.name}")
    return predictions


def pool_component_moments(moments: np.ndarray, multiplicities: np.ndarray) -> float:
    """Pool centered weighted-Pearson moments under one component count vector."""
    pooled = np.zeros((2, 6), dtype=np.float64)
    for index, multiplicity in enumerate(multiplicities):
        count = int(multiplicity)
        if count == 0:
            continue
        for target_index in range(2):
            contribution = moments[index, target_index].copy()
            contribution[[0, 3, 4, 5]] *= count
            pooled[target_index] = _merge_moments(pooled[target_index], contribution)
    return float(np.mean([_wp_from_moments(pooled[k]) for k in range(2)]))


def bootstrap_multiplicities(component_ids: np.ndarray) -> np.ndarray:
    """Make one immutable component-bootstrap count matrix shared by all arms."""
    if component_ids.ndim != 1 or len(np.unique(component_ids)) != len(component_ids):
        raise ValueError("bootstrap requires unique one-dimensional component ids")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    counts = np.zeros((BOOTSTRAP_COUNT, len(component_ids)), dtype=np.int16)
    for draw in range(BOOTSTRAP_COUNT):
        selected = rng.integers(0, len(component_ids), len(component_ids))
        counts[draw] = np.bincount(selected, minlength=len(component_ids))
    if not np.all(counts.sum(axis=1) == len(component_ids)):
        raise ValueError("bootstrap component multiplicities do not conserve sample size")
    return counts


def bootstrap_effect(draws: np.ndarray, point: float, components: int) -> dict:
    """Summarize fixed-count paired bootstrap draws for one registered contrast."""
    if (
        draws.shape != (BOOTSTRAP_COUNT,)
        or not np.isfinite(draws).all()
        or not math.isfinite(point)
    ):
        raise ValueError("invalid bootstrap contrast draws")
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean_diff": float(point),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "ci95": [float(low), float(high)],
        "p_le_0": float(np.mean(draws <= 0)),
        "n_components": int(components),
        "n_boot": BOOTSTRAP_COUNT,
        "seed": BOOTSTRAP_SEED,
        "metric_id": METRIC_ID,
        "paired_component_resampling": True,
    }


def compute_seed_contrasts(
    seed: int,
    arm_wps: dict[str, float],
    arm_draws: dict[str, np.ndarray],
    component_count: int,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Calculate 20 paired effects, two joint interactions, and four primaries."""
    pairs: dict[str, dict] = {}
    for candidate, reference, family in PAIR_CONTRASTS:
        point = arm_wps[candidate] - arm_wps[reference]
        draws = arm_draws[candidate] - arm_draws[reference]
        name = f"{candidate}_minus_{reference}"
        pairs[name] = {
            "seed": seed,
            "family": family,
            "candidate": candidate,
            "reference": reference,
            **bootstrap_effect(draws, point, component_count),
        }
    interactions: dict[str, dict] = {}
    for optimizer in OPTIMIZERS:
        key_sc = f"sine_embed__{optimizer}__warmcos_47_427"
        key_bc = f"base__{optimizer}__warmcos_47_427"
        key_sm = f"sine_embed__{optimizer}__constant_mean_warmcos"
        key_bm = f"base__{optimizer}__constant_mean_warmcos"
        point = (arm_wps[key_sc] - arm_wps[key_bc]) - (arm_wps[key_sm] - arm_wps[key_bm])
        draws = (arm_draws[key_sc] - arm_draws[key_bc]) - (arm_draws[key_sm] - arm_draws[key_bm])
        name = f"sine_activation_x_warmcos_shape__{optimizer}"
        interactions[name] = {
            "seed": seed,
            "family": "activation_by_schedule_shape_interaction",
            "optimizer": optimizer,
            "contrast": f"({key_sc}-{key_bc})-({key_sm}-{key_bm})",
            **bootstrap_effect(draws, point, component_count),
        }
    primary = {
        name: value
        for name, value in pairs.items()
        if value["family"] == "warmcos_minus_constant_mean"
    }
    if len(pairs) != 20 or len(interactions) != 2 or len(primary) != 4:
        raise AssertionError("registered E480 contrast family counts changed")
    return pairs, interactions, primary


def main() -> None:
    """Validate E480 fit proofs, score 24 arms and export paired raw selections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempt-record", type=Path, default=DEFAULT_ATTEMPT_RECORD)
    parser.add_argument("--sealed-spec", type=Path, default=SEALED_FIT_SPEC)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"fit run directory missing: {run}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scorer output: {output}")
    spec = json.loads(args.sealed_spec.read_text())
    attempt = validate_attempt_record(args.attempt_record.resolve(), spec, run)
    report = json.loads((run / "fit_manifest.json").read_text())
    seal_identity = validate_sealed_fit(run, report, spec)
    traces = validate_lr_traces(run)
    cohort = json.loads(COHORT_PATH.read_text())
    nodes = cohort.get("cohort", [])
    fit_nodes = [node for node in nodes if node.get("role") == "fit"]
    test_nodes = [node for node in nodes if node.get("role") == "test"]
    cells = validate_cells(run, report, nodes, traces)
    if len(fit_nodes) != 24 or len(test_nodes) != 16 or len(cells) != 24:
        raise AssertionError("fixed E480 fit/test/cell counts changed")
    if args.device == "cuda":
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA scoring requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(1)

    fit_manifest, fit_x, _fit_y, _fit_need, _fit_score, loaded_fit_nodes = e470.load_cohort(
        include_test=False
    )
    all_manifest, all_x, all_y, _all_need, all_score, loaded_nodes = e470.load_cohort(
        include_test=True
    )
    if [node["row_group"] for node in loaded_fit_nodes] != [
        node["row_group"] for node in fit_nodes
    ]:
        raise ValueError("fit-only normalization loader order differs from frozen cohort")
    if [node["row_group"] for node in loaded_nodes] != [node["row_group"] for node in nodes]:
        raise ValueError("score loader order differs from frozen cohort")
    normalizer_mean, normalizer_scale = validate_external_normalizer(
        fit_manifest, fit_x, loaded_fit_nodes, all_manifest, all_x, loaded_nodes
    )
    independent_mean, independent_scale = derive_fit_normalizer_independently(fit_nodes)
    if not np.array_equal(normalizer_mean, independent_mean) or not np.array_equal(
        normalizer_scale, independent_scale
    ):
        raise ValueError("E470 loader normalizer differs from independent fit-only derivation")
    if fit_x.shape != (24, 20_000, 42) or all_x.shape != (40, 20_000, 42):
        raise ValueError(
            f"unexpected normalized fit/test feature shapes: {fit_x.shape}, {all_x.shape}"
        )
    test_indices = [index for index, node in enumerate(nodes) if node["role"] == "test"]
    test_x = all_x[test_indices]
    clipped_targets = all_y[test_indices]
    selected_mask = all_score[test_indices]
    raw_targets = read_raw_test_targets(test_nodes, clipped_targets, selected_mask)
    counts = selected_mask.sum(axis=1).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("each registered component must contribute selected rows")
    component_ids = np.asarray([node["component"] for node in test_nodes], dtype=np.int64)
    if len(np.unique(component_ids)) != 16:
        raise ValueError("component identities must be unique for paired bootstrap")
    sorted_order = np.argsort(component_ids, kind="stable")
    bootstrap_components = component_ids[sorted_order]
    multiplicities = bootstrap_multiplicities(bootstrap_components)

    arm_scores: dict[str, dict] = {}
    arm_moments: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}
    raw_selected_targets = raw_targets[selected_mask]
    selected_predictions: dict[str, np.ndarray] = {}
    for seed, activation, optimizer, schedule in CELLS:
        arm_key = f"seed{seed}__{activation}__{optimizer}__{schedule}"
        within_seed = f"{activation}__{optimizer}__{schedule}"
        cell = cells[(seed, activation, optimizer, schedule)]
        checkpoint = run / cell["checkpoint"]
        prediction = predict_test_sequences(checkpoint, activation, test_x, device)
        if not np.isfinite(prediction).all():
            raise ValueError(f"nonfinite raw full-sequence prediction for {arm_key}")
        official = global_wp(raw_targets, prediction, selected_mask)
        if official.get("metric_id") != METRIC_ID or official.get("selected_rows") != int(
            counts.sum()
        ):
            raise ValueError(f"official metric ID or selected-row total differs for {arm_key}")
        moments, _ = paired_component_moments(raw_targets, prediction, prediction, selected_mask)
        arm_moments[seed][within_seed] = moments[sorted_order]
        selected_predictions[arm_key] = prediction[selected_mask].copy()
        arm_scores[arm_key] = {
            "seed": seed,
            "activation": activation,
            "optimizer": optimizer,
            "schedule": schedule,
            "wp": float(official["wp"]),
            "t0": float(official["t0"]),
            "t1": float(official["t1"]),
            "selected_rows": int(official["selected_rows"]),
            "component_row_counts": counts.tolist(),
            "parameters": EXPECTED_PARAMETERS,
            "updates": UPDATES,
        }

    per_seed_pairs: dict[str, dict] = {}
    per_seed_interactions: dict[str, dict] = {}
    per_seed_primary: dict[str, dict] = {}
    per_seed_bootstrap: dict[str, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        within_seed_wps = {arm: arm_scores[f"seed{seed}__{arm}"]["wp"] for arm in arm_moments[seed]}
        within_seed_draws: dict[str, np.ndarray] = {}
        for arm, moments in arm_moments[seed].items():
            within_seed_draws[arm] = np.asarray(
                [pool_component_moments(moments, counts) for counts in multiplicities],
                dtype=np.float64,
            )
            if within_seed_draws[arm].shape != (BOOTSTRAP_COUNT,):
                raise ValueError("bootstrap draw shape invalid")
        pairs, interactions, primary = compute_seed_contrasts(
            seed, within_seed_wps, within_seed_draws, len(component_ids)
        )
        per_seed_pairs[str(seed)] = pairs
        per_seed_interactions[str(seed)] = interactions
        per_seed_primary[str(seed)] = primary
        per_seed_bootstrap[str(seed)] = within_seed_draws

    total_pairs = sum(len(value) for value in per_seed_pairs.values())
    total_interactions = sum(len(value) for value in per_seed_interactions.values())
    total_primary = sum(len(value) for value in per_seed_primary.values())
    if (total_pairs, total_interactions, total_primary) != (40, 4, 8):
        raise AssertionError(
            "E480 must retain 40 paired effects, four interactions and eight primary effects"
        )

    arm_order = [
        f"seed{seed}__{activation}__{optimizer}__{schedule}"
        for seed, activation, optimizer, schedule in CELLS
    ]
    output.mkdir(parents=True)
    export_path = output / "selected_exports.npz"
    np.savez_compressed(
        export_path,
        arm_names=np.asarray(arm_order),
        targets_raw=raw_selected_targets,
        predictions_raw=np.stack([selected_predictions[arm] for arm in arm_order]),
        selected_mask=selected_mask,
        component_ids=component_ids,
        row_groups=np.asarray([node["row_group"] for node in test_nodes], dtype=np.int64),
        selected_row_counts=counts,
        component_bootstrap_multiplicities=multiplicities,
    )
    result = {
        "kind": "independent E480 two-seed activation-schedule scorer",
        "metric_id": METRIC_ID,
        "scope": "two fixed fit initializations; same 16 previously exposed components, no seed pooling or public forecast",
        "fit_seeds": list(SEEDS),
        "fit_manifest_sha256": sha256_file(run / "fit_manifest.json"),
        "lr_trace_receipt_sha256": sha256_file(run / "lr_traces.json"),
        "runner_attempt_receipt": attempt,
        "sealed_identity": seal_identity,
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "normalizer": {
            "source": "independently rederived from raw E480 fit-role rows; compared exactly with frozen E470 loader",
            "identity_model_buffers_required": True,
            "fit_mean_sha256": hashlib.sha256(normalizer_mean.astype("<f4").tobytes()).hexdigest(),
            "fit_scale_sha256": hashlib.sha256(
                normalizer_scale.astype("<f4").tobytes()
            ).hexdigest(),
        },
        "scorer_sources_sha256": {
            "scorer": sha256_file(Path(__file__)),
            "official_gmetric": sha256_file(ROOT / "src/wnn/gmetric.py"),
            "official_evaluation": sha256_file(ROOT / "src/wnn/evaluation.py"),
            "fit_manifest_sources": report["source_sha256"],
        },
        "component_ids": component_ids.tolist(),
        "row_groups": [node["row_group"] for node in test_nodes],
        "selected_row_counts": counts.tolist(),
        "selected_rows_total": int(counts.sum()),
        "arm_count": len(arm_scores),
        "arm_scores": arm_scores,
        "pair_contrast_count": total_pairs,
        "pair_contrasts_by_seed": per_seed_pairs,
        "joint_interaction_count": total_interactions,
        "joint_interactions_by_seed": per_seed_interactions,
        "primary_warmcos_minus_constant_mean_count": total_primary,
        "primary_warmcos_minus_constant_mean_by_seed": per_seed_primary,
        "bootstrap": {
            "method": "official centered weighted-Pearson component moments pooled with wnn.gmetric._merge_moments",
            "multiplicity_hash": hashlib.sha256(multiplicities.astype("<i2").tobytes()).hexdigest(),
            "multiplicities_shared_across_all_24_arms_and_all_contrasts": True,
            "resampling_unit": "fixed exposed component ID",
            "n_boot": BOOTSTRAP_COUNT,
            "seed": BOOTSTRAP_SEED,
            "no_seed_pooling": True,
            "component_note": "all 16 components have prior exposure; exploratory paired inference only",
        },
        "selected_exports_path": export_path.name,
        "selected_exports_sha256": sha256_file(export_path),
    }
    (output / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "arms": len(arm_scores),
                "contrasts": total_pairs + total_interactions,
                "primary": total_primary,
                "selected_rows": int(counts.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

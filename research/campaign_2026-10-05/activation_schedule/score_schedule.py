"""Independently validate and score the sealed E476 12-cell fit matrix."""

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
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
SEALED_FIT_SPEC = HERE / "e476_twelve_fit_v1.sealed.json"
DEFAULT_RUN = HERE / "runs/twelve_v1"
DEFAULT_OUTPUT = HERE / "eda/twelve_v1"
DEFAULT_ATTEMPT_RECORD = (
    ROOT / "research/campaign_2026-10-04/runner/attempts/e476-activation-twelve-fit-v1.json"
)
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
DATA_PATH = ROOT / "datasets/valid.parquet"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E474_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402

from wnn.campaign import RUNTIME_FILES, runtime_identity  # noqa: E402
from wnn.campaign_spec import digest as campaign_spec_digest  # noqa: E402
from wnn.campaign_spec import validate_spec  # noqa: E402
from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402
from wnn.gmetric import (  # noqa: E402
    METRIC_CLIP,
    METRIC_ID,
    _merge_moments,
    _wp_from_moments,
    global_wp,
)
from wnn.nn import RNNModel  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

SEED = 271_828
BOOTSTRAP_COUNT = 2_000
BOOTSTRAP_SEED = 271_828
UPDATES = 474
CHUNK_ROWS = 256
EXPECTED_PARAMETERS = 61_058
SAMPLE_STEPS = (0, 46, 236, 473)
EXPECTED_MANIFEST_SOURCES = {
    "research/campaign_2026-10-05/activation_schedule/fit_activation_schedule.py",
    "research/campaign_2026-10-05/activation_schedule/tests.py",
    "research/campaign_2026-10-05/activation_schedule/PREREG.md",
    "research/campaign_2026-10-05/activation_placement/fit_activation_placement.py",
    "research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py",
    "src/wnn/optim.py",
    "src/wnn/nn.py",
}
EXPECTED_SEALED_ARTIFACTS = (
    "research/campaign_2026-10-05/activation_schedule/runs/twelve_v1/fit_manifest.json",
    "research/campaign_2026-10-05/activation_schedule/runs/twelve_v1/lr_traces.json",
)
ACTIVATIONS = ("base", "sine_embed")
OPTIMIZERS = ("adamw", "muon_adamw")
SCHEDULES = ("constant_peak", "warmcos_47_427", "constant_mean_warmcos")
ARMS = tuple(
    (activation, optimizer, schedule)
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
    """E476 readout model with the independently reconstructed sine embedding."""

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
    """Hash an LR trace in the exact little-endian float64 representation."""
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def expected_lr_traces() -> dict[str, np.ndarray]:
    """Reconstruct all three preregistered 474-update traces independently."""
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


def validate_lr_traces(run: Path, report: dict) -> dict[str, np.ndarray]:
    """Verify the saved trace receipt, exact values, hashes, areas and telemetry LR."""
    path = run / "lr_traces.json"
    receipt = json.loads(path.read_text())
    if report.get("lr_trace_receipt") != "lr_traces.json" or receipt.get("updates") != UPDATES:
        raise ValueError("LR trace receipt pointer or update count is invalid")
    if set(receipt.get("protocols", {})) != set(SCHEDULES):
        raise ValueError("the saved trace set differs from the preregistered schedules")
    expected = expected_lr_traces()
    for name, values in expected.items():
        saved = np.asarray(receipt["protocols"][name].get("values"), dtype=np.float64)
        if saved.shape != (UPDATES,) or not np.array_equal(saved, values):
            raise ValueError(f"saved LR values differ from fixed trace {name}")
        digest = trace_digest(values)
        metadata = receipt["protocols"][name]
        if metadata.get("sha256") != digest:
            raise ValueError(f"LR trace digest mismatch for {name}")
        if not math.isclose(
            float(metadata.get("sum", float("nan"))), float(values.sum()), rel_tol=0, abs_tol=1e-15
        ):
            raise ValueError(f"LR trace sum mismatch for {name}")
        if not math.isclose(
            float(metadata.get("mean", float("nan"))),
            float(values.mean()),
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"LR trace mean mismatch for {name}")
    reference = np.asarray(receipt.get("e474_source_reference_trace"), dtype=np.float64)
    if reference.shape != (UPDATES,) or not np.array_equal(reference, np.full(UPDATES, 0.001)):
        raise ValueError("E474 constant source reference trace changed")
    if receipt.get("e474_source_reference_trace_sha256") != trace_digest(reference):
        raise ValueError("E474 reference LR trace hash mismatch")
    if not math.isclose(
        float(expected["constant_mean_warmcos"].sum()),
        float(expected["warmcos_47_427"].sum()),
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("constant-mean and warm-cosine arithmetic LR areas differ")
    return expected


def validate_sealed_spec_core(spec: dict) -> dict:
    """Validate the campaign seal map and E476-specific artifact contract."""
    checked = validate_spec(spec)
    if checked["name"] != "e476_activation_schedule_twelve_fit_v1" or checked["kind"] != "gpu":
        raise ValueError("unexpected E476 fit specification identity")
    if tuple(checked["artifacts"]) != EXPECTED_SEALED_ARTIFACTS:
        raise ValueError("sealed artifact map must declare exactly fit_manifest and lr_traces")
    if checked["environment"] != {}:
        raise ValueError("sealed E476 environment map differs from the reviewed source")
    return checked


def validate_attempt_record(attempt_path: Path, spec: dict, run: Path) -> dict:
    """Require a clean runner finish with unchanged runtime, OOM counters, and artifacts."""
    checked_spec = validate_sealed_spec_core(spec)
    record = json.loads(attempt_path.read_text())
    if record.get("id") != "e476-activation-twelve-fit-v1" or record.get("kind") != "gpu":
        raise ValueError("runner attempt identity or kind differs from E476")
    if record.get("spec") != checked_spec:
        raise ValueError("runner attempt's embedded spec differs from the sealed spec")
    if record.get("spec_sha256") != campaign_spec_digest(checked_spec):
        raise ValueError("runner attempt spec digest differs from the sealed spec")
    actual_argv = record.get("argv")
    if (
        not isinstance(actual_argv, list)
        or len(actual_argv) != len(checked_spec["argv"])
        or actual_argv[1:] != checked_spec["argv"][1:]
        or not isinstance(actual_argv[0], str)
        or not Path(actual_argv[0]).is_absolute()
    ):
        raise ValueError("runner command does not match the sealed E476 argument vector")
    runtime = runtime_identity(ROOT)
    if set(record.get("runtime_sha256", {})) != set(RUNTIME_FILES):
        raise ValueError("runner runtime hash map has a different file set")
    if record["runtime_sha256"] != runtime:
        raise ValueError("runner runtime files differ from the attempt receipt")
    if (
        record.get("status") != "finished"
        or record.get("exit_code") != 0
        or record.get("signals") != []
        or record.get("teardown_verified") is not True
        or record.get("failure") is not None
        or record.get("failure_scope") is not None
        or record.get("failure_category") is not None
    ):
        raise ValueError("runner attempt did not finish cleanly with verified teardown")
    unit = record.get("unit_properties", {})
    if unit.get("Result") != "success" or unit.get("ExecMainStatus") not in (0, "0"):
        raise ValueError("runner systemd service did not report a successful exit")

    oom_keys = ("oom", "oom_kill", "oom_group_kill")
    for initial_key, final_key in (
        ("initial_memory_events", "final_memory_events"),
        ("initial_parent_memory_events", "final_parent_memory_events"),
    ):
        initial, final = record.get(initial_key), record.get(final_key)
        if not isinstance(initial, dict) or not isinstance(final, dict):
            raise ValueError(f"runner OOM snapshots are missing: {initial_key}/{final_key}")
        for key in oom_keys:
            before, after = initial.get(key), final.get(key)
            if type(before) is not int or type(after) is not int or before < 0 or after < 0:
                raise ValueError(f"invalid runner OOM counter: {initial_key}/{key}")
            if after != before:
                raise ValueError(f"runner OOM counter increased: {initial_key}/{key}")

    expected_artifacts = []
    for relative_path in checked_spec["artifacts"]:
        path = ROOT / relative_path
        if not path.is_file():
            raise ValueError(f"declared E476 artifact is missing: {relative_path}")
        expected_artifacts.append(
            {"path": relative_path, "sha256": sha256_file(path), "size": path.stat().st_size}
        )
    if record.get("artifacts") != expected_artifacts:
        raise ValueError("runner artifact receipt differs from the exact sealed artifact set")
    expected_run = ROOT / Path(checked_spec["artifacts"][0]).parent
    if run.resolve() != expected_run.resolve():
        raise ValueError("scoring run directory differs from the runner's sealed artifacts")
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


def validate_sealed_identity(run: Path, report: dict) -> dict:
    """Validate run, code, cohort and dataset identities against the sealed fit spec."""
    spec = validate_sealed_spec_core(json.loads(SEALED_FIT_SPEC.read_text()))
    expected_run = ROOT / Path(spec["artifacts"][0]).parent
    if run.resolve() != expected_run.resolve():
        raise ValueError("run path differs from the sealed E476 artifact directory")
    if report.get("status") != "all_12_complete_test_unread":
        raise ValueError("run is incomplete or test rows were read during fitting")
    if not report.get("fit_only") or report.get("exposed_test_groups_loaded") is not False:
        raise ValueError("fit manifest does not certify fit-only training")
    code = {entry["path"]: entry["sha256"] for entry in spec["code"]}
    sealed_inputs = {entry["path"]: entry for entry in spec["inputs"]}
    sealed_content = {entry["path"]: entry["sha256"] for entry in spec["code"] + spec["inputs"]}
    # Check every sealed code dependency, not only the subset named by the fit
    # manifest. The manifest itself must still match each source it recorded.
    for relative_path, digest in code.items():
        local = ROOT / relative_path
        if not local.is_file() or sha256_file(local) != digest:
            raise ValueError(f"local source no longer matches sealed spec: {relative_path}")
    local_inputs: dict[str, str] = {}
    for relative_path, entry in sealed_inputs.items():
        local = ROOT / relative_path
        if not local.is_file() or local.stat().st_size != entry["size"]:
            raise ValueError(f"local input size differs from sealed spec: {relative_path}")
        if relative_path != "datasets/valid.parquet":
            digest = sha256_file(local)
            if digest != entry["sha256"]:
                raise ValueError(f"local input differs from sealed spec: {relative_path}")
            local_inputs[relative_path] = digest
    if set(report.get("source_sha256", {})) != EXPECTED_MANIFEST_SOURCES:
        raise ValueError("fit manifest source list differs from the fixed E476 dependency set")
    for relative_path, digest in report["source_sha256"].items():
        if sealed_content.get(relative_path) != digest:
            raise ValueError(f"fit source differs from sealed spec: {relative_path}")
        local = ROOT / relative_path
        if not local.is_file() or sha256_file(local) != digest:
            raise ValueError(f"local fit source no longer matches seal: {relative_path}")
    if (
        report.get("cohort_sha256")
        != sealed_inputs["research/campaign_2026-10-05/target_mechanics/cohort.json"]["sha256"]
    ):
        raise ValueError("fit cohort hash differs from sealed input")
    if sha256_file(COHORT_PATH) != report["cohort_sha256"]:
        raise ValueError("local cohort bytes differ from the recorded fit cohort")
    dataset_identity = sealed_inputs.get("datasets/valid.parquet")
    if dataset_identity is None or not DATA_PATH.is_file():
        raise ValueError("sealed validation dataset is missing")
    if DATA_PATH.stat().st_size != dataset_identity["size"]:
        raise ValueError("local validation dataset size differs from the fit seal")
    if report.get("dataset_sha256") != dataset_identity["sha256"]:
        raise ValueError("fit manifest dataset digest differs from the sealed input")
    # This full digest read is deliberately bounded-memory; the 5.6 GB parquet is an input artifact.
    local_data_hash = sha256_file(DATA_PATH)
    if local_data_hash != dataset_identity["sha256"]:
        raise ValueError("local validation dataset bytes differ from the sealed input")
    local_inputs["datasets/valid.parquet"] = local_data_hash
    expected_manifest_path, expected_trace_path = (ROOT / path for path in spec["artifacts"])
    if (
        expected_manifest_path != run / "fit_manifest.json"
        or expected_trace_path != run / "lr_traces.json"
    ):
        raise ValueError("sealed artifact paths do not match the run directory")
    if (
        not expected_trace_path.is_file()
        or json.loads(expected_manifest_path.read_text()) != report
    ):
        raise ValueError("sealed fit manifest or LR trace artifact is missing/mismatched")
    if len(report.get("completed_cells", [])) != len(ARMS):
        raise ValueError("not all 12 fixed cells are present")
    if report.get("seed") != SEED or report.get("parameters_per_cell") != EXPECTED_PARAMETERS:
        raise ValueError("seed or parameter count differs from the fixed design")
    if report.get("updates_per_fit") != UPDATES or report.get("sampling_steps_zero_based") != list(
        SAMPLE_STEPS
    ):
        raise ValueError("training update or telemetry design changed")
    fixed_manifest_fields = {
        "experiment": "E476 activation x optimizer x learning-rate protocol factorial",
        "cohort_rows": 24,
        "activations": list(ACTIVATIONS),
        "optimizers": list(OPTIMIZERS),
        "schedules": list(SCHEDULES),
        "batch": 4,
        "chunk_rows": CHUNK_ROWS,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "gradient_clip": 1.0,
    }
    for field, expected_value in fixed_manifest_fields.items():
        if report.get(field) != expected_value:
            raise ValueError(f"fit manifest field {field} differs from the fixed design")
    optimizer_details = {
        "adamw": "all parameters; betas=(0.9,0.999), wd=0.01",
        "muon_adamw": "Muon only on gru.weight_ih_l0, weight_hh_l0, weight_ih_l1, weight_hh_l1; 3 blocks each; AdamW on input/head/biases",
    }
    if report.get("optimizer_details") != optimizer_details:
        raise ValueError("optimizer scope differs from the fixed E476 design")
    precision = report.get("precision", {})
    if (
        precision.get("device_math") != "FP32 IEEE"
        or precision.get("autocast") is not False
        or precision.get("torch.backends.fp32_precision") != "ieee"
        or precision.get("torch.backends.cuda.matmul.fp32_precision") != "ieee"
        or precision.get("torch.backends.cudnn.fp32_precision") != "ieee"
        or precision.get("torch.backends.cudnn.rnn.fp32_precision") != "ieee"
    ):
        raise ValueError("fit manifest precision policy differs from sealed FP32 IEEE")
    return {
        "fit_spec_name": spec["name"],
        "fit_spec_version": spec["version"],
        "sealed_dataset_sha256": dataset_identity["sha256"],
        "sealed_dataset_bytes": dataset_identity["size"],
        "local_dataset_sha256": local_data_hash,
        "sealed_input_sha256": local_inputs,
        "sealed_artifact_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (expected_manifest_path, expected_trace_path)
        },
        "cohort_sha256": report["cohort_sha256"],
        "fit_source_sha256": report["source_sha256"],
    }


def validate_cohort_and_cells(
    run: Path, report: dict, traces: dict[str, np.ndarray]
) -> tuple[list[dict], list[dict], dict[tuple[str, str, str], dict]]:
    """Validate exact fit/test roles, seeded fit order, telemetry and checkpoint hashes."""
    source = json.loads(COHORT_PATH.read_text())
    nodes = source.get("cohort", [])
    fit_nodes = [node for node in nodes if node["role"] == "fit"]
    test_nodes = [node for node in nodes if node["role"] == "test"]
    if len(fit_nodes) != 24 or len(test_nodes) != 16:
        raise ValueError("frozen cohort must contain 24 fit and 16 test components")
    fit_components = {node["component"] for node in fit_nodes}
    test_components = {node["component"] for node in test_nodes}
    if len(fit_components) != 24 or len(test_components) != 16 or fit_components & test_components:
        raise ValueError("fit/test components overlap or are not unique")
    fit_row_groups = [node["row_group"] for node in fit_nodes]
    test_row_groups = [node["row_group"] for node in test_nodes]
    if len(set(fit_row_groups + test_row_groups)) != 40:
        raise ValueError("fit/test row groups overlap or are duplicated")
    expected_order = list(range(len(fit_nodes)))
    np.random.default_rng(SEED).shuffle(expected_order)
    expected_fit_order = [fit_nodes[index]["row_group"] for index in expected_order]
    actual_cells: dict[tuple[str, str, str], dict] = {}
    observed = set()
    for cell in report["completed_cells"]:
        key = (cell.get("activation"), cell.get("optimizer"), cell.get("schedule"))
        if key not in ARMS or key in observed:
            raise ValueError(f"unregistered or duplicate fitted cell: {key}")
        observed.add(key)
        if cell.get("parameters") != EXPECTED_PARAMETERS or cell.get("updates") != UPDATES:
            raise ValueError(f"wrong parameter/update count for {key}")
        if (
            cell.get("parameters_changed") is not True
            or cell.get("changed_parameter_tensors", 0) <= 0
        ):
            raise ValueError(f"cell did not update parameters: {key}")
        if cell.get("fit_row_groups") != expected_fit_order:
            raise ValueError(f"fit group order differs from the seed recipe: {key}")
        if set(cell["fit_row_groups"]) & set(test_row_groups):
            raise ValueError(f"test role appears in training provenance: {key}")
        schedule = key[2]
        if cell.get("lr_trace_sha256") != trace_digest(traces[schedule]):
            raise ValueError(f"cell LR hash mismatch: {key}")
        telemetry = cell.get("telemetry", [])
        if [entry.get("step_zero_based") for entry in telemetry] != list(SAMPLE_STEPS):
            raise ValueError(f"fixed telemetry samples missing or reordered: {key}")
        for entry in telemetry:
            step = int(entry["step_zero_based"])
            batch = step // 79
            chunk_start = (step % 79) * CHUNK_ROWS
            expected_group_batch = expected_fit_order[batch * 4 : (batch + 1) * 4]
            if entry.get("lr") != float(traces[schedule][step]):
                raise ValueError(f"telemetry LR differs from fixed trace: {key}, step {step}")
            gradient_norm = float(entry.get("gradient_norm_preclip", float("nan")))
            fit_loss = float(entry.get("fit_loss", float("nan")))
            if not math.isfinite(gradient_norm) or gradient_norm < 0:
                raise ValueError(f"invalid telemetry gradient norm: {key}, step {step}")
            if not math.isfinite(fit_loss) or fit_loss < 0:
                raise ValueError(f"invalid telemetry fit loss: {key}, step {step}")
            updates = entry.get("parameter_update_rms", {})
            if set(updates) != {"input_projection", "recurrent_core", "hidden_head", "output_head"}:
                raise ValueError(f"telemetry family set changed for {key}, step {step}")
            if any(
                not math.isfinite(float(value)) or float(value) < 0 for value in updates.values()
            ):
                raise ValueError(f"invalid telemetry parameter delta for {key}, step {step}")
            # The full traversal plus fixed step determines this exact batch and chunk.
            if (
                batch >= 6
                or len(expected_group_batch) != 4
                or chunk_start not in range(0, 20_000, CHUNK_ROWS)
            ):
                raise ValueError(
                    f"sampled telemetry step has invalid fit location: {key}, step {step}"
                )
        checkpoint_name = cell.get("checkpoint")
        expected_name = "__".join(key) + ".pt"
        if checkpoint_name != expected_name:
            raise ValueError(f"checkpoint name mismatch for {key}")
        checkpoint = run / checkpoint_name
        if not checkpoint.is_file() or sha256_file(checkpoint) != cell.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint missing or hash mismatch for {key}")
        actual_cells[key] = cell
    if observed != set(ARMS):
        raise ValueError("completed cell identities differ from the fixed 12-cell matrix")
    return fit_nodes, test_nodes, actual_cells


def verify_raw_selected_targets(
    test_nodes: list[dict], targets: np.ndarray, masks: np.ndarray
) -> None:
    """Re-read test labels and prove selected raw values are finite before clipping."""
    parquet = pq.ParquetFile(DATA_PATH)
    for index, node in enumerate(test_nodes):
        table = parquet.read_row_group(
            node["row_group"],
            columns=["t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        raw = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")])
        raw = np.asarray(raw, dtype=np.float64)
        mask = table["need_prediction"].to_numpy().astype(bool)
        mask &= table["is_scored"].to_numpy().astype(bool)
        if not np.array_equal(mask, masks[index]):
            raise ValueError(f"official selection mask changed for row group {node['row_group']}")
        if not mask.any() or not np.isfinite(raw[mask]).all():
            raise ValueError(f"selected raw targets are empty or nonfinite in {node['row_group']}")
        expected = np.clip(raw[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float32)
        if not np.array_equal(expected, targets[index][mask]):
            raise ValueError(f"clipped target export differs from source for {node['row_group']}")


def make_model(activation: str) -> RNNModel:
    """Independently instantiate the exact E476 base or sine-embedding model."""
    model_type = RNNModel if activation == "base" else SineEmbeddingRNN
    if activation not in ACTIVATIONS:
        raise ValueError(f"unregistered activation: {activation}")
    model = model_type(42, hidden=64, layers=2, dropout=0.0, n_out=2, cell="gru", enc="linear")
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise ValueError("reconstructed E476 model does not have 61,058 parameters")
    return model


def predict_test_sequences(
    checkpoint: Path,
    activation: str,
    x_test: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Stream each test component in order, resetting state and checking all outputs."""
    model = make_model(activation).to(device).float()
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    if any(not torch.isfinite(value).all() for value in state_dict.values()):
        raise ValueError(f"checkpoint contains nonfinite parameters: {checkpoint.name}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    predictions = np.empty((len(x_test), x_test.shape[1], 2), dtype=np.float32)
    precision_context = (
        e474.full_ieee_float32() if device.type == "cuda" else contextlib.nullcontext()
    )
    with precision_context, torch.no_grad():
        for component_index, sequence in enumerate(x_test):
            state = None
            parts = []
            for start in range(0, sequence.shape[0], CHUNK_ROWS):
                chunk = torch.from_numpy(sequence[start : start + CHUNK_ROWS][None]).to(
                    device=device, dtype=torch.float32
                )
                output, state = model(chunk, state)
                if not torch.isfinite(output).all() or not torch.isfinite(state).all():
                    raise ValueError(f"nonfinite full sequence in {checkpoint.name}")
                parts.append(output[0].cpu().numpy())
            predictions[component_index] = np.concatenate(parts, axis=0)
    if not np.isfinite(predictions).all():
        raise ValueError(f"nonfinite full-sequence output in {checkpoint.name}")
    return predictions


def pool_component_moments(moments: np.ndarray, multiplicities: np.ndarray) -> float:
    """Pool centered official moments under a component-resample count vector."""
    pooled = np.zeros((2, 6), dtype=np.float64)
    for index, multiplicity in enumerate(multiplicities):
        count = int(multiplicity)
        if count == 0:
            continue
        for target_index in range(2):
            contribution = moments[index, target_index].copy()
            contribution[[0, 3, 4, 5]] *= count
            pooled[target_index] = _merge_moments(pooled[target_index], contribution)
    values = [_wp_from_moments(pooled[target_index]) for target_index in range(2)]
    return float(np.mean(values))


def interaction_bootstrap(
    moments: dict[str, np.ndarray], components: np.ndarray, keys: tuple[str, str, str, str]
) -> dict:
    """Bootstrap an activation-by-schedule difference-of-differences jointly."""
    sorted_order = np.argsort(components, kind="stable")
    sorted_moments = {key: moments[key][sorted_order] for key in keys}
    count = len(components)
    draws = np.empty(BOOTSTRAP_COUNT, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for index in range(BOOTSTRAP_COUNT):
        selected = rng.integers(0, count, count)
        multiplicities = np.bincount(selected, minlength=count)
        wp_values = {
            key: pool_component_moments(value, multiplicities)
            for key, value in sorted_moments.items()
        }
        sine_cos, base_cos, sine_mean, base_mean = (wp_values[key] for key in keys)
        draws[index] = (sine_cos - base_cos) - (sine_mean - base_mean)
    point_scores = {
        key: pool_component_moments(value, np.ones(count, dtype=np.int64))
        for key, value in sorted_moments.items()
    }
    sine_cos, base_cos, sine_mean, base_mean = (point_scores[key] for key in keys)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "mean_diff": float((sine_cos - base_cos) - (sine_mean - base_mean)),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "ci95": [float(lo), float(hi)],
        "p_le_0": float(np.mean(draws <= 0)),
        "n_components": int(count),
        "n_boot": BOOTSTRAP_COUNT,
        "seed": BOOTSTRAP_SEED,
        "metric_id": METRIC_ID,
        "joint_component_resampling": True,
    }


def main() -> None:
    """Validate all sealed E476 cells, score the test cohort, and export contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempt-record", type=Path, default=DEFAULT_ATTEMPT_RECORD)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scorer output: {output}")
    sealed_spec = json.loads(SEALED_FIT_SPEC.read_text())
    attempt_receipt = validate_attempt_record(args.attempt_record.resolve(), sealed_spec, run)
    report = json.loads((run / "fit_manifest.json").read_text())
    data_identity = validate_sealed_identity(run, report)
    traces = validate_lr_traces(run, report)
    fit_nodes, test_nodes, cells = validate_cohort_and_cells(run, report, traces)
    if len(fit_nodes) != 24 or len(test_nodes) != 16 or len(cells) != 12:
        raise AssertionError("fixed fit/test/cell counts changed")
    if args.device == "cuda":
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA scoring requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(1)

    manifest, all_x, all_y, all_need, all_score, loaded_nodes = e470.load_cohort(include_test=True)
    original_cohort = json.loads(COHORT_PATH.read_text())["cohort"]
    if manifest.get("cohort") != original_cohort:
        raise ValueError("scoring loader cohort differs from frozen source nodes")
    if [node["row_group"] for node in loaded_nodes if node["role"] == "test"] != [
        node["row_group"] for node in test_nodes
    ]:
        raise ValueError("test row-group order differs from the frozen cohort")
    test_indices = [index for index, node in enumerate(loaded_nodes) if node["role"] == "test"]
    x_test = all_x[test_indices]
    target_test = all_y[test_indices]
    score_test = all_score[test_indices]
    if x_test.shape != (16, 20_000, 42) or target_test.shape != (16, 20_000, 2):
        raise ValueError("unexpected test feature/target shape")
    verify_raw_selected_targets(test_nodes, target_test, score_test)
    counts = score_test.sum(axis=1).astype(np.int64)
    if np.any(counts <= 0):
        raise ValueError("every fixed test component must contribute selected rows")
    component_ids = np.asarray([node["component"] for node in test_nodes])
    if len(np.unique(component_ids)) != 16:
        raise ValueError("component bootstrap requires 16 unique component identities")

    scored: dict[str, dict] = {}
    moments_by_arm: dict[str, np.ndarray] = {}
    selected_predictions: dict[str, np.ndarray] = {}
    common_targets = target_test[score_test]
    for activation, optimizer_name, schedule in ARMS:
        arm = "__".join((activation, optimizer_name, schedule))
        cell = cells[(activation, optimizer_name, schedule)]
        full_prediction = predict_test_sequences(
            run / cell["checkpoint"], activation, x_test, device
        )
        e470.ensure_finite_prediction(full_prediction, arm)
        selected = full_prediction[score_test]
        if not np.isfinite(selected).all():
            raise ValueError(f"nonfinite selected prediction for {arm}")
        official = global_wp(
            common_targets[None],
            selected[None],
            np.ones((1, len(common_targets)), dtype=bool),
        )
        if official["metric_id"] != METRIC_ID or official["selected_rows"] != int(counts.sum()):
            raise ValueError(f"official metric identity/count mismatch for {arm}")
        arm_moments, _ = paired_component_moments(
            target_test, full_prediction, full_prediction, score_test
        )
        moments_by_arm[arm] = arm_moments
        selected_predictions[arm] = selected.copy()
        scored[arm] = {
            "activation": activation,
            "optimizer": optimizer_name,
            "schedule": schedule,
            "wp": float(official["wp"]),
            "t0": float(official["t0"]),
            "t1": float(official["t1"]),
            "selected_rows": int(official["selected_rows"]),
            "parameters": EXPECTED_PARAMETERS,
            "updates": UPDATES,
            "sequence_row_counts": counts.tolist(),
        }

    contrasts: dict[str, dict] = {}
    for candidate, reference, family in PAIR_CONTRASTS:
        estimate = paired_component_bootstrap(
            moments_by_arm[candidate],
            moments_by_arm[reference],
            component_ids,
            n_boot=BOOTSTRAP_COUNT,
            seed=BOOTSTRAP_SEED,
            centered=True,
        )
        point_candidate = scored[candidate]["wp"]
        point_reference = scored[reference]["wp"]
        if not np.isclose(estimate["candidate_wp"], point_candidate, atol=1e-7, rtol=0):
            raise ValueError(f"paired bootstrap candidate score mismatch: {candidate}")
        if not np.isclose(estimate["reference_wp"], point_reference, atol=1e-7, rtol=0):
            raise ValueError(f"paired bootstrap reference score mismatch: {reference}")
        contrasts[f"{candidate}_minus_{reference}"] = {
            "family": family,
            "candidate": candidate,
            "reference": reference,
            **estimate,
        }

    interactions = {}
    for optimizer_name in OPTIMIZERS:
        keys = (
            f"sine_embed__{optimizer_name}__warmcos_47_427",
            f"base__{optimizer_name}__warmcos_47_427",
            f"sine_embed__{optimizer_name}__constant_mean_warmcos",
            f"base__{optimizer_name}__constant_mean_warmcos",
        )
        effect = interaction_bootstrap(moments_by_arm, component_ids, keys)
        expected_interaction = float(
            (scored[keys[0]]["wp"] - scored[keys[1]]["wp"])
            - (scored[keys[2]]["wp"] - scored[keys[3]]["wp"])
        )
        if not np.isclose(effect["mean_diff"], expected_interaction, atol=1e-7, rtol=0):
            raise ValueError("interaction point estimate differs from official arm scores")
        interactions[f"sine_activation_x_warmcos_shape__{optimizer_name}"] = {
            "family": "activation_by_schedule_shape_interaction",
            "optimizer": optimizer_name,
            "contrast": f"({keys[0]} - {keys[1]}) - ({keys[2]} - {keys[3]})",
            **effect,
        }
    if len(contrasts) != 20 or len(interactions) != 2:
        raise AssertionError("fixed 20 pair contrasts and 2 interactions changed")

    output.mkdir(parents=True)
    arm_order = ["__".join(arm) for arm in ARMS]
    np.savez_compressed(
        output / "selected_exports.npz",
        arm_names=np.asarray(arm_order),
        targets=common_targets,
        predictions=np.stack([selected_predictions[arm] for arm in arm_order]),
        row_counts=counts,
        component_ids=component_ids.astype(str),
        row_groups=np.asarray([node["row_group"] for node in test_nodes], dtype=np.int64),
    )
    result = {
        "kind": "independent E476 activation-schedule scorer",
        "metric": METRIC_ID,
        "scope": "one fixed optimizer seed and 16 previously exposed diagnostic components; no seed pooling or public forecast",
        "fit_seed": SEED,
        "fit_result_sha256": sha256_file(run / "fit_manifest.json"),
        "lr_trace_receipt_sha256": sha256_file(run / "lr_traces.json"),
        "runner_attempt_receipt": attempt_receipt,
        "sealed_identity": data_identity,
        "cohort_sha256": report["cohort_sha256"],
        "source_sha256": {
            "scorer": sha256_file(Path(__file__)),
            "official_gmetric": sha256_file(ROOT / "src/wnn/gmetric.py"),
            "official_evaluation": sha256_file(ROOT / "src/wnn/evaluation.py"),
            "fit_manifest_sources": report["source_sha256"],
        },
        "component_ids": component_ids.tolist(),
        "row_groups": [node["row_group"] for node in test_nodes],
        "selected_rows": int(counts.sum()),
        "sequence_row_counts": counts.tolist(),
        "arm_count": len(scored),
        "arm_scores": scored,
        "pair_contrast_count": len(contrasts),
        "pair_contrasts": contrasts,
        "interaction_count": len(interactions),
        "interactions": interactions,
        "bootstrap": {
            "helper": "wnn.evaluation.paired_component_bootstrap with centered Welford moments",
            "interaction_pooler": "wnn.gmetric centered-moment merge and official weighted-Pearson primitive",
            "resampling_unit": "component identity; paired identical component multiplicities across arms",
            "n_boot": BOOTSTRAP_COUNT,
            "seed": BOOTSTRAP_SEED,
            "cohort_note": "all 16 components were exposed previously; exploratory reporting only",
        },
        "selected_export_sha256": None,
    }
    result["selected_export_sha256"] = sha256_file(output / "selected_exports.npz")
    (output / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "arms": len(scored),
                "contrasts": len(contrasts) + len(interactions),
                "selected_rows": int(counts.sum()),
            }
        )
    )


if __name__ == "__main__":
    main()

"""Independent E474 checkpoint exporter and component-paired grader.

This script is intentionally separate from the fit actor. It refuses to score
unless the campaign attempt exited successfully, proved child teardown, and
still matches its sealed spec and declared artifacts. It reconstructs the
registered RNNs from ``wnn.nn.RNNModel`` and never imports the actor's model
class or selects a fit based on exposed test outcomes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))

import fit_activation_optimizer as e470  # noqa: E402

from wnn.campaign import digest, runtime_identity  # noqa: E402
from wnn.campaign_spec import content_file, validate_spec, verify_files  # noqa: E402
from wnn.gmetric import (  # noqa: E402
    METRIC_CLIP,
    METRIC_ID,
    _centered_moments,
    _merge_moments,
    _wp_from_moments,
    global_wp,
)
from wnn.nn import RNNModel  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
DATA_PATH = ROOT / "datasets/valid.parquet"
EXPECTED_COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
EXPECTED_DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
SEEDS = (271828, 271829)
FEATURES = 42
HIDDEN = 64
LAYERS = 2
PARAMETERS = 61_058
ROWS_PER_GROUP = 20_000
CHUNK_ROWS = 256
UPDATES = 474
TELEMETRY_STEPS = (0, 46, 236, 473)
BOOTSTRAP_COUNT = 2_000
BOOTSTRAP_SEED = 474_271_828
ARMS = (
    "base",
    "sine_embed",
    "slope_embed",
    "sine_head",
    "slope_head",
    "sine_both",
)
ARM_FORMULAS = {
    "base": ("gelu", "gelu"),
    "sine_embed": ("sine", "gelu"),
    "slope_embed": ("slope", "gelu"),
    "sine_head": ("gelu", "sine"),
    "slope_head": ("gelu", "slope"),
    "sine_both": ("sine", "sine"),
}
PAIRED_CONTRASTS = (
    ("sine_embed", "base", "sine_embed_minus_base"),
    ("slope_embed", "base", "slope_embed_minus_base"),
    ("sine_embed", "slope_embed", "sine_embed_minus_slope_embed"),
    ("sine_head", "base", "sine_head_minus_base"),
    ("slope_head", "base", "slope_head_minus_base"),
    ("sine_head", "slope_head", "sine_head_minus_slope_head"),
)
EXPECTED_ACTIVATION_FORMULAS = {
    "gelu": "F.gelu(x, approximate='none')",
    "sine": "F.gelu(x, approximate='none') + 0.1*sin(x)",
    "slope": "F.gelu(x, approximate='none') + 0.1*x",
}


def sha256_file(path: Path) -> str:
    """Hash file contents with bounded memory."""
    digest_value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest_value.update(block)
    return digest_value.hexdigest()


class ScoringActivation(nn.Module):
    """Independent pointwise E474 activation formulas with no state."""

    def __init__(self, formula: str) -> None:
        super().__init__()
        if formula not in EXPECTED_ACTIVATION_FORMULAS:
            raise ValueError(f"unknown registered activation: {formula}")
        self.formula = formula

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = F.gelu(value, approximate="none")
        if self.formula == "gelu":
            return base
        if self.formula == "sine":
            return base + 0.1 * torch.sin(value)
        if self.formula == "slope":
            return base + 0.1 * value
        raise AssertionError("activation dispatch is exhaustive")


class ScoringRNN(RNNModel):
    """Independent RNNModel reconstruction for one registered placement arm."""

    def __init__(self, embedding_formula: str, head_formula: str) -> None:
        super().__init__(
            n_in=FEATURES,
            hidden=HIDDEN,
            layers=LAYERS,
            dropout=0.0,
            n_out=2,
            cell="gru",
            enc="linear",
        )
        self.embedding_formula = embedding_formula
        self.head_formula = head_formula
        self.embedding_activation = ScoringActivation(embedding_formula)
        self.head[1] = ScoringActivation(head_formula)

    def encode(self, normalized_input: torch.Tensor) -> torch.Tensor:
        return self.embedding_activation(self.inp(normalized_input))


@contextlib.contextmanager
def full_ieee_float32(device: torch.device) -> Iterator[None]:
    """Match the E474 FP32 IEEE scoring arithmetic without importing its model."""
    if device.type != "cuda":
        with torch.autocast(device_type="cpu", enabled=False):
            yield
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA scoring requested, but CUDA is unavailable")
    backend_settings = (
        (torch.backends, "fp32_precision"),
        (torch.backends.cuda.matmul, "fp32_precision"),
        (torch.backends.cudnn, "fp32_precision"),
        (torch.backends.cudnn.rnn, "fp32_precision"),
    )
    if not all(hasattr(owner, name) for owner, name in backend_settings):
        raise RuntimeError("this scorer requires PyTorch per-backend FP32 precision controls")
    previous = [getattr(owner, name) for owner, name in backend_settings]
    try:
        for owner, name in backend_settings:
            setattr(owner, name, "ieee")
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        for (owner, name), value in zip(backend_settings, previous, strict=True):
            setattr(owner, name, value)


def expected_arm_rows() -> list[dict[str, Any]]:
    """Return the exact frozen seed-major arm ordering."""
    return [
        {
            "seed": seed,
            "arm": arm,
            "embedding_activation": ARM_FORMULAS[arm][0],
            "head_activation": ARM_FORMULAS[arm][1],
        }
        for seed in SEEDS
        for arm in ARMS
    ]


def _check_mem_events(initial: dict, final: dict, label: str) -> None:
    """Reject incomplete OOM receipts or increased worker OOM counters."""
    keys = ("oom", "oom_kill", "oom_group_kill")
    if not all(key in initial and key in final for key in keys):
        raise ValueError(f"{label} memory-event receipt is incomplete")
    if any(final[key] > initial[key] for key in keys):
        raise ValueError(f"{label} observed an OOM event during the fit")


def validate_artifact_records(root: Path, declared: set[str], records: object) -> dict[str, dict]:
    """Require unique runner-shaped artifact receipts matching current bytes."""
    if not isinstance(records, list):
        raise ValueError("successful attempt lacks artifact identities")
    by_path = {item.get("path"): item for item in records if isinstance(item, dict)}
    if len(by_path) != len(records) or set(by_path) != declared:
        raise ValueError("attempt artifact identities differ from the sealed artifact list")
    for relative in sorted(declared):
        actual = content_file(root, relative)
        if actual != by_path[relative]:
            raise ValueError(f"post-fit artifact differs from attempt receipt: {relative}")
    return by_path


def validate_attempt_record(attempt_path: Path, run_dir: Path) -> tuple[dict, dict]:
    """Require a successful, torn-down GPU attempt and verify its sealed files."""
    record = json.loads(attempt_path.read_text())
    if not isinstance(record, dict) or record.get("id") != attempt_path.stem:
        raise ValueError("attempt filename and recorded ID differ")
    spec = validate_spec(record.get("spec"))
    if record.get("spec_sha256") != digest(spec):
        raise ValueError("attempt spec hash does not match its embedded sealed spec")
    if record.get("kind") != "gpu" or spec["kind"] != "gpu":
        raise ValueError("E474 full fit must be bound to a GPU campaign attempt")
    if record.get("runtime_sha256") != runtime_identity(ROOT):
        raise ValueError(
            "attempt runtime file-hash map differs from current sealed campaign runtime"
        )
    if (
        record.get("status") != "finished"
        or record.get("exit_code") != 0
        or record.get("failure") is not None
        or record.get("signals") != []
        or record.get("teardown_verified") is not True
    ):
        raise ValueError("campaign attempt did not finish cleanly with verified teardown")
    if (
        not isinstance(record.get("peak_observed_vram_bytes"), int)
        or record["peak_observed_vram_bytes"] >= spec["limits"]["gpu_stop_bytes"]
        or not isinstance(record.get("peak_ram_bytes"), int)
        or record["peak_ram_bytes"] > spec["limits"]["ram_bytes"]
    ):
        raise ValueError("attempt resource peaks are missing or exceed sealed limits")
    _check_mem_events(
        record.get("initial_memory_events", {}),
        record.get("final_memory_events", {}),
        "worker",
    )
    _check_mem_events(
        record.get("initial_parent_memory_events", {}),
        record.get("final_parent_memory_events", {}),
        "aggregate parent",
    )

    argv = spec["argv"]
    actor_relpath = "research/campaign_2026-10-05/activation_placement/fit_activation_placement.py"
    if actor_relpath not in argv and str(ROOT / actor_relpath) not in argv:
        raise ValueError("sealed command does not invoke the E474 fit actor")
    try:
        output_arg = argv[argv.index("--output") + 1]
    except (ValueError, IndexError) as error:
        raise ValueError("sealed E474 command has no --output directory") from error
    sealed_run = Path(output_arg)
    if not sealed_run.is_absolute():
        sealed_run = ROOT / sealed_run
    if sealed_run.resolve() != run_dir.resolve():
        raise ValueError("requested fit directory differs from the sealed command output")

    declared_artifacts = set(spec["artifacts"])
    manifest_path = str((run_dir / "fit_manifest.json").resolve().relative_to(ROOT.resolve()))
    if declared_artifacts != {manifest_path}:
        raise ValueError("E474 attempt must declare exactly its sealed fit_manifest.json output")
    validate_artifact_records(ROOT, declared_artifacts, record.get("artifacts"))

    # Verify every content-bound source and input, including the multi-GB parquet,
    # with a streaming hash before interpreting any checkpoint or test prediction.
    verify_files(ROOT, [*spec["code"], *spec["inputs"]])
    return record, spec


def validate_sealed_inputs(spec: dict, report: dict) -> dict:
    """Bind report, cohort, dataset, and actor hashes before fit-only reconstruction."""
    code_by_path = {entry["path"]: entry for entry in spec["code"]}
    input_by_path = {entry["path"]: entry for entry in spec["inputs"]}
    actor_paths = {
        "research/campaign_2026-10-05/activation_placement/fit_activation_placement.py": "fit_activation_placement.py",
        "research/campaign_2026-10-05/activation_placement/tests.py": "tests.py",
        "research/campaign_2026-10-05/activation_placement/gpu_probe.py": "gpu_probe.py",
    }
    required_code = {*actor_paths, "src/wnn/nn.py"}
    if not required_code <= set(code_by_path):
        raise ValueError("sealed code must include E474 actor/tests/probe and wnn.nn architecture")
    expected_actor_hashes = {
        local_name: code_by_path[path]["sha256"] for path, local_name in actor_paths.items()
    }
    if report.get("source_sha256") != expected_actor_hashes:
        raise ValueError("fit manifest source hashes differ from the sealed E474 code")
    recipe_path = "research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py"
    if (
        recipe_path not in code_by_path
        or report.get("recipe_source_sha256") != code_by_path[recipe_path]["sha256"]
    ):
        raise ValueError("fit feature/objective source hash differs from sealed E470 source")
    if "datasets/valid.parquet" not in input_by_path:
        raise ValueError("sealed fit spec does not bind datasets/valid.parquet")
    if input_by_path["datasets/valid.parquet"]["sha256"] != EXPECTED_DATASET_SHA256:
        raise ValueError("sealed validation dataset differs from E470 cohort data")
    if report.get("dataset_sha256") != input_by_path["datasets/valid.parquet"]["sha256"]:
        raise ValueError("fit manifest dataset hash differs from sealed input")
    if "research/campaign_2026-10-05/target_mechanics/cohort.json" not in input_by_path:
        raise ValueError("sealed fit spec does not bind the frozen cohort")
    cohort_identity = input_by_path["research/campaign_2026-10-05/target_mechanics/cohort.json"]
    if cohort_identity["sha256"] != EXPECTED_COHORT_SHA256:
        raise ValueError("sealed cohort differs from the fixed E474 cohort")
    if report.get("cohort_sha256") != cohort_identity["sha256"]:
        raise ValueError("fit manifest cohort hash differs from sealed cohort")

    return {
        "dataset": input_by_path["datasets/valid.parquet"],
        "cohort": cohort_identity,
        "source_sha256": expected_actor_hashes,
    }


def validate_fit_manifest(
    report: dict, run_dir: Path, fit_nodes: list[dict], spec: dict
) -> list[dict]:
    """Require all registered fits, exact group order, telemetry, and checkpoints."""
    expected_top_level = {
        "kind": "twelve-fit localized activation placement screen",
        "status": "all_12_fit_arms_complete_test_unread",
        "fit_only": True,
        "test_rows_loaded": False,
        "seeds": list(SEEDS),
        "hidden": HIDDEN,
        "trainable_parameters_per_arm": PARAMETERS,
        "input_features": FEATURES,
        "batch": 4,
        "chunk_rows": CHUNK_ROWS,
        "updates_per_fit": UPDATES,
    }
    for key, expected in expected_top_level.items():
        if report.get(key) != expected:
            raise ValueError(f"fit manifest {key!r} is {report.get(key)!r}, expected {expected!r}")
    if report.get("optimizer") != {
        "name": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "schedule": "constant",
        "gradient_clip": 1.0,
    }:
        raise ValueError(
            "fit optimizer/training objective recipe differs from E474 preregistration"
        )
    if report.get("planned_arms") != expected_arm_rows():
        raise ValueError("planned arm matrix or seed-major order differs from E474 preregistration")
    telemetry_protocol = report.get("telemetry_protocol", {})
    if telemetry_protocol.get("steps_zero_based") != list(TELEMETRY_STEPS):
        raise ValueError("fit telemetry step indices differ from the preregistration")
    if telemetry_protocol.get("families") != ["inp", "gru", "head"]:
        raise ValueError("fit telemetry parameter families differ from preregistration")

    source_rows = [node["row_group"] for node in fit_nodes]
    test_nodes = [
        node for node in json.loads(COHORT_PATH.read_text())["cohort"] if node["role"] == "test"
    ]
    test_groups = {node["row_group"] for node in test_nodes}
    if len(test_nodes) != 16 or len({node["component"] for node in test_nodes}) != 16:
        raise ValueError("fixed exposed test panel is not 16 distinct components")
    if test_groups & set(source_rows):
        raise ValueError("fit/test row-group leakage in the frozen cohort")
    rows = report.get("arms")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("partial E474 fit matrix refused")
    expected = expected_arm_rows()
    validated = []
    for summary, plan in zip(rows, expected, strict=True):
        for key, value in plan.items():
            if summary.get(key) != value:
                raise ValueError(f"fit arm order/identity mismatch at {plan}")
        seed = plan["seed"]
        shuffled = list(range(24))
        np.random.default_rng(seed).shuffle(shuffled)
        expected_order = [fit_nodes[index]["row_group"] for index in shuffled]
        if summary.get("fit_row_groups") != expected_order:
            raise ValueError(f"fit row-group order differs from seeded training for {plan}")
        if set(summary["fit_row_groups"]) & test_groups:
            raise ValueError(f"fit arm includes a test row group: {plan}")
        if summary.get("parameters") != PARAMETERS or summary.get("updates") != UPDATES:
            raise ValueError(f"parameter count or 474-update invariant failed: {plan}")
        elapsed = summary.get("fit_seconds")
        if not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"invalid fit duration in {plan}")
        expected_formula = {
            "embedding": EXPECTED_ACTIVATION_FORMULAS[plan["embedding_activation"]],
            "head": EXPECTED_ACTIVATION_FORMULAS[plan["head_activation"]],
        }
        if summary.get("activation_formulas") != expected_formula:
            raise ValueError(f"activation formula metadata differs in {plan}")
        sampling = summary.get("telemetry_sampling_rule", {})
        if sampling.get("steps_zero_based") != list(TELEMETRY_STEPS):
            raise ValueError(f"telemetry indices differ in {plan}")
        telemetry = summary.get("telemetry")
        if not isinstance(telemetry, list) or [
            entry.get("update_index_zero_based") for entry in telemetry
        ] != list(TELEMETRY_STEPS):
            raise ValueError(f"missing or reordered telemetry samples in {plan}")
        for entry in telemetry:
            step = entry["update_index_zero_based"]
            batch_index, offset = divmod(step, 79)
            expected_batch = expected_order[batch_index * 4 : (batch_index + 1) * 4]
            if (
                entry.get("fit_row_groups") != expected_batch
                or entry.get("chunk_start") != offset * CHUNK_ROWS
            ):
                raise ValueError(
                    f"telemetry row-group/chunk provenance mismatch in {plan} at {step}"
                )
            if set(entry.get("preclip_gradient_rms", {})) != {"inp", "gru", "head"}:
                raise ValueError(f"telemetry gradient families differ in {plan} at {step}")
            if set(entry.get("parameter_update_rms", {})) != {"inp", "gru", "head"}:
                raise ValueError(f"telemetry update families differ in {plan} at {step}")
            values = [
                *entry["preclip_gradient_rms"].values(),
                *entry["parameter_update_rms"].values(),
            ]
            if any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                for value in values
            ):
                raise ValueError(f"invalid telemetry values in {plan} at {step}")

        checkpoint_name = summary.get("checkpoint")
        expected_name = f"seed{seed}_{plan['arm']}.pt"
        if checkpoint_name != expected_name or Path(checkpoint_name).name != checkpoint_name:
            raise ValueError(f"unexpected checkpoint path in {plan}")
        checkpoint_path = run_dir / checkpoint_name
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != summary.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"checkpoint is absent or SHA-256 differs from fit manifest: {plan}")
        validated.append({**summary, "checkpoint_path": checkpoint_path})

    if (
        report.get("precision", {}).get("autocast") is not False
        or report.get("precision", {}).get("device_math") != "FP32 IEEE"
    ):
        raise ValueError("full fit precision receipt is missing or not IEEE FP32")
    return validated


def reconstruct_model(arm: str, checkpoint: Path, device: torch.device) -> ScoringRNN:
    """Strict-load one independently constructed registered E474 model."""
    if arm not in ARM_FORMULAS:
        raise ValueError(f"unregistered activation-placement arm: {arm}")
    model = ScoringRNN(*ARM_FORMULAS[arm])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is not a state dictionary: {checkpoint.name}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"checkpoint layout mismatch: missing={missing}, unexpected={unexpected}")
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETERS:
        raise ValueError("independent scorer model does not have 61,058 trainable parameters")
    if any(
        not torch.isfinite(tensor).all().item()
        for tensor in state.values()
        if torch.is_floating_point(tensor)
    ):
        raise ValueError(f"checkpoint contains nonfinite state: {checkpoint.name}")
    if not torch.equal(model.mean, torch.zeros_like(model.mean)) or not torch.equal(
        model.std, torch.ones_like(model.std)
    ):
        raise ValueError(
            "checkpoint normalization buffers are not identity; refusing double normalization"
        )
    return model.to(device=device, dtype=torch.float32).eval()


def check_base_parity(checkpoint: Path) -> dict[str, float]:
    """Verify the independent base reconstruction against unmodified RNNModel."""
    scored = ScoringRNN("gelu", "gelu")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    scored.load_state_dict(state, strict=True)
    reference = RNNModel(
        n_in=FEATURES,
        hidden=HIDDEN,
        layers=LAYERS,
        dropout=0.0,
        n_out=2,
        cell="gru",
        enc="linear",
    )
    reference.load_state_dict(state, strict=True)
    scored.eval()
    reference.eval()
    values = torch.linspace(-2.0, 2.0, steps=2 * 37 * FEATURES, dtype=torch.float32).reshape(
        2, 37, FEATURES
    )
    with torch.inference_mode():
        score_pred, score_state = scored(values)
        ref_pred, ref_state = reference(values)
    torch.testing.assert_close(score_pred, ref_pred, atol=0.0, rtol=0.0)
    torch.testing.assert_close(score_state, ref_state, atol=0.0, rtol=0.0)
    return {
        "max_prediction_abs": float((score_pred - ref_pred).abs().max()),
        "max_state_abs": float((score_state - ref_state).abs().max()),
    }


def load_test_panel(
    fit_nodes: list[dict], fit_only: dict, test_data_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Read only the fixed 16 test groups and reproduce E470 causal normalization."""
    cohort = json.loads(COHORT_PATH.read_text())
    nodes = [node for node in cohort["cohort"] if node["role"] == "test"]
    if len(fit_nodes) != 24 or len(nodes) != 16:
        raise ValueError("expected the fixed fit-only 24/16 component split")
    fit_components = {node["component"] for node in fit_nodes}
    test_components = {node["component"] for node in nodes}
    if fit_components & test_components or len(test_components) != 16:
        raise ValueError("fit and test components overlap or test components are duplicated")
    row_groups = [node["row_group"] for node in nodes]
    if len(set(row_groups)) != 16:
        raise ValueError("test row groups are not unique")

    mean = np.asarray(fit_only["normalization"]["fit_mean"], dtype=np.float32)
    scale = np.asarray(fit_only["normalization"]["fit_scale"], dtype=np.float32)
    parquet = pq.ParquetFile(test_data_path)
    xs, targets, masks = [], [], []
    for node in nodes:
        table = parquet.read_row_group(
            node["row_group"],
            columns=[*e470.FEATURES, "t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        if table.num_rows != ROWS_PER_GROUP:
            raise ValueError(f"wrong row count in test row group {node['row_group']}")
        raw_features = np.column_stack([table[column].to_numpy() for column in e470.FEATURES])
        causal = e470.causal_features(raw_features)
        normalized = np.clip((causal - mean) / scale, -8.0, 8.0).astype(np.float32)
        raw_target = np.column_stack([table["t0"].to_numpy(), table["t1"].to_numpy()])
        try:
            raw_target = np.asarray(raw_target, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("test target columns must be numeric") from error
        selected = table["need_prediction"].to_numpy().astype(bool)
        selected &= table["is_scored"].to_numpy().astype(bool)
        if not selected.any() or not np.isfinite(raw_target[selected]).all():
            raise ValueError(
                f"invalid selected labels or empty score mask in row group {node['row_group']}"
            )
        if not np.isfinite(normalized).all():
            raise ValueError(
                f"causal normalized test features are nonfinite in row group {node['row_group']}"
            )
        xs.append(normalized)
        targets.append(raw_target)
        masks.append(selected)
    return np.stack(xs), np.stack(targets), np.stack(masks), nodes


def finite_prediction(values: np.ndarray, arm: str) -> np.ndarray:
    """Reject nonfinite raw full-sequence predictions before any metric clipping."""
    try:
        prediction = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"raw prediction for {arm} is not numeric") from error
    if prediction.shape[-1] != 2 or not np.isfinite(prediction).all():
        raise ValueError(f"raw full-sequence prediction for {arm} is invalid before clipping")
    return prediction


def predict_test_panel(
    model: ScoringRNN, features: np.ndarray, device: torch.device, arm_key: str
) -> np.ndarray:
    """Stream each test component in 256-row chunks with an explicit state reset."""
    if features.shape != (16, ROWS_PER_GROUP, FEATURES):
        raise ValueError(f"unexpected normalized test tensor shape: {features.shape}")
    output = np.empty((16, ROWS_PER_GROUP, 2), dtype=np.float32)
    with torch.inference_mode():
        for batch_start in range(0, 16, 4):
            ids = list(range(batch_start, min(batch_start + 4, 16)))
            state = None
            for position in range(0, ROWS_PER_GROUP, CHUNK_ROWS):
                xx = torch.from_numpy(features[ids, position : position + CHUNK_ROWS]).to(
                    device=device, dtype=torch.float32
                )
                prediction, state = model(xx, state)
                if not torch.isfinite(prediction).all() or not torch.isfinite(state).all():
                    raise ValueError(f"nonfinite raw model output in {arm_key} at row {position}")
                output[ids, position : position + CHUNK_ROWS] = prediction.detach().cpu().numpy()
                state = state.detach()
    return finite_prediction(output, arm_key)


def component_moments(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build stable centered weighted-Pearson moments for each test component."""
    if targets.shape != predictions.shape or targets.shape != (16, ROWS_PER_GROUP, 2):
        raise ValueError("component targets and predictions must align as (16, 20000, 2)")
    if mask.shape != (16, ROWS_PER_GROUP) or not mask.any():
        raise ValueError("component scoring mask has an invalid shape or no selected rows")
    result = np.zeros((16, 2, 6), dtype=np.float64)
    for index in range(16):
        y = targets[index, mask[index]]
        p = predictions[index, mask[index]]
        if not (np.isfinite(y).all() and np.isfinite(p).all()):
            raise ValueError(
                f"nonfinite selected target or raw prediction in component index {index}"
            )
        clipped_y = np.clip(y, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        clipped_p = np.clip(p, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        for target_index in range(2):
            result[index, target_index] = _centered_moments(
                clipped_y[:, target_index], clipped_p[:, target_index]
            )
    return result


def score_moments_by_target(moments: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Merge component centered moments and compute the two official target WPs."""
    score = np.zeros(2, dtype=np.float64)
    for target_index in range(2):
        pooled = np.zeros(6, dtype=np.float64)
        for component_index, count in enumerate(counts):
            for _ in range(int(count)):
                pooled = _merge_moments(pooled, moments[component_index, target_index])
        score[target_index] = _wp_from_moments(pooled)
    return score


def joint_component_bootstrap(
    arm_moments: dict[str, np.ndarray],
    point_scores: dict[str, dict],
    n_boot: int = BOOTSTRAP_COUNT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict]:
    """Bootstrap all arms on identical component draws, including the interaction."""
    contrast_keys = [item[2] for item in PAIRED_CONTRASTS] + [
        "sine_both_minus_sine_embed_minus_sine_head_plus_base"
    ]
    sampled = {key: np.empty((n_boot, 3), dtype=np.float64) for key in contrast_keys}
    rng = np.random.default_rng(seed)
    n_components = 16
    for boot_index in range(n_boot):
        selected = rng.integers(0, n_components, size=n_components)
        counts = np.bincount(selected, minlength=n_components)
        scores = {
            arm: score_moments_by_target(moments, counts) for arm, moments in arm_moments.items()
        }
        for candidate, reference, key in PAIRED_CONTRASTS:
            sampled[key][boot_index, :2] = scores[candidate] - scores[reference]
            sampled[key][boot_index, 2] = float(np.mean(sampled[key][boot_index, :2]))
        interaction = (
            scores["sine_both"] - scores["sine_embed"] - scores["sine_head"] + scores["base"]
        )
        interaction_key = contrast_keys[-1]
        sampled[interaction_key][boot_index, :2] = interaction
        sampled[interaction_key][boot_index, 2] = float(np.mean(interaction))

    point = {
        arm: np.asarray([values["t0"], values["t1"]], dtype=np.float64)
        for arm, values in point_scores.items()
    }
    result = {}
    for candidate, reference, key in PAIRED_CONTRASTS:
        point_delta = point[candidate] - point[reference]
        distribution = sampled[key]
        result[key] = summarize_bootstrap(distribution, point_delta, n_boot, seed)
        result[key].update({"candidate": candidate, "reference": reference})
    interaction_key = contrast_keys[-1]
    interaction_point = (
        point["sine_both"] - point["sine_embed"] - point["sine_head"] + point["base"]
    )
    result[interaction_key] = summarize_bootstrap(
        sampled[interaction_key], interaction_point, n_boot, seed
    )
    result[interaction_key]["formula"] = "WP(sine_both) - WP(sine_embed) - WP(sine_head) + WP(base)"
    result[interaction_key]["arms"] = ["sine_both", "sine_embed", "sine_head", "base"]
    result[interaction_key]["joint_component_resampling"] = True
    return result


def summarize_bootstrap(
    samples: np.ndarray, point_delta: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Summarize paired two-target and mean-WP bootstrap distributions."""
    names = ("t0", "t1", "wp")
    if samples.shape != (n_boot, 3) or point_delta.shape != (2,):
        raise ValueError("bootstrap samples or point contrast have unexpected shape")
    point_all = np.asarray([point_delta[0], point_delta[1], point_delta.mean()])
    return {
        "point_delta": dict(zip(names, point_all.tolist(), strict=True)),
        "bootstrap": {
            name: {
                "mean": float(samples[:, index].mean()),
                "sd": float(samples[:, index].std(ddof=1)),
                "ci95": np.percentile(samples[:, index], [2.5, 97.5]).tolist(),
                "p_le_zero": float(np.mean(samples[:, index] <= 0.0)),
            }
            for index, name in enumerate(names)
        },
        "n_components": 16,
        "n_boot": n_boot,
        "seed": seed,
        "component_draws_shared_across_all_arms": True,
        "metric_id": METRIC_ID,
    }


def validate_sources_and_attempt(
    attempt_path: Path, run_dir: Path
) -> tuple[dict, dict, dict, list[dict], np.ndarray, np.ndarray, list[dict]]:
    """Run all provenance checks before reading any exposed test rows."""
    attempt, spec = validate_attempt_record(attempt_path, run_dir)
    report_path = run_dir / "fit_manifest.json"
    report = json.loads(report_path.read_text())
    validate_sealed_inputs(spec, report)
    # Recompute fit-only normalization from sealed cohort/data bytes. The
    # trainer's auxiliary fit_only_cohort.json is intentionally not represented
    # as a runner-digested artifact in this campaign spec.
    fit_manifest, fit_x, _fit_y, _fit_need, _fit_score, fit_nodes = e470.load_cohort(
        include_test=False
    )
    cohort_nodes = [
        node for node in json.loads(COHORT_PATH.read_text())["cohort"] if node["role"] == "fit"
    ]
    if fit_nodes != cohort_nodes or len(fit_nodes) != 24:
        raise ValueError("E470 loader did not reconstruct exactly the sealed 24 fit-role nodes")
    mean = np.asarray(fit_manifest["normalization"]["fit_mean"], dtype=np.float32)
    scale = np.asarray(fit_manifest["normalization"]["fit_scale"], dtype=np.float32)
    if mean.shape != (FEATURES,) or scale.shape != (FEATURES,):
        raise ValueError("recomputed fit-only normalization must have 42 channels")
    if not (np.isfinite(mean).all() and np.isfinite(scale).all() and (scale >= 0.001).all()):
        raise ValueError("recomputed fit-only normalization is invalid")
    del fit_x, _fit_y, _fit_need, _fit_score
    validated_arms = validate_fit_manifest(report, run_dir, fit_nodes, spec)
    return attempt, spec, report, fit_nodes, mean, scale, validated_arms


def score_run(
    run_dir: Path,
    attempt_path: Path,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Validate a completed run, export 12-arm predictions, and compute contrasts."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite scorer output: {output_dir}")
    attempt, spec, report, fit_nodes, fit_mean, fit_scale, validated_arms = (
        validate_sources_and_attempt(attempt_path, run_dir)
    )
    fit_only = {"normalization": {"fit_mean": fit_mean.tolist(), "fit_scale": fit_scale.tolist()}}
    test_features, raw_targets, mask, test_nodes = load_test_panel(fit_nodes, fit_only, DATA_PATH)
    if device.type == "cuda":
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA scorer requested but the device is unavailable")
    torch.set_num_threads(1)

    score_summaries: dict[str, dict] = {}
    moments_by_seed: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}
    predictions_for_export: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    parity_by_seed: dict[str, dict[str, float]] = {}
    test_components = np.asarray([node["component"] for node in test_nodes], dtype=np.int64)
    test_row_groups = [node["row_group"] for node in test_nodes]
    if len(np.unique(test_components)) != 16:
        raise ValueError("test export must preserve one row sequence per distinct component")

    with full_ieee_float32(device):
        for summary in validated_arms:
            seed = summary["seed"]
            arm = summary["arm"]
            key = f"seed{seed}_{arm}"
            checkpoint = summary["checkpoint_path"]
            if arm == "base":
                parity_by_seed[str(seed)] = check_base_parity(checkpoint)
            model = reconstruct_model(arm, checkpoint, device)
            raw_predictions = predict_test_panel(model, test_features, device, key)
            selected_targets = raw_targets[mask]
            selected_predictions = raw_predictions[mask]
            if not (
                np.isfinite(selected_targets).all() and np.isfinite(selected_predictions).all()
            ):
                raise ValueError(f"selected raw arrays are nonfinite in {key}")
            official = global_wp(
                raw_targets,
                raw_predictions,
                mask,
            )
            moments = component_moments(raw_targets, raw_predictions, mask)
            all_counts = np.ones(16, dtype=np.int64)
            target_scores_from_moments = score_moments_by_target(moments, all_counts)
            if not np.allclose(
                target_scores_from_moments,
                [official["t0"], official["t1"]],
                atol=2e-10,
                rtol=0,
            ):
                raise ValueError(f"component moments do not reproduce official WP for {key}")
            score_summaries[key] = {
                "seed": seed,
                "arm": arm,
                "wp": float(official["wp"]),
                "t0": float(official["t0"]),
                "t1": float(official["t1"]),
                "selected_rows": int(official["selected_rows"]),
                "parameters": summary["parameters"],
                "updates": summary["updates"],
                "checkpoint": summary["checkpoint"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
            }
            moments_by_seed[seed][arm] = moments
            predictions_for_export[key] = (selected_targets.copy(), selected_predictions.copy())
            del model, raw_predictions

    contrasts = {}
    for seed in SEEDS:
        seed_scores = {arm: score_summaries[f"seed{seed}_{arm}"] for arm in ARMS}
        contrasts[str(seed)] = joint_component_bootstrap(
            moments_by_seed[seed], seed_scores, n_boot=BOOTSTRAP_COUNT, seed=BOOTSTRAP_SEED
        )

    output_dir.mkdir(parents=True)
    prediction_exports = {}
    selected_counts = np.asarray(
        [mask[index].sum() for index in range(len(test_nodes))], dtype=np.int64
    )
    selected_row_indices = np.concatenate(
        [np.flatnonzero(mask[index]) for index in range(len(test_nodes))]
    ).astype(np.int32)
    component_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(selected_counts, dtype=np.int64)]
    )
    for key, (selected_targets, selected_predictions) in predictions_for_export.items():
        filename = f"{key}_selected_predictions.npz"
        output_path = output_dir / filename
        np.savez_compressed(
            output_path,
            targets_raw=selected_targets.astype(np.float32, copy=False),
            predictions_raw=selected_predictions.astype(np.float32, copy=False),
            components=test_components,
            row_groups=np.asarray(test_row_groups, dtype=np.int64),
            selected_counts=selected_counts,
            component_offsets=component_offsets,
            selected_row_indices=selected_row_indices,
        )
        prediction_exports[filename] = {
            "sha256": sha256_file(output_path),
            "selected_rows": len(selected_targets),
            "columns": ["t0", "t1"],
            "values_are_raw_before_metric_clipping": True,
        }

    scorer_path = Path(__file__).resolve()
    result = {
        "kind": "independent E474 activation-placement scorer",
        "status": "all_12_scored",
        "metric_id": METRIC_ID,
        "metric_clip": METRIC_CLIP,
        "scope": "16 historically exposed diagnostic test components; exploratory and not fresh validation/public-score evidence",
        "attempt": {
            "id": attempt["id"],
            "status": attempt["status"],
            "exit_code": attempt["exit_code"],
            "teardown_verified": attempt["teardown_verified"],
            "spec_sha256": attempt["spec_sha256"],
            "fit_result_artifacts": attempt["artifacts"],
        },
        "fit_manifest_sha256": sha256_file(run_dir / "fit_manifest.json"),
        "fit_normalization_source": "deterministically recomputed from the sealed 24 fit row groups; auxiliary fit_only_cohort.json is not claimed as a campaign artifact",
        "fit_source_sha256": report["source_sha256"],
        "recipe_source_sha256": report["recipe_source_sha256"],
        "metric_source_sha256": sha256_file(ROOT / "src/wnn/gmetric.py"),
        "scorer_sha256": sha256_file(scorer_path),
        "dataset_identity": {
            "path": "datasets/valid.parquet",
            "sha256": EXPECTED_DATASET_SHA256,
            "size": DATA_PATH.stat().st_size,
        },
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "seed_order": list(SEEDS),
        "arms_per_seed": list(ARMS),
        "components": test_components.tolist(),
        "test_row_groups": test_row_groups,
        "test_sequences": len(test_nodes),
        "selected_rows_per_component": {
            str(node["component"]): int(mask[index].sum()) for index, node in enumerate(test_nodes)
        },
        "baseline_reconstruction_parity": parity_by_seed,
        "arm_scores": score_summaries,
        "paired_contrasts_by_seed": contrasts,
        "bootstrap": {
            "method": "joint component bootstrap; same 16 component draws shared by every arm in each replicate",
            "resampling_unit": "connected component; each source test row group is one component",
            "n_boot": BOOTSTRAP_COUNT,
            "seed": BOOTSTRAP_SEED,
            "interaction": "computed inside each joint resample from four arm-level global WPs, never by combining independent confidence intervals",
            "multiplicity": "all 6 paired contrasts plus the preregistered interaction retained per seed; no selection or correction",
        },
        "prediction_exports": prediction_exports,
        "prediction_export_directory": str(output_dir),
        "selected_targets_validated_from_source_rows": True,
        "raw_predictions_finite_before_clipping": True,
        "test_rows_used_for_training": False,
    }
    (output_dir / "score_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "arms": len(score_summaries),
                "rows": int(mask.sum()),
                "metric": METRIC_ID,
            }
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, required=True, help="completed E474 fit output directory"
    )
    parser.add_argument(
        "--attempt-record", type=Path, required=True, help="campaign attempts/<id>.json proof"
    )
    parser.add_argument("--output", type=Path, required=True, help="fresh scorer export directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    score_run(
        args.run.resolve(),
        args.attempt_record.resolve(),
        torch.device(args.device),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()

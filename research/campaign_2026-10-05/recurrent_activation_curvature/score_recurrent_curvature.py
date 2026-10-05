"""Independently score the sealed E481 recurrent-candidate curvature fits."""

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
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
SEALED_SPEC = HERE / "e481_eighteen_fit_v1.sealed.json"
DEFAULT_RUN = HERE / "runs/eighteen_v1"
DEFAULT_OUTPUT = HERE / "scores/eighteen_v1"
DEFAULT_ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e481-candidate-curvature-eighteen-fit-v1.json"
)
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
DATA_PATH = ROOT / "datasets/valid.parquet"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E474_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402

from wnn.campaign import RUNTIME_FILES, runtime_identity  # noqa: E402
from wnn.campaign_spec import digest as spec_digest  # noqa: E402
from wnn.campaign_spec import validate_spec  # noqa: E402
from wnn.evaluation import paired_component_moments  # noqa: E402
from wnn.gmetric import (  # noqa: E402
    METRIC_CLIP,
    METRIC_ID,
    _merge_moments,
    _wp_from_moments,
    global_wp,
)
from wnn.resources import configure_torch  # noqa: E402

SEEDS = (271_828, 271_829, 271_830)
OPTIMIZERS = ("adamw", "muon_adamw")
SHAPES = ("baseline", "sine", "cubic")
CELLS = tuple(
    (seed, optimizer, shape) for seed in SEEDS for optimizer in OPTIMIZERS for shape in SHAPES
)
FEATURES, HIDDEN, SEQUENCE_ROWS = 42, 64, 20_000
CHUNK_ROWS, UPDATES, EXPECTED_PARAMETERS = 256, 474, 61_058
TELEMETRY_STEPS = (0, 46, 236, 473)
BOOTSTRAP_COUNT, BOOTSTRAP_SEED = 2_000, 271_828
EXPECTED_SPEC_NAME = "e481_candidate_curvature_eighteen_fit_v1"
EXPECTED_ATTEMPT_ID = "e481-candidate-curvature-eighteen-fit-v1"
EXPECTED_DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
EXPECTED_COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
ACTOR_SHA256 = "3d97d2d07b116b3cf04b365c33e3d3caed2c96c35108bd28f3a6ff63b54d65cc"
TESTS_SHA256 = "ae5e277b1551d2b6c338aa90e83bb1fb7ffff0d29271d33f1f3498caa8e69ee2"
PROBE_SHA256 = "e311e022232807c407d4fe56720c9e16d6474af04528064f05886487e9fd077a"
PREFLIGHT_SHA256 = "83d99ffaf1b037ccf28d62dfc5d40b58a984dd7a2b49f4dea0cea9ae1f3c72b4"
E470_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E474_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
SOURCE_PATHS = {
    "fit_recurrent_curvature.py": "research/campaign_2026-10-05/recurrent_activation_curvature/fit_recurrent_curvature.py",
    "tests.py": "research/campaign_2026-10-05/recurrent_activation_curvature/tests.py",
    "gpu_probe.py": "research/campaign_2026-10-05/recurrent_activation_curvature/gpu_probe.py",
}
EXPECTED_SOURCE_HASHES = {
    "fit_recurrent_curvature.py": ACTOR_SHA256,
    "tests.py": TESTS_SHA256,
    "gpu_probe.py": PROBE_SHA256,
}
PAIR_SHAPES = (("sine", "baseline"), ("cubic", "baseline"), ("cubic", "sine"))


@torch.jit.script
def transform_candidate(u: Tensor, shape_code: int) -> Tensor:
    """Apply one registered parameter-free candidate map."""
    if shape_code == 0:
        return u
    if shape_code == 1:
        return u + 0.1 * torch.sin(math.pi * u)
    if shape_code == 2:
        return u + (0.1 * math.pi) * u * (1.0 - u * u)
    raise ValueError("unregistered E481 candidate shape")


@torch.jit.script
def candidate_scan(
    x: Tensor,
    h0: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    shape_code: int,
) -> tuple[Tensor, Tensor]:
    """Evaluate the reset-after GRU candidate recurrence in r/z/n order."""
    input_affine = F.linear(x, weight_ih, bias_ih)
    outputs = torch.jit.annotate(list[Tensor], [])
    hidden = h0
    for step in range(x.size(1)):
        i_r, i_z, i_n = input_affine[:, step, :].chunk(3, dim=1)
        h_r, h_z, h_n = F.linear(hidden, weight_hh, bias_hh).chunk(3, dim=1)
        reset = torch.sigmoid(i_r + h_r)
        update = torch.sigmoid(i_z + h_z)
        u = torch.tanh(i_n + reset * h_n)
        candidate = transform_candidate(u, shape_code)
        hidden = (1.0 - update) * candidate + update * hidden
        outputs.append(hidden)
    return torch.stack(outputs, dim=1), hidden


class CurvatureModel(nn.Module):
    """Independent E481 checkpoint-compatible 42→64→GRU×2→concat head model."""

    def __init__(self, shape: str):
        super().__init__()
        if shape not in SHAPES:
            raise ValueError(f"unregistered shape: {shape}")
        self.shape = shape
        self.shape_code = SHAPES.index(shape)
        self.register_buffer("mean", torch.zeros(FEATURES))
        self.register_buffer("std", torch.ones(FEATURES))
        self.inp = nn.Linear(FEATURES, HIDDEN)
        self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 2))
        if sum(parameter.numel() for parameter in self.parameters()) != EXPECTED_PARAMETERS:
            raise ValueError("E481 model parameter count differs from 61,058")

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Process one causal chunk and return its per-layer carried state."""
        if x.ndim != 3 or x.shape[-1] != FEATURES:
            raise ValueError("expected [batch,time,42] input")
        encoded = F.gelu(self.inp((x - self.mean) / self.std))
        batch = encoded.shape[0]
        if state is None:
            hidden = encoded.new_zeros((2, batch, HIDDEN))
        else:
            if state.shape != (2, batch, HIDDEN):
                raise ValueError("expected carried state [2,batch,64]")
            hidden = state
        layer_input = encoded
        finals: list[Tensor] = []
        for layer in range(2):
            layer_input, final = candidate_scan(
                layer_input,
                hidden[layer],
                getattr(self.gru, f"weight_ih_l{layer}"),
                getattr(self.gru, f"weight_hh_l{layer}"),
                getattr(self.gru, f"bias_ih_l{layer}"),
                getattr(self.gru, f"bias_hh_l{layer}"),
                self.shape_code,
            )
            finals.append(final)
        prediction = self.head(torch.cat((layer_input, encoded), dim=-1))
        return prediction, torch.stack(finals)


def file_sha256(path: Path) -> str:
    """Hash a file in bounded memory blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_identity_buffers(state: dict[str, Tensor]) -> None:
    """Require identity model-side normalization buffers in every checkpoint."""
    if "mean" not in state or "std" not in state:
        raise ValueError("checkpoint lacks external-normalization identity buffers")
    if not torch.equal(state["mean"].cpu(), torch.zeros(FEATURES)):
        raise ValueError("checkpoint mean buffer is not identity zero")
    if not torch.equal(state["std"].cpu(), torch.ones(FEATURES)):
        raise ValueError("checkpoint std buffer is not identity one")


def validate_spec_identity(spec: dict) -> dict:
    """Check E481 sealed spec name, command, run and output artifact map."""
    checked = validate_spec(spec)
    if checked.get("name") != EXPECTED_SPEC_NAME or checked.get("kind") != "gpu":
        raise ValueError("unexpected E481 spec identity or kind")
    expected_run = "research/campaign_2026-10-05/recurrent_activation_curvature/runs/eighteen_v1"
    if checked.get("argv", [None])[1] != SOURCE_PATHS["fit_recurrent_curvature.py"]:
        raise ValueError("E481 spec does not invoke the registered fit actor")
    if expected_run not in checked.get("argv", []):
        raise ValueError("E481 runner output argument differs from registered run")
    if checked.get("artifacts") != [f"{expected_run}/fit_manifest.json"]:
        raise ValueError("E481 artifact map must contain the fixed fit manifest")
    code_hashes = {entry["path"]: entry["sha256"] for entry in checked.get("code", [])}
    for name, relative in SOURCE_PATHS.items():
        path = ROOT / relative
        if code_hashes.get(relative) != EXPECTED_SOURCE_HASHES[name]:
            raise ValueError(f"sealed spec does not bind E481 {name}")
        if file_sha256(path) != EXPECTED_SOURCE_HASHES[name]:
            raise ValueError(f"local E481 {name} differs from reviewed source")
    for relative, expected in (
        (
            "research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py",
            E470_SHA256,
        ),
        (
            "research/campaign_2026-10-05/activation_placement/fit_activation_placement.py",
            E474_SHA256,
        ),
    ):
        if file_sha256(ROOT / relative) != expected or code_hashes.get(relative) != expected:
            raise ValueError(f"E481 spec or local tree differs from frozen helper: {relative}")
    inputs = {entry["path"]: entry["sha256"] for entry in checked.get("inputs", [])}
    required_inputs = {
        "datasets/valid.parquet": EXPECTED_DATASET_SHA256,
        "research/campaign_2026-10-05/target_mechanics/cohort.json": EXPECTED_COHORT_SHA256,
        "research/campaign_2026-10-05/recurrent_activation_curvature/preflight_cpu_v2/fit_manifest.json": PREFLIGHT_SHA256,
    }
    for relative, expected in required_inputs.items():
        if inputs.get(relative) != expected or file_sha256(ROOT / relative) != expected:
            raise ValueError(f"E481 sealed input identity differs: {relative}")
    preflight_path = (
        ROOT
        / "research/campaign_2026-10-05/recurrent_activation_curvature/preflight_cpu_v2/fit_manifest.json"
    )
    preflight = json.loads(preflight_path.read_text())
    if (
        preflight.get("status") != "cpu_fit_only_one_update_per_arm_passed_test_unread"
        or preflight.get("test_rows_loaded") is not False
        or len(preflight.get("arms", [])) != 18
    ):
        raise ValueError("E481 fit-only CPU preflight receipt differs")
    return checked


def validate_attempt(attempt_path: Path, spec: dict, run: Path) -> dict:
    """Require runner success, exact runtime, artifact hash, and no new OOMs."""
    checked = validate_spec_identity(spec)
    record = json.loads(attempt_path.read_text())
    if record.get("id") != EXPECTED_ATTEMPT_ID or record.get("kind") != "gpu":
        raise ValueError("E481 attempt identity differs")
    if record.get("spec") != checked or record.get("spec_sha256") != spec_digest(checked):
        raise ValueError("E481 attempt does not bind the exact sealed spec")
    argv = record.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not Path(argv[0]).is_absolute()
        or argv[1:] != checked["argv"][1:]
    ):
        raise ValueError("E481 attempt command differs from sealed command")
    current_runtime = runtime_identity(ROOT)
    if (
        not isinstance(record.get("runtime_sha256"), dict)
        or set(record["runtime_sha256"]) != set(RUNTIME_FILES)
        or record["runtime_sha256"] != current_runtime
    ):
        raise ValueError("E481 runtime file hashes differ from current runner")
    if (
        record.get("status") != "finished"
        or record.get("exit_code") != 0
        or record.get("signals") != []
        or record.get("teardown_verified") is not True
        or any(
            record.get(key) is not None for key in ("failure", "failure_scope", "failure_category")
        )
    ):
        raise ValueError("E481 attempt did not finish cleanly with verified teardown")
    unit = record.get("unit_properties", {})
    if unit.get("Result") != "success" or unit.get("ExecMainStatus") not in (0, "0"):
        raise ValueError("E481 systemd unit reports failure")
    oom_counters = ("oom", "oom_kill", "oom_group_kill")
    for before_key, after_key in (
        ("initial_memory_events", "final_memory_events"),
        ("initial_parent_memory_events", "final_parent_memory_events"),
    ):
        before, after = record.get(before_key), record.get(after_key)
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError(f"E481 attempt lacks OOM snapshots: {before_key}")
        for counter in oom_counters:
            a, b = before.get(counter), after.get(counter)
            if type(a) is not int or type(b) is not int or a < 0 or b < 0 or a != b:
                raise ValueError(f"E481 OOM counter increased or malformed: {before_key}/{counter}")
    expected_artifacts = []
    for relative in checked["artifacts"]:
        artifact_path = ROOT / relative
        if not artifact_path.is_file():
            raise ValueError(f"E481 runner output missing: {relative}")
        expected_artifacts.append(
            {
                "path": relative,
                "sha256": file_sha256(artifact_path),
                "size": artifact_path.stat().st_size,
            }
        )
    if record.get("artifacts") != expected_artifacts:
        raise ValueError("E481 attempt artifact inventory differs")
    expected_run = ROOT / Path(checked["artifacts"][0]).parent
    if expected_run.resolve() != run.resolve():
        raise ValueError("E481 run directory differs from the fixed spec")
    return {
        "attempt_id": record["id"],
        "attempt_sha256": file_sha256(attempt_path),
        "spec_sha256": record["spec_sha256"],
        "runtime_sha256": current_runtime,
        "artifacts": expected_artifacts,
        "worker_oom_counters_unchanged": True,
        "parent_oom_counters_unchanged": True,
        "teardown_verified": True,
    }


def derive_fit_normalizer(fit_nodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Independently recompute E470 fit-only normalization from raw fit rows."""
    if len(fit_nodes) != 24 or any(node.get("role") != "fit" for node in fit_nodes):
        raise ValueError("normalizer requires exactly the 24 fit-role nodes")
    parquet = pq.ParquetFile(DATA_PATH)
    features = []
    for node in fit_nodes:
        table = parquet.read_row_group(node["row_group"], columns=e470.FEATURES, use_threads=False)
        raw = np.column_stack([table[column].to_numpy() for column in e470.FEATURES])
        features.append(e470.causal_features(raw))
    x = np.stack(features)
    mean = x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(axis=(0, 1), dtype=np.float64), 0.001).astype(np.float32)
    if (
        mean.shape != (FEATURES,)
        or scale.shape != (FEATURES,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
    ):
        raise ValueError("invalid independently derived fit normalizer")
    return mean, scale


def validate_normalizer(
    fit_manifest: dict,
    fit_x: np.ndarray,
    fit_nodes: list[dict],
    all_manifest: dict,
    all_x: np.ndarray,
    all_nodes: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Compare fit-only and full-cohort loader transforms and raw-fit stats."""
    fit_norm = fit_manifest.get("normalization")
    if not isinstance(fit_norm, dict) or fit_norm != all_manifest.get("normalization"):
        raise ValueError("fit and all-cohort loader normalization differs")
    mean = np.asarray(fit_norm.get("fit_mean"), dtype=np.float32)
    scale = np.asarray(fit_norm.get("fit_scale"), dtype=np.float32)
    direct_mean, direct_scale = derive_fit_normalizer(fit_nodes)
    if not np.array_equal(mean, direct_mean) or not np.array_equal(scale, direct_scale):
        raise ValueError("frozen loader normalization differs from independent fit-only derivation")
    if (
        mean.shape != (FEATURES,)
        or scale.shape != (FEATURES,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("invalid fit-only normalization vectors")
    fit_map = {node["row_group"]: i for i, node in enumerate(fit_nodes)}
    all_map = {node["row_group"]: i for i, node in enumerate(all_nodes)}
    if len(fit_map) != len(fit_nodes):
        raise ValueError("duplicate fit groups for normalization validation")
    for row_group, fit_index in fit_map.items():
        if row_group not in all_map or not np.array_equal(
            fit_x[fit_index], all_x[all_map[row_group]]
        ):
            raise ValueError("test-inclusive loader changed a fit-normalized sequence")
    if not np.isfinite(all_x).all():
        raise ValueError("normalized cohort contains nonfinite features")
    return mean, scale


def validate_fit_manifest(
    run: Path, report: dict, spec: dict, nodes: list[dict]
) -> tuple[dict, dict]:
    """Validate fit provenance, fixed cells, row orders, telemetry and checkpoints."""
    checked = validate_spec_identity(spec)
    if (
        report.get("status") != "all_18_fit_arms_complete_test_unread"
        or report.get("fit_only") is not True
        or report.get("test_rows_loaded") is not False
    ):
        raise ValueError(
            "E481 manifest is incomplete or does not certify test-unread fit-only training"
        )
    fixed = {
        "seeds": list(SEEDS),
        "optimizers": list(OPTIMIZERS),
        "shapes": list(SHAPES),
        "planned_fit_count": 18,
        "parameters_per_arm": EXPECTED_PARAMETERS,
        "updates_per_full_fit": UPDATES,
        "batch": 4,
        "chunk_rows": CHUNK_ROWS,
        "telemetry_steps_zero_based": list(TELEMETRY_STEPS),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "cohort_sha256": EXPECTED_COHORT_SHA256,
    }
    for key, expected in fixed.items():
        if report.get(key) != expected:
            raise ValueError(f"E481 manifest field differs: {key}")
    if (
        e474.file_hash(COHORT_PATH) != EXPECTED_COHORT_SHA256
        or e470.file_hash(DATA_PATH) != EXPECTED_DATASET_SHA256
    ):
        raise ValueError("E481 cohort or dataset hash changed")
    if report.get("parent_source_sha256") != {"e470": E470_SHA256, "e474": E474_SHA256}:
        raise ValueError("E481 frozen parent helper identities differ")
    precision = report.get("precision", {})
    if (
        precision.get("device_math") != "FP32 IEEE"
        or precision.get("autocast") is not False
        or precision.get("torch.backends.fp32_precision") != "ieee"
        or precision.get("torch.backends.cuda.matmul.fp32_precision") != "ieee"
        or precision.get("torch.backends.cudnn.fp32_precision") != "ieee"
        or precision.get("torch.backends.cudnn.rnn.fp32_precision") != "ieee"
    ):
        raise ValueError("E481 fit precision receipt differs from required IEEE FP32")
    for key, relative in SOURCE_PATHS.items():
        expected = EXPECTED_SOURCE_HASHES[key]
        if report.get("source_sha256", {}).get(key) != expected:
            raise ValueError(f"E481 report source hash differs: {key}")
        sealed_code = {item["path"]: item["sha256"] for item in checked.get("code", [])}
        if sealed_code.get(relative) != expected:
            raise ValueError(f"E481 spec source differs from report: {key}")
    fit_groups = [node for node in nodes if node.get("role") == "fit"]
    test_groups = [node for node in nodes if node.get("role") == "test"]
    if len(nodes) != 40 or len(fit_groups) != 24 or len(test_groups) != 16:
        raise ValueError("E481 cohort must retain exactly 24 fit and 16 test sequences")
    fit_components = {node["component"] for node in fit_groups}
    test_components = {node["component"] for node in test_groups}
    fit_row_groups = {node["row_group"] for node in fit_groups}
    test_row_groups = {node["row_group"] for node in test_groups}
    if len(fit_components) != 24 or len(test_components) != 16 or fit_components & test_components:
        raise ValueError("E481 component roles overlap or contain duplicates")
    if len(fit_row_groups | test_row_groups) != 40 or fit_row_groups & test_row_groups:
        raise ValueError("E481 row-group roles overlap or contain duplicates")
    recorded_fit = report.get("fit_only_cohort", {})
    if recorded_fit.get("cohort") != fit_groups or recorded_fit.get("normalization") is None:
        raise ValueError("E481 fit-only cohort manifest differs from canonical cohort")
    expected_keys = set(CELLS)
    arms = report.get("arms", [])
    if len(arms) != len(CELLS):
        raise ValueError("E481 manifest does not contain all 18 cells")
    observed = {}
    for arm in arms:
        key = (arm.get("seed"), arm.get("optimizer"), arm.get("shape"))
        if key not in expected_keys or key in observed:
            raise ValueError(f"unknown or duplicate E481 cell: {key}")
        seed, optimizer, shape = key
        order = list(range(24))
        np.random.default_rng(seed).shuffle(order)
        expected_order = [fit_groups[index]["row_group"] for index in order]
        if arm.get("fit_row_groups") != expected_order:
            raise ValueError(f"E481 fit traversal mismatch: {key}")
        if arm.get("updates") != UPDATES or arm.get("parameters") != EXPECTED_PARAMETERS:
            raise ValueError(f"E481 cell update/count differs: {key}")
        telemetry = arm.get("telemetry")
        if not isinstance(telemetry, list) or [
            entry.get("update_index_zero_based") for entry in telemetry
        ] != list(TELEMETRY_STEPS):
            raise ValueError(f"E481 telemetry cadence differs: {key}")
        for sample in telemetry:
            batch_index = sample["update_index_zero_based"] // math.ceil(SEQUENCE_ROWS / CHUNK_ROWS)
            expected_batch = expected_order[batch_index * 4 : (batch_index + 1) * 4]
            if sample.get("fit_row_groups") != expected_batch:
                raise ValueError(f"E481 telemetry uses unexpected fit batch: {key}")
            for field in ("preclip_gradient_rms", "parameter_update_rms"):
                families = sample.get(field, {})
                if set(families) != {"inp", "gru", "head"} or any(
                    not math.isfinite(float(v)) or float(v) < 0 for v in families.values()
                ):
                    raise ValueError(f"invalid E481 {field}: {key}")
        checkpoint_name = arm.get("checkpoint")
        expected_checkpoint = f"seed{seed}_{optimizer}_{shape}.pt"
        if checkpoint_name != expected_checkpoint or Path(checkpoint_name).name != checkpoint_name:
            raise ValueError(f"invalid E481 checkpoint filename: {key}")
        checkpoint = run / checkpoint_name
        if not checkpoint.is_file() or file_sha256(checkpoint) != arm.get("checkpoint_sha256"):
            raise ValueError(f"E481 checkpoint hash differs: {key}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError(f"E481 checkpoint contains nonfinite tensors: {key}")
        validate_identity_buffers(state)
        model = CurvatureModel(shape)
        if set(state) != set(model.state_dict()):
            raise ValueError(f"E481 checkpoint keys differ from independent model: {key}")
        observed[key] = arm
    if set(observed) != expected_keys:
        raise ValueError("E481 cell matrix is incomplete")
    return observed, {"validated_cells": 18, "fit_roles": 24, "test_roles": 16}


def validate_raw_targets(
    raw: np.ndarray, mask: np.ndarray, clipped: np.ndarray, row_group: int
) -> np.ndarray:
    """Reject raw nonfinite labels before clipping and verify the official mask."""
    raw = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    clipped = np.asarray(clipped, dtype=np.float32)
    if (
        raw.ndim != 2
        or raw.shape[-1] != 2
        or mask.shape != raw.shape[:1]
        or clipped.shape != raw.shape
    ):
        raise ValueError(f"malformed E481 label arrays for row group {row_group}")
    if not mask.any() or not np.isfinite(raw).all() or not np.isfinite(raw[mask]).all():
        raise ValueError(f"empty or nonfinite raw E481 labels for row group {row_group}")
    if not np.array_equal(np.clip(raw, -METRIC_CLIP, METRIC_CLIP).astype(np.float32), clipped):
        raise ValueError(
            f"official clipped labels differ from raw source for row group {row_group}"
        )
    return raw


def read_raw_test_targets(
    test_nodes: list[dict], clipped: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    """Read official raw test labels, checking selected masks before clipping."""
    expected = (len(test_nodes), SEQUENCE_ROWS, 2)
    if clipped.shape != expected or selected.shape != expected[:2]:
        raise ValueError("E481 test targets/masks do not match fixed 16×20,000 roles")
    parquet = pq.ParquetFile(DATA_PATH)
    raw_targets = np.empty(expected, dtype=np.float64)
    for index, node in enumerate(test_nodes):
        table = parquet.read_row_group(
            node["row_group"],
            columns=["t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        raw = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")]).astype(np.float64)
        source_mask = table["need_prediction"].to_numpy().astype(bool) & table[
            "is_scored"
        ].to_numpy().astype(bool)
        if not np.array_equal(source_mask, selected[index]):
            raise ValueError(f"E481 selected mask differs for row group {node['row_group']}")
        raw_targets[index] = validate_raw_targets(
            raw, source_mask, clipped[index], node["row_group"]
        )
    return raw_targets


@contextlib.contextmanager
def ieee_context(device: torch.device):
    """Use the fit's full IEEE policy for CUDA scoring."""
    if device.type == "cuda":
        with e474.full_ieee_float32():
            yield
    else:
        with torch.autocast(device_type="cpu", enabled=False):
            yield


def predict_sequences(
    checkpoint: Path, shape: str, x: np.ndarray, device: torch.device
) -> np.ndarray:
    """Predict full test sequences with chunk-carried recurrent state."""
    model = CurvatureModel(shape).to(device).float()
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    if any(not torch.isfinite(value).all() for value in state_dict.values()):
        raise ValueError(f"nonfinite E481 checkpoint: {checkpoint.name}")
    validate_identity_buffers(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    predictions = np.empty((len(x), x.shape[1], 2), dtype=np.float32)
    with ieee_context(device), torch.no_grad():
        for index, sequence in enumerate(x):
            state = None
            parts = []
            for start in range(0, sequence.shape[0], CHUNK_ROWS):
                chunk = torch.from_numpy(sequence[start : start + CHUNK_ROWS][None]).to(
                    device=device, dtype=torch.float32
                )
                output, state = model(chunk, state)
                if not torch.isfinite(output).all() or not torch.isfinite(state).all():
                    raise ValueError(f"nonfinite prediction/state: {checkpoint.name}")
                parts.append(output[0].cpu().numpy())
            predictions[index] = np.concatenate(parts, axis=0)
    if not np.isfinite(predictions).all():
        raise ValueError(f"nonfinite full-sequence predictions: {checkpoint.name}")
    return predictions


def bootstrap_counts(component_ids: np.ndarray) -> np.ndarray:
    """Generate shared multiplicities after sorting the 16 component IDs."""
    if component_ids.ndim != 1 or len(component_ids) != 16 or len(np.unique(component_ids)) != 16:
        raise ValueError("E481 bootstrap requires 16 unique sorted components")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    counts = np.zeros((BOOTSTRAP_COUNT, len(component_ids)), dtype=np.int16)
    for draw in range(BOOTSTRAP_COUNT):
        selected = rng.integers(0, len(component_ids), len(component_ids))
        counts[draw] = np.bincount(selected, minlength=len(component_ids))
    if not np.all(counts.sum(axis=1) == len(component_ids)):
        raise ValueError("E481 bootstrap counts do not conserve component count")
    return counts


def pool_moments(moments: np.ndarray, counts: np.ndarray) -> float:
    """Recompute global WP from component-wise centered moments."""
    pooled = np.zeros((2, 6), dtype=np.float64)
    for index, multiplicity in enumerate(counts):
        if multiplicity == 0:
            continue
        for target in range(2):
            contribution = moments[index, target].copy()
            contribution[[0, 3, 4, 5]] *= int(multiplicity)
            pooled[target] = _merge_moments(pooled[target], contribution)
    return float(np.mean([_wp_from_moments(pooled[target]) for target in range(2)]))


def contrast_summary(draws: np.ndarray, point: float) -> dict:
    """Summarize one paired WP contrast under common multiplicity draws."""
    if (
        draws.shape != (BOOTSTRAP_COUNT,)
        or not np.isfinite(draws).all()
        or not math.isfinite(point)
    ):
        raise ValueError("invalid E481 contrast draws")
    return {
        "mean_diff": float(point),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
        "p_le_0": float(np.mean(draws <= 0)),
        "n_components": 16,
        "n_boot": BOOTSTRAP_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "metric_id": METRIC_ID,
    }


def registered_contrasts(scores: dict[str, float], draws: dict[str, np.ndarray]) -> dict[str, dict]:
    """Compute six shape pairs, three optimizer pairs, and one joint interaction per seed."""
    results = {}
    for optimizer in OPTIMIZERS:
        for candidate, reference in PAIR_SHAPES:
            a = f"{optimizer}__{candidate}"
            b = f"{optimizer}__{reference}"
            results[f"{a}_minus_{b}"] = {
                "family": "shape_within_optimizer",
                "candidate": a,
                "reference": b,
                **contrast_summary(draws[a] - draws[b], scores[a] - scores[b]),
            }
    for shape in SHAPES:
        candidate = f"muon_adamw__{shape}"
        reference = f"adamw__{shape}"
        results[f"{candidate}_minus_{reference}"] = {
            "family": "optimizer_within_shape",
            "candidate": candidate,
            "reference": reference,
            **contrast_summary(
                draws[candidate] - draws[reference], scores[candidate] - scores[reference]
            ),
        }
    muon_diff = draws["muon_adamw__cubic"] - draws["muon_adamw__sine"]
    adam_diff = draws["adamw__cubic"] - draws["adamw__sine"]
    interaction = (scores["muon_adamw__cubic"] - scores["muon_adamw__sine"]) - (
        scores["adamw__cubic"] - scores["adamw__sine"]
    )
    results["curvature_cubic_minus_sine_x_muon_minus_adamw"] = {
        "family": "shape_by_optimizer_interaction",
        "candidate": "(muon_adamw:cubic-sine)",
        "reference": "(adamw:cubic-sine)",
        **contrast_summary(muon_diff - adam_diff, interaction),
    }
    if len(results) != 10:
        raise AssertionError("expected ten registered E481 contrasts per seed")
    return results


def main() -> None:
    """Validate sealed E481 provenance, score 18 arms, and export paired data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--attempt-record", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--sealed-spec", type=Path, default=SEALED_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite E481 score output: {output}")
    if not run.is_dir():
        raise FileNotFoundError(f"E481 run directory missing: {run}")
    spec = json.loads(args.sealed_spec.read_text())
    attempt = validate_attempt(args.attempt_record.resolve(), spec, run)
    report = json.loads((run / "fit_manifest.json").read_text())
    cohort = json.loads(COHORT_PATH.read_text())
    nodes = cohort.get("cohort", [])
    cells, role_receipt = validate_fit_manifest(run, report, spec, nodes)
    fit_nodes = [node for node in nodes if node["role"] == "fit"]
    test_nodes = [node for node in nodes if node["role"] == "test"]
    if args.device == "cuda":
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for E481 scoring but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    fit_manifest, fit_x, *_fit_rest, loaded_fit_nodes = e470.load_cohort(include_test=False)
    all_manifest, all_x, all_y, _all_need, all_selected, loaded_nodes = e470.load_cohort(
        include_test=True
    )
    if [node["row_group"] for node in loaded_fit_nodes] != [
        node["row_group"] for node in fit_nodes
    ]:
        raise ValueError("fit-only loader role/order differs from canonical cohort")
    if [node["row_group"] for node in loaded_nodes] != [node["row_group"] for node in nodes]:
        raise ValueError("test scoring loader role/order differs from canonical cohort")
    mean, scale = validate_normalizer(
        fit_manifest, fit_x, fit_nodes, all_manifest, all_x, loaded_nodes
    )
    if fit_x.shape != (24, SEQUENCE_ROWS, FEATURES) or all_x.shape != (40, SEQUENCE_ROWS, FEATURES):
        raise ValueError("E481 normalized cohort shape differs from registered 24/16 split")
    test_indices = [i for i, node in enumerate(nodes) if node["role"] == "test"]
    test_x, clipped_targets, selected_mask = (
        all_x[test_indices],
        all_y[test_indices],
        all_selected[test_indices],
    )
    raw_targets = read_raw_test_targets(test_nodes, clipped_targets, selected_mask)
    row_counts = selected_mask.sum(axis=1).astype(np.int64)
    if np.any(row_counts <= 0):
        raise ValueError("E481 test components must all have selected rows")
    component_ids = np.asarray([node["component"] for node in test_nodes], dtype=np.int64)
    order = np.argsort(component_ids, kind="stable")
    sorted_components = component_ids[order]
    multiplicities = bootstrap_counts(sorted_components)
    arm_scores: dict[str, dict] = {}
    arm_moments: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}
    selected_predictions: dict[str, np.ndarray] = {}
    selected_targets = raw_targets[selected_mask]
    for seed, optimizer, shape in CELLS:
        arm_key = f"seed{seed}__{optimizer}__{shape}"
        within = f"{optimizer}__{shape}"
        arm = cells[(seed, optimizer, shape)]
        prediction = predict_sequences(run / arm["checkpoint"], shape, test_x, device)
        metric = global_wp(raw_targets, prediction, selected_mask)
        if metric.get("metric_id") != METRIC_ID or metric.get("selected_rows") != int(
            row_counts.sum()
        ):
            raise ValueError(f"official E481 metric metadata differs: {arm_key}")
        moments, _ = paired_component_moments(raw_targets, prediction, prediction, selected_mask)
        arm_moments[seed][within] = moments[order]
        selected_predictions[arm_key] = prediction[selected_mask].copy()
        arm_scores[arm_key] = {
            "seed": seed,
            "optimizer": optimizer,
            "shape": shape,
            "wp": float(metric["wp"]),
            "t0": float(metric["t0"]),
            "t1": float(metric["t1"]),
            "selected_rows": int(metric["selected_rows"]),
            "component_row_counts": row_counts.tolist(),
            "parameters": EXPECTED_PARAMETERS,
            "updates": UPDATES,
        }
    per_seed_contrasts = {}
    per_seed_draws = {}
    per_seed_contrast_draws = {}
    for seed in SEEDS:
        point_scores = {key: arm_scores[f"seed{seed}__{key}"]["wp"] for key in arm_moments[seed]}
        draws = {
            key: np.asarray(
                [pool_moments(moment, multiplicity) for multiplicity in multiplicities],
                dtype=np.float64,
            )
            for key, moment in arm_moments[seed].items()
        }
        per_seed_contrasts[str(seed)] = registered_contrasts(point_scores, draws)
        per_seed_draws[str(seed)] = draws
        contrast_draws = {}
        for optimizer in OPTIMIZERS:
            for candidate, reference in PAIR_SHAPES:
                contrast_draws[f"{optimizer}__{candidate}_minus_{optimizer}__{reference}"] = (
                    draws[f"{optimizer}__{candidate}"] - draws[f"{optimizer}__{reference}"]
                )
        for shape in SHAPES:
            contrast_draws[f"muon_adamw__{shape}_minus_adamw__{shape}"] = (
                draws[f"muon_adamw__{shape}"] - draws[f"adamw__{shape}"]
            )
        contrast_draws["curvature_cubic_minus_sine_x_muon_minus_adamw"] = (
            draws["muon_adamw__cubic"]
            - draws["muon_adamw__sine"]
            - draws["adamw__cubic"]
            + draws["adamw__sine"]
        )
        if len(contrast_draws) != 10:
            raise AssertionError("E481 bootstrap contrast draw map is incomplete")
        per_seed_contrast_draws[str(seed)] = contrast_draws
    contrast_count = sum(len(values) for values in per_seed_contrasts.values())
    if contrast_count != 30:
        raise AssertionError(f"E481 expected 30 contrasts, got {contrast_count}")
    output.mkdir(parents=True)
    export_path = output / "selected_exports.npz"
    arm_order = [f"seed{seed}__{optimizer}__{shape}" for seed, optimizer, shape in CELLS]
    np.savez_compressed(
        export_path,
        arm_names=np.asarray(arm_order),
        targets_raw=selected_targets,
        predictions_raw=np.stack([selected_predictions[name] for name in arm_order]),
        selected_mask=selected_mask,
        component_ids=component_ids,
        sorted_component_ids=sorted_components,
        row_groups=np.asarray([node["row_group"] for node in test_nodes], dtype=np.int64),
        selected_row_counts=row_counts,
        component_bootstrap_multiplicities=multiplicities,
        arm_bootstrap_wp=np.stack(
            [per_seed_draws[str(seed)][f"{optimizer}__{shape}"] for seed, optimizer, shape in CELLS]
        ),
        contrast_names=np.asarray(
            [f"seed{seed}__{name}" for seed in SEEDS for name in per_seed_contrast_draws[str(seed)]]
        ),
        contrast_bootstrap_diffs=np.stack(
            [
                per_seed_contrast_draws[str(seed)][name]
                for seed in SEEDS
                for name in per_seed_contrast_draws[str(seed)]
            ]
        ),
    )
    result = {
        "kind": "independent E481 recurrent candidate curvature scorer",
        "metric_id": METRIC_ID,
        "scope": "three fixed fit seeds; same 16 previously exposed components; exploratory only; no seed pooling",
        "fit_seeds": list(SEEDS),
        "attempt_receipt": attempt,
        "fit_manifest_sha256": file_sha256(run / "fit_manifest.json"),
        "fit_provenance": role_receipt,
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "normalizer": {
            "source": "independent raw fit-role derivation compared with frozen E470 loader",
            "fit_mean_sha256": hashlib.sha256(mean.astype("<f4").tobytes()).hexdigest(),
            "fit_scale_sha256": hashlib.sha256(scale.astype("<f4").tobytes()).hexdigest(),
            "identity_model_buffers_required": True,
        },
        "source_sha256": {
            "scorer": file_sha256(Path(__file__)),
            "official_gmetric": file_sha256(ROOT / "src/wnn/gmetric.py"),
            "official_evaluation": file_sha256(ROOT / "src/wnn/evaluation.py"),
            "actor_sources": report["source_sha256"],
        },
        "component_ids": component_ids.tolist(),
        "sorted_component_ids": sorted_components.tolist(),
        "row_groups": [node["row_group"] for node in test_nodes],
        "selected_row_counts": row_counts.tolist(),
        "selected_rows_total": int(row_counts.sum()),
        "arm_count": len(arm_scores),
        "arm_scores": arm_scores,
        "contrast_count": contrast_count,
        "contrasts_by_seed": per_seed_contrasts,
        "bootstrap": {
            "method": "official centered weighted-Pearson moments and shared component multiplicity draws",
            "n_boot": BOOTSTRAP_COUNT,
            "seed": BOOTSTRAP_SEED,
            "multiplicity_sha256": hashlib.sha256(
                multiplicities.astype("<i2").tobytes()
            ).hexdigest(),
            "multiplicities_shared_across_all_18_arms_and_contrasts": True,
            "arm_draws_saved": True,
            "all_30_contrast_draws_saved": True,
            "resampling_unit": "numeric-sorted fixed component ID",
            "no_seed_pooling": True,
        },
        "selected_exports_path": export_path.name,
        "selected_exports_sha256": file_sha256(export_path),
    }
    (output / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "arms": 18,
                "contrasts": contrast_count,
                "selected_rows": int(row_counts.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

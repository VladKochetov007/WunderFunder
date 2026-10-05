"""Independent E485 fit-only verifier and paired snapshot scorer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E471_DIR = ROOT / "research/campaign_2026-10-05/recurrent_activation"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
DATA_PATH = ROOT / "datasets/valid.parquet"
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
BOOTSTRAP_COUNT, BOOTSTRAP_SEED = 2_000, 271_828
SEEDS, ALPHAS, SNAPSHOT_EPOCHS = (271_836, 271_837, 271_838), (0.0, 0.1), (1, 2, 4)
ROWS, FEATURES, HIDDEN, BATCH, CHUNK_ROWS = 20_000, 42, 128, 4, 256
UPDATES_PER_EPOCH, UPDATES_PER_TRAJECTORY = 474, 1_896
EXPECTED_PARAMETERS = 236_802
TELEMETRY_STEPS = (0, 46, 236, 473)
METRIC_CLIP = 2.0
FIT_SOURCE_SHA256 = "d9a1b9d243a9c9e50082fe0d9e7d98186d63cb0c566fcae0a20c00ccd42c4b48"
TEST_SOURCE_SHA256 = "4f0579a73a7492ce960582d7f744f9b29152e7b5860219593c9fc63a89005534"
PROBE_SOURCE_SHA256 = "7f3b9a808eb8adbd0856b61fe8bd36ba53df2c1cd1455b697eb748b880d444fd"
UPDATE_GATES_SHA256 = "52d547abd2da11cd712e5973b2ef1ac2eca70b449cbd143abf8141bea221d164"
E470_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E471_SCAN_SHA256 = "998bceb024b5c29e81e73efc5d0ebe434d53c7356cfcb947f7ca36dd625426cf"
E474_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
WNN_NN_SHA256 = "388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3"
WNN_OPTIM_SHA256 = "620569a752fb611adf69976f1dbed1b7882e641672c58175005f5d3f819d2991"
WNN_RESOURCES_SHA256 = "06a624ad963dfc2b0c75ce717278de5f1bb85f48518a99e519b697a371d4cf09"
DESIGN_SHA256 = "bcbf17ec7b632909e9e8792cdf11436822dd4dcac5326befa506c8dcb96e70cc"
CRITIC_SHA256 = "7e5a2457636febfbec13f9cff7d4205757e0cb7de4efbfe9b2deb679481bbdeb"
COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
CLIP_NORM, LR, WEIGHT_DECAY, BETAS = 1.0, 0.001, 0.01, (0.9, 0.999)
MUON_NAMES = {
    f"gru.{prefix}_l{layer}"
    for prefix in ("weight_ih", "weight_hh")
    for layer in range(2)
}
CONTRASTS = (
    ("shape_alpha01_minus_alpha00_epoch1", (0.1, 1), (0.0, 1)),
    ("shape_alpha01_minus_alpha00_epoch2", (0.1, 2), (0.0, 2)),
    ("shape_alpha01_minus_alpha00_epoch4", (0.1, 4), (0.0, 4)),
    ("duration_alpha00_epoch2_minus_epoch1", (0.0, 2), (0.0, 1)),
    ("duration_alpha00_epoch4_minus_epoch1", (0.0, 4), (0.0, 1)),
    ("duration_alpha01_epoch2_minus_epoch1", (0.1, 2), (0.1, 1)),
    ("duration_alpha01_epoch4_minus_epoch1", (0.1, 4), (0.1, 1)),
    ("interaction_shape_by_duration_epoch4_minus_epoch1", (0.1, 4), (0.1, 1)),
)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E471_DIR))
sys.path.insert(0, str(E474_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
from wnn.campaign import RUNTIME_FILES, runtime_identity  # noqa: E402
from wnn.campaign_spec import digest as spec_digest  # noqa: E402
from wnn.campaign_spec import validate_spec  # noqa: E402
from wnn.evaluation import _score_component_moments, paired_component_moments  # noqa: E402
from wnn.gmetric import METRIC_ID, global_wp  # noqa: E402


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Fail closed when any registered evidence differs."""
    if not condition:
        raise ValueError(message)


def finite(value: Any, label: str) -> float:
    """Convert a finite numeric receipt scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is nonfinite")
    return result


def tensor_state_sha256(state: dict[str, Tensor]) -> str:
    """Hash tensor state by sorted names, dtype, shape, and contiguous bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.jit.script
def independent_candidate_scan(
    x: Tensor,
    h0: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor]:
    """Independent reset-after PyTorch-GRU equation with candidate-only shape."""
    input_gates = F.linear(x, weight_ih, bias_ih)
    outputs = torch.jit.annotate(list[Tensor], [])
    hidden = h0
    for step in range(x.size(1)):
        input_reset, input_update, input_candidate = input_gates[:, step, :].chunk(3, dim=1)
        hidden_gates = F.linear(hidden, weight_hh, bias_hh)
        hidden_reset, hidden_update, hidden_candidate = hidden_gates.chunk(3, dim=1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        u = torch.tanh(input_candidate + reset * hidden_candidate)
        candidate = u + alpha * torch.sin(math.pi * u)
        hidden = (1.0 - update) * candidate + update * hidden
        outputs.append(hidden)
    return torch.stack(outputs, dim=1), hidden


class IndependentDurationRNN(e474.PlacementRNN):
    """Rebuild the H128 native topology, replacing only recurrent candidate math."""

    def __init__(self, alpha: float) -> None:
        if alpha not in ALPHAS:
            raise ValueError(f"unregistered E485 alpha {alpha}")
        super().__init__("gelu", "gelu", hidden=HIDDEN)
        self.alpha = float(alpha)

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != FEATURES:
            raise ValueError("E485 inputs must be [batch,time,42]")
        encoded = self.encode((x - self.mean) / self.std)
        batch = encoded.shape[0]
        if state is None:
            hidden_state = encoded.new_zeros((2, batch, HIDDEN))
        else:
            if state.shape != (2, batch, HIDDEN):
                raise ValueError("E485 state must be [2,batch,128]")
            hidden_state = state
        layer_input = encoded
        final_states: list[Tensor] = []
        for layer in range(2):
            layer_output, final_hidden = independent_candidate_scan(
                layer_input,
                hidden_state[layer],
                getattr(self.gru, f"weight_ih_l{layer}"),
                getattr(self.gru, f"weight_hh_l{layer}"),
                getattr(self.gru, f"bias_ih_l{layer}"),
                getattr(self.gru, f"bias_hh_l{layer}"),
                self.alpha,
            )
            layer_input = layer_output
            final_states.append(final_hidden)
        final_state = torch.stack(final_states, dim=0)
        prediction = self.head(torch.cat((layer_input, encoded), dim=-1))
        if not (torch.isfinite(prediction).all() and torch.isfinite(final_state).all()):
            raise ValueError("E485 scorer produced nonfinite prediction/state")
        return prediction, final_state


def validate_parent_sources() -> dict[str, str]:
    """Verify all E485 model, data-processing, optimizer, and prereg sources."""
    expected = {
        "fit_duration.py": (HERE / "fit_duration.py", FIT_SOURCE_SHA256),
        "tests.py": (HERE / "tests.py", TEST_SOURCE_SHA256),
        "gpu_probe.py": (HERE / "gpu_probe.py", PROBE_SOURCE_SHA256),
        "GPU_PROBE_UPDATE_GATES.md": (HERE / "GPU_PROBE_UPDATE_GATES.md", UPDATE_GATES_SHA256),
        "E470": (E470_DIR / "fit_activation_optimizer.py", E470_SHA256),
        "E471 scan": (E471_DIR / "fit_recurrent_candidate.py", E471_SCAN_SHA256),
        "E474": (E474_DIR / "fit_activation_placement.py", E474_SHA256),
        "wnn.nn": (ROOT / "src/wnn/nn.py", WNN_NN_SHA256),
        "wnn.optim": (ROOT / "src/wnn/optim.py", WNN_OPTIM_SHA256),
        "wnn.resources": (ROOT / "src/wnn/resources.py", WNN_RESOURCES_SHA256),
        "design": (HERE / "E485_DESIGN.md", DESIGN_SHA256),
        "blind critic": (HERE / "SPEC_CRITIC.md", CRITIC_SHA256),
        "cohort": (COHORT_PATH, COHORT_SHA256),
        "dataset": (DATA_PATH, DATASET_SHA256),
    }
    observed = {name: sha256_file(path) for name, (path, _digest) in expected.items()}
    for name, (_path, digest) in expected.items():
        require(observed[name] == digest, f"E485 source/data identity differs: {name}")
    return observed


def validate_spec_sources(spec: dict, expected_artifact: str) -> dict[str, str]:
    """Verify the complete campaign spec's repository-local byte inventory."""
    checked = validate_spec(spec)
    require(checked == spec, "E485 spec is not canonical")
    require(spec.get("kind") == "gpu" and spec.get("artifacts") == [expected_artifact],
            "E485 spec kind/artifact differs")
    code, inputs = spec.get("code"), spec.get("inputs")
    require(isinstance(code, list) and isinstance(inputs, list), "E485 code/input inventory missing")
    observed: dict[str, str] = {}
    for label, entries in (("code", code), ("input", inputs)):
        seen = set()
        for entry in entries:
            relative = entry.get("path")
            require(isinstance(relative, str) and relative not in seen, f"duplicate E485 {label} path")
            seen.add(relative)
            rel = Path(relative)
            require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe E485 path {relative}")
            path = ROOT / rel
            require(not path.is_symlink() and path.is_file(), f"missing/symlink E485 {label}: {relative}")
            require(path.stat().st_size == entry.get("size"), f"E485 {label} size differs: {relative}")
            digest = sha256_file(path)
            require(digest == entry.get("sha256"), f"E485 {label} hash differs: {relative}")
            observed[relative] = digest
    required = {
        "research/campaign_2026-10-05/activation_duration/fit_duration.py": FIT_SOURCE_SHA256,
        "research/campaign_2026-10-05/activation_duration/tests.py": TEST_SOURCE_SHA256,
        "research/campaign_2026-10-05/activation_duration/gpu_probe.py": PROBE_SOURCE_SHA256,
        "research/campaign_2026-10-05/activation_duration/GPU_PROBE_UPDATE_GATES.md": UPDATE_GATES_SHA256,
        "research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py": E470_SHA256,
        "research/campaign_2026-10-05/recurrent_activation/fit_recurrent_candidate.py": E471_SCAN_SHA256,
        "research/campaign_2026-10-05/activation_placement/fit_activation_placement.py": E474_SHA256,
        "src/wnn/nn.py": WNN_NN_SHA256,
        "src/wnn/optim.py": WNN_OPTIM_SHA256,
        "src/wnn/resources.py": WNN_RESOURCES_SHA256,
    }
    code_map = {entry["path"]: entry["sha256"] for entry in code}
    for path, digest in required.items():
        require(code_map.get(path) == digest and observed.get(path) == digest,
                f"E485 spec omits or changes required source {path}")
    input_map = {entry["path"]: entry["sha256"] for entry in inputs}
    for path, digest in {
        "datasets/valid.parquet": DATASET_SHA256,
        "research/campaign_2026-10-05/target_mechanics/cohort.json": COHORT_SHA256,
        "research/campaign_2026-10-05/activation_duration/E485_DESIGN.md": DESIGN_SHA256,
        "research/campaign_2026-10-05/activation_duration/SPEC_CRITIC.md": CRITIC_SHA256,
    }.items():
        require(input_map.get(path) == digest and observed.get(path) == digest,
                f"E485 spec omits or changes required input {path}")
    return observed


def validate_attempt(path: Path, spec_path: Path, phase: str) -> tuple[dict, dict, Path]:
    """Verify successful local attempt, runtime, resources, and one proof artifact."""
    spec = json.loads(spec_path.read_text())
    checked = validate_spec(spec)
    record = json.loads(path.read_text())
    require(record.get("kind") == "gpu" and record.get("spec") == checked
            and record.get("spec_sha256") == spec_digest(checked),
            f"E485 {phase} attempt does not bind its spec")
    argv = record.get("argv")
    require(isinstance(argv, list) and argv and Path(argv[0]).is_absolute()
            and argv[1:] == checked.get("argv", [])[1:], f"E485 {phase} argv mismatch")
    require(record.get("status") == "finished" and record.get("exit_code") == 0
            and record.get("signals") == [] and record.get("teardown_verified") is True
            and all(record.get(key) is None for key in ("failure", "failure_scope", "failure_category")),
            f"E485 {phase} attempt did not finish cleanly")
    unit = record.get("unit_properties", {})
    require(unit.get("Result") == "success" and unit.get("ExecMainStatus") in (0, "0"),
            f"E485 {phase} systemd unit failed")
    runtime = runtime_identity(ROOT)
    require(record.get("runtime_sha256") == runtime and set(runtime) == set(RUNTIME_FILES),
            f"E485 {phase} runtime identity differs")
    oom_keys = ("oom", "oom_kill", "oom_group_kill")
    for before_key, after_key in (("initial_memory_events", "final_memory_events"),
                                  ("initial_parent_memory_events", "final_parent_memory_events")):
        before, after = record.get(before_key), record.get(after_key)
        require(isinstance(before, dict) and isinstance(after, dict), f"E485 {phase} OOM events missing")
        require(all(type(before.get(k)) is int and type(after.get(k)) is int
                    and before[k] >= 0 and before[k] == after[k] for k in oom_keys),
                f"E485 {phase} OOM event increased")
    require(all(record["final_memory_events"].get(key) == 0 for key in oom_keys),
            f"E485 {phase} worker OOM counter nonzero")
    limits = checked.get("limits", {})
    require(type(record.get("peak_ram_bytes")) is int
            and record["peak_ram_bytes"] < limits.get("ram_bytes", 0),
            f"E485 {phase} host RAM cap exceeded/missing")
    require(type(record.get("peak_observed_vram_bytes")) is int
            and record["peak_observed_vram_bytes"] < limits.get("gpu_stop_bytes", 0),
            f"E485 {phase} physical GPU cap exceeded/missing")
    artifact_paths = checked.get("artifacts", [])
    require(len(artifact_paths) == 1 and isinstance(artifact_paths[0], str),
            f"E485 {phase} must declare exactly one proof artifact")
    artifact_rel = artifact_paths[0]
    rel = Path(artifact_rel)
    require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe E485 artifact path {artifact_rel}")
    artifact = ROOT / rel
    require(not artifact.is_symlink() and artifact.is_file(), f"E485 {phase} artifact missing")
    expected_inventory = [{"path": artifact_rel, "sha256": sha256_file(artifact),
                           "size": artifact.stat().st_size}]
    require(record.get("artifacts") == expected_inventory, f"E485 {phase} artifact inventory differs")
    require(str(artifact.parent.relative_to(ROOT)) in checked.get("argv", []),
            f"E485 {phase} output path differs from artifact directory")
    return record, checked, artifact


def expected_precision(report: dict) -> None:
    """Require saved full-IEEE backend flags for the four-epoch run."""
    precision = report.get("active_precision", {})
    require(precision.get("device_math") == "FP32 IEEE" and precision.get("autocast") is False
            and all(precision.get(key) == "ieee" for key in (
                "torch.backends.fp32_precision", "torch.backends.cuda.matmul.fp32_precision",
                "torch.backends.cudnn.fp32_precision", "torch.backends.cudnn.rnn.fp32_precision")),
            "E485 full fit lacks inside-context IEEE precision receipt")


def validate_probe_report(report: dict, probe_attempt: dict, fit_attempt: dict,
                          fit_manifest: dict) -> dict:
    """Validate probe parity, strict update-error gates, all six cells, and continuity."""
    require(report.get("kind") == "E485 actual H128 B4x256 CUDA probe"
            and report.get("status") == "passed" and report.get("fit_only") is True
            and report.get("test_rows_loaded") is False, "E485 CUDA probe did not pass fit-only status")
    require(report.get("dataset_sha256") == DATASET_SHA256
            and report.get("cohort_sha256") == COHORT_SHA256, "E485 probe data identity differs")
    require(report.get("source_sha256") == {
        "fit_duration.py": FIT_SOURCE_SHA256,
        "tests.py": TEST_SOURCE_SHA256,
        "gpu_probe.py": PROBE_SOURCE_SHA256,
        "GPU_PROBE_UPDATE_GATES.md": UPDATE_GATES_SHA256,
    }, "E485 probe source hashes differ")
    require(report.get("parent_source_sha256") == {
        "e470": E470_SHA256, "e471_scan": E471_SCAN_SHA256, "e474": E474_SHA256,
        "wnn.nn": WNN_NN_SHA256, "wnn.optim": WNN_OPTIM_SHA256,
        "wnn.resources": WNN_RESOURCES_SHA256,
    }, "E485 probe parent-source hashes differ")
    require(report.get("active_precision_inside_context", {}).get("device_math") == "FP32 IEEE"
            and all(report["active_precision_inside_context"].get(key) == "ieee" for key in (
                "torch.backends.fp32_precision", "torch.backends.cuda.matmul.fp32_precision",
                "torch.backends.cudnn.fp32_precision", "torch.backends.cudnn.rnn.fp32_precision")),
            "E485 probe did not capture active precision inside context")
    require(report.get("optimizer_cuda_orthogonalization") == "existing wnn.optim BF16 orthogonalizer",
            "E485 probe misstates CUDA Muon orthogonalization")
    limits = report.get("resource_limits", {})
    require(limits.get("allocator_cap_bytes") == 2 * 1024**3
            and limits.get("observed_stop_bytes") == 3_000_000_000,
            "E485 probe resource limits differ")
    memory = limits.get("after", {})
    require(memory.get("peak_allocated_bytes", 2**63) <= 2 * 1024**3
            and memory.get("observed_used_after_bytes", 2**63) <= 3_000_000_000,
            "E485 probe memory receipt exceeds limits")
    helper = report.get("shared_gradient_helper_exactness", {})
    require(helper.get("clip_activated") is True and helper.get("parameters_byte_exact") is True
            and helper.get("optimizer_state_byte_exact") is True,
            "E485 shared-gradient frozen-helper parity failed")
    native_checks = report.get("native_alpha_zero_parity", [])
    require(len(native_checks) == len(SEEDS), "E485 native alpha-zero checks incomplete")
    for check in native_checks:
        require(check.get("seed") in SEEDS and check.get("mode") == "train"
                and check.get("state_shape") == [2, 4, HIDDEN]
                and check.get("cpu_initial_bytes_match_native_and_custom") is True,
                "E485 native/custom actual-shape parity metadata differs")
        for gate, data in check.get("update_delta_gates", {}).items():
            require(data.get("passed") is True, f"E485 probe's registered delta gate failed: {gate}")
            ref = finite(data.get("reference_update_rms"), f"E485 reference update RMS {gate}")
            err = finite(data.get("error_update_rms"), f"E485 update RMS error {gate}")
            require(err <= 0.02 * ref + 1e-10,
                    f"E485 stricter controller RMS gate failed: {gate} ({err}/{ref})")
            maxima = data.get("max_abs_error_by_tensor", {})
            require(maxima and all(finite(value, f"E485 max update error {gate}") <= 2e-5
                                   for value in maxima.values()),
                    f"E485 stricter controller max-absolute gate failed: {gate}")
    expected_cells = {(seed, alpha) for seed in SEEDS for alpha in ALPHAS}
    observed_cells = set()
    updates = report.get("six_actual_fit_updates", [])
    for update in updates:
        key = (update.get("seed"), update.get("alpha"))
        require(key in expected_cells and key not in observed_cells,
                f"unknown/duplicate E485 probe fit update {key}")
        observed_cells.add(key)
        require(update.get("batch") == BATCH and update.get("chunk_rows") == CHUNK_ROWS
                and update.get("state_shape") == [2, BATCH, HIDDEN]
                and update.get("parameters_changed") is True and update.get("parameters_finite") is True,
                f"E485 probe update shape/finite/change differs: {key}")
        finite(update.get("loss"), f"E485 probe loss {key}")
    require(observed_cells == expected_cells, "E485 probe omitted a seed/shape cell")
    projection = report.get("timing_projection", {})
    projected = finite(projection.get("projected_matrix_seconds"), "E485 full-matrix projection")
    require(projection.get("projection_fits_cap") is True and projected <= 7_200
            and projection.get("admission_cap_seconds") == 7_200,
            "E485 measured projection exceeds its admission budget")
    require(probe_attempt.get("environment", {}).get("CUDA_VISIBLE_DEVICES")
            == fit_attempt.get("environment", {}).get("CUDA_VISIBLE_DEVICES"),
            "E485 probe/full fit CUDA visibility differs")
    require(fit_manifest.get("source_sha256") == report.get("source_sha256"),
            "E485 probe and fit source identities differ")
    return {"probe_cells": len(observed_cells), "native_parity_seeds": len(native_checks),
            "strict_update_delta_gates_passed": sum(
                len(c.get("update_delta_gates", {})) for c in native_checks),
            "helper_exact": True, "measured_projection_seconds": projected}


def validate_checkpoint(path: Path, expected_name: str) -> tuple[IndependentDurationRNN, dict[str, Tensor]]:
    """Load and validate an exact H128 snapshot without actor-model imports."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected_keys = set(IndependentDurationRNN(0.0).state_dict())
    require(isinstance(state, dict) and set(state) == expected_keys,
            f"E485 checkpoint state keys differ: {expected_name}")
    require(all(isinstance(value, Tensor) and torch.isfinite(value).all()
                for value in state.values()), f"E485 checkpoint contains nonfinite/malformed values: {expected_name}")
    require(torch.equal(state["mean"], torch.zeros(FEATURES))
            and torch.equal(state["std"], torch.ones(FEATURES)),
            f"E485 checkpoint model buffers differ from identity: {expected_name}")
    model = IndependentDurationRNN(0.0)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"E485 checkpoint shape mismatch: {expected_name}") from exc
    require(sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS,
            f"E485 independent parameter count differs: {expected_name}")
    return model, state


def canonical_fit_nodes() -> tuple[list[dict], list[dict]]:
    """Read role descriptors and reject fit/test component overlap."""
    require(sha256_file(COHORT_PATH) == COHORT_SHA256, "E485 cohort bytes differ")
    document = json.loads(COHORT_PATH.read_text())
    all_nodes = document.get("cohort", [])
    fit = [node for node in all_nodes if node.get("role") == "fit"]
    test = [node for node in all_nodes if node.get("role") == "test"]
    require(len(all_nodes) == 40 and len(fit) == 24 and len(test) == 16,
            "E485 cohort roles/counts differ")
    require(not ({n["component"] for n in fit} & {n["component"] for n in test}),
            "E485 fit and test components overlap")
    return fit, test


def validate_normalizer(manifest: dict, fit_nodes: list[dict], fit_x: np.ndarray,
                        fit_only_manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the E470 fit-only normalizer from raw fit features."""
    require(len(fit_nodes) == 24 and all(node["role"] == "fit" for node in fit_nodes),
            "E485 normalizer must use exactly 24 fit roles")
    parquet = pq.ParquetFile(DATA_PATH)
    raw_features = []
    for node in fit_nodes:
        table = parquet.read_row_group(node["row_group"], columns=e470.FEATURES, use_threads=False)
        raw = np.column_stack([table[column].to_numpy() for column in e470.FEATURES])
        raw_features.append(e470.causal_features(raw))
    raw_x = np.stack(raw_features)
    mean = raw_x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(raw_x.std(axis=(0, 1), dtype=np.float64), 0.001).astype(np.float32)
    normalization = fit_only_manifest.get("normalization", {})
    observed_mean = np.asarray(normalization.get("fit_mean"), dtype=np.float32)
    observed_scale = np.asarray(normalization.get("fit_scale"), dtype=np.float32)
    require(np.array_equal(mean, observed_mean) and np.array_equal(scale, observed_scale),
            "E485 loader fit normalizer differs from raw fit-role recomputation")
    require(manifest.get("cohort_manifest", {}).get("normalization") == normalization,
            "E485 fit-only cohort normalizer differs from loader normalizer")
    require(fit_x.shape == (24, ROWS, FEATURES) and np.isfinite(fit_x).all(),
            "E485 normalized fit features have invalid shape/values")
    for index, raw_sequence in enumerate(raw_x):
        normalized = np.clip((raw_sequence - mean) / scale, -8, 8).astype(np.float32)
        require(np.array_equal(normalized, fit_x[index]),
                f"E485 normalized features differ at fit row group {fit_nodes[index]['row_group']}")
    return mean, scale


def expected_epoch_order(seed: int, epoch_index: int, fit_nodes: list[dict]) -> list[int]:
    """Return row groups under the exact registered canonical-order shuffle."""
    order = np.arange(24)
    np.random.default_rng(seed + epoch_index).shuffle(order)
    return [fit_nodes[int(index)]["row_group"] for index in order]


def validate_telemetry(epoch: dict, seed: int, alpha: float, epoch_index: int,
                       row_groups: list[int]) -> None:
    """Check all four actual progress samples and their batch/chunk coordinates."""
    telemetry = epoch.get("telemetry")
    require(isinstance(telemetry, list)
            and [row.get("epoch_step_index_zero_based") for row in telemetry]
            == list(TELEMETRY_STEPS), f"E485 telemetry cadence differs {seed}/{alpha}/{epoch_index}")
    for row in telemetry:
        step = row["epoch_step_index_zero_based"]
        require(row.get("global_update_index_zero_based") == epoch_index * UPDATES_PER_EPOCH + step,
                f"E485 telemetry global update index differs {seed}/{alpha}/{epoch_index}/{step}")
        require(row.get("epoch_index_zero_based") == epoch_index,
                f"E485 telemetry epoch differs {seed}/{alpha}/{epoch_index}/{step}")
        batch_index = step // math.ceil(ROWS / CHUNK_ROWS)
        expected_batch = row_groups[batch_index * BATCH:(batch_index + 1) * BATCH]
        expected_chunk = (step % math.ceil(ROWS / CHUNK_ROWS)) * CHUNK_ROWS
        require(row.get("row_groups") == expected_batch and row.get("chunk_start") == expected_chunk,
                f"E485 telemetry sampled rows/chunk differ {seed}/{alpha}/{epoch_index}/{step}")
        for field in ("loss",):
            require(finite(row.get(field), f"E485 telemetry {field}") >= 0,
                    f"E485 telemetry {field} is negative")
        for field in ("preclip_gradient_rms", "parameter_update_rms"):
            families = row.get(field)
            require(isinstance(families, dict) and set(families) == {"inp", "gru", "head"},
                    f"E485 telemetry family names differ: {field}")
            require(all(finite(value, f"E485 {field} family") >= 0 for value in families.values()),
                    f"E485 telemetry family is invalid: {field}")


def validate_trajectory(run: Path, trajectory: dict, seed: int, alpha: float,
                        fit_nodes: list[dict]) -> dict[tuple[int, float, int], dict]:
    """Validate four continuous epochs, full order receipts, checkpoints and progress."""
    require(trajectory.get("seed") == seed and trajectory.get("alpha") == alpha
            and trajectory.get("status") == "finished"
            and trajectory.get("parameters") == EXPECTED_PARAMETERS
            and trajectory.get("updates") == UPDATES_PER_TRAJECTORY,
            f"E485 trajectory identity/status/count differs: {seed}/{alpha}")
    require(finite(trajectory.get("fit_seconds"), "E485 trajectory duration") > 0,
            f"E485 trajectory duration is invalid: {seed}/{alpha}")
    epochs = trajectory.get("epochs")
    require(isinstance(epochs, list) and len(epochs) == 4,
            f"E485 trajectory must have all four epochs: {seed}/{alpha}")
    checkpoints: dict[tuple[int, float, int], dict] = {}
    cumulative_clipped = 0
    all_fit_rows = 0
    for epoch_index, epoch in enumerate(epochs):
        epoch_number = epoch_index + 1
        expected_rows = expected_epoch_order(seed, epoch_index, fit_nodes)
        require(epoch.get("epoch") == epoch_number and epoch.get("updates_this_epoch") == UPDATES_PER_EPOCH
                and epoch.get("total_updates") == (epoch_index + 1) * UPDATES_PER_EPOCH,
                f"E485 epoch/update count differs: {seed}/{alpha}/{epoch_number}")
        require(epoch.get("fit_row_groups_in_order") == expected_rows
                and len(expected_rows) == 24 and len(set(expected_rows)) == 24,
                f"E485 row traversal differs: {seed}/{alpha}/{epoch_number}")
        clipped = epoch.get("clipped_updates")
        require(type(clipped) is int and 0 <= clipped <= UPDATES_PER_EPOCH,
                f"E485 clipped-step count invalid: {seed}/{alpha}/{epoch_number}")
        cumulative_clipped += clipped
        require(epoch.get("cumulative_clipped_updates") == cumulative_clipped,
                f"E485 cumulative clipping count differs: {seed}/{alpha}/{epoch_number}")
        validate_telemetry(epoch, seed, alpha, epoch_index, expected_rows)
        snapshot = epoch.get("snapshot")
        if epoch_number in SNAPSHOT_EPOCHS:
            expected_name = f"seed{seed}_alpha{alpha:.1f}_epoch{epoch_number}.pt"
            require(isinstance(snapshot, dict) and snapshot.get("checkpoint") == expected_name,
                    f"E485 checkpoint receipt missing: {seed}/{alpha}/{epoch_number}")
            checkpoint_path = run / expected_name
            require(checkpoint_path.is_file() and not checkpoint_path.is_symlink()
                    and sha256_file(checkpoint_path) == snapshot.get("sha256"),
                    f"E485 checkpoint SHA differs: {expected_name}")
            _model, state = validate_checkpoint(checkpoint_path, expected_name)
            checkpoints[(seed, alpha, epoch_number)] = {
                "checkpoint": expected_name,
                "checkpoint_sha256": snapshot["sha256"],
                "state": state,
            }
        else:
            require(snapshot is None, f"E485 unexpected epoch snapshot: {seed}/{alpha}/{epoch_number}")
        all_fit_rows += len(epoch.get("fit_row_groups_in_order", []))
    require(all_fit_rows == 96 and trajectory.get("clipped_updates") == cumulative_clipped,
            f"E485 trajectory row/clipping summary differs: {seed}/{alpha}")
    progress_path = run / f"seed{seed}_alpha{alpha:.1f}_progress.json"
    require(progress_path.is_file() and not progress_path.is_symlink(),
            f"E485 per-epoch progress receipt missing: {seed}/{alpha}")
    progress = json.loads(progress_path.read_text())
    require(progress.get("seed") == seed and progress.get("alpha") == alpha
            and progress.get("completed_epochs") == 4
            and progress.get("updates") == UPDATES_PER_TRAJECTORY
            and progress.get("clipped_updates") == cumulative_clipped
            and progress.get("epochs") == epochs,
            f"E485 progress/fit manifest diverges: {seed}/{alpha}")
    return checkpoints


def validate_fit_report(run: Path, report: dict, fit_only_manifest: dict,
                        fit_nodes: list[dict], fit_x: np.ndarray) -> dict[tuple[int, float, int], dict]:
    """Verify fit-only contract, exact sources, all six trajectories and 18 snapshots."""
    require(report.get("kind") == "E485 recurrent activation duration screen"
            and report.get("status") == "all_six_trajectories_complete_test_unread"
            and report.get("fit_only") is True and report.get("test_rows_loaded") is False,
            "E485 full-fit report incomplete or violates test boundary")
    require(report.get("seeds") == list(SEEDS) and report.get("alphas") == list(ALPHAS)
            and report.get("hidden") == HIDDEN and report.get("parameters_per_trajectory") == EXPECTED_PARAMETERS
            and report.get("epochs") == 4 and report.get("snapshots_per_trajectory") == list(SNAPSHOT_EPOCHS)
            and report.get("updates_per_epoch") == UPDATES_PER_EPOCH
            and report.get("batch_size") == BATCH and report.get("chunk_rows") == CHUNK_ROWS,
            "E485 registered matrix/shape differs")
    require(report.get("candidate") == "u=tanh(v); n=u+alpha*sin(pi*u); h'=(1-z)*n+z*h",
            "E485 candidate equation label differs")
    require(report.get("optimizer") == {
        "name": "MuonAdamW", "muon_names": sorted(MUON_NAMES), "muon_blocks": 3,
        "muon_beta1": 0.9, "adam_betas": list(BETAS), "lr": LR,
        "weight_decay": WEIGHT_DECAY, "schedule": "constant", "global_clip_norm": CLIP_NORM,
        "state_continues_across_epochs": True,
        "cuda_muon_orthogonalization": "existing wnn.optim BF16 implementation",
    }, "E485 optimizer recipe/parameter partition differs")
    require(report.get("shuffle") == "canonical np.arange(24), default_rng(seed+epoch_index).shuffle",
            "E485 cohort shuffle recipe differs")
    require(report.get("dataset_sha256") == DATASET_SHA256
            and report.get("cohort_sha256") == COHORT_SHA256,
            "E485 fit data/cohort hash differs")
    require(report.get("design_sha256") == DESIGN_SHA256
            and report.get("blind_critic_sha256") == CRITIC_SHA256
            and report.get("probe_update_gates_sha256") == UPDATE_GATES_SHA256,
            "E485 registered design/source gate hashes differ")
    parent = {"e470": E470_SHA256, "e471_scan": E471_SCAN_SHA256, "e474": E474_SHA256,
              "wnn.nn": WNN_NN_SHA256, "wnn.optim": WNN_OPTIM_SHA256,
              "wnn.resources": WNN_RESOURCES_SHA256}
    require(report.get("parent_source_sha256") == parent, "E485 parent source hashes differ")
    source_map = {"fit_duration.py": FIT_SOURCE_SHA256, "tests.py": TEST_SOURCE_SHA256,
                  "gpu_probe.py": PROBE_SOURCE_SHA256,
                  "GPU_PROBE_UPDATE_GATES.md": UPDATE_GATES_SHA256}
    require(report.get("source_sha256") == source_map, "E485 actor source hash map differs")
    expected_precision(report)
    require(report.get("cohort_manifest") == fit_only_manifest,
            "E485 fit-only cohort manifest differs from independent loader")
    manifest_file = run / "fit_only_cohort.json"
    require(manifest_file.is_file() and json.loads(manifest_file.read_text()) == fit_only_manifest,
            "E485 standalone fit-only cohort receipt differs")
    expected_order = {}
    for seed in SEEDS:
        orders = [expected_epoch_order(seed, epoch, fit_nodes) for epoch in range(4)]
        expected_order[seed] = orders
        require(any(orders[0] != order for order in orders[1:]),
                f"E485 epoch order did not change for seed {seed}")
    trajectories = report.get("trajectories", [])
    require(len(trajectories) == 6, "E485 fit manifest omits trajectories")
    expected_keys = [(seed, alpha) for seed in SEEDS for alpha in ALPHAS]
    observed_keys = [(item.get("seed"), item.get("alpha")) for item in trajectories]
    require(observed_keys == expected_keys, "E485 trajectory order/identities differ")
    checkpoints: dict[tuple[int, float, int], dict] = {}
    for trajectory in trajectories:
        key = (trajectory["seed"], trajectory["alpha"])
        checkpoints.update(validate_trajectory(run, trajectory["seed"], trajectory["alpha"], fit_nodes))
    require(len(checkpoints) == 18, "E485 full run must contain all 18 snapshots")
    for seed in SEEDS:
        for epoch in SNAPSHOT_EPOCHS:
            baseline = checkpoints[(seed, 0.0, epoch)]["state"]
            sine = checkpoints[(seed, 0.1, epoch)]["state"]
            require(set(baseline) == set(sine), f"E485 paired snapshot state keys differ: {seed}/{epoch}")
    require(fit_x.shape == (24, ROWS, FEATURES) and np.isfinite(fit_x).all(),
            "E485 independent fit loader shape/finite check failed")
    return checkpoints


def validate_run(run: Path, fit_spec_path: Path, fit_attempt_path: Path,
                 probe_run: Path, probe_spec_path: Path, probe_attempt_path: Path):
    """Run all fit/probe/source/attempt gates without opening test-role groups."""
    fit_attempt, fit_spec, fit_artifact = validate_attempt(fit_attempt_path, fit_spec_path, "fit")
    probe_attempt, probe_spec, probe_artifact = validate_attempt(probe_attempt_path, probe_spec_path, "probe")
    require(run.resolve() == fit_artifact.parent.resolve(), "E485 fit directory differs from proof artifact")
    require(probe_run.resolve() == probe_artifact.parent.resolve(), "E485 probe directory differs from proof artifact")
    validate_spec_sources(fit_spec, str(fit_artifact.relative_to(ROOT)))
    validate_spec_sources(probe_spec, str(probe_artifact.relative_to(ROOT)))
    parent_sources = validate_parent_sources()
    fit_nodes, _test_nodes = canonical_fit_nodes()
    fit_only_manifest, fit_x, _fit_y, _need, _score, loaded_nodes = e470.load_cohort(include_test=False)
    require(loaded_nodes == fit_nodes, "E470 fit-only loader order differs from canonical roles")
    mean, scale = validate_normalizer(fit_only_manifest, fit_nodes, fit_x, fit_only_manifest)
    report = json.loads(fit_artifact.read_text())
    checkpoints = validate_fit_report(run, report, fit_only_manifest, fit_nodes, fit_x)
    probe = json.loads(probe_artifact.read_text())
    probe_receipt = validate_probe_report(probe, probe_attempt, fit_attempt, report)
    proof = {
        "fit_attempt_id": fit_attempt["id"], "fit_attempt_sha256": sha256_file(fit_attempt_path),
        "fit_spec_sha256": sha256_file(fit_spec_path), "fit_manifest_sha256": sha256_file(fit_artifact),
        "probe_attempt_id": probe_attempt["id"], "probe_attempt_sha256": sha256_file(probe_attempt_path),
        "probe_spec_sha256": sha256_file(probe_spec_path), "probe_artifact_sha256": sha256_file(probe_artifact),
        "source_hashes": parent_sources, "snapshot_count": len(checkpoints),
        "normalizer_sha256": hashlib.sha256(mean.astype("<f4").tobytes() + scale.astype("<f4").tobytes()).hexdigest(),
        "probe": probe_receipt, "test_boundary": "all 24 fit roles verified; no test roles loaded",
    }
    return proof, checkpoints, report, fit_only_manifest, fit_nodes

"""Independently verify and score the frozen E484 twelve-fit run.

The scorer reconstructs all six readouts itself over the frozen native GRU.
Test sequences are inaccessible to the fit-only verifier; scoring requires a
separate explicit flag after the fit, probe, and controller receipts pass.
"""

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
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
E477_DIR = ROOT / "research/campaign_2026-10-05/activation_head_mixture"
ACTOR_FIT = HERE / "fit_context_mixture_v2.py"
ACTOR_PROBE = HERE / "gpu_probe_v2.py"
ACTOR_TESTS = HERE / "tests_v2.py"
DATA_PATH = ROOT / "datasets/valid.parquet"
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
BOOTSTRAP_COUNT, BOOTSTRAP_SEED = 2_000, 271_834
SEEDS = (271_834, 271_835)
ARMS = ("gelu", "fixed_mix", "static_mix", "recurrent_mix", "projection_mix", "self_mix")
MIXTURE_ARMS = ARMS[1:]
DYNAMIC_SOURCES = {
    "recurrent_mix": "recurrent",
    "projection_mix": "projection",
    "self_mix": "head_self",
}
EXPECTED_PARAMETERS = {"gelu": 61_058, "fixed_mix": 61_058, **{arm: 61_122 for arm in ARMS[2:]}}
FEATURES, HIDDEN, ROWS, CHUNK_ROWS = 42, 64, 20_000, 256
BATCH, UPDATES = 4, 474
TELEMETRY_STEPS = (0, 46, 236, 473)
GATE_EPSILON, METRIC_CLIP = 1e-6, 2.0
FIT_SOURCE_V2_SHA256 = "2dec963a966ba30e1dc3deed40472fd2d0129a2d24fa07585a3a7ed4e2e1dcbc"
PROBE_SOURCE_V2_SHA256 = "06e4b056dd0c9ea3eeab3d7725918801a6796b68d7f82b734e191aa366a2018d"
TEST_SOURCE_V2_SHA256 = "0d7030f0197c949c3cd72ea0b1e338cfe8b86e66b7de69b9f00853a6f154c9d2"
FIT_SPEC_V2_SHA256 = "33f0c8d9213a90c938f2986b7fdf964b6add3c076376d37fa2eec994934d4fe4"
PROBE_SPEC_V2_SHA256 = "2c08f4e92a02f717ed4209d4d41c05cf09b05623cf61e305facb285bf7c0317b"
E470_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E474_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
E477_SHA256 = "2d4b1b103dd31912fea7ab24a63e6ce176e532dca8d0dfa2cdb32aebd9e567d7"
WNN_NN_SHA256 = "388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3"
WNN_RESOURCES_SHA256 = "06a624ad963dfc2b0c75ce717278de5f1bb85f48518a99e519b697a371d4cf09"
GMETRIC_SHA256 = "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa"
EVALUATION_SHA256 = "96cebf2c8d564493f5a3a866b8d879964247f885587a455ebb1449e937230d0e"
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
DESIGN_SHA256 = "e2657b85e31c10d0bab3edab066ba71faa6f2eee79c5cddfd9aff86c6f89b398"
CRITIC_SHA256 = "7f1325b9f9517199bb0a2af08a4a86a7625c917814458bf17a254fe9b79303d4"
CRITIC_RESPONSE_SHA256 = "9325be932eb0427db6631bc616b35dfbe285da78c876df99c966b19b8817315e"
CLIP_NORM, LR, WEIGHT_DECAY, BETAS = 1.0, 1e-3, 0.01, (0.9, 0.999)
CONTRASTS = (
    ("recurrent_minus_static", "recurrent_mix", "static_mix"),
    ("recurrent_minus_projection", "recurrent_mix", "projection_mix"),
    ("recurrent_minus_self", "recurrent_mix", "self_mix"),
    ("static_minus_fixed", "static_mix", "fixed_mix"),
    ("recurrent_minus_fixed", "recurrent_mix", "fixed_mix"),
    ("projection_minus_fixed", "projection_mix", "fixed_mix"),
    ("self_minus_fixed", "self_mix", "fixed_mix"),
    ("fixed_minus_gelu", "fixed_mix", "gelu"),
)
REGISTERED_CONTRASTS_PER_SEED = (
    "recurrent_mix - static_mix",
    "recurrent_mix - projection_mix",
    "recurrent_mix - self_mix",
    "static_mix - fixed_mix",
    "recurrent_mix - fixed_mix",
    "projection_mix - fixed_mix",
    "self_mix - fixed_mix",
    "gelu - fixed_mix",
)
FIT_OOM_KEYS = ("oom", "oom_kill", "oom_group_kill")

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E474_DIR))
sys.path.insert(0, str(E477_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402

from wnn.campaign import RUNTIME_FILES, runtime_identity  # noqa: E402
from wnn.campaign_spec import digest as spec_digest  # noqa: E402
from wnn.campaign_spec import validate_spec  # noqa: E402
from wnn.evaluation import _score_component_moments, paired_component_moments  # noqa: E402
from wnn.gmetric import METRIC_ID, global_wp  # noqa: E402


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading its full contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    """Fail closed with an actionable provenance error."""
    if not ok:
        raise ValueError(message)


def finite(value: Any, label: str) -> float:
    """Parse a finite scalar receipt field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is nonfinite")
    return result


class ManualGate(nn.Module):
    """Independent E484 readout equation with the actor-compatible state key."""

    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unregistered E484 arm: {arm}")
        self.arm = arm
        if arm in ("static_mix", *DYNAMIC_SOURCES):
            self.gate_coeff = nn.Parameter(torch.zeros(HIDDEN))

    def forward_details(self, hidden: Tensor, recurrent: Tensor, projection: Tensor):
        gelu = F.gelu(hidden, approximate="none")
        if self.arm == "gelu":
            return gelu, None
        leaky = F.leaky_relu(hidden, negative_slope=0.01)
        if self.arm == "fixed_mix":
            gate = torch.full_like(hidden, 0.5)
        elif self.arm == "static_mix":
            gate = torch.sigmoid(self.gate_coeff).view(1, 1, HIDDEN).expand_as(hidden)
        else:
            context = {
                "recurrent": recurrent,
                "projection": projection,
                "head_self": hidden,
            }[DYNAMIC_SOURCES[self.arm]]
            rms = torch.sqrt(context.square().mean(dim=-1, keepdim=True) + GATE_EPSILON)
            normalized = context / rms
            gate = torch.sigmoid(self.gate_coeff.view(1, 1, HIDDEN) * normalized)
        return gate * gelu + (1.0 - gate) * leaky, gate

    def forward(self, hidden: Tensor, recurrent: Tensor, projection: Tensor) -> Tensor:
        """Return the registered mixture output."""
        return self.forward_details(hidden, recurrent, projection)[0]


class IndependentE484Model(nn.Module):
    """Reconstruct the E474 native GRU and six E484 heads without actor imports."""

    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unregistered E484 arm: {arm}")
        self.register_buffer("mean", torch.zeros(FEATURES))
        self.register_buffer("std", torch.ones(FEATURES))
        self.inp = nn.Linear(FEATURES, HIDDEN)
        self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(
            nn.Linear(2 * HIDDEN, HIDDEN), ManualGate(arm), nn.Linear(HIDDEN, 2)
        )
        self.arm = arm

    def forward_details(self, x: Tensor, state: Tensor | None = None):
        """Return predictions, final native GRU state, gate and its context."""
        if x.ndim != 3 or x.shape[-1] != FEATURES:
            raise ValueError("E484 inputs must have shape [batch,time,42]")
        if not torch.isfinite(x).all():
            raise ValueError("E484 model input is nonfinite")
        projection = F.gelu(self.inp((x - self.mean) / self.std), approximate="none")
        recurrent, final_state = self.gru(projection, state)
        hidden = self.head[0](torch.cat((recurrent, projection), dim=-1))
        activated, gate = self.head[1].forward_details(hidden, recurrent, projection)
        prediction = self.head[2](activated)
        if not torch.isfinite(prediction).all() or not torch.isfinite(final_state).all():
            raise ValueError("E484 scorer produced nonfinite output/state")
        if gate is not None and not torch.isfinite(gate).all():
            raise ValueError("E484 scorer produced nonfinite gate")
        return prediction, final_state, gate

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return predictions and carried two-layer GRU state."""
        prediction, final_state, _gate = self.forward_details(x, state)
        return prediction, final_state


def expected_state_keys(arm: str) -> set[str]:
    """Build expected checkpoint keys without depending on actor model code."""
    keys = {
        "mean",
        "std",
        "inp.weight",
        "inp.bias",
        "gru.weight_ih_l0",
        "gru.weight_hh_l0",
        "gru.bias_ih_l0",
        "gru.bias_hh_l0",
        "gru.weight_ih_l1",
        "gru.weight_hh_l1",
        "gru.bias_ih_l1",
        "gru.bias_hh_l1",
        "head.0.weight",
        "head.0.bias",
        "head.2.weight",
        "head.2.bias",
    }
    if arm in ARMS[2:]:
        keys.add("head.1.gate_coeff")
    return keys


def validate_identity_buffers(state: dict[str, Tensor]) -> None:
    """Require identity model buffers; input normalization lives outside the checkpoint."""
    require("mean" in state and "std" in state, "checkpoint lacks mean/std buffers")
    require(
        torch.equal(state["mean"].cpu(), torch.zeros(FEATURES))
        and torch.equal(state["std"].cpu(), torch.ones(FEATURES)),
        "model buffers are not the registered identity values",
    )


def validate_checkpoint(path: Path, arm: dict, seed: int, name: str) -> IndependentE484Model:
    """Verify one checkpoint's bytes, exact shapes, finite tensors and parameter count."""
    expected_name = f"seed{seed}_{name}.pt"
    require(
        arm.get("checkpoint") == expected_name and path.name == expected_name,
        f"unexpected checkpoint name for {seed}/{name}",
    )
    require(
        path.is_file() and sha256_file(path) == arm.get("checkpoint_sha256"),
        f"checkpoint SHA mismatch for {seed}/{name}",
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(state, dict), "checkpoint is not a tensor state dictionary")
    require(set(state) == expected_state_keys(name), f"checkpoint keys differ: {seed}/{name}")
    require(
        all(isinstance(x, Tensor) and torch.isfinite(x).all() for x in state.values()),
        f"checkpoint has malformed/nonfinite tensors: {seed}/{name}",
    )
    validate_identity_buffers(state)
    model = IndependentE484Model(name)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint tensor shapes differ: {seed}/{name}") from exc
    expected_count = EXPECTED_PARAMETERS[name]
    actual_count = sum(parameter.numel() for parameter in model.parameters())
    require(
        actual_count == expected_count, f"independent model parameter count differs: {seed}/{name}"
    )
    require(arm.get("parameters") == expected_count, f"fit parameter count differs: {seed}/{name}")
    return model


def _canonical_path(root: Path, relative: str) -> Path:
    """Resolve a safe repository-relative path and reject traversal/symlinks."""
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe path: {relative}")
    path = root / rel
    require(not path.is_symlink(), f"symlink is not accepted: {relative}")
    return path


def validate_spec_sources(spec: dict, expected_artifact: str) -> dict[str, str]:
    """Validate the sealed source and input inventory against exact local bytes."""
    checked = validate_spec(spec)
    require(checked == spec, "campaign spec is not canonical")
    require(spec.get("kind") == "gpu", "E484 full fits require GPU campaign spec")
    require(spec.get("artifacts") == [expected_artifact], "E484 artifact path differs")
    code = spec.get("code")
    inputs = spec.get("inputs")
    require(isinstance(code, list) and isinstance(inputs, list), "spec source/input lists missing")
    observed = {}
    for label, entries in (("code", code), ("input", inputs)):
        seen = set()
        for entry in entries:
            relative = entry.get("path")
            require(isinstance(relative, str) and relative not in seen, f"duplicate {label} path")
            seen.add(relative)
            path = _canonical_path(ROOT, relative)
            require(
                path.is_file() and path.stat().st_size == entry.get("size"),
                f"sealed {label} missing/size mismatch: {relative}",
            )
            digest = sha256_file(path)
            require(digest == entry.get("sha256"), f"sealed {label} hash mismatch: {relative}")
            observed[relative] = digest
    required_sources = {
        "research/campaign_2026-10-05/activation_context_mixture/fit_context_mixture_v2.py": FIT_SOURCE_V2_SHA256,
        "research/campaign_2026-10-05/activation_context_mixture/gpu_probe_v2.py": PROBE_SOURCE_V2_SHA256,
        "research/campaign_2026-10-05/activation_context_mixture/tests_v2.py": TEST_SOURCE_V2_SHA256,
        "research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py": E470_SHA256,
        "research/campaign_2026-10-05/activation_placement/fit_activation_placement.py": E474_SHA256,
        "research/campaign_2026-10-05/activation_head_mixture/fit_activation_head_mixture.py": E477_SHA256,
        "src/wnn/nn.py": WNN_NN_SHA256,
        "src/wnn/resources.py": WNN_RESOURCES_SHA256,
    }
    code_map = {entry["path"]: entry["sha256"] for entry in code}
    for relative, digest in required_sources.items():
        require(
            code_map.get(relative) == digest and observed.get(relative) == digest,
            f"E484 spec/local source identity differs: {relative}",
        )
    input_map = {entry["path"]: entry["sha256"] for entry in inputs}
    required_inputs = {
        "datasets/valid.parquet": DATASET_SHA256,
        "research/campaign_2026-10-05/target_mechanics/cohort.json": COHORT_SHA256,
        "research/campaign_2026-10-05/activation_context_mixture/E484_DESIGN_V2.md": DESIGN_SHA256,
        "research/campaign_2026-10-05/activation_context_mixture/SPEC_CRITIC.md": CRITIC_SHA256,
        "research/campaign_2026-10-05/activation_context_mixture/SPEC_CRITIC_V2_RESPONSE.md": CRITIC_RESPONSE_SHA256,
    }
    for relative, digest in required_inputs.items():
        require(
            input_map.get(relative) == digest and observed.get(relative) == digest,
            f"E484 sealed input identity differs: {relative}",
        )
    return observed


def validate_attempt(path: Path, spec_path: Path, expected_phase: str) -> tuple[dict, dict, Path]:
    """Verify controller/systemd success, exact spec/runtime/artifact, and OOM counters."""
    expected_spec_sha = FIT_SPEC_V2_SHA256 if expected_phase == "fit" else PROBE_SPEC_V2_SHA256
    require(
        sha256_file(spec_path) == expected_spec_sha,
        f"E484 {expected_phase} sealed spec SHA differs",
    )
    spec = json.loads(spec_path.read_text())
    checked = validate_spec(spec)
    expected_name = (
        "e484_context_mixture_twelve_fit_v2"
        if expected_phase == "fit"
        else "e484_context_mixture_actual_gpu_probe_v2"
    )
    expected_id = (
        "e484-context-mixture-twelve-fit-v2"
        if expected_phase == "fit"
        else "e484-context-mixture-real-gpu-probe-v2"
    )
    require(spec.get("name") == expected_name, f"wrong E484 {expected_phase} spec name")
    record = json.loads(path.read_text())
    require(
        record.get("id") == expected_id and record.get("kind") == "gpu",
        f"wrong E484 {expected_phase} attempt identity",
    )
    require(
        record.get("spec") == checked and record.get("spec_sha256") == spec_digest(checked),
        f"E484 {expected_phase} attempt does not bind exact spec",
    )
    argv = record.get("argv")
    require(
        isinstance(argv, list)
        and len(argv)
        and Path(argv[0]).is_absolute()
        and argv[1:] == checked.get("argv", [])[1:],
        f"E484 {expected_phase} attempt command differs",
    )
    require(
        record.get("status") == "finished"
        and record.get("exit_code") == 0
        and record.get("signals") == []
        and record.get("teardown_verified") is True
        and all(record.get(k) is None for k in ("failure", "failure_scope", "failure_category")),
        f"E484 {expected_phase} attempt did not finish cleanly",
    )
    unit = record.get("unit_properties", {})
    require(
        unit.get("Result") == "success" and unit.get("ExecMainStatus") in (0, "0"),
        f"E484 {expected_phase} systemd unit failed",
    )
    runtime = runtime_identity(ROOT)
    require(
        record.get("runtime_sha256") == runtime and set(runtime) == set(RUNTIME_FILES),
        f"E484 {expected_phase} runner runtime hashes differ",
    )
    for before_key, after_key in (
        ("initial_memory_events", "final_memory_events"),
        ("initial_parent_memory_events", "final_parent_memory_events"),
    ):
        before, after = record.get(before_key), record.get(after_key)
        require(
            isinstance(before, dict) and isinstance(after, dict),
            f"E484 {expected_phase} OOM snapshots missing",
        )
        require(
            all(
                type(before.get(k)) is int
                and type(after.get(k)) is int
                and before[k] >= 0
                and before[k] == after[k]
                for k in FIT_OOM_KEYS
            ),
            f"E484 {expected_phase} OOM counter increased",
        )
    final_worker = record.get("final_memory_events", {})
    require(
        all(final_worker.get(key) == 0 for key in FIT_OOM_KEYS),
        f"E484 {expected_phase} worker OOM counter nonzero",
    )
    limits = checked.get("limits", {})
    peak_ram = record.get("peak_ram_bytes")
    peak_vram = record.get("peak_observed_vram_bytes")
    require(
        type(peak_ram) is int and peak_ram < limits.get("ram_bytes", 0),
        f"E484 {expected_phase} RAM ceiling exceeded/missing",
    )
    require(
        type(peak_vram) is int and peak_vram < limits.get("gpu_stop_bytes", 0),
        f"E484 {expected_phase} physical GPU stop exceeded/missing",
    )
    artifact_rel = checked.get("artifacts", [None])[0]
    require(
        isinstance(artifact_rel, str) and len(checked["artifacts"]) == 1,
        f"E484 {expected_phase} should declare one proof artifact",
    )
    artifact = _canonical_path(ROOT, artifact_rel)
    require(artifact.is_file(), f"E484 {expected_phase} artifact missing")
    expected_artifacts = [
        {"path": artifact_rel, "sha256": sha256_file(artifact), "size": artifact.stat().st_size}
    ]
    require(
        record.get("artifacts") == expected_artifacts,
        f"E484 {expected_phase} attempt artifact inventory differs",
    )
    expected_run_arg = checked.get("argv", [])
    require(
        str(artifact.parent.relative_to(ROOT)) in expected_run_arg,
        f"E484 {expected_phase} output argument does not match artifact",
    )
    return record, checked, artifact


def canonical_nodes() -> tuple[list[dict], list[dict], dict]:
    """Read the immutable cohort role descriptors, without reading parquet labels."""
    require(sha256_file(COHORT_PATH) == COHORT_SHA256, "E484 cohort hash changed")
    document = json.loads(COHORT_PATH.read_text())
    all_nodes = document.get("cohort", [])
    fit_nodes = [node for node in all_nodes if node.get("role") == "fit"]
    test_nodes = [node for node in all_nodes if node.get("role") == "test"]
    require(
        len(all_nodes) == 40 and len(fit_nodes) == 24 and len(test_nodes) == 16,
        "E484 cohort roles/counts differ",
    )
    require(
        len({n["component"] for n in fit_nodes}) == 24
        and len({n["component"] for n in test_nodes}) == 16,
        "E484 component roles duplicate",
    )
    require(
        not ({n["component"] for n in fit_nodes} & {n["component"] for n in test_nodes}),
        "E484 fit/test components overlap",
    )
    return fit_nodes, test_nodes, document


def validate_probe(
    probe_path: Path,
    probe_attempt_path: Path,
    probe_attempt: dict,
    probe_spec: dict,
    fit_attempt: dict,
    fit_report: dict,
) -> dict:
    """Validate successful probe parity, exact B4×256 updates, and fit continuity."""
    probe = json.loads(probe_path.read_text())
    require(
        probe.get("kind") == "E484 actual CUDA fit-only update probe"
        and probe.get("status") == "all_12_cells_finite_changed"
        and probe.get("fit_only") is True
        and probe.get("test_rows_loaded") is False,
        "E484 actual GPU probe did not pass the registered fit-only status",
    )
    require(
        probe.get("dataset_sha256") == DATASET_SHA256
        and probe.get("cohort_sha256") == COHORT_SHA256,
        "E484 probe data/cohort identity differs",
    )
    require(
        probe.get("peak_allocated_bytes", 2**63) <= 2 * 1024**3
        and probe.get("peak_reserved_bytes", 2**63) <= 2 * 1024**3,
        "E484 probe exceeded CUDA allocator cap",
    )
    precision = probe.get("precision", {})
    require(
        precision.get("device_math") == "FP32 IEEE"
        and precision.get("autocast") is False
        and precision.get("torch.backends.fp32_precision") == "ieee"
        and precision.get("torch.backends.cuda.matmul.fp32_precision") == "ieee"
        and precision.get("torch.backends.cudnn.fp32_precision") == "ieee"
        and precision.get("torch.backends.cudnn.rnn.fp32_precision") == "ieee",
        "E484 probe did not capture active IEEE precision settings",
    )
    parity = probe.get("parity", {})
    require(
        parity.get("device") == "cuda" and len(parity.get("checks", [])) == 2,
        "E484 CUDA native parity receipt missing",
    )
    expected_refs = {"gelu": "gelu", "fixed_mix": "same_channel_mix"}
    for item in parity["checks"]:
        arm = item.get("arm")
        require(
            arm in expected_refs
            and item.get("reference") == expected_refs[arm]
            and item.get("exact_forward_state_input_and_parameter_gradients") is True
            and item.get("clipping_exercised") is True
            and item.get("exact_clipped_adamw_step") is True,
            f"E484 native parity incomplete for {arm}",
        )
        require(
            finite(item.get("preclip_gradient_norm"), "parity gradient norm") > CLIP_NORM,
            f"E484 parity did not exercise clipping for {arm}",
        )
    probe_sources = probe.get("actor_source_sha256", {})
    fit_sources = fit_report.get("actor_source_sha256", {})
    require(
        probe_sources == fit_sources, "E484 worker source identity changed between probe and fit"
    )
    require(
        probe.get("parent_source_sha256") == fit_report.get("parent_source_sha256"),
        "E484 parent source identity changed between probe and fit",
    )
    require(
        probe.get("device") == fit_report.get("device"), "E484 probe and fit device names differ"
    )
    probe_env, fit_env = probe_attempt.get("environment", {}), fit_attempt.get("environment", {})
    require(
        probe_env.get("CUDA_VISIBLE_DEVICES") == fit_env.get("CUDA_VISIBLE_DEVICES"),
        "E484 probe and full fit used different CUDA visibility",
    )
    probe_fit_manifest_hash = hashlib.sha256(
        json.dumps(fit_report.get("fit_only_cohort"), sort_keys=True).encode()
    ).hexdigest()
    require(
        probe.get("fit_manifest_hash") == probe_fit_manifest_hash,
        "E484 probe fit-only cohort hash differs from full-fit manifest",
    )
    arms = probe.get("arms", [])
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    found = set()
    for item in arms:
        key = (item.get("seed"), item.get("arm"))
        require(key in expected and key not in found, f"unknown/duplicate E484 probe arm: {key}")
        found.add(key)
        require(
            item.get("batch") == BATCH
            and item.get("chunk_rows") == CHUNK_ROWS
            and item.get("updates") == 1
            and item.get("parameters") == EXPECTED_PARAMETERS[key[1]]
            and type(item.get("changed_parameter_tensors")) is int
            and item.get("changed_parameter_tensors") > 0,
            f"E484 probe shape/update/count differs: {key}",
        )
        finite(item.get("loss"), f"E484 probe loss {key}")
        expected_order = np.arange(24)
        np.random.default_rng(key[0]).shuffle(expected_order)
        nodes, _test_nodes, _doc = canonical_nodes()
        expected_rows = [nodes[int(i)]["row_group"] for i in expected_order[:4]]
        require(item.get("fit_row_groups") == expected_rows, f"E484 probe fit batch differs: {key}")
        optimizer = item.get("optimizer", {})
        require(
            optimizer.get("name") == "AdamW"
            and optimizer.get("lr") == LR
            and optimizer.get("weight_decay") == WEIGHT_DECAY
            and optimizer.get("betas") == list(BETAS),
            f"E484 probe optimizer differs: {key}",
        )
    require(found == expected, "E484 probe omitted one or more of 12 registered cells")
    return {
        "probe_attempt_id": probe_attempt.get("id"),
        "probe_attempt_sha256": sha256_file(probe_attempt_path),
        "probe_artifact_sha256": sha256_file(probe_path),
        "probe_source_sha256": probe_sources,
        "probe_device": probe.get("device"),
        "probe_cells": len(found),
        "native_parity_passed": True,
        "test_rows_loaded": False,
    }


def _expected_precision(report: dict) -> None:
    precision = report.get("precision", {})
    require(
        precision.get("device_math") == "FP32 IEEE"
        and precision.get("autocast") is False
        and precision.get("torch.backends.fp32_precision") == "ieee"
        and precision.get("torch.backends.cuda.matmul.fp32_precision") == "ieee"
        and precision.get("torch.backends.cudnn.fp32_precision") == "ieee"
        and precision.get("torch.backends.cudnn.rnn.fp32_precision") == "ieee",
        "E484 fit precision metadata is not captured as IEEE within its context",
    )


def validate_arm_telemetry(
    arm: dict, seed: int, name: str, nodes: list[dict], order_rows: list[int]
) -> None:
    """Validate full row traversal and the four exact progress samples."""
    require(
        arm.get("seed") == seed and arm.get("arm") == name, f"E484 arm key differs: {seed}/{name}"
    )
    require(
        arm.get("updates") == UPDATES and arm.get("parameters") == EXPECTED_PARAMETERS[name],
        f"E484 update/count differs: {seed}/{name}",
    )
    require(arm.get("fit_row_groups") == order_rows, f"E484 row order differs: {seed}/{name}")
    telemetry = arm.get("telemetry")
    require(
        isinstance(telemetry, list)
        and [row.get("update_index_zero_based") for row in telemetry] == list(TELEMETRY_STEPS),
        f"E484 telemetry cadence differs: {seed}/{name}",
    )
    for row in telemetry:
        step = row["update_index_zero_based"]
        batch_index = step // math.ceil(ROWS / CHUNK_ROWS)
        expected_batch = order_rows[batch_index * BATCH : (batch_index + 1) * BATCH]
        expected_chunk = (step % math.ceil(ROWS / CHUNK_ROWS)) * CHUNK_ROWS
        require(
            row.get("fit_row_groups") == expected_batch
            and row.get("chunk_start") == expected_chunk,
            f"E484 sampled batch/chunk differs: {seed}/{name}/{step}",
        )
        for key in ("loss", "state_abs_max"):
            require(
                finite(row.get(key), f"E484 {key}") >= 0
                if key == "state_abs_max"
                else math.isfinite(finite(row.get(key), f"E484 {key}")),
                f"E484 invalid telemetry {key}: {seed}/{name}/{step}",
            )
        for family_field in ("preclip_gradient_rms", "parameter_update_rms"):
            family = row.get(family_field)
            require(
                isinstance(family, dict) and set(family) == {"inp", "gru", "head"},
                f"E484 telemetry families differ: {seed}/{name}/{family_field}",
            )
            require(
                all(finite(value, f"E484 {family_field}") >= 0 for value in family.values()),
                f"E484 telemetry has invalid value: {seed}/{name}/{family_field}",
            )
        gate = row.get("gate")
        if name == "gelu":
            require(gate is None, "E484 GELU arm unexpectedly records a gate")
        else:
            require(isinstance(gate, dict), f"E484 mixture arm lacks gate telemetry: {seed}/{name}")
            for key in ("min", "max", "mean", "saturated_fraction_at_0_01"):
                value = finite(gate.get(key), f"E484 gate {key}")
                require(
                    0.0 <= value <= 1.0, f"E484 gate diagnostic out of [0,1]: {seed}/{name}/{key}"
                )


def expected_optimizer(model: IndependentE484Model, actual: dict, arm_name: str) -> None:
    """Require the single exact E474 AdamW group covers each model parameter once."""
    names = list(dict(model.named_parameters()))
    require(
        actual.get("name") == "AdamW"
        and actual.get("lr") == LR
        and actual.get("weight_decay") == WEIGHT_DECAY
        and actual.get("betas") == list(BETAS)
        and actual.get("groups") == 1
        and actual.get("parameter_count") == EXPECTED_PARAMETERS[arm_name]
        and actual.get("parameter_names") == names,
        f"E484 optimizer group membership or settings differ for {arm_name}",
    )


def validate_fit_manifest(
    run: Path,
    report: dict,
    spec: dict,
    fit_nodes: list[dict],
    fit_only_manifest: dict,
    fit_x: np.ndarray,
) -> dict[tuple[int, str], dict]:
    """Verify exact arm receipts/checkpoint bytes and independent fit-only normalization."""
    require(
        report.get("kind") == "E484 causal context-conditioned activation mixture"
        and report.get("status") == "all_12_fit_arms_complete_test_unread"
        and report.get("fit_only") is True
        and report.get("test_rows_loaded") is False,
        "E484 fit manifest incomplete or test boundary failed",
    )
    require(
        report.get("seeds") == list(SEEDS)
        and report.get("arms") == list(ARMS)
        and report.get("fit_count") == 12
        and report.get("updates_per_fit") == UPDATES
        and report.get("batch") == BATCH
        and report.get("chunk_rows") == CHUNK_ROWS
        and report.get("telemetry_steps_zero_based") == list(TELEMETRY_STEPS),
        "E484 fit matrix or cadence differs",
    )
    require(
        report.get("parameters_per_arm") == EXPECTED_PARAMETERS, "E484 parameter count map differs"
    )
    require(
        report.get("registered_contrasts_per_seed") == list(REGISTERED_CONTRASTS_PER_SEED)
        and report.get("registered_contrast_count") == 16,
        "E484 registered contrast inventory differs",
    )
    require(
        report.get("optimizer")
        == {
            "name": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "betas": list(BETAS),
            "schedule": "constant",
            "clip_norm": CLIP_NORM,
        },
        "E484 optimizer recipe differs",
    )
    require(
        report.get("gate", {}).get("context_rms_epsilon") == GATE_EPSILON
        and report.get("gate", {}).get("context_centered") is False
        and report.get("gate", {}).get("initial_coefficients") == 0.0
        and report.get("gate", {}).get("extra_parameters_per_learned_gate") == HIDDEN,
        "E484 gate specification differs",
    )
    require(
        report.get("dataset_sha256") == DATASET_SHA256
        and report.get("cohort_sha256") == COHORT_SHA256,
        "E484 fit data/cohort identity differs",
    )
    require(
        report.get("fit_only_cohort") == fit_only_manifest,
        "E484 manifest fit-only cohort/normalizer differs from local fit loader",
    )
    require(
        report.get("parent_source_sha256")
        == {
            "e470": E470_SHA256,
            "e474": E474_SHA256,
            "e477": E477_SHA256,
            "wnn.nn": WNN_NN_SHA256,
            "wnn.resources": WNN_RESOURCES_SHA256,
        },
        "E484 frozen parent source identity differs",
    )
    _expected_precision(report)
    actor_sources = {
        "fit_context_mixture_v2.py": FIT_SOURCE_V2_SHA256,
        "tests_v2.py": TEST_SOURCE_V2_SHA256,
        "gpu_probe_v2.py": PROBE_SOURCE_V2_SHA256,
    }
    require(
        report.get("actor_source_sha256") == actor_sources,
        "E484 trainer manifest actor source hashes differ",
    )
    for seed in SEEDS:
        torch.manual_seed(seed)
        initial = IndependentE484Model("gelu").state_dict()
        require(
            report.get("initial_state_sha256", {}).get(str(seed)) == tensor_state_sha256(initial),
            f"E484 initialization hash differs for seed {seed}",
        )
    require(
        fit_x.shape == (24, ROWS, FEATURES)
        and np.isfinite(fit_x).all()
        and len(fit_nodes) == 24
        and all(node.get("role") == "fit" for node in fit_nodes),
        "E484 fit-only data loader returned bad roles/arrays",
    )
    order_by_seed = {}
    for seed in SEEDS:
        order = np.arange(24)
        np.random.default_rng(seed).shuffle(order)
        order_by_seed[seed] = [fit_nodes[int(i)]["row_group"] for i in order]
    arms = report.get("fit_manifest_arms", [])
    expected = {(seed, name) for seed in SEEDS for name in ARMS}
    observed = {}
    for arm in arms:
        key = (arm.get("seed"), arm.get("arm"))
        require(key in expected and key not in observed, f"unknown/duplicate E484 fit arm {key}")
        seed, name = key
        validate_arm_telemetry(arm, seed, name, fit_nodes, order_by_seed[seed])
        model = validate_checkpoint(run / str(arm.get("checkpoint")), arm, seed, name)
        expected_optimizer(model, arm.get("optimizer", {}), name)
        finite(arm.get("fit_seconds"), f"E484 fit duration {key}")
        # Each per-arm file and JSONL row is a second receipt, not a score source.
        arm_json_path = run / f"arm_seed{seed}_{name}.json"
        require(
            arm_json_path.is_file() and json.loads(arm_json_path.read_text()) == arm,
            f"E484 per-arm JSON receipt differs: {key}",
        )
        observed[key] = arm
    require(set(observed) == expected, "E484 fit manifest omits registered arm")
    rows_path = run / "fit_arms.jsonl"
    require(rows_path.is_file(), "E484 per-arm JSONL receipt missing")
    jsonl = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    require(
        len(jsonl) == 12 and jsonl == arms,
        "E484 per-arm JSONL entries differ from the fit manifest",
    )
    if (run / "FIT_FAILURE.json").exists():
        raise ValueError("E484 run directory contains a failure receipt")
    return observed


def tensor_state_sha256(state: dict[str, Tensor]) -> str:
    """Independently fingerprint initial tensor keys, types, shapes and bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_fit_normalizer(
    fit_manifest: dict, fit_x: np.ndarray, fit_nodes: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Independently recompute fit-only statistics and normalized sequences."""
    if len(fit_nodes) != 24 or any(node.get("role") != "fit" for node in fit_nodes):
        raise ValueError("E484 normalizer requires the 24 canonical fit roles")
    parquet = pq.ParquetFile(DATA_PATH)
    raw_feature_arrays = []
    for node in fit_nodes:
        table = parquet.read_row_group(node["row_group"], columns=e470.FEATURES, use_threads=False)
        raw = np.column_stack([table[column].to_numpy() for column in e470.FEATURES])
        raw_feature_arrays.append(e470.causal_features(raw))
    raw_fit = np.stack(raw_feature_arrays)
    mean = raw_fit.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(raw_fit.std(axis=(0, 1), dtype=np.float64), 0.001).astype(np.float32)
    normalization = fit_manifest.get("normalization", {})
    observed_mean = np.asarray(normalization.get("fit_mean"), dtype=np.float32)
    observed_scale = np.asarray(normalization.get("fit_scale"), dtype=np.float32)
    require(
        mean.shape == (FEATURES,)
        and scale.shape == (FEATURES,)
        and np.isfinite(mean).all()
        and np.isfinite(scale).all()
        and (scale > 0).all(),
        "E484 independently recomputed normalizer is invalid",
    )
    require(
        np.array_equal(mean, observed_mean) and np.array_equal(scale, observed_scale),
        "E484 loader normalizer differs from raw-fit recomputation",
    )
    require(
        fit_x.shape == (24, ROWS, FEATURES) and np.isfinite(fit_x).all(),
        "E484 normalized fit feature matrix has invalid shape/values",
    )
    for index, raw_sequence in enumerate(raw_fit):
        normalized = np.clip((raw_sequence - mean) / scale, -8, 8).astype(np.float32)
        require(
            np.array_equal(normalized, fit_x[index]),
            f"E484 fit feature normalization differs at row group {fit_nodes[index]['row_group']}",
        )
    return mean, scale


def validate_full_run(
    run: Path,
    fit_spec_path: Path,
    fit_attempt_path: Path,
    probe_run: Path,
    probe_spec_path: Path,
    probe_attempt_path: Path,
):
    """Validate all proof gates without loading any held-out sequence or label."""
    fit_attempt, fit_spec, fit_artifact = validate_attempt(fit_attempt_path, fit_spec_path, "fit")
    probe_attempt, probe_spec, probe_artifact = validate_attempt(
        probe_attempt_path, probe_spec_path, "probe"
    )
    require(
        run.resolve() == fit_artifact.parent.resolve(),
        "E484 fit directory differs from attempt artifact",
    )
    require(
        probe_run.resolve() == probe_artifact.parent.resolve(),
        "E484 probe directory differs from attempt artifact",
    )
    validate_spec_sources(fit_spec, str(fit_artifact.relative_to(ROOT)))
    validate_spec_sources(probe_spec, str(probe_artifact.relative_to(ROOT)))
    fit_nodes, _test_nodes, _cohort = canonical_nodes()
    fit_manifest, fit_x, _fit_y, _fit_need, _fit_score, loaded_nodes = e470.load_cohort(
        include_test=False
    )
    require(loaded_nodes == fit_nodes, "E470 fit-only loader role/order differs from cohort")
    mean, scale = validate_fit_normalizer(fit_manifest, fit_x, fit_nodes)
    report = json.loads(fit_artifact.read_text())
    cells = validate_fit_manifest(run, report, fit_spec, fit_nodes, fit_manifest, fit_x)
    probe_proof = validate_probe(
        probe_artifact, probe_attempt_path, probe_attempt, probe_spec, fit_attempt, report
    )
    proof = {
        "fit_attempt_id": fit_attempt.get("id"),
        "fit_attempt_sha256": sha256_file(fit_attempt_path),
        "fit_spec_sha256": fit_attempt.get("spec_sha256"),
        "fit_manifest_sha256": sha256_file(fit_artifact),
        "fit_source_sha256": report.get("actor_source_sha256"),
        "fit_cell_count": len(cells),
        "fit_only_normalizer_sha256": hashlib.sha256(
            mean.astype("<f4").tobytes() + scale.astype("<f4").tobytes()
        ).hexdigest(),
        "probe": probe_proof,
        "fit_test_boundary": "validated before test loader; no test rows read",
    }
    return proof, cells, report, fit_manifest, fit_x, fit_nodes


def _bootstrap_counts(component_ids: np.ndarray) -> np.ndarray:
    """Generate shared 2,000 component draws in numeric-sorted component order."""
    if component_ids.ndim != 1 or len(component_ids) != 16 or len(np.unique(component_ids)) != 16:
        raise ValueError("E484 bootstrap requires 16 distinct components")
    order = np.argsort(component_ids, kind="stable")
    sorted_ids = component_ids[order]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    counts = np.zeros((BOOTSTRAP_COUNT, 16), dtype=np.int16)
    for i in range(BOOTSTRAP_COUNT):
        counts[i] = np.bincount(rng.integers(0, 16, 16), minlength=16)
    return sorted_ids, order, counts


def _read_raw_test_targets(
    test_nodes: list[dict], clipped: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    """Read all raw target rows and reject nonfinite values before metric clipping."""
    expected = (16, ROWS, 2)
    require(
        clipped.shape == expected and selected.shape == expected[:2],
        "E484 test label/mask shape differs",
    )
    parquet = pq.ParquetFile(DATA_PATH)
    raw_targets = np.empty(expected, dtype=np.float64)
    for i, node in enumerate(test_nodes):
        table = parquet.read_row_group(
            node["row_group"],
            columns=["t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        raw = np.column_stack((table["t0"].to_numpy(), table["t1"].to_numpy())).astype(np.float64)
        source_mask = table["need_prediction"].to_numpy().astype(bool) & table[
            "is_scored"
        ].to_numpy().astype(bool)
        require(
            np.array_equal(source_mask, selected[i]),
            f"E484 source selection mask mismatch at row group {node['row_group']}",
        )
        require(
            np.isfinite(raw).all() and np.isfinite(raw[source_mask]).all(),
            f"E484 raw test targets nonfinite at row group {node['row_group']}",
        )
        require(
            np.array_equal(np.clip(raw, -METRIC_CLIP, METRIC_CLIP).astype(np.float32), clipped[i]),
            f"E484 clipped/raw test target mismatch at row group {node['row_group']}",
        )
        raw_targets[i] = raw
    return raw_targets


def _predict_sequence(
    model: IndependentE484Model, sequence: np.ndarray, device: torch.device
) -> np.ndarray:
    """Run full causal sequence with state carried across fixed chunks."""
    model.eval()
    outputs = []
    state = None
    with torch.no_grad():
        for start in range(0, ROWS, CHUNK_ROWS):
            chunk = torch.from_numpy(sequence[start : start + CHUNK_ROWS][None]).to(
                device=device, dtype=torch.float32
            )
            prediction, state = model(chunk, state)
            if not torch.isfinite(prediction).all() or not torch.isfinite(state).all():
                raise ValueError("E484 scorer got nonfinite chunk output/state")
            outputs.append(prediction[0].cpu().numpy())
    result = np.concatenate(outputs, axis=0)
    if result.shape != (ROWS, 2) or not np.isfinite(result).all():
        raise ValueError("E484 full-sequence prediction invalid")
    return result


def _contrast_result(point: float, draws: np.ndarray) -> dict:
    """Summarize a paired component bootstrap contrast."""
    require(
        draws.shape == (BOOTSTRAP_COUNT,) and np.isfinite(draws).all() and math.isfinite(point),
        "invalid E484 contrast bootstrap",
    )
    return {
        "point_diff": float(point),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
        "p_le_0": float(np.mean(draws <= 0)),
        "components": 16,
        "bootstrap_count": BOOTSTRAP_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "metric_id": METRIC_ID,
    }


def _score_component_moments_by_target(moments: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Pool centered component moments and return target-zero/target-one WPs."""
    if moments.shape != (16, 2, 6) or counts.shape != (16,):
        raise ValueError("E484 targetwise bootstrap moments have invalid shape")
    result = np.zeros(2, dtype=np.float64)
    for target in range(2):
        target_moments = moments[:, target, :]
        weight = counts * target_moments[:, 0]
        total = float(weight.sum())
        if total <= 0:
            continue
        active = np.flatnonzero(weight > 0)
        anchor_y, anchor_p = target_moments[active[0], 1:3]
        dy = target_moments[:, 1] - anchor_y
        dp = target_moments[:, 2] - anchor_p
        mean_y = np.sum(weight * dy) / total
        mean_p = np.sum(weight * dp) / total
        variance_y = np.sum(counts * target_moments[:, 3] + weight * (dy - mean_y) ** 2)
        variance_p = np.sum(counts * target_moments[:, 4] + weight * (dp - mean_p) ** 2)
        covariance = np.sum(counts * target_moments[:, 5] + weight * (dy - mean_y) * (dp - mean_p))
        if np.sqrt(variance_y / total) > 1e-8 and np.sqrt(variance_p / total) > 1e-8:
            result[target] = np.clip(covariance / np.sqrt(variance_y * variance_p), -1, 1)
    return result


def score_validated_run(
    run: Path,
    cells: dict[tuple[int, str], dict],
    fit_manifest: dict,
    fit_x: np.ndarray,
    fit_nodes: list[dict],
    device: torch.device,
):
    """Load test roles after proof validation and score all arms without selection."""
    all_manifest, all_x, clipped_y, _need, selected, all_nodes = e470.load_cohort(include_test=True)
    canonical = json.loads(COHORT_PATH.read_text())["cohort"]
    require(
        all_nodes == canonical and all_x.shape == (40, ROWS, FEATURES),
        "E484 test-inclusive loader role/order differs",
    )
    require(
        fit_manifest["normalization"] == all_manifest.get("normalization"),
        "E484 test loader normalization differs from fit-only loader",
    )
    all_fit_pos = [i for i, node in enumerate(all_nodes) if node["role"] == "fit"]
    all_test_pos = [i for i, node in enumerate(all_nodes) if node["role"] == "test"]
    require(
        [all_nodes[i] for i in all_fit_pos] == fit_nodes and len(all_test_pos) == 16,
        "E484 test roles do not preserve canonical fit/test order",
    )
    test_nodes = [all_nodes[i] for i in all_test_pos]
    test_x = all_x[all_test_pos]
    test_clipped = clipped_y[all_test_pos]
    test_selected = selected[all_test_pos]
    raw_targets = _read_raw_test_targets(test_nodes, test_clipped, test_selected)
    component_ids = np.asarray([node["component"] for node in test_nodes], dtype=np.int64)
    row_groups = np.asarray([node["row_group"] for node in test_nodes], dtype=np.int64)
    sorted_components, component_order, multiplicities = _bootstrap_counts(component_ids)
    row_counts = test_selected.sum(axis=1).astype(np.int64)
    require((row_counts > 0).all(), "E484 test component has no selected rows")
    scores, moments, predictions = {}, {}, {}
    for seed in SEEDS:
        for arm_name in ARMS:
            key = (seed, arm_name)
            model = validate_checkpoint(run / cells[key]["checkpoint"], cells[key], seed, arm_name)
            model.to(device=device, dtype=torch.float32)
            with (
                e474.full_ieee_float32()
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            ):
                prediction = np.empty((16, ROWS, 2), dtype=np.float32)
                for i, sequence in enumerate(test_x):
                    prediction[i] = _predict_sequence(model, sequence, device)
            if not np.isfinite(prediction).all():
                raise ValueError(f"E484 prediction matrix nonfinite for {key}")
            metric = global_wp(raw_targets, prediction, test_selected)
            require(
                metric.get("metric_id") == METRIC_ID
                and metric.get("selected_rows") == int(row_counts.sum()),
                f"E484 official metric receipt differs: {key}",
            )
            moments_arm, _ = paired_component_moments(
                raw_targets, prediction, prediction, test_selected
            )
            name = f"seed{seed}__{arm_name}"
            scores[name] = {
                "seed": seed,
                "arm": arm_name,
                "parameters": EXPECTED_PARAMETERS[arm_name],
                "updates": UPDATES,
                "wp": float(metric["wp"]),
                "t0": float(metric["t0"]),
                "t1": float(metric["t1"]),
                "selected_rows": int(metric["selected_rows"]),
                "component_row_counts": row_counts.tolist(),
            }
            moments[name] = moments_arm[component_order]
            predictions[name] = prediction
    bootstrap_wp = {}
    bootstrap_target_wp = {}
    for key, arm_moments in moments.items():
        bootstrap_wp[key] = np.asarray(
            [_score_component_moments(arm_moments, draw) for draw in multiplicities],
            dtype=np.float64,
        )
        bootstrap_target_wp[key] = np.stack(
            [_score_component_moments_by_target(arm_moments, draw) for draw in multiplicities]
        )
    contrast_results = {}
    contrast_draws = {}
    target_contrast_results = {}
    target_contrast_draws = {}
    for seed in SEEDS:
        for label, candidate_arm, reference_arm in CONTRASTS:
            candidate_key = f"seed{seed}__{candidate_arm}"
            reference_key = f"seed{seed}__{reference_arm}"
            name = f"seed{seed}__{label}"
            point = scores[candidate_key]["wp"] - scores[reference_key]["wp"]
            draws = bootstrap_wp[candidate_key] - bootstrap_wp[reference_key]
            contrast_results[name] = {
                "seed": seed,
                "contrast": label,
                "candidate": candidate_arm,
                "reference": reference_arm,
                **_contrast_result(point, draws),
            }
            contrast_draws[name] = draws
            target_point = np.asarray(
                [
                    scores[candidate_key]["t0"] - scores[reference_key]["t0"],
                    scores[candidate_key]["t1"] - scores[reference_key]["t1"],
                ],
                dtype=np.float64,
            )
            target_draws = bootstrap_target_wp[candidate_key] - bootstrap_target_wp[reference_key]
            target_contrast_results[name] = {
                "seed": seed,
                "contrast": label,
                "candidate": candidate_arm,
                "reference": reference_arm,
                "targets": {
                    f"t{target}": _contrast_result(target_point[target], target_draws[:, target])
                    for target in range(2)
                },
            }
            target_contrast_draws[name] = target_draws
    require(
        len(scores) == 12 and len(contrast_results) == 16, "E484 score/contrast inventory differs"
    )
    arms_order = [f"seed{seed}__{arm}" for seed in SEEDS for arm in ARMS]
    result = {
        "kind": "independent E484 causal context-mixture grade",
        "metric_id": METRIC_ID,
        "scope": "two fit seeds; same 16 exposed components; mechanism diagnostic only",
        "fit_cells": 12,
        "contrast_count": 16,
        "no_seed_pooling": True,
        "arm_scores": scores,
        "registered_contrasts": contrast_results,
        "registered_target_contrasts": target_contrast_results,
        "component_ids": component_ids.tolist(),
        "sorted_component_ids": sorted_components.tolist(),
        "row_groups": row_groups.tolist(),
        "selected_row_counts": row_counts.tolist(),
        "selected_rows_total": int(row_counts.sum()),
        "bootstrap": {
            "method": "official centered weighted-Pearson moments; paired component resampling",
            "draws": BOOTSTRAP_COUNT,
            "seed": BOOTSTRAP_SEED,
            "multiplicities_shared_across_all_arms_and_contrasts": True,
            "unit": "numeric-sorted fixed component ID",
        },
    }
    exports = {
        "arm_names": np.asarray(arms_order),
        "targets_raw": raw_targets,
        "predictions_raw": np.stack([predictions[name] for name in arms_order]),
        "selected_mask": test_selected,
        "component_ids": component_ids,
        "sorted_component_ids": sorted_components,
        "row_groups": row_groups,
        "selected_row_counts": row_counts,
        "component_bootstrap_multiplicities": multiplicities,
        "arm_bootstrap_wp": np.stack([bootstrap_wp[name] for name in arms_order]),
        "contrast_names": np.asarray(list(contrast_draws)),
        "contrast_bootstrap_diffs": np.stack(list(contrast_draws.values())),
        "arm_bootstrap_wp_by_target": np.stack([bootstrap_target_wp[name] for name in arms_order]),
        "target_contrast_bootstrap_diffs": np.stack(list(target_contrast_draws.values())),
    }
    return result, exports


def main() -> None:
    """Require a fit-only proof pass before a separate scoring invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fit-spec", type=Path, required=True)
    parser.add_argument("--fit-attempt", type=Path, required=True)
    parser.add_argument("--probe-run", type=Path, required=True)
    parser.add_argument("--probe-spec", type=Path, required=True)
    parser.add_argument("--probe-attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--verify-fit-only", action="store_true")
    modes.add_argument("--score", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite E484 grader output: {output}")
    proof, cells, report, fit_manifest, fit_x, fit_nodes = validate_full_run(
        args.run.resolve(),
        args.fit_spec.resolve(),
        args.fit_attempt.resolve(),
        args.probe_run.resolve(),
        args.probe_spec.resolve(),
        args.probe_attempt.resolve(),
    )
    if args.verify_fit_only:
        print(
            json.dumps(
                {
                    "status": "all_12_fit_receipts_and_checkpoints_verified_test_unread",
                    "fit_cells": len(cells),
                    "probe_cells": proof["probe"]["probe_cells"],
                    "test_rows_loaded": False,
                    "proof": proof,
                },
                indent=2,
            ),
            flush=True,
        )
        return
    if args.device == "cuda":
        from wnn.resources import configure_torch

        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA scoring requested but unavailable")
    torch.set_num_threads(1)
    score, exports = score_validated_run(
        args.run.resolve(), cells, fit_manifest, fit_x, fit_nodes, torch.device(args.device)
    )
    output.mkdir(parents=True)
    score["fit_only_proof"] = proof
    score["grader_sha256"] = sha256_file(Path(__file__))
    npz_path = output / "paired_components.npz"
    temporary_npz = output / "paired_components.tmp.npz"
    np.savez_compressed(temporary_npz, **exports)
    temporary_npz.replace(npz_path)
    score["raw_paired_exports"] = {"path": npz_path.name, "sha256": sha256_file(npz_path)}
    temp_result = output / "scores.tmp.json"
    temp_result.write_text(json.dumps(score, indent=2, allow_nan=False) + "\n")
    temp_result.replace(output / "scores.json")
    print(
        json.dumps(
            {
                "output": str(output),
                "scores": len(score["arm_scores"]),
                "contrasts": len(score["registered_contrasts"]),
                "selected_rows": score["selected_rows_total"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

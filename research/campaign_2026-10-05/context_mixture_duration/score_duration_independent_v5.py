"""Independently verify E488's 12 trajectories and score only after proof gates.

The scorer reconstructs the six-head-free four-arm E484 model, derives the
train-only feature normalization itself, and validates controller/probe/fit
receipts before its score path reads any test-role parquet row group.
"""

from __future__ import annotations

import argparse
import contextlib
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

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets/valid.parquet"
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
E477_DIR = ROOT / "research/campaign_2026-10-05/activation_head_mixture"
E484_DIR = ROOT / "research/campaign_2026-10-05/activation_context_mixture"
E488_DIR = ROOT / "research/campaign_2026-10-05/context_mixture_duration"

FEATURES = (
    "i0_p0",
    "i0_p11",
    "i0_v0",
    "i0_v11",
    "i0_dp0",
    "i0_dv0",
    "i1_p0",
    "i1_p11",
    "i1_v0",
    "i1_v11",
    "i1_dp0",
    "i1_dv0",
    *(f"a{i}" for i in range(8)),
)
SEEDS = (271839, 271840, 271841)
ARMS = ("gelu", "fixed_mix", "static_mix", "recurrent_mix")
EPOCHS = (1, 3)
ROWS, FEATURES_DIM, RAW_FEATURES = 20_000, 42, 20
HIDDEN, CHUNK_ROWS, BATCH = 64, 256, 4
UPDATES_PER_EPOCH, TOTAL_UPDATES = 474, 1422
TELEMETRY_STEPS = (0, 46, 236, 473, 947, 1421)
BOOTSTRAP_COUNT, BOOTSTRAP_SEED = 2_000, 271_839
METRIC_CLIP = 2.0
BASE_PARAMETERS, GATED_PARAMETERS = 61_058, 61_122
LR, WEIGHT_DECAY, BETAS, CLIP_NORM = 0.001, 0.01, (0.9, 0.999), 1.0
GATE_EPSILON = 1e-6
MAX_GPU_ALLOCATOR = 2 * 1024**3
MAX_GPU_OBSERVED = 3_000_000_000
FIT_OOM_KEYS = ("oom", "oom_kill", "oom_group_kill")

EXPECTED_HASHES = {
    "fit_duration.py": "89d38cd71da3ff8905d51da60d369bbe04b07d26789ab242085d59d384ac299e",
    "tests_duration.py": "29a2b247d5f32e32c5242325c0bbce35f511a44fc32d01c429d29f252d33413d",
    "gpu_probe_duration.py": "3e1f383c9364e70b2ea0e4846a15de5281090326a1d5fe634061dbaebea5747f",
    "gpu_probe_duration_v2.py": "d7abd232ebbec9a10ec5dff2415103ff75ce672633e512034426669347489d7b",
    "E488_DESIGN.md": "3764a095504af6da66a4faa908a47ff5dd3f94266249d5bea7c07cf611148824",
    "SPEC_CRITIC.md": "ccc5caba6eef86b928e31cc78ce65602ad528a6549b0017a3d91ffcf1a3aa3e1",
    "CONTROLLER_CLARIFICATIONS.md": "d382f867bd8c889c5c63f40228ba1133e0f52d2a9e396860fbf15009fced0a9c",
    "INDEPENDENT_REVIEW.md": "85fc20e06e5894a48f7ea9851d44bd9a934c03cd878b51a0c153dec61a316e3a",
    "E484_actor.py": "2dec963a966ba30e1dc3deed40472fd2d0129a2d24fa07585a3a7ed4e2e1dcbc",
    "E470.py": "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9",
    "E474.py": "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068",
    "E477.py": "2d4b1b103dd31912fea7ab24a63e6ce176e532dca8d0dfa2cdb32aebd9e567d7",
    "wnn.nn.py": "388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3",
    "wnn.resources.py": "06a624ad963dfc2b0c75ce717278de5f1bb85f48518a99e519b697a371d4cf09",
    "wnn.gmetric.py": "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa",
    "wnn.evaluation.py": "96cebf2c8d564493f5a3a866b8d879964247f885587a455ebb1449e937230d0e",
    "dataset": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
    "cohort": "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2",
    "RESOURCE_ONLY_ADMISSION_AMENDMENT_V3.md": "9309269da0c263053f8fd767142e29dc6a127ac30477c35074e6589535ef9072",
    "RESOURCE_ONLY_ADMISSION_AMENDMENT_V3_REVIEW.md": "d99329047aa1d6e21b662019e1e6687ea0f06c44ae1218dbf5d692034d1020b4",
    "ROOT_FULL_FIT_ADMISSION_V3.md": "4ac484c54c5638355f292bd3f4a24a0852f93e7a35fdd75da49b874d79a029e6",
    "e488_twelve_fit_v3.sealed.json": "5230798a975dfc3b207cf855eccf38ef990ece7fe92578b0c6bf866b5b525936",
    "e488_gpu_probe_v2.sealed.json": "353eb321768c31567853c8c2e9314a57dd2a3dbf84f7564b1b70e8d9110afdaf",
    "actual_probe_v2": "6db1e875175f6b7631a1ca126738ebae40ea7db576e9788a60ff04635c6730ba",
}

FIT_CODE_PATHS = frozenset(
    (
        "research/campaign_2026-10-05/activation_context_mixture/fit_context_mixture_v2.py",
        "research/campaign_2026-10-05/activation_context_mixture/gpu_probe_v2.py",
        "research/campaign_2026-10-05/activation_context_mixture/tests_v2.py",
        "research/campaign_2026-10-05/activation_head_mixture/fit_activation_head_mixture.py",
        "research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py",
        "research/campaign_2026-10-05/activation_placement/fit_activation_placement.py",
        "research/campaign_2026-10-05/context_mixture_duration/CONTROLLER_CLARIFICATIONS.md",
        "research/campaign_2026-10-05/context_mixture_duration/E488_DESIGN.md",
        "research/campaign_2026-10-05/context_mixture_duration/INDEPENDENT_REVIEW.md",
        "research/campaign_2026-10-05/context_mixture_duration/RESOURCE_ONLY_ADMISSION_AMENDMENT_V3.md",
        "research/campaign_2026-10-05/context_mixture_duration/RESOURCE_ONLY_ADMISSION_AMENDMENT_V3_REVIEW.md",
        "research/campaign_2026-10-05/context_mixture_duration/ROOT_FULL_FIT_ADMISSION_V3.md",
        "research/campaign_2026-10-05/context_mixture_duration/SPEC_CRITIC.md",
        "research/campaign_2026-10-05/context_mixture_duration/fit_duration.py",
        "research/campaign_2026-10-05/context_mixture_duration/gpu_probe_duration.py",
        "research/campaign_2026-10-05/context_mixture_duration/gpu_probe_duration_v2.py",
        "research/campaign_2026-10-05/context_mixture_duration/tests_duration.py",
        "scripts/run_campaign.py",
        "src/wnn/__init__.py",
        "src/wnn/campaign.py",
        "src/wnn/campaign_local.py",
        "src/wnn/campaign_spec.py",
        "src/wnn/gmetric.py",
        "src/wnn/nn.py",
        "src/wnn/optim.py",
        "src/wnn/resources.py",
    )
)
FIT_INPUT_PATHS = frozenset(
    (
        "datasets/valid.parquet",
        "research/campaign_2026-10-04/runner/attempts/e484-context-mixture-real-gpu-probe-v1.json",
        "research/campaign_2026-10-04/runner/attempts/e488-context-duration-real-gpu-probe-v2.json",
        "research/campaign_2026-10-04/runner/logs/e484-context-mixture-real-gpu-probe-v1.log",
        "research/campaign_2026-10-05/activation_context_mixture/ACTOR_SOURCE_NOTE_V2.md",
        "research/campaign_2026-10-05/activation_context_mixture/E484_DESIGN_V2.md",
        "research/campaign_2026-10-05/activation_context_mixture/GPU_PROBE_FAILURE_V1.md",
        "research/campaign_2026-10-05/activation_context_mixture/INDEPENDENT_REVIEW.md",
        "research/campaign_2026-10-05/activation_context_mixture/LITERATURE.md",
        "research/campaign_2026-10-05/activation_context_mixture/ROOT_VERSIONED_SOURCE_REVIEW_V2.md",
        "research/campaign_2026-10-05/activation_context_mixture/SPEC_CRITIC.md",
        "research/campaign_2026-10-05/activation_context_mixture/SPEC_CRITIC_V2_RESPONSE.md",
        "research/campaign_2026-10-05/activation_context_mixture/preflight_cpu_v5/fit_manifest.json",
        "research/campaign_2026-10-05/context_mixture_duration/e488_gpu_probe_v2.sealed.json",
        "research/campaign_2026-10-05/context_mixture_duration/gpu_probe_v2/probe.json",
        "research/campaign_2026-10-05/context_mixture_duration/preflight_cpu_v1/fit_manifest.json",
        "research/campaign_2026-10-05/target_mechanics/cohort.json",
    )
)
PROBE_CODE_PATHS = FIT_CODE_PATHS - {
    "research/campaign_2026-10-05/context_mixture_duration/RESOURCE_ONLY_ADMISSION_AMENDMENT_V3.md",
    "research/campaign_2026-10-05/context_mixture_duration/RESOURCE_ONLY_ADMISSION_AMENDMENT_V3_REVIEW.md",
    "research/campaign_2026-10-05/context_mixture_duration/ROOT_FULL_FIT_ADMISSION_V3.md",
}
PROBE_INPUT_PATHS = FIT_INPUT_PATHS - {
    "research/campaign_2026-10-04/runner/attempts/e488-context-duration-real-gpu-probe-v2.json",
    "research/campaign_2026-10-05/context_mixture_duration/e488_gpu_probe_v2.sealed.json",
    "research/campaign_2026-10-05/context_mixture_duration/gpu_probe_v2/probe.json",
}

CONTRASTS = (
    ("recurrent_minus_static", "recurrent_mix", "static_mix"),
    ("recurrent_minus_fixed", "recurrent_mix", "fixed_mix"),
    ("static_minus_fixed", "static_mix", "fixed_mix"),
    ("fixed_minus_gelu", "fixed_mix", "gelu"),
)

sys.path.insert(0, str(ROOT / "src"))
from wnn.campaign import RUNTIME_FILES, runtime_identity  # noqa: E402
from wnn.campaign_spec import digest as spec_digest  # noqa: E402
from wnn.campaign_spec import validate_spec  # noqa: E402
from wnn.evaluation import _score_component_moments, paired_component_moments  # noqa: E402
from wnn.gmetric import METRIC_ID, global_wp  # noqa: E402


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory for provenance checks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    """Raise a descriptive proof failure at the trust boundary."""
    if not ok:
        raise ValueError(message)


def finite(value: Any, label: str) -> float:
    """Require a finite numeric receipt field, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is nonfinite")
    return result


class ManualGate(nn.Module):
    """Independent registered GELU/LeakyReLU head gate implementation."""

    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unregistered E488 arm: {arm}")
        self.arm = arm
        if arm in ("static_mix", "recurrent_mix"):
            self.gate_coeff = nn.Parameter(torch.zeros(HIDDEN))

    def forward(self, hidden: Tensor, recurrent: Tensor) -> tuple[Tensor, Tensor | None]:
        """Return head activation and optional gate values."""
        gelu = F.gelu(hidden, approximate="none")
        if self.arm == "gelu":
            return gelu, None
        leaky = F.leaky_relu(hidden, negative_slope=0.01)
        if self.arm == "fixed_mix":
            gate = torch.full_like(hidden, 0.5)
        elif self.arm == "static_mix":
            gate = torch.sigmoid(self.gate_coeff).view(1, 1, HIDDEN).expand_as(hidden)
        else:
            context = recurrent
            context = context / torch.sqrt(
                context.square().mean(dim=-1, keepdim=True) + GATE_EPSILON
            )
            gate = torch.sigmoid(self.gate_coeff.view(1, 1, HIDDEN) * context)
        return gate * gelu + (1.0 - gate) * leaky, gate


class IndependentE488Model(nn.Module):
    """Recreate frozen E474 topology with an independently written E484 gate."""

    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unregistered E488 arm: {arm}")
        self.register_buffer("mean", torch.zeros(FEATURES_DIM))
        self.register_buffer("std", torch.ones(FEATURES_DIM))
        self.inp = nn.Linear(FEATURES_DIM, HIDDEN)
        self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(
            nn.Linear(2 * HIDDEN, HIDDEN), ManualGate(arm), nn.Linear(HIDDEN, 2)
        )
        self.arm = arm

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Compute one causal chunk and its carry state; reject nonfinite values."""
        if x.ndim != 3 or x.shape[-1] != FEATURES_DIM or not torch.isfinite(x).all():
            raise ValueError("E488 scorer input must be finite [batch,time,42]")
        projection = F.gelu(self.inp((x - self.mean) / self.std), approximate="none")
        recurrent, final_state = self.gru(projection, state)
        hidden = self.head[0](torch.cat((recurrent, projection), dim=-1))
        activated, _gate = self.head[1](hidden, recurrent)
        prediction = self.head[2](activated)
        if not torch.isfinite(prediction).all() or not torch.isfinite(final_state).all():
            raise ValueError("E488 scorer output/state is nonfinite")
        return prediction, final_state


def expected_state_keys(arm: str) -> set[str]:
    """Declare valid checkpoint keys without importing the actor model."""
    names = {
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
    if arm in ("static_mix", "recurrent_mix"):
        names.add("head.1.gate_coeff")
    return names


def validate_identity_buffers(state: dict[str, Tensor]) -> None:
    """Require external normalization and identity values in model buffers."""
    require("mean" in state and "std" in state, "checkpoint lacks mean/std buffers")
    require(torch.equal(state["mean"].cpu(), torch.zeros(FEATURES_DIM)), "nonidentity model mean")
    require(torch.equal(state["std"].cpu(), torch.ones(FEATURES_DIM)), "nonidentity model std")


def validate_checkpoint(run: Path, record: dict, seed: int, arm: str) -> IndependentE488Model:
    """Check one epoch checkpoint's manifest identity, tensors and parameter count."""
    expected_name = f"seed{seed}_{arm}_epoch{record['epoch_number']}.pt"
    relative = record.get("path")
    require(relative == expected_name, f"unexpected E488 checkpoint path {seed}/{arm}")
    checkpoint = (run / relative).resolve()
    require(checkpoint.is_relative_to(run.resolve()), "checkpoint path escapes fit run")
    require(not (run / relative).is_symlink(), "checkpoint symlink is not accepted")
    require(checkpoint.is_file(), f"missing E488 checkpoint {relative}")
    require(sha256_file(checkpoint) == record.get("sha256"), f"checkpoint hash differs {relative}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(
        isinstance(state, dict) and set(state) == expected_state_keys(arm), "checkpoint keys differ"
    )
    require(
        all(isinstance(value, Tensor) and torch.isfinite(value).all() for value in state.values()),
        f"checkpoint contains malformed/nonfinite values {relative}",
    )
    validate_identity_buffers(state)
    model = IndependentE488Model(arm)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"checkpoint shapes do not match the independent model {relative}"
        ) from error
    expected_parameters = (
        GATED_PARAMETERS if arm in ("static_mix", "recurrent_mix") else BASE_PARAMETERS
    )
    require(
        sum(parameter.numel() for parameter in model.parameters()) == expected_parameters,
        f"independent parameter count differs for {seed}/{arm}",
    )
    require(
        record.get("state_sha256") == state_tensor_sha256(state),
        f"checkpoint state hash differs {relative}",
    )
    return model


def state_tensor_sha256(state: dict[str, Tensor]) -> str:
    """Hash sorted state names, dtype, shape and exact CPU bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def expected_parent_sources() -> dict[str, str]:
    """Return the immutable model/runtime parent identity map."""
    return {
        "e470": EXPECTED_HASHES["E470.py"],
        "e474": EXPECTED_HASHES["E474.py"],
        "e477": EXPECTED_HASHES["E477.py"],
        "e484_actor": EXPECTED_HASHES["E484_actor.py"],
        "wnn.nn": EXPECTED_HASHES["wnn.nn.py"],
        "wnn.resources": EXPECTED_HASHES["wnn.resources.py"],
    }


def expected_actor_sources() -> dict[str, str]:
    """Return the actor's source receipt keys and exact hashes."""
    return {
        "fit_duration.py": EXPECTED_HASHES["fit_duration.py"],
        "tests_duration.py": EXPECTED_HASHES["tests_duration.py"],
        "gpu_probe_duration.py": EXPECTED_HASHES["gpu_probe_duration.py"],
    }


def check_scientific_sources() -> dict[str, str]:
    """Check all E488 and inherited parent files before any data loading."""
    paths = {
        "fit_duration.py": HERE / "fit_duration.py",
        "tests_duration.py": HERE / "tests_duration.py",
        "gpu_probe_duration.py": HERE / "gpu_probe_duration.py",
        "gpu_probe_duration_v2.py": HERE / "gpu_probe_duration_v2.py",
        "E488_DESIGN.md": HERE / "E488_DESIGN.md",
        "SPEC_CRITIC.md": HERE / "SPEC_CRITIC.md",
        "CONTROLLER_CLARIFICATIONS.md": HERE / "CONTROLLER_CLARIFICATIONS.md",
        "INDEPENDENT_REVIEW.md": HERE / "INDEPENDENT_REVIEW.md",
        "E484_actor.py": E484_DIR / "fit_context_mixture_v2.py",
        "E470.py": E470_DIR / "fit_activation_optimizer.py",
        "E474.py": E474_DIR / "fit_activation_placement.py",
        "E477.py": E477_DIR / "fit_activation_head_mixture.py",
        "wnn.nn.py": ROOT / "src/wnn/nn.py",
        "wnn.resources.py": ROOT / "src/wnn/resources.py",
        "wnn.gmetric.py": ROOT / "src/wnn/gmetric.py",
        "wnn.evaluation.py": ROOT / "src/wnn/evaluation.py",
        "dataset": DATA_PATH,
        "cohort": COHORT_PATH,
        "RESOURCE_ONLY_ADMISSION_AMENDMENT_V3.md": HERE / "RESOURCE_ONLY_ADMISSION_AMENDMENT_V3.md",
        "RESOURCE_ONLY_ADMISSION_AMENDMENT_V3_REVIEW.md": HERE
        / "RESOURCE_ONLY_ADMISSION_AMENDMENT_V3_REVIEW.md",
        "ROOT_FULL_FIT_ADMISSION_V3.md": HERE / "ROOT_FULL_FIT_ADMISSION_V3.md",
        "e488_twelve_fit_v3.sealed.json": HERE / "e488_twelve_fit_v3.sealed.json",
        "e488_gpu_probe_v2.sealed.json": HERE / "e488_gpu_probe_v2.sealed.json",
        "actual_probe_v2": HERE / "gpu_probe_v2/probe.json",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in EXPECTED_HASHES.items():
        require(observed.get(name) == expected, f"E488 source/input identity changed: {name}")
    return observed


def safe_repo_file(relative: str) -> Path:
    """Resolve a repository-relative path without traversal or symlinks."""
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe path: {relative}")
    path = ROOT / rel
    require(not path.is_symlink(), f"symlink not allowed in sealed evidence: {relative}")
    return path


def validate_spec_file(path: Path, artifact: str) -> dict:
    """Validate canonical local bytes for every spec code and input entry."""
    spec = json.loads(path.read_text())
    checked = validate_spec(spec)
    require(checked == spec, f"noncanonical E488 campaign spec: {path}")
    require(spec.get("kind") == "gpu", "E488 probe/full-fit requires a GPU spec")
    require(spec.get("artifacts") == [artifact], f"wrong E488 declared artifact: {artifact}")
    for category in ("code", "inputs"):
        entries = spec.get(category)
        require(isinstance(entries, list), f"missing E488 {category} manifest")
        for entry in entries:
            file_path = safe_repo_file(entry["path"])
            require(file_path.is_file(), f"missing sealed {category} file {entry['path']}")
            require(
                file_path.stat().st_size == entry["size"], f"sealed size differs {entry['path']}"
            )
            require(
                sha256_file(file_path) == entry["sha256"], f"sealed hash differs {entry['path']}"
            )
    return spec


def validate_attempt(
    attempt_path: Path,
    spec_path: Path,
    artifact_relative: str,
    expected_phase: str,
) -> tuple[dict, dict, Path]:
    """Validate controller completion, exact runtime/artifact and no new OOMs."""
    spec = validate_spec_file(spec_path, artifact_relative)
    record = json.loads(attempt_path.read_text())
    require(
        isinstance(record.get("id"), str) and record["id"], f"missing {expected_phase} attempt ID"
    )
    require(record.get("kind") == "gpu", f"wrong {expected_phase} attempt kind")
    require(
        record.get("spec") == spec and record.get("spec_sha256") == spec_digest(spec),
        f"{expected_phase} attempt does not bind the sealed spec",
    )
    argv = record.get("argv")
    require(
        isinstance(argv, list)
        and len(argv) >= 2
        and Path(argv[0]).is_absolute()
        and argv[1:] == spec.get("argv", [])[1:],
        f"{expected_phase} command differs from sealed argv",
    )
    require(
        record.get("status") == "finished"
        and record.get("exit_code") == 0
        and record.get("signals") == []
        and record.get("teardown_verified") is True
        and all(
            record.get(name) is None for name in ("failure", "failure_scope", "failure_category")
        ),
        f"{expected_phase} attempt did not finish cleanly",
    )
    unit = record.get("unit_properties", {})
    require(
        unit.get("Result") == "success" and unit.get("ExecMainStatus") in (0, "0"),
        f"{expected_phase} systemd unit failed",
    )
    runtime = runtime_identity(ROOT)
    require(
        record.get("runtime_sha256") == runtime and set(runtime) == set(RUNTIME_FILES),
        f"{expected_phase} runner runtime hashes differ",
    )
    for before_key, after_key in (
        ("initial_memory_events", "final_memory_events"),
        ("initial_parent_memory_events", "final_parent_memory_events"),
    ):
        before, after = record.get(before_key), record.get(after_key)
        require(
            isinstance(before, dict) and isinstance(after, dict),
            f"{expected_phase} OOM snapshots missing",
        )
        require(
            all(
                type(before.get(key)) is int
                and type(after.get(key)) is int
                and before[key] >= 0
                and before[key] == after[key]
                for key in FIT_OOM_KEYS
            ),
            f"{expected_phase} attempt increased an OOM counter",
        )
    require(
        all(record.get("final_memory_events", {}).get(key) == 0 for key in FIT_OOM_KEYS),
        f"{expected_phase} worker OOM counters are nonzero",
    )
    limits = spec["limits"]
    peak_ram, peak_vram = record.get("peak_ram_bytes"), record.get("peak_observed_vram_bytes")
    require(
        type(peak_ram) is int and peak_ram < limits["ram_bytes"],
        f"{expected_phase} RAM cap exceeded",
    )
    require(
        type(peak_vram) is int and peak_vram < limits["gpu_stop_bytes"],
        f"{expected_phase} GPU stop exceeded",
    )
    artifacts = record.get("artifacts")
    require(
        isinstance(artifacts, list) and len(artifacts) == 1,
        f"{expected_phase} artifact receipt count differs",
    )
    artifact_path = safe_repo_file(artifact_relative)
    require(artifact_path.is_file(), f"missing {expected_phase} artifact {artifact_relative}")
    actual = {
        "path": artifact_relative,
        "size": artifact_path.stat().st_size,
        "sha256": sha256_file(artifact_path),
    }
    require(artifacts == [actual], f"{expected_phase} artifact receipt differs")
    require(
        str(Path(artifact_relative).parent) in spec["argv"],
        f"{expected_phase} output path is not in argv",
    )
    return record, spec, artifact_path


def canonical_fit_test_nodes() -> tuple[list[dict], list[dict], dict]:
    """Read immutable row-role metadata without touching the parquet labels."""
    require(sha256_file(COHORT_PATH) == EXPECTED_HASHES["cohort"], "E488 cohort identity differs")
    document = json.loads(COHORT_PATH.read_text())
    nodes = document.get("cohort")
    require(isinstance(nodes, list) and len(nodes) == 40, "E488 cohort row inventory differs")
    fit_nodes = [node for node in nodes if node.get("role") == "fit"]
    test_nodes = [node for node in nodes if node.get("role") == "test"]
    require(len(fit_nodes) == 24 and len(test_nodes) == 16, "E488 fit/test component counts differ")
    require(len({node["component"] for node in fit_nodes}) == 24, "duplicate E488 fit component")
    require(len({node["component"] for node in test_nodes}) == 16, "duplicate E488 test component")
    require(
        not (
            {node["component"] for node in fit_nodes} & {node["component"] for node in test_nodes}
        ),
        "E488 fit/test components overlap",
    )
    return fit_nodes, test_nodes, document


def causal_features(raw: np.ndarray) -> np.ndarray:
    """Reproduce E470's point-in-time feature transform after raw finite checks."""
    values = np.asarray(raw)
    require(values.shape == (ROWS, RAW_FEATURES), f"unexpected raw feature shape: {values.shape}")
    require(np.isfinite(values).all(), "raw features are nonfinite before causal transform")
    difference = np.diff(values, axis=0, prepend=values[:1])
    event = np.any(difference[:, :2] != 0, axis=1)
    index = np.arange(len(values))
    last_event = np.maximum.accumulate(np.where(event, index, 0))
    features = np.nan_to_num(
        np.column_stack((values, difference, np.log1p(index - last_event), event)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    require(
        features.shape == (ROWS, FEATURES_DIM) and np.isfinite(features).all(),
        "E488 causal feature transform is malformed/nonfinite",
    )
    return features


def derive_fit_normalizer(fit_nodes: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only fit-role feature columns and independently derive E470 statistics."""
    parquet = pq.ParquetFile(DATA_PATH)
    sequences = []
    for node in fit_nodes:
        table = parquet.read_row_group(node["row_group"], columns=list(FEATURES), use_threads=False)
        raw = np.column_stack([table[name].to_numpy() for name in FEATURES])
        sequences.append(causal_features(raw))
    features = np.stack(sequences)
    if features.shape != (24, ROWS, FEATURES_DIM):
        raise ValueError("E488 independent fit feature shape differs")
    mean = features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(features.std(axis=(0, 1), dtype=np.float64), 0.001).astype(np.float32)
    normalized = np.clip((features - mean) / scale, -8.0, 8.0).astype(np.float32)
    require(np.isfinite(normalized).all(), "normalized fit features are nonfinite")
    return mean, scale, normalized


def canonical_epoch_order(seed: int, epoch_index: int) -> np.ndarray:
    """Return E488's zero-based paired shuffle over canonical fit positions."""
    require(seed in SEEDS and epoch_index in (0, 1, 2), "invalid E488 seed/epoch index")
    order = np.arange(24)
    np.random.default_rng(seed + epoch_index).shuffle(order)
    return order


def expected_precision(value: dict, label: str) -> None:
    """Require active full-IEEE FP32 receipt captured inside the context."""
    require(
        value.get("device_math") == "FP32 IEEE"
        and value.get("autocast") is False
        and value.get("torch.backends.fp32_precision") == "ieee"
        and value.get("torch.backends.cuda.matmul.fp32_precision") == "ieee"
        and value.get("torch.backends.cudnn.fp32_precision") == "ieee"
        and value.get("torch.backends.cudnn.rnn.fp32_precision") == "ieee",
        f"{label} precision policy is not full IEEE FP32",
    )


def validate_probe_report(
    probe: dict,
    probe_attempt: dict,
    probe_attempt_path: Path,
    probe_spec: dict,
    fit_nodes: list[dict],
    fit_mean: np.ndarray,
    fit_scale: np.ndarray,
) -> dict:
    """Require an all-arm actual-shape probe and amended budget projection."""
    require(
        probe.get("kind") == "E488 actual-shape CUDA probe"
        and probe.get("status") == "passed"
        and probe.get("test_rows_loaded") is False
        and probe.get("cells_complete") == 12,
        "E488 actual-shape probe did not complete all twelve fit-only cells",
    )
    require(probe.get("device_count_visible") == 1, "E488 probe must see exactly one GPU")
    expected_precision(probe.get("active_precision_inside_full_ieee_context", {}), "E488 probe")
    require(
        probe.get("probe_v2_sha256") == EXPECTED_HASHES["gpu_probe_duration_v2.py"],
        "E488 v2 probe wrapper hash differs",
    )
    require(probe.get("sources") == expected_actor_sources(), "E488 probe actor source map differs")
    expected_fit_groups = [node["row_group"] for node in fit_nodes]
    require(
        probe.get("fit_components") == expected_fit_groups, "E488 probe fit group order differs"
    )
    parity = probe.get("reference_parity", {})
    require(
        parity.get("device") == "cuda" and len(parity.get("checks", [])) == 2,
        "E488 reference parity receipt is incomplete",
    )
    expected_refs = {"gelu": "gelu", "fixed_mix": "same_channel_mix"}
    for item in parity["checks"]:
        arm = item.get("arm")
        require(
            arm in expected_refs and item.get("reference") == expected_refs[arm],
            f"unexpected E488 native parity arm: {arm}",
        )
        require(
            item.get("exact_forward_state_input_and_parameter_gradients") is True
            and item.get("exact_clipped_adamw_step") is True,
            f"E488 native reference parity failed for {arm}",
        )
        require(
            finite(item.get("preclip_gradient_norm"), "E488 parity norm") > CLIP_NORM,
            f"E488 parity did not exercise clipping for {arm}",
        )
    probe_normalizer = probe.get("fit_only_normalization", {})
    _expect_normalizer(probe_normalizer, fit_mean, fit_scale, "E488 probe")
    allocator_peak = probe.get("parity_memory", {}).get("peak_allocated_bytes")
    require(
        type(allocator_peak) is int and allocator_peak <= MAX_GPU_ALLOCATOR,
        "E488 parity allocator peak exceeds cap",
    )
    records = probe.get("cells", [])
    expected_cells = {(seed, arm) for seed in SEEDS for arm in ARMS}
    found: set[tuple[int, str]] = set()
    for cell in records:
        key = (cell.get("seed"), cell.get("arm"))
        require(
            key in expected_cells and key not in found, f"unknown/duplicate E488 probe cell {key}"
        )
        found.add(key)
        require(cell.get("changed_parameter_tensors", 0) > 0, f"E488 probe did not update {key}")
        require(cell.get("state_shape") == [2, 4, HIDDEN], f"E488 probe state shape differs {key}")
        finite(cell.get("loss"), f"E488 probe loss {key}")
        memory = cell.get("memory", {})
        peak = memory.get("peak_allocated_bytes")
        require(
            type(peak) is int and peak <= MAX_GPU_ALLOCATOR,
            f"E488 probe allocator cap exceeded {key}",
        )
        update_telemetry = cell.get("telemetry", {})
        preclip_norm = finite(
            update_telemetry.get("preclip_global_norm"), f"E488 probe gradient norm {key}"
        )
        clip_factor = finite(update_telemetry.get("clip_factor"), f"E488 probe clip factor {key}")
        require(
            0.0 < clip_factor <= 1.0
            and math.isclose(
                clip_factor,
                min(1.0, CLIP_NORM / (preclip_norm + 1e-6)),
                rel_tol=0,
                abs_tol=1e-12,
            ),
            f"E488 actual-shape probe clip factor disagrees with gradient norm: {key}",
        )
        for field in ("preclip_gradient_rms", "parameter_update_rms"):
            families = update_telemetry.get(field)
            require(
                isinstance(families, dict) and set(families) == {"inp", "gru", "head"},
                f"E488 probe {field} family keys differ: {key}",
            )
            require(
                all(
                    finite(value, f"E488 probe {field}/{family}") >= 0
                    for family, value in families.items()
                ),
                f"E488 probe {field} contains negative values: {key}",
            )
        head_update = update_telemetry.get("head_update_rms")
        require(
            isinstance(head_update, dict) and set(head_update) == {"head"},
            f"E488 probe head update RMS family differs: {key}",
        )
        require(
            math.isclose(
                finite(head_update["head"], "E488 probe head update RMS"),
                update_telemetry["parameter_update_rms"]["head"],
                rel_tol=0,
                abs_tol=1e-12,
            ),
            f"E488 probe head update RMS disagrees with family telemetry: {key}",
        )
    require(found == expected_cells, "E488 probe omitted a seed/arm cell")
    projection = verify_reviewed_projection(probe, probe_attempt)
    require(
        probe_spec.get("name") == "e488_context_duration_gpu_probe_v2",
        "E488 actual-shape probe spec name differs",
    )
    require(
        probe_spec.get("limits", {}).get("gpu_memory_bytes") == MAX_GPU_ALLOCATOR,
        "E488 probe sealed allocator limit differs",
    )
    return {
        "probe_attempt_id": probe_attempt.get("id"),
        "probe_spec_sha256": sha256_file(E488_DIR / "e488_gpu_probe_v2.sealed.json"),
        "probe_attempt_sha256": sha256_file(probe_attempt_path),
        "probe_fit_cells": len(found),
        "charged_projection_seconds": projection["charged_projection_seconds"],
        "amended_allowance_seconds": 1800,
    }


def verify_reviewed_projection(probe: dict, probe_attempt: dict) -> dict[str, float]:
    """Recompute original and controller-charged v3 projections from evidence."""
    actual_charge = finite(probe_attempt.get("elapsed_seconds"), "E488 probe charged seconds")
    body_probe_seconds = finite(probe.get("probe_seconds"), "E488 wrapper probe seconds")
    fit_update_projection = finite(probe.get("full_fit_projection_seconds"), "E488 fit projection")
    safety = finite(probe.get("projection_safety_multiplier"), "E488 projection margin")
    overhead = finite(probe.get("fixed_full_fit_overhead_margin_seconds"), "E488 fixed overhead")
    wrapper_projected = body_probe_seconds + fit_update_projection * safety + overhead
    require(
        math.isclose(
            finite(probe.get("projected_study_seconds_with_margin"), "E488 wrapper projection"),
            wrapper_projected,
            rel_tol=0,
            abs_tol=1e-9,
        ),
        "E488 wrapper projection arithmetic differs from its registered terms",
    )
    require(
        math.isclose(wrapper_projected, 1075.2630104514828, rel_tol=0, abs_tol=1e-6),
        "E488 original 900-second failed projection evidence differs",
    )
    charged_projected = actual_charge + fit_update_projection * safety + overhead
    require(
        math.isclose(charged_projected, 1081.277130055489, rel_tol=0, abs_tol=0.02),
        "E488 charged probe-plus-fit projection differs from reviewed amendment",
    )
    require(
        charged_projected <= 1800.0,
        f"E488 charged amended projection exceeds 1800s: {charged_projected}",
    )
    require(wrapper_projected <= 1800.0, "E488 wrapper projection exceeds amended allowance")
    return {
        "wrapper_projection_seconds": wrapper_projected,
        "charged_projection_seconds": charged_projected,
    }


def expected_contrast_names() -> list[str]:
    """List the four epoch contrasts at epochs one/three and one interaction."""
    names = []
    for epoch in EPOCHS:
        names.extend(
            [
                f"recurrent_{epoch} - static_{epoch}",
                f"recurrent_{epoch} - fixed_{epoch}",
                f"static_{epoch} - fixed_{epoch}",
                f"fixed_{epoch} - gelu_{epoch}",
            ]
        )
    names.append("(recurrent_3 - static_3) - (recurrent_1 - static_1)")
    return names


def validate_fit_spec(spec: dict) -> None:
    """Enforce the one sealed v3 fit recipe and its amendment evidence."""
    require(spec.get("name") == "e488_context_duration_twelve_fit_v3", "wrong E488 v3 fit spec")
    require(
        spec.get("artifacts")
        == [
            "research/campaign_2026-10-05/context_mixture_duration/runs/twelve_v3/fit_manifest.json"
        ],
        "E488 v3 fit artifact path differs",
    )
    require(
        spec.get("argv", [])[1:]
        == [
            "research/campaign_2026-10-05/context_mixture_duration/fit_duration.py",
            "--run-fits",
            "--device",
            "cuda",
            "--output",
            "research/campaign_2026-10-05/context_mixture_duration/runs/twelve_v3",
        ],
        "E488 v3 fit command differs",
    )
    require(
        spec.get("limits", {}).get("timeout_seconds") == 1788,
        "E488 v3 worker timeout does not charge the probe first",
    )
    require(
        spec.get("limits", {}).get("gpu_memory_bytes") == MAX_GPU_ALLOCATOR
        and spec.get("limits", {}).get("gpu_stop_bytes") == MAX_GPU_OBSERVED,
        "E488 v3 GPU limits differ from the registered caps",
    )
    require(
        spec.get("limits", {}).get("cpus") == 2
        and spec.get("limits", {}).get("ram_bytes") == 6 * 1024**3,
        "E488 v3 worker envelope differs",
    )
    code_hashes = {item["path"]: item["sha256"] for item in spec["code"]}
    inputs = {item["path"]: item["sha256"] for item in spec["inputs"]}
    require(
        set(code_hashes) == FIT_CODE_PATHS and set(inputs) == FIT_INPUT_PATHS,
        "E488 v3 fit code/input role inventory differs from the sealed fixture",
    )
    expected_code = {
        "research/campaign_2026-10-05/context_mixture_duration/fit_duration.py": EXPECTED_HASHES[
            "fit_duration.py"
        ],
        "research/campaign_2026-10-05/context_mixture_duration/tests_duration.py": EXPECTED_HASHES[
            "tests_duration.py"
        ],
        "research/campaign_2026-10-05/context_mixture_duration/gpu_probe_duration_v2.py": EXPECTED_HASHES[
            "gpu_probe_duration_v2.py"
        ],
        "research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py": EXPECTED_HASHES[
            "E470.py"
        ],
        "research/campaign_2026-10-05/activation_placement/fit_activation_placement.py": EXPECTED_HASHES[
            "E474.py"
        ],
        "research/campaign_2026-10-05/activation_head_mixture/fit_activation_head_mixture.py": EXPECTED_HASHES[
            "E477.py"
        ],
        "research/campaign_2026-10-05/activation_context_mixture/fit_context_mixture_v2.py": EXPECTED_HASHES[
            "E484_actor.py"
        ],
        "src/wnn/nn.py": EXPECTED_HASHES["wnn.nn.py"],
        "src/wnn/resources.py": EXPECTED_HASHES["wnn.resources.py"],
        "src/wnn/gmetric.py": EXPECTED_HASHES["wnn.gmetric.py"],
        "research/campaign_2026-10-05/context_mixture_duration/E488_DESIGN.md": EXPECTED_HASHES[
            "E488_DESIGN.md"
        ],
        "research/campaign_2026-10-05/context_mixture_duration/SPEC_CRITIC.md": EXPECTED_HASHES[
            "SPEC_CRITIC.md"
        ],
        "research/campaign_2026-10-05/context_mixture_duration/CONTROLLER_CLARIFICATIONS.md": EXPECTED_HASHES[
            "CONTROLLER_CLARIFICATIONS.md"
        ],
        "research/campaign_2026-10-05/context_mixture_duration/INDEPENDENT_REVIEW.md": EXPECTED_HASHES[
            "INDEPENDENT_REVIEW.md"
        ],
        "research/campaign_2026-10-05/context_mixture_duration/RESOURCE_ONLY_ADMISSION_AMENDMENT_V3.md": EXPECTED_HASHES[
            "RESOURCE_ONLY_ADMISSION_AMENDMENT_V3.md"
        ],
        "research/campaign_2026-10-05/context_mixture_duration/RESOURCE_ONLY_ADMISSION_AMENDMENT_V3_REVIEW.md": EXPECTED_HASHES[
            "RESOURCE_ONLY_ADMISSION_AMENDMENT_V3_REVIEW.md"
        ],
        "research/campaign_2026-10-05/context_mixture_duration/ROOT_FULL_FIT_ADMISSION_V3.md": EXPECTED_HASHES[
            "ROOT_FULL_FIT_ADMISSION_V3.md"
        ],
    }
    for path, expected in expected_code.items():
        require(code_hashes.get(path) == expected, f"E488 fit spec source differs: {path}")
    # evaluation.py is imported by this independent grader for metric/bootstrap
    # helpers; it is intentionally a grader-only pin, not an E488 trainer input.
    grader_only_path = "src/wnn/evaluation.py"
    require(
        EXPECTED_HASHES["wnn.evaluation.py"] == sha256_file(ROOT / grader_only_path)
        and grader_only_path not in code_hashes,
        "E488 grader-only evaluation dependency was confused with fit code",
    )
    required_inputs = {
        "research/campaign_2026-10-05/context_mixture_duration/e488_gpu_probe_v2.sealed.json": EXPECTED_HASHES[
            "e488_gpu_probe_v2.sealed.json"
        ],
        "research/campaign_2026-10-05/context_mixture_duration/gpu_probe_v2/probe.json": EXPECTED_HASHES[
            "actual_probe_v2"
        ],
        "datasets/valid.parquet": EXPECTED_HASHES["dataset"],
        "research/campaign_2026-10-05/target_mechanics/cohort.json": EXPECTED_HASHES["cohort"],
    }
    for path, expected in required_inputs.items():
        require(inputs.get(path) == expected, f"E488 v3 fit spec omitted or changed input: {path}")


def validate_probe_spec_roles(spec: dict) -> None:
    """Require the exact sealed probe code/input role inventory."""
    require(spec.get("name") == "e488_context_duration_gpu_probe_v2", "wrong E488 probe spec")
    code_paths = {item["path"] for item in spec.get("code", [])}
    input_paths = {item["path"] for item in spec.get("inputs", [])}
    require(
        code_paths == PROBE_CODE_PATHS and input_paths == PROBE_INPUT_PATHS,
        "E488 v2 probe code/input role inventory differs from the sealed fixture",
    )
    code_hashes = {item["path"]: item["sha256"] for item in spec["code"]}
    require(
        code_hashes.get(
            "research/campaign_2026-10-05/context_mixture_duration/gpu_probe_duration_v2.py"
        )
        == EXPECTED_HASHES["gpu_probe_duration_v2.py"],
        "E488 actual probe wrapper source differs",
    )


def validate_full_evidence(
    fit_spec_path: Path,
    fit_attempt_path: Path,
    fit_run: Path,
    probe_spec_path: Path,
    probe_attempt_path: Path,
    probe_path: Path,
) -> tuple[dict, dict, list[dict], list[dict], np.ndarray, np.ndarray, dict, dict]:
    """Check every source/attempt/probe/fit receipt before any test row-group read."""
    source_receipt = check_scientific_sources()
    require(
        fit_spec_path == HERE / "e488_twelve_fit_v3.sealed.json",
        "E488 scorer requires the exact sealed v3 fit spec",
    )
    require(
        probe_spec_path == HERE / "e488_gpu_probe_v2.sealed.json",
        "E488 scorer requires the exact sealed v2 probe spec",
    )
    fit_relative = (
        "research/campaign_2026-10-05/context_mixture_duration/runs/twelve_v3/fit_manifest.json"
    )
    probe_relative = "research/campaign_2026-10-05/context_mixture_duration/gpu_probe_v2/probe.json"
    fit_attempt, fit_spec, fit_artifact = validate_attempt(
        fit_attempt_path, fit_spec_path, fit_relative, "full fit"
    )
    probe_attempt, probe_spec, probe_artifact = validate_attempt(
        probe_attempt_path, probe_spec_path, probe_relative, "actual-shape probe"
    )
    validate_fit_spec(fit_spec)
    validate_probe_spec_roles(probe_spec)
    require(
        fit_attempt.get("id") == "e488-context-duration-twelve-fit-v3",
        "E488 full fit attempt ID differs",
    )
    require(
        probe_attempt.get("id") == "e488-context-duration-real-gpu-probe-v2",
        "E488 actual probe attempt ID differs",
    )
    require(
        fit_run.resolve() == (HERE / "runs/twelve_v3").resolve(),
        "E488 scorer requires the registered runs/twelve_v3 directory",
    )
    require(
        fit_artifact.resolve() == (fit_run / "fit_manifest.json").resolve(),
        "E488 fit artifact does not point to fit_manifest.json in the registered run",
    )
    require(
        probe_artifact.resolve() == probe_path.resolve(),
        "E488 probe path differs from sealed artifact",
    )
    require(
        sha256_file(probe_path) == EXPECTED_HASHES["actual_probe_v2"],
        "E488 admitted actual probe artifact hash differs",
    )
    fit_env, probe_env = fit_attempt.get("environment", {}), probe_attempt.get("environment", {})
    require(
        fit_env.get("CUDA_VISIBLE_DEVICES") == probe_env.get("CUDA_VISIBLE_DEVICES"),
        "E488 full fit and probe used different CUDA visibility",
    )
    require(
        fit_attempt.get("elapsed_seconds", 0) <= fit_spec["limits"]["timeout_seconds"],
        "E488 full fit exceeded its charged worker timeout",
    )
    total_charge = finite(probe_attempt.get("elapsed_seconds"), "E488 probe charge") + finite(
        fit_attempt.get("elapsed_seconds"), "E488 full-fit charge"
    )
    require(
        total_charge <= 1800.0, "E488 charged probe+fit wall time exceeds amended 1,800 seconds"
    )

    fit_nodes, test_nodes, _cohort = canonical_fit_test_nodes()
    fit_mean, fit_scale, _normalized_fit = derive_fit_normalizer(fit_nodes)
    fit_manifest = json.loads(fit_artifact.read_text())
    cells, manifest_receipt = validate_fit_manifest(
        fit_run, fit_manifest, fit_nodes, fit_mean, fit_scale
    )
    probe_report = json.loads(probe_artifact.read_text())
    probe_receipt = validate_probe_report(
        probe_report,
        probe_attempt,
        probe_attempt_path,
        probe_spec,
        fit_nodes,
        fit_mean,
        fit_scale,
    )
    require(
        probe_report.get("sources") == fit_manifest.get("actor_source_sha256"),
        "E488 probe and fit actor source maps differ",
    )
    require(
        probe_report.get("probe_v2_sha256") == source_receipt["gpu_probe_duration_v2.py"],
        "E488 probe v2 source differs from local bytes",
    )
    return (
        fit_manifest,
        probe_report,
        fit_nodes,
        test_nodes,
        fit_mean,
        fit_scale,
        cells,
        {
            "status": "all_fit_only_evidence_passed_test_rows_unread",
            "fit_attempt_id": fit_attempt["id"],
            "fit_attempt_sha256": sha256_file(fit_attempt_path),
            "fit_attempt_elapsed_seconds": fit_attempt["elapsed_seconds"],
            "fit_spec_sha256": sha256_file(fit_spec_path),
            "fit_manifest_sha256": sha256_file(fit_artifact),
            "fit_manifest_bytes": fit_artifact.stat().st_size,
            "probe_receipt": probe_receipt,
            "charged_probe_plus_fit_seconds": total_charge,
            "source_sha256": source_receipt,
            "normalizer_sha256": hashlib.sha256(
                fit_mean.tobytes() + fit_scale.tobytes()
            ).hexdigest(),
            "fit_only_components": len(fit_nodes),
            "test_components_known_from_cohort_manifest_only": len(test_nodes),
            "snapshot_count": len(cells),
            "seed_initial_state_sha256": manifest_receipt["initial_state_sha256"],
        },
    )


def _expect_normalizer(value: dict, mean: np.ndarray, scale: np.ndarray, label: str) -> None:
    """Compare JSON float32 mean/scale exactly to independent fit-role derivation."""
    observed_mean = np.asarray(value.get("fit_mean"), dtype=np.float32)
    observed_scale = np.asarray(value.get("fit_scale"), dtype=np.float32)
    require(
        observed_mean.shape == (FEATURES_DIM,) and observed_scale.shape == (FEATURES_DIM,),
        f"{label} normalizer dimensions differ",
    )
    require(
        np.isfinite(observed_mean).all() and np.isfinite(observed_scale).all(),
        f"{label} normalizer is nonfinite",
    )
    require(
        np.array_equal(observed_mean, mean) and np.array_equal(observed_scale, scale),
        f"{label} normalizer differs from independent fit-only recomputation",
    )


def validate_sample_telemetry(
    row: dict, seed: int, arm: str, fit_nodes: list[dict], step: int
) -> None:
    """Check six global sample locations against exact epoch/batch/chunk order."""
    epoch_index = step // UPDATES_PER_EPOCH
    epoch_local = step % UPDATES_PER_EPOCH
    batch_index, chunk_index = divmod(epoch_local, 79)
    chunk_start = chunk_index * CHUNK_ROWS
    epoch_positions = canonical_epoch_order(seed, epoch_index)
    expected_groups = [
        fit_nodes[int(index)]["row_group"]
        for index in epoch_positions[batch_index * BATCH : (batch_index + 1) * BATCH]
    ]
    require(
        row.get("global_update_zero_based") == step and row.get("update_index_zero_based") == step,
        f"E488 telemetry update index differs {seed}/{arm}/{step}",
    )
    require(
        row.get("epoch_index_zero_based") == epoch_index
        and row.get("batch_index_zero_based") == batch_index
        and row.get("chunk_start") == chunk_start,
        f"E488 telemetry position differs {seed}/{arm}/{step}",
    )
    require(
        row.get("fit_row_groups") == expected_groups,
        f"E488 telemetry fit order differs {seed}/{arm}/{step}",
    )
    for field in (
        "loss",
        "preclip_global_norm",
        "clip_factor",
        "state_abs_max",
        "gradient_rms_preclip",
    ):
        value = finite(row.get(field), f"E488 telemetry {field}")
        require(value >= 0, f"negative E488 telemetry {field}")
    require(0.0 < row["clip_factor"] <= 1.0, "E488 clipping factor outside (0,1]")
    expected_clip = min(1.0, CLIP_NORM / (row["preclip_global_norm"] + 1e-6))
    require(
        math.isclose(row["clip_factor"], expected_clip, rel_tol=0, abs_tol=1e-12),
        "E488 clipping factor differs from its preclip norm",
    )
    require(
        math.isclose(
            row["gradient_rms_preclip"], row["preclip_global_norm"], rel_tol=0, abs_tol=1e-12
        ),
        "E488 duplicate global gradient norm telemetry differs",
    )
    for field in ("preclip_gradient_rms", "parameter_update_rms"):
        families = row.get(field)
        require(
            isinstance(families, dict) and set(families) == {"inp", "gru", "head"},
            f"E488 {field} family keys differ",
        )
        for value in families.values():
            require(
                finite(value, f"E488 {field} family") >= 0, f"negative E488 {field} family value"
            )
    head_update = row.get("head_update_rms")
    require(
        isinstance(head_update, dict) and set(head_update) == {"head"},
        "E488 head update RMS family keys differ",
    )
    require(
        math.isclose(
            finite(head_update["head"], "E488 head update RMS"),
            row["parameter_update_rms"]["head"],
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "E488 head update RMS disagrees with family update telemetry",
    )
    gate = row.get("gate")
    coefficient = row.get("gate_coefficient_rms")
    if arm == "gelu":
        require(
            gate is None and coefficient is None, "E488 GELU arm unexpectedly has gate telemetry"
        )
    else:
        require(isinstance(gate, dict), f"E488 mixture gate telemetry missing {seed}/{arm}/{step}")
        for field in ("min", "max", "mean", "saturated_fraction_at_0_01"):
            value = finite(gate.get(field), f"E488 gate {field}")
            require(0 <= value <= 1, f"E488 gate summary outside [0,1]: {field}")
        require(
            gate["min"] <= gate["mean"] <= gate["max"],
            "E488 gate summary has an inconsistent min/mean/max order",
        )
        if arm == "fixed_mix":
            require(
                gate["min"] == 0.5 and gate["max"] == 0.5 and gate["mean"] == 0.5,
                "E488 fixed mixture gate is not constant 0.5",
            )
            require(coefficient is None, "E488 fixed gate should not have coefficients")
        else:
            require(
                finite(coefficient, "E488 learned gate coefficient RMS") >= 0,
                "E488 learned gate coefficient RMS is negative",
            )


def validate_trajectory(
    run: Path, trajectory: dict, fit_nodes: list[dict], initial_hash: str
) -> dict[tuple[int, str, int], tuple[dict, IndependentE488Model]]:
    """Validate one continuous 1,422-update trajectory and both saved snapshots."""
    seed, arm = trajectory.get("seed"), trajectory.get("arm")
    require(seed in SEEDS and arm in ARMS, "unknown E488 trajectory seed/arm")
    require(
        trajectory.get("status") == "complete" and trajectory.get("epochs_completed") == 3,
        f"incomplete E488 trajectory {seed}/{arm}",
    )
    require(
        trajectory.get("global_updates") == TOTAL_UPDATES
        and trajectory.get("total_updates") == TOTAL_UPDATES
        and trajectory.get("updates_per_epoch") == UPDATES_PER_EPOCH,
        f"E488 trajectory update total differs {seed}/{arm}",
    )
    require(
        trajectory.get("optimizer_state_steps") == [TOTAL_UPDATES],
        f"E488 AdamW step counter did not persist across epochs {seed}/{arm}",
    )
    optimizer = trajectory.get("optimizer", {})
    require(
        optimizer
        == {
            "name": "AdamW",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "betas": list(BETAS),
            "groups": 1,
        },
        f"E488 optimizer config differs {seed}/{arm}",
    )
    orders = trajectory.get("epoch_orders", [])
    require(len(orders) == 3, f"E488 trajectory must record three epoch orders {seed}/{arm}")
    for epoch_index, order in enumerate(orders):
        positions = canonical_epoch_order(seed, epoch_index)
        expected_groups = [fit_nodes[int(index)]["row_group"] for index in positions]
        require(
            order.get("epoch_index_zero_based") == epoch_index
            and order.get("row_groups") == expected_groups,
            f"E488 canonical shuffle order differs {seed}/{arm}/epoch{epoch_index}",
        )
    telemetry = trajectory.get("telemetry")
    require(
        isinstance(telemetry, list)
        and [row.get("global_update_zero_based") for row in telemetry] == list(TELEMETRY_STEPS),
        f"E488 trajectory telemetry cadence differs {seed}/{arm}",
    )
    for row, step in zip(telemetry, TELEMETRY_STEPS, strict=True):
        validate_sample_telemetry(row, seed, arm, fit_nodes, step)
    checkpoints = trajectory.get("checkpoints", [])
    require(
        len(checkpoints) == 2 and [item.get("epoch_number") for item in checkpoints] == [1, 3],
        f"E488 snapshot count/epochs differ {seed}/{arm}",
    )
    validated = {}
    for checkpoint in checkpoints:
        epoch_number = checkpoint["epoch_number"]
        expected_step = epoch_number * UPDATES_PER_EPOCH
        require(
            checkpoint.get("global_updates") == expected_step,
            f"E488 checkpoint update location differs {seed}/{arm}/epoch{epoch_number}",
        )
        model = validate_checkpoint(run, checkpoint, seed, arm)
        validated[(seed, arm, epoch_number)] = (checkpoint, model)
        sidecar_path = run / f"checkpoint_seed{seed}_{arm}_epoch{epoch_number}.json"
        require(
            sidecar_path.is_file() and json.loads(sidecar_path.read_text()) == checkpoint,
            f"E488 checkpoint sidecar differs {seed}/{arm}/epoch{epoch_number}",
        )
    trajectory_path = run / f"trajectory_seed{seed}_{arm}.json"
    require(
        trajectory_path.is_file() and json.loads(trajectory_path.read_text()) == trajectory,
        f"E488 trajectory sidecar differs {seed}/{arm}",
    )
    progress_path = run / f"progress_seed{seed}_{arm}.json"
    require(progress_path.is_file(), f"missing E488 final progress receipt {seed}/{arm}")
    progress = json.loads(progress_path.read_text())
    require(
        progress.get("status") == "epoch_3_complete"
        and progress.get("epochs_completed") == 3
        and progress.get("global_updates") == TOTAL_UPDATES
        and progress.get("optimizer_state_steps") == [TOTAL_UPDATES]
        and progress.get("telemetry") == telemetry
        and progress.get("epoch_orders") == orders
        and progress.get("checkpoints") == checkpoints,
        f"E488 progress receipt disagrees with continuous trajectory {seed}/{arm}",
    )
    return validated


def validate_fit_manifest(
    run: Path, manifest: dict, fit_nodes: list[dict], fit_mean: np.ndarray, fit_scale: np.ndarray
) -> tuple[dict[tuple[int, str, int], tuple[dict, IndependentE488Model]], dict]:
    """Validate every trajectory, normalizer, init digest and saved checkpoint."""
    require(
        manifest.get("kind") == "E488 continuous context-mixture duration study"
        and manifest.get("status") == "all_12_trajectories_complete_test_unread"
        and manifest.get("test_rows_loaded") is False,
        "E488 fit manifest is incomplete or crossed the test boundary",
    )
    require(
        manifest.get("seeds") == list(SEEDS) and manifest.get("arms") == list(ARMS),
        "E488 seed/arm inventory differs",
    )
    require(
        manifest.get("epochs") == 3
        and manifest.get("updates_per_epoch") == UPDATES_PER_EPOCH
        and manifest.get("total_updates_per_trajectory") == TOTAL_UPDATES,
        "E488 continuous update count differs",
    )
    require(
        manifest.get("registered_contrasts_per_seed")
        == [
            "recurrent_1 - static_1",
            "recurrent_1 - fixed_1",
            "static_1 - fixed_1",
            "fixed_1 - gelu_1",
            "recurrent_3 - static_3",
            "recurrent_3 - fixed_3",
            "static_3 - fixed_3",
            "fixed_3 - gelu_3",
            "(recurrent_3 - static_3) - (recurrent_1 - static_1)",
        ],
        "E488 preregistered contrast inventory differs",
    )
    expected_precision(manifest.get("precision", {}), "E488 fit manifest")
    require(
        manifest.get("parent_source_sha256") == expected_parent_sources(),
        "E488 frozen parent source receipt differs",
    )
    require(
        manifest.get("actor_source_sha256") == expected_actor_sources(),
        "E488 actor source receipt differs",
    )
    fit_cohort = manifest.get("fit_only_cohort", {})
    require(fit_cohort.get("cohort") == fit_nodes, "E488 fit-only cohort/order differs")
    normalization = fit_cohort.get("normalization", {})
    _expect_normalizer(normalization, fit_mean, fit_scale, "fit manifest")
    initial_hashes = manifest.get("seed_initial_state_sha256", {})
    require(
        set(initial_hashes) == {str(seed) for seed in SEEDS},
        "E488 seed initialization receipt incomplete",
    )
    initial_states = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = IndependentE488Model("gelu")
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        digest = state_tensor_sha256(state)
        require(digest == initial_hashes[str(seed)], f"E488 initial bytes differ for seed {seed}")
        initial_states[seed] = digest
    require(
        len(set(initial_states.values())) == len(SEEDS), "E488 fresh seed initializations collided"
    )
    trajectories = manifest.get("trajectories", [])
    expected_keys = {(seed, arm) for seed in SEEDS for arm in ARMS}
    found = set()
    cells: dict[tuple[int, str, int], tuple[dict, IndependentE488Model]] = {}
    for trajectory in trajectories:
        key = (trajectory.get("seed"), trajectory.get("arm"))
        require(
            key in expected_keys and key not in found, f"unknown/duplicate E488 trajectory {key}"
        )
        found.add(key)
        cells.update(validate_trajectory(run, trajectory, fit_nodes, initial_hashes[str(key[0])]))
    require(
        found == expected_keys and len(cells) == 24,
        "E488 expected 12 trajectories and 24 saved snapshots",
    )
    return cells, {"initial_state_sha256": initial_states, "snapshot_count": len(cells)}


def _selected_test_rows(test_nodes: list[dict], mean: np.ndarray, scale: np.ndarray):
    """Load test rows only after proof validation and return scored sequences."""
    parquet = pq.ParquetFile(DATA_PATH)
    features, targets, masks = [], [], []
    for node in test_nodes:
        table = parquet.read_row_group(
            node["row_group"],
            columns=[*FEATURES, "t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        raw_features = np.column_stack([table[name].to_numpy() for name in FEATURES])
        transformed = causal_features(raw_features)
        normalized = np.clip((transformed - mean) / scale, -8.0, 8.0).astype(np.float32)
        raw_targets = np.column_stack((table["t0"].to_numpy(), table["t1"].to_numpy()))
        selection = table["need_prediction"].to_numpy().astype(bool) & table[
            "is_scored"
        ].to_numpy().astype(bool)
        require(
            raw_targets.shape == (ROWS, 2) and selection.shape == (ROWS,),
            f"E488 test row shape differs: {node['row_group']}",
        )
        require(np.isfinite(raw_targets).all(), f"nonfinite raw test targets: {node['row_group']}")
        require(
            selection.any(), f"E488 test component has no scored need rows: {node['component']}"
        )
        require(
            np.isfinite(normalized).all(),
            f"nonfinite normalized test features: {node['row_group']}",
        )
        features.append(normalized)
        targets.append(raw_targets)
        masks.append(selection)
    return np.stack(features), np.stack(targets), np.stack(masks)


def predict_sequence(
    model: IndependentE488Model, sequence: np.ndarray, device: torch.device
) -> np.ndarray:
    """Predict one complete component while carrying GRU state across chunks."""
    model.to(device=device, dtype=torch.float32).eval()
    state = None
    outputs = []
    with torch.no_grad():
        for start in range(0, ROWS, CHUNK_ROWS):
            chunk = torch.from_numpy(sequence[start : start + CHUNK_ROWS][None]).to(
                device=device, dtype=torch.float32
            )
            prediction, state = model(chunk, state)
            outputs.append(prediction[0].detach().cpu().numpy())
    result = np.concatenate(outputs, axis=0)
    require(
        result.shape == (ROWS, 2) and np.isfinite(result).all(),
        "E488 full-sequence prediction is malformed/nonfinite",
    )
    return result


def score_target_moments(moments: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Pool centered weighted-Pearson moments independently for t0/t1."""
    require(
        moments.shape == (16, 2, 6) and counts.shape == (16,),
        "E488 target-wise moments/count dimensions differ",
    )
    result = np.zeros(2, dtype=np.float64)
    for target in range(2):
        side = moments[:, target, :]
        weights = counts * side[:, 0]
        total = float(weights.sum())
        if total <= 0:
            continue
        active = np.flatnonzero(weights > 0)
        anchor_target, anchor_prediction = side[active[0], 1:3]
        dy = side[:, 1] - anchor_target
        dp = side[:, 2] - anchor_prediction
        mean_y = np.sum(weights * dy) / total
        mean_p = np.sum(weights * dp) / total
        variance_y = np.sum(counts * side[:, 3] + weights * (dy - mean_y) ** 2)
        variance_p = np.sum(counts * side[:, 4] + weights * (dp - mean_p) ** 2)
        covariance = np.sum(counts * side[:, 5] + weights * (dy - mean_y) * (dp - mean_p))
        if math.sqrt(variance_y / total) > 1e-8 and math.sqrt(variance_p / total) > 1e-8:
            result[target] = np.clip(covariance / math.sqrt(variance_y * variance_p), -1.0, 1.0)
    return result


def bootstrap_summary(
    point: float, draws: np.ndarray, seed: int, target: str | None = None
) -> dict:
    """Summarize a within-seed paired component-bootstrap contrast."""
    require(
        draws.shape == (BOOTSTRAP_COUNT,) and np.isfinite(draws).all() and math.isfinite(point),
        "invalid E488 bootstrap contrast",
    )
    return {
        "point_diff": float(point),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
        "p_le_0": float(np.mean(draws <= 0)),
        "seed": seed,
        "target": target,
        "components": 16,
        "bootstrap_count": BOOTSTRAP_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "metric_id": METRIC_ID,
    }


def score_validated_run(
    fit_run: Path,
    cells: dict[tuple[int, str, int], tuple[dict, IndependentE488Model]],
    fit_nodes: list[dict],
    test_nodes: list[dict],
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    export_path: Path,
) -> dict:
    """Load labels only after fit-only evidence passes, then score all 24 snapshots."""
    test_x, raw_targets, selected = _selected_test_rows(test_nodes, mean, scale)
    component_ids = np.asarray([node["component"] for node in test_nodes], dtype=np.int64)
    row_groups = np.asarray([node["row_group"] for node in test_nodes], dtype=np.int64)
    order = np.argsort(component_ids, kind="stable")
    sorted_component_ids = component_ids[order]
    require(len(np.unique(sorted_component_ids)) == 16, "E488 test components are not unique")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw_indices = rng.integers(0, len(order), size=(BOOTSTRAP_COUNT, len(order)))
    multiplicities = np.stack(
        [np.bincount(sample, minlength=len(order)) for sample in draw_indices]
    ).astype(np.int64)
    counts_by_component = selected.sum(axis=1).astype(np.int64)
    require((counts_by_component > 0).all(), "E488 component selection count is empty")

    arm_scores: dict[str, dict] = {}
    moments: dict[str, np.ndarray] = {}
    target_moments: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    for seed in SEEDS:
        for arm in ARMS:
            for epoch in EPOCHS:
                key = (seed, arm, epoch)
                checkpoint, model = cells[key]
                model.to(device=device, dtype=torch.float32)
                prediction = np.empty((16, ROWS, 2), dtype=np.float32)
                context = (
                    torch.autocast(device_type="cuda", enabled=False)
                    if device.type == "cuda"
                    else torch.autocast(device_type="cpu", enabled=False)
                )
                with context:
                    for index, sequence in enumerate(test_x):
                        prediction[index] = predict_sequence(model, sequence, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                require(np.isfinite(prediction).all(), f"nonfinite full E488 prediction {key}")
                metric = global_wp(raw_targets, prediction, selected)
                require(
                    metric.get("metric_id") == METRIC_ID
                    and metric.get("selected_rows") == int(counts_by_component.sum()),
                    f"official global WP receipt differs {key}",
                )
                pair_moments, _ = paired_component_moments(
                    raw_targets, prediction, prediction, selected
                )
                label = f"seed{seed}__{arm}__epoch{epoch}"
                arm_scores[label] = {
                    "seed": seed,
                    "arm": arm,
                    "epoch": epoch,
                    "parameters": GATED_PARAMETERS
                    if arm in ("static_mix", "recurrent_mix")
                    else BASE_PARAMETERS,
                    "updates": epoch * UPDATES_PER_EPOCH,
                    "wp": float(metric["wp"]),
                    "t0": float(metric["t0"]),
                    "t1": float(metric["t1"]),
                    "metric_id": METRIC_ID,
                    "selected_rows": int(metric["selected_rows"]),
                    "component_row_counts": counts_by_component.tolist(),
                }
                moments[label] = pair_moments[order]
                target_moments[label] = pair_moments[order]
                predictions[label] = prediction

    arm_bootstrap: dict[str, np.ndarray] = {}
    target_bootstrap: dict[str, np.ndarray] = {}
    for key, component_stats in moments.items():
        wp_draws, target_draws = [], []
        for counts in multiplicities:
            wp_draws.append(_score_component_moments(component_stats, counts))
            target_draws.append(score_target_moments(target_moments[key], counts))
        arm_bootstrap[key] = np.asarray(wp_draws, dtype=np.float64)
        target_bootstrap[key] = np.asarray(target_draws, dtype=np.float64)

    global_contrasts: dict[str, dict] = {}
    target_contrasts: dict[str, dict] = {}
    for seed in SEEDS:
        for epoch in EPOCHS:
            for short_name, candidate, reference in CONTRASTS:
                candidate_key = f"seed{seed}__{candidate}__epoch{epoch}"
                reference_key = f"seed{seed}__{reference}__epoch{epoch}"
                contrast_key = f"seed{seed}__epoch{epoch}__{short_name}"
                point = arm_scores[candidate_key]["wp"] - arm_scores[reference_key]["wp"]
                draws = arm_bootstrap[candidate_key] - arm_bootstrap[reference_key]
                global_contrasts[contrast_key] = {
                    "seed": seed,
                    "epoch": epoch,
                    "contrast": short_name,
                    "candidate": candidate,
                    "reference": reference,
                    **bootstrap_summary(point, draws, seed),
                }
                per_target = {}
                for target_index in range(2):
                    target_name = f"t{target_index}"
                    target_point = (
                        arm_scores[candidate_key][target_name]
                        - arm_scores[reference_key][target_name]
                    )
                    target_draws = (
                        target_bootstrap[candidate_key][:, target_index]
                        - target_bootstrap[reference_key][:, target_index]
                    )
                    per_target[target_name] = bootstrap_summary(
                        target_point, target_draws, seed, target_name
                    )
                target_contrasts[contrast_key] = {
                    "seed": seed,
                    "epoch": epoch,
                    "contrast": short_name,
                    "targets": per_target,
                }
        high_rec = f"seed{seed}__recurrent_mix__epoch3"
        high_static = f"seed{seed}__static_mix__epoch3"
        low_rec = f"seed{seed}__recurrent_mix__epoch1"
        low_static = f"seed{seed}__static_mix__epoch1"
        interaction_name = f"seed{seed}__duration_interaction"
        interaction_point = (
            arm_scores[high_rec]["wp"]
            - arm_scores[high_static]["wp"]
            - arm_scores[low_rec]["wp"]
            + arm_scores[low_static]["wp"]
        )
        interaction_draws = (
            arm_bootstrap[high_rec]
            - arm_bootstrap[high_static]
            - arm_bootstrap[low_rec]
            + arm_bootstrap[low_static]
        )
        global_contrasts[interaction_name] = {
            "seed": seed,
            "contrast": "(recurrent_3 - static_3) - (recurrent_1 - static_1)",
            "candidate": "recurrent_minus_static_epoch3",
            "reference": "recurrent_minus_static_epoch1",
            **bootstrap_summary(interaction_point, interaction_draws, seed),
        }
        per_target = {}
        for target_index in range(2):
            target_name = f"t{target_index}"
            target_point = (
                arm_scores[high_rec][target_name]
                - arm_scores[high_static][target_name]
                - arm_scores[low_rec][target_name]
                + arm_scores[low_static][target_name]
            )
            target_draws = (
                target_bootstrap[high_rec][:, target_index]
                - target_bootstrap[high_static][:, target_index]
                - target_bootstrap[low_rec][:, target_index]
                + target_bootstrap[low_static][:, target_index]
            )
            per_target[target_name] = bootstrap_summary(
                target_point, target_draws, seed, target_name
            )
        target_contrasts[interaction_name] = {
            "seed": seed,
            "contrast": "(recurrent_3 - static_3) - (recurrent_1 - static_1)",
            "targets": per_target,
        }

    require(
        len(arm_scores) == 24 and len(global_contrasts) == 27 and len(target_contrasts) == 27,
        "E488 score/contrast inventory differs from registered 24/27/27 counts",
    )
    archive = {
        "component_ids": component_ids,
        "sorted_component_ids": sorted_component_ids,
        "row_groups": row_groups,
        "raw_targets": raw_targets,
        "selected_mask": selected,
        "bootstrap_component_indices": draw_indices,
        "bootstrap_multiplicities": multiplicities,
    }
    archive.update({f"prediction__{key}": value for key, value in predictions.items()})
    np.savez_compressed(export_path, **archive)
    return {
        "kind": "independent E488 duration evaluation",
        "metric_id": METRIC_ID,
        "scope": "three fit seeds; same 16 exposed components; duration mechanism diagnostic only",
        "trajectory_count": 12,
        "snapshot_count": 24,
        "registered_global_contrast_count": 27,
        "registered_target_contrast_count": 27,
        "no_seed_pooling": True,
        "selected_rows_total": int(counts_by_component.sum()),
        "component_ids": component_ids.tolist(),
        "sorted_component_ids": sorted_component_ids.tolist(),
        "row_groups": row_groups.tolist(),
        "component_row_counts": counts_by_component.tolist(),
        "bootstrap": {
            "method": "official centered weighted-Pearson moments; paired component resampling",
            "draws": BOOTSTRAP_COUNT,
            "seed": BOOTSTRAP_SEED,
            "multiplicities_shared_across_all_24_snapshots_and_contrasts": True,
            "unit": "numeric-sorted fixed component ID",
        },
        "arm_scores": arm_scores,
        "registered_global_contrasts": global_contrasts,
        "registered_target_contrasts": target_contrasts,
        "raw_selected_export": export_path.name,
        "test_labels_used_for_fitting_or_selection": False,
    }


@contextlib.contextmanager
def ieee_float32(device: torch.device):
    """Temporarily force non-autocast IEEE FP32 for GPU inference."""
    if device.type != "cuda":
        with torch.autocast(device_type="cpu", enabled=False):
            yield
        return
    backend_settings = (
        (torch.backends, "fp32_precision"),
        (torch.backends.cuda.matmul, "fp32_precision"),
        (torch.backends.cudnn, "fp32_precision"),
        (torch.backends.cudnn.rnn, "fp32_precision"),
    )
    if not all(hasattr(owner, key) for owner, key in backend_settings):
        raise RuntimeError("scoring requires PyTorch full IEEE FP32 backend controls")
    previous = [getattr(owner, key) for owner, key in backend_settings]
    try:
        for owner, key in backend_settings:
            setattr(owner, key, "ieee")
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        for (owner, key), value in zip(backend_settings, previous, strict=True):
            setattr(owner, key, value)


def main() -> None:
    """Verify E488 receipts; test labels require the explicit score mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("verify-only", "score"),
        default="verify-only",
        help="verify-only never loads test row groups; score requires explicit selection",
    )
    parser.add_argument("--fit-run", type=Path, default=HERE / "runs/twelve_v3")
    parser.add_argument("--fit-spec", type=Path, default=HERE / "e488_twelve_fit_v3.sealed.json")
    parser.add_argument("--fit-attempt", type=Path, required=True)
    parser.add_argument("--probe-spec", type=Path, default=HERE / "e488_gpu_probe_v2.sealed.json")
    parser.add_argument("--probe-attempt", type=Path, required=True)
    parser.add_argument("--probe-json", type=Path, default=HERE / "gpu_probe_v2/probe.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite E488 grader output: {args.output}")
    torch.set_num_threads(1)
    evidence = validate_full_evidence(
        args.fit_spec,
        args.fit_attempt,
        args.fit_run,
        args.probe_spec,
        args.probe_attempt,
        args.probe_json,
    )
    fit_manifest, _probe, fit_nodes, test_nodes, fit_mean, fit_scale, cells, proof = evidence
    args.output.mkdir(parents=True)
    (args.output / "proof.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    if args.mode == "verify-only":
        print(json.dumps(proof, indent=2, allow_nan=False), flush=True)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA scoring requested but CUDA is unavailable")
    export = args.output / "selected_predictions.npz"
    with ieee_float32(torch.device(args.device)):
        score = score_validated_run(
            args.fit_run,
            cells,
            fit_nodes,
            test_nodes,
            fit_mean,
            fit_scale,
            torch.device(args.device),
            export,
        )
    score["selected_export_sha256"] = sha256_file(export)
    score["selected_export_bytes"] = export.stat().st_size
    score["proof_sha256"] = sha256_file(args.output / "proof.json")
    (args.output / "scores.json").write_text(json.dumps(score, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": "scored",
                "arm_snapshots": 24,
                "global_contrasts": 27,
                "target_contrasts": 27,
                "test_rows_loaded": True,
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

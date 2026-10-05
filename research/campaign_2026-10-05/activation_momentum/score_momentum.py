"""Independent E483 scorer for recurrent Muon momentum by candidate shape."""

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
HERE = ROOT / "research/campaign_2026-10-05/activation_momentum"
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
E481_DIR = ROOT / "research/campaign_2026-10-05/recurrent_activation_curvature"
DATA_PATH = ROOT / "datasets/valid.parquet"
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
DESIGN_PATH = HERE / "E483_DESIGN.md"
CRITIC_PATH = HERE / "SPEC_CRITIC.md"
ACTOR_PATH = HERE / "fit_momentum.py"
ACTOR_TESTS_PATH = HERE / "tests.py"
PROBE_PATH = HERE / "gpu_probe.py"
PREFLIGHT_PATH = HERE / "preflight_cpu_v3/fit_manifest.json"
DEFAULT_RUN = HERE / "runs/eighteen_v1"
DEFAULT_SPEC = HERE / "e483_eighteen_fit_v1.sealed.json"
DEFAULT_ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e483-isolated-muon-momentum-eighteen-fit-v1.json"
)
DEFAULT_OUTPUT = HERE / "eda/eighteen_v1"

sys.path.insert(0, str(ROOT / "src"))
from wnn.campaign import RUNTIME_FILES, runtime_identity  # noqa: E402
from wnn.campaign_spec import digest as spec_digest  # noqa: E402
from wnn.campaign_spec import validate_spec  # noqa: E402
from wnn.gmetric import (  # noqa: E402
    METRIC_CLIP,
    METRIC_ID,
    _centered_moments,
    _merge_moments,
    _wp_from_moments,
    global_wp,
)

SEEDS = (271_831, 271_832, 271_833)
SHAPES = ("baseline", "sine")
MOMENTA = (0.8, 0.9, 0.95)
ARMS = tuple((seed, shape, beta) for seed in SEEDS for shape in SHAPES for beta in MOMENTA)
FEATURE_COLUMNS = (
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
FEATURES, HIDDEN, ROWS, CHUNK, UPDATES = 42, 64, 20_000, 256, 474
CHUNKS_PER_SEQUENCE = math.ceil(ROWS / CHUNK)
PARAMETERS = 61_058
BETAS = (0.9, 0.999)
LR, WEIGHT_DECAY = 0.001, 0.01
TELEMETRY_STEPS = (0, 46, 236, 473)
MUON_NAMES = {"gru.weight_ih_l0", "gru.weight_hh_l0", "gru.weight_ih_l1", "gru.weight_hh_l1"}
BOOTSTRAP_COUNT, BOOTSTRAP_SEED = 2_000, 271_828
EXPECTED_HASHES = {
    "fit_momentum.py": "54f4fbc66f69e165a5b21ff740faa556ef368d326ebbeb24736d68a36440bbe8",
    "tests.py": "d63b46acc9abae85e6ebac2bdda0c89de01832f197c89b50f926d3fe675e9a42",
    "gpu_probe.py": "af3ac0fba649d0027841f07ccd551798222f57f3da569e84b7064393a4f775c4",
    "preflight_cpu_v3/fit_manifest.json": "d9fb30e5c809fceded545ab39ff9779def9b16ee467586723381fa868cc4cb13",
    "E483_DESIGN.md": "fa5aa9ec2a7f897f7ed31f7a99d9ee0b8a681cce73146ebd367d729f6a1e8288",
    "SPEC_CRITIC.md": "9cd04e02fc0f4d5a0dd324d531f8d38d6f48261e3d1f87fd4aeddfc754da0c28",
}
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
PARENT_HASHES = {
    "e470": "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9",
    "e474": "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068",
    "e481": "3d97d2d07b116b3cf04b365c33e3d3caed2c96c35108bd28f3a6ff63b54d65cc",
    "wnn.nn": "388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3",
    "wnn.optim": "620569a752fb611adf69976f1dbed1b7882e641672c58175005f5d3f819d2991",
    "wnn.resources": "06a624ad963dfc2b0c75ce717278de5f1bb85f48518a99e519b697a371d4cf09",
}
OFFICIAL_METRIC_SHA256 = "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa"
PARENT_PATHS = {
    "e470": E470_DIR / "fit_activation_optimizer.py",
    "e474": E474_DIR / "fit_activation_placement.py",
    "e481": E481_DIR / "fit_recurrent_curvature.py",
    "wnn.nn": ROOT / "src/wnn/nn.py",
    "wnn.optim": ROOT / "src/wnn/optim.py",
    "wnn.resources": ROOT / "src/wnn/resources.py",
}
SOURCE_PATHS = {
    "fit_momentum.py": ACTOR_PATH,
    "tests.py": ACTOR_TESTS_PATH,
    "gpu_probe.py": PROBE_PATH,
}


def sha256_file(path: Path) -> str:
    """Hash one file using bounded-size reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sources() -> dict[str, str]:
    """Bind the registered E483, E470/E474/E481, metric, and runtime sources."""
    expected_paths = {
        "fit_momentum.py": ACTOR_PATH,
        "tests.py": ACTOR_TESTS_PATH,
        "gpu_probe.py": PROBE_PATH,
        "preflight_cpu_v3/fit_manifest.json": PREFLIGHT_PATH,
        "E483_DESIGN.md": DESIGN_PATH,
        "SPEC_CRITIC.md": CRITIC_PATH,
    }
    actual = {name: sha256_file(path) for name, path in expected_paths.items()}
    for name, expected in EXPECTED_HASHES.items():
        if actual[name] != expected:
            raise ValueError(f"E483 source/preflight digest changed: {name}")
    for name, path in PARENT_PATHS.items():
        actual[name] = sha256_file(path)
        if actual[name] != PARENT_HASHES[name]:
            raise ValueError(f"frozen E483 parent source changed: {name}")
    actor_source = ACTOR_PATH.read_text()
    placement_source = PARENT_PATHS["e474"].read_text()
    if "with e474.full_ieee_float32():" not in actor_source:
        raise ValueError(
            "E483 fit source does not run the full matrix inside the full-IEEE context"
        )
    if (
        'setattr(owner, name, "ieee")' not in placement_source
        or 'torch.autocast(device_type="cuda", enabled=False)' not in placement_source
    ):
        raise ValueError(
            "frozen E474 helper source does not implement the registered full-IEEE policy"
        )
    return actual


@torch.jit.script
def candidate_transform(u: Tensor, sine: bool) -> Tensor:
    if sine:
        return u + 0.1 * torch.sin(math.pi * u)
    return u


@torch.jit.script
def reset_after_scan(
    x: Tensor,
    h0: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    sine: bool,
) -> tuple[Tensor, Tensor]:
    """Independent r/z/n reset-after GRU scan with explicit carried state."""
    input_affine = F.linear(x, weight_ih, bias_ih)
    outputs = torch.jit.annotate(list[Tensor], [])
    hidden = h0
    for step in range(x.size(1)):
        i_r, i_z, i_n = input_affine[:, step, :].chunk(3, dim=1)
        h_r, h_z, h_n = F.linear(hidden, weight_hh, bias_hh).chunk(3, dim=1)
        reset = torch.sigmoid(i_r + h_r)
        update = torch.sigmoid(i_z + h_z)
        u = torch.tanh(i_n + reset * h_n)
        candidate = candidate_transform(u, sine)
        hidden = (1.0 - update) * candidate + update * hidden
        outputs.append(hidden)
    return torch.stack(outputs, dim=1), hidden


class IndependentMomentumModel(nn.Module):
    """Standalone checkpoint-compatible H64 x 2 reset-after GRU model."""

    def __init__(self, sine: bool) -> None:
        super().__init__()
        self.sine = sine
        self.register_buffer("mean", torch.zeros(FEATURES))
        self.register_buffer("std", torch.ones(FEATURES))
        self.inp = nn.Linear(FEATURES, HIDDEN)
        self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 2))
        if sum(parameter.numel() for parameter in self.parameters()) != PARAMETERS:
            raise ValueError("independent E483 model does not contain 61,058 parameters")

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != FEATURES:
            raise ValueError("expected [batch,time,42] features")
        encoded = F.gelu(self.inp((x - self.mean) / self.std))
        batch = encoded.shape[0]
        hidden = encoded.new_zeros((2, batch, HIDDEN)) if state is None else state
        if hidden.shape != (2, batch, HIDDEN):
            raise ValueError("expected state [2,batch,64]")
        layer_input = encoded
        finals: list[Tensor] = []
        for layer in range(2):
            layer_input, final = reset_after_scan(
                layer_input,
                hidden[layer],
                getattr(self.gru, f"weight_ih_l{layer}"),
                getattr(self.gru, f"weight_hh_l{layer}"),
                getattr(self.gru, f"bias_ih_l{layer}"),
                getattr(self.gru, f"bias_hh_l{layer}"),
                self.sine,
            )
            finals.append(final)
        prediction = self.head(torch.cat((layer_input, encoded), dim=-1))
        return prediction, torch.stack(finals)


def state_hash(state: dict[str, Tensor]) -> str:
    """Reproduce the actor's documented initialization hash encoding."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def expected_initial_state_hashes() -> dict[str, str]:
    """Reconstruct registered per-seed RNNModel initializations independently."""
    from wnn.nn import RNNModel

    result: dict[str, str] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = RNNModel(FEATURES, hidden=HIDDEN, layers=2, dropout=0.0, n_out=2)
        result[str(seed)] = state_hash(model.state_dict())
    return result


def canonical_cohort() -> tuple[list[dict], list[dict]]:
    """Validate the frozen 24-fit/16-test split without opening parquet rows."""
    cohort = json.loads(COHORT_PATH.read_text())
    nodes = cohort.get("cohort", [])
    fit_nodes = [node for node in nodes if node.get("role") == "fit"]
    test_nodes = [node for node in nodes if node.get("role") == "test"]
    if len(nodes) != 40 or len(fit_nodes) != 24 or len(test_nodes) != 16:
        raise ValueError("E483 requires exactly 24 fit and 16 exposed test nodes")
    if (
        len({node["component"] for node in fit_nodes}) != 24
        or len({node["component"] for node in test_nodes}) != 16
    ):
        raise ValueError("E483 component IDs are duplicated")
    if {node["component"] for node in fit_nodes} & {node["component"] for node in test_nodes}:
        raise ValueError("E483 fit/test component roles overlap")
    if len({node["row_group"] for node in nodes}) != 40:
        raise ValueError("E483 row groups are duplicated across roles")
    return fit_nodes, sorted(test_nodes, key=lambda item: int(item["component"]))


def validate_spec_identity(spec_raw: dict[str, Any], run: Path) -> dict[str, Any]:
    """Require a sealed E483 GPU fit spec bound to frozen sources and fit data."""
    spec = validate_spec(spec_raw)
    if spec["kind"] != "gpu" or spec["name"] != "e483_isolated_recurrent_momentum_eighteen_fits_v1":
        raise ValueError("unexpected E483 fit spec identity/kind")
    limits = spec["limits"]
    if (
        limits["cpus"] > 2
        or limits["ram_bytes"] > 6 * 1024**3
        or limits["gpu_memory_bytes"] > 2 * 1024**3
        or limits["gpu_stop_bytes"] > 3_000_000_000
        or limits["timeout_seconds"] > 7_200
    ):
        raise ValueError("E483 fit spec exceeds registered worker/resource bounds")
    run_rel = run.resolve().relative_to(ROOT.resolve()).as_posix()
    if spec["artifacts"] != [f"{run_rel}/fit_manifest.json"]:
        raise ValueError("E483 fit spec must declare exactly the fit manifest")
    argv = spec["argv"]
    if (
        len(argv) < 7
        or argv[1] != "research/campaign_2026-10-05/activation_momentum/fit_momentum.py"
    ):
        raise ValueError("E483 fit spec does not call the frozen actor")
    if not all(token in argv for token in ("--run-fits", "--device", "cuda", "--output", run_rel)):
        raise ValueError(
            "E483 fit spec must run the complete CUDA matrix to the fixed run directory"
        )
    code = {item["path"]: item["sha256"] for item in spec["code"]}
    required_sources = {
        "research/campaign_2026-10-05/activation_momentum/fit_momentum.py": EXPECTED_HASHES[
            "fit_momentum.py"
        ],
        "research/campaign_2026-10-05/activation_momentum/tests.py": EXPECTED_HASHES["tests.py"],
        "research/campaign_2026-10-05/activation_momentum/gpu_probe.py": EXPECTED_HASHES[
            "gpu_probe.py"
        ],
        "src/wnn/gmetric.py": OFFICIAL_METRIC_SHA256,
    }
    for _name, (path, expected) in PARENT_PATHS.items():
        required_sources[path.relative_to(ROOT).as_posix()] = expected
    for path, expected in required_sources.items():
        if code.get(path) != expected:
            raise ValueError(f"E483 sealed source hash mismatch: {path}")
    inputs = {item["path"]: item["sha256"] for item in spec["inputs"]}
    required_inputs = {
        "datasets/valid.parquet": DATASET_SHA256,
        "research/campaign_2026-10-05/target_mechanics/cohort.json": COHORT_SHA256,
        "research/campaign_2026-10-05/activation_momentum/E483_DESIGN.md": EXPECTED_HASHES[
            "E483_DESIGN.md"
        ],
        "research/campaign_2026-10-05/activation_momentum/SPEC_CRITIC.md": EXPECTED_HASHES[
            "SPEC_CRITIC.md"
        ],
        "research/campaign_2026-10-05/activation_momentum/preflight_cpu_v3/fit_manifest.json": EXPECTED_HASHES[
            "preflight_cpu_v3/fit_manifest.json"
        ],
    }
    for path, expected in required_inputs.items():
        if inputs.get(path) != expected:
            raise ValueError(f"E483 sealed input hash mismatch: {path}")
    for item in [*spec["code"], *spec["inputs"]]:
        path = ROOT / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["size"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"E483 sealed source/input changed on disk: {item['path']}")
    check_sources()
    return spec


def validate_attempt(spec_raw: dict[str, Any], attempt_path: Path, run: Path) -> dict[str, Any]:
    """Fail closed on runner status, runtime, artifacts, and OOM deltas."""
    spec = validate_spec_identity(spec_raw, run)
    record = json.loads(attempt_path.read_text())
    if (
        record.get("id") != "e483-isolated-muon-momentum-eighteen-fit-v1"
        or record.get("kind") != "gpu"
    ):
        raise ValueError("unexpected E483 runner attempt identity")
    if record.get("spec") != spec or record.get("spec_sha256") != spec_digest(spec):
        raise ValueError("E483 attempt does not bind the exact sealed fit spec")
    argv = record.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not Path(argv[0]).is_absolute()
        or argv[1:] != spec["argv"][1:]
    ):
        raise ValueError("E483 runner command differs from sealed argv")
    current_runtime = runtime_identity(ROOT)
    if record.get("runtime_sha256") != current_runtime or set(current_runtime) != set(
        RUNTIME_FILES
    ):
        raise ValueError("E483 runtime hashes differ from current runner")
    if (
        record.get("status") != "finished"
        or record.get("exit_code") != 0
        or record.get("signals") != []
        or record.get("teardown_verified") is not True
        or any(
            record.get(key) is not None for key in ("failure", "failure_scope", "failure_category")
        )
    ):
        raise ValueError("E483 fit attempt did not finish cleanly")
    unit = record.get("unit_properties", {})
    if unit.get("Result") != "success" or unit.get("ExecMainStatus") not in (0, "0"):
        raise ValueError("E483 systemd unit reports failure")
    for before_key, after_key in (
        ("initial_memory_events", "final_memory_events"),
        ("initial_parent_memory_events", "final_parent_memory_events"),
    ):
        before, after = record.get(before_key), record.get(after_key)
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError(f"E483 attempt lacks OOM snapshots: {before_key}")
        for counter in ("oom", "oom_kill", "oom_group_kill"):
            a, b = before.get(counter), after.get(counter)
            if type(a) is not int or type(b) is not int or a < 0 or b < 0 or a != b:
                raise ValueError(f"E483 OOM counter changed or malformed: {before_key}/{counter}")
    artifacts = []
    for relative in spec["artifacts"]:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"E483 fit artifact missing: {relative}")
        artifacts.append(
            {"path": relative, "sha256": sha256_file(path), "size": path.stat().st_size}
        )
    if record.get("artifacts") != artifacts:
        raise ValueError("E483 attempt artifact inventory differs")
    if (ROOT / spec["artifacts"][0]).parent.resolve() != run.resolve():
        raise ValueError("E483 run path differs from fit spec")
    return {
        "attempt_id": record["id"],
        "attempt_sha256": sha256_file(attempt_path),
        "spec_sha256": record["spec_sha256"],
        "runtime_sha256": current_runtime,
        "artifacts": artifacts,
        "worker_oom_counters_unchanged": True,
        "parent_oom_counters_unchanged": True,
        "teardown_verified": True,
    }


def verify_preflight() -> dict[str, Any]:
    receipt = json.loads(PREFLIGHT_PATH.read_text())
    if (
        receipt.get("status") != "cpu_fit_only_one_update_per_arm_passed_test_unread"
        or receipt.get("test_rows_loaded") is not False
        or receipt.get("planned_fit_count") != 18
        or receipt.get("source_sha256")
        != {
            "fit_momentum.py": EXPECTED_HASHES["fit_momentum.py"],
            "tests.py": EXPECTED_HASHES["tests.py"],
            "gpu_probe.py": EXPECTED_HASHES["gpu_probe.py"],
        }
    ):
        raise ValueError("E483 CPU fit-only preflight receipt differs")
    return receipt


def causal_features(raw: np.ndarray) -> np.ndarray:
    """Independent reproduction of the frozen causal raw feature transform."""
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("expected 20 raw input channels")
    difference = np.diff(values, axis=0, prepend=values[:1])
    event = np.any(difference[:, :2] != 0, axis=1)
    index = np.arange(len(values))
    last = np.maximum.accumulate(np.where(event, index, 0))
    transformed = np.column_stack((values, difference, np.log1p(index - last), event))
    return np.nan_to_num(transformed, nan=0, posinf=0, neginf=0).astype(np.float32)


def read_features(nodes: list[dict]) -> np.ndarray:
    parquet = pq.ParquetFile(DATA_PATH)
    rows = []
    for node in nodes:
        table = parquet.read_row_group(
            node["row_group"], columns=list(FEATURE_COLUMNS), use_threads=False
        )
        raw = np.column_stack([table[column].to_numpy() for column in FEATURE_COLUMNS])
        if raw.shape != (ROWS, len(FEATURE_COLUMNS)):
            raise ValueError(f"unexpected rows/features in group {node['row_group']}")
        rows.append(causal_features(raw))
    return np.stack(rows)


def derive_fit_normalizer(fit_nodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if len(fit_nodes) != 24 or any(node.get("role") != "fit" for node in fit_nodes):
        raise ValueError("normalization requires exactly the 24 canonical fit-role groups")
    fit_x = read_features(fit_nodes)
    mean = fit_x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(fit_x.std(axis=(0, 1), dtype=np.float64), 0.001).astype(np.float32)
    return mean, scale


def validate_optimizer_groups(
    groups: list[dict[str, Any]], expected_names: set[str], beta: float
) -> None:
    """Validate exact recurrent-Muon selection and unchanged AdamW groups."""
    if not isinstance(groups, list) or len(groups) != 2:
        raise ValueError("expected exactly one Muon and one AdamW parameter group")
    all_names: list[str] = []
    seen_muon = 0
    for group in groups:
        names = group.get("names")
        if not isinstance(names, list) or names != sorted(set(names)):
            raise ValueError("optimizer parameter names are malformed")
        all_names.extend(names)
        if group.get("lr") != LR or group.get("weight_decay") != WEIGHT_DECAY:
            raise ValueError("optimizer LR/decay changed")
        if group.get("muon"):
            seen_muon += 1
            if (
                set(names) != MUON_NAMES
                or group.get("betas") != [beta, BETAS[1]]
                or group.get("blocks") != 3
                or group.get("ns_steps") != 5
                or group.get("rms") != 0.2
            ):
                raise ValueError("recurrent Muon group differs from registered settings")
        elif group.get("betas") != list(BETAS):
            raise ValueError("non-Muon AdamW beta values changed")
    if seen_muon != 1 or len(all_names) != len(set(all_names)) or set(all_names) != expected_names:
        raise ValueError("optimizer groups do not exactly partition model parameters")


def validate_fit_manifest(
    run: Path, report: dict[str, Any], fit_nodes: list[dict]
) -> tuple[dict[tuple[int, str, float], dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray]:
    if (
        report.get("kind") != "E483 full fit matrix"
        or report.get("status") != "all_18_fit_arms_complete_test_unread"
        or report.get("fit_only") is not True
        or report.get("test_rows_loaded") is not False
    ):
        raise ValueError("E483 manifest lacks complete test-unread fit status")
    fixed = {
        "seeds": list(SEEDS),
        "shapes": list(SHAPES),
        "muon_beta1": list(MOMENTA),
        "planned_fit_count": 18,
        "updates_per_fit": UPDATES,
        "dataset_sha256": DATASET_SHA256,
        "cohort_sha256": COHORT_SHA256,
    }
    for key, expected in fixed.items():
        if report.get(key) != expected:
            raise ValueError(f"E483 fit manifest field differs: {key}")
    if report.get("parent_source_sha256") != PARENT_HASHES:
        raise ValueError("E483 frozen parent source identities differ")
    expected_source_receipt = {
        name: EXPECTED_HASHES[name] for name in ("fit_momentum.py", "tests.py", "gpu_probe.py")
    }
    if report.get("source_sha256") != expected_source_receipt:
        raise ValueError("E483 actor source receipt differs")
    expected_fit_cohort = [
        node for node in json.loads(COHORT_PATH.read_text())["cohort"] if node["role"] == "fit"
    ]
    fit_only = report.get("fit_only_cohort", {})
    if fit_only.get("cohort") != expected_fit_cohort or len(fit_only.get("cohort", [])) != 24:
        raise ValueError("E483 fit-only cohort role/order differs")
    if fit_only.get("normalization") is None:
        raise ValueError("E483 fit manifest lacks fit-only normalization")

    mean, scale = derive_fit_normalizer(fit_nodes)
    normalization = fit_only["normalization"]
    if not np.array_equal(np.asarray(normalization.get("fit_mean"), np.float32), mean):
        raise ValueError(
            "E483 fit-only normalizer mean differs from independent fit-row recomputation"
        )
    if not np.array_equal(np.asarray(normalization.get("fit_scale"), np.float32), scale):
        raise ValueError(
            "E483 fit-only normalizer scale differs from independent fit-row recomputation"
        )
    init_hashes = expected_initial_state_hashes()
    if report.get("initial_state_sha256") != init_hashes or len(set(init_hashes.values())) != 3:
        raise ValueError("E483 seed initializations differ from independent reconstruction")

    expected_keys = set(ARMS)
    arms = report.get("arms", [])
    if len(arms) != 18:
        raise ValueError("E483 fit manifest must contain all 18 arms")
    observed: dict[tuple[int, str, float], dict[str, Any]] = {}
    named_parameters = set(IndependentMomentumModel(False).state_dict()) - {"mean", "std"}
    # State dict keys include module names; trainable names match these non-buffer keys.
    expected_names = named_parameters
    for arm in arms:
        key = (arm.get("seed"), arm.get("shape"), arm.get("muon_beta1"))
        if key not in expected_keys or key in observed:
            raise ValueError(f"unknown or duplicate E483 cell: {key}")
        seed, shape, beta = key
        order = np.arange(24)
        np.random.default_rng(seed).shuffle(order)
        expected_order = [fit_nodes[int(index)]["row_group"] for index in order]
        if arm.get("fit_row_groups") != expected_order:
            raise ValueError(f"E483 fit traversal differs for {key}")
        if arm.get("updates") != UPDATES or arm.get("parameters") != PARAMETERS:
            raise ValueError(f"E483 update/parameter count differs for {key}")
        clipped = arm.get("clipped_updates")
        if type(clipped) is not int or not 0 <= clipped <= UPDATES:
            raise ValueError(f"invalid whole-run clipping count for {key}")
        telemetry = arm.get("telemetry")
        if not isinstance(telemetry, list) or [
            item.get("update_index_zero_based") for item in telemetry
        ] != list(TELEMETRY_STEPS):
            raise ValueError(f"E483 telemetry cadence differs for {key}")
        for sample in telemetry:
            step = sample["update_index_zero_based"]
            batch_index, chunk_index = divmod(step, CHUNKS_PER_SEQUENCE)
            if (
                sample.get("fit_row_groups")
                != expected_order[batch_index * 4 : (batch_index + 1) * 4]
            ):
                raise ValueError(f"E483 telemetry fit batch differs for {key}")
            if sample.get("chunk_start") != chunk_index * CHUNK:
                raise ValueError(f"E483 telemetry chunk differs for {key}")
            if (
                not math.isfinite(float(sample.get("preclip_gradient_norm", float("nan"))))
                or float(sample["preclip_gradient_norm"]) < 0
            ):
                raise ValueError(f"E483 telemetry preclip norm invalid for {key}")
            if bool(sample.get("clipped")) != (float(sample["preclip_gradient_norm"]) > 1.0):
                raise ValueError(f"E483 telemetry clipping flag disagrees for {key}")
            if (
                not math.isfinite(float(sample.get("state_abs_max", float("nan"))))
                or float(sample["state_abs_max"]) > 1.0001
            ):
                raise ValueError(f"E483 recurrent state bound telemetry invalid for {key}")
            for field in ("preclip_gradient_rms", "parameter_update_rms"):
                values = sample.get(field, {})
                if set(values) != {"inp", "gru", "head"} or any(
                    not math.isfinite(float(v)) or float(v) < 0 for v in values.values()
                ):
                    raise ValueError(f"E483 telemetry {field} invalid for {key}")
        try:
            validate_optimizer_groups(arm.get("optimizer_groups"), expected_names, beta)
        except ValueError as error:
            raise ValueError(f"E483 optimizer receipt differs for {key}: {error}") from error
        checkpoint_name = arm.get("checkpoint")
        expected_checkpoint = f"seed{seed}_{shape}_momentum{beta}.pt"
        if checkpoint_name != expected_checkpoint or Path(checkpoint_name).name != checkpoint_name:
            raise ValueError(f"E483 checkpoint filename differs for {key}")
        checkpoint = run / checkpoint_name
        if not checkpoint.is_file() or sha256_file(checkpoint) != arm.get("checkpoint_sha256"):
            raise ValueError(f"E483 checkpoint hash differs for {key}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError(f"nonfinite checkpoint tensor for {key}")
        model = IndependentMomentumModel(shape == "sine")
        if set(state) != set(model.state_dict()):
            raise ValueError(f"checkpoint key map differs for {key}")
        if not torch.equal(state["mean"], torch.zeros(FEATURES)) or not torch.equal(
            state["std"], torch.ones(FEATURES)
        ):
            raise ValueError(f"identity normalizer buffers differ for {key}")
        observed[key] = arm
    if set(observed) != expected_keys:
        raise ValueError("E483 fit matrix incomplete")
    if (
        list((run / "fit_manifest.json").parent.glob("*.pt"))
        and len(list((run / "fit_manifest.json").parent.glob("*.pt"))) != 18
    ):
        raise ValueError("E483 run has unexpected checkpoint file count")
    return (
        observed,
        {"validated_arms": 18, "fit_roles": 24, "test_roles_declared_only": 16},
        mean,
        scale,
    )


def normalize_features(raw_features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid fit-only normalizer")
    return np.clip((raw_features - mean) / scale, -8.0, 8.0).astype(np.float32)


def validate_raw_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Reject nonfinite raw targets before any metric clipping."""
    try:
        raw = np.asarray(raw_labels, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("raw E483 targets must be numeric") from error
    if raw.ndim != 2 or raw.shape[-1] != 2 or not np.isfinite(raw).all():
        raise ValueError("raw E483 targets must be finite with exactly two columns")
    return raw


def load_test_data(
    test_nodes: list[dict], mean: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read exposed test groups only when explicit scoring is authorized."""
    parquet = pq.ParquetFile(DATA_PATH)
    xs, targets, need_masks, scored_masks = [], [], [], []
    for node in test_nodes:
        table = parquet.read_row_group(
            node["row_group"],
            columns=[*FEATURE_COLUMNS, "t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        raw_x = np.column_stack([table[column].to_numpy() for column in FEATURE_COLUMNS])
        raw_y = validate_raw_labels(
            np.column_stack([table[column].to_numpy() for column in ("t0", "t1")])
        )
        if raw_x.shape != (ROWS, len(FEATURE_COLUMNS)) or raw_y.shape != (ROWS, 2):
            raise ValueError(f"unexpected E483 test group shape: {node['row_group']}")
        xs.append(normalize_features(causal_features(raw_x), mean, scale))
        targets.append(raw_y)
        need_masks.append(table["need_prediction"].to_numpy().astype(bool))
        scored_masks.append(table["is_scored"].to_numpy().astype(bool))
    x = np.stack(xs)
    y_clip = np.stack(targets)
    need = np.stack(need_masks)
    scored = np.stack(scored_masks)
    selected = need & scored
    if not np.isfinite(x).all() or not selected.any(axis=1).all():
        raise ValueError("invalid exposed test features or empty selected sequence")
    return x, y_clip, need, scored, selected


def predict_full(
    model: IndependentMomentumModel, x: np.ndarray, device: torch.device
) -> np.ndarray:
    model = model.to(device=device, dtype=torch.float32).eval()
    predictions = np.empty((len(x), ROWS, 2), dtype=np.float32)
    state = None
    with full_ieee_float32(device), torch.no_grad():
        for start in range(0, ROWS, CHUNK):
            chunk = torch.from_numpy(x[:, start : start + CHUNK]).to(
                device=device, dtype=torch.float32
            )
            output, state = model(chunk, state)
            if not torch.isfinite(output).all() or not torch.isfinite(state).all():
                raise ValueError(f"nonfinite prediction/state at chunk {start}")
            predictions[:, start : start + output.shape[1]] = output.cpu().numpy()
            state = state.detach()
    if predictions.shape != (len(x), ROWS, 2) or not np.isfinite(predictions).all():
        raise ValueError("nonfinite full-sequence prediction")
    return predictions


@contextlib.contextmanager
def full_ieee_float32(device: torch.device):
    """Match the fit's IEEE FP32 CUDA policy without importing actor code."""
    if device.type == "cpu":
        with torch.autocast(device_type="cpu", enabled=False):
            yield
        return
    controls = (
        (torch.backends, "fp32_precision"),
        (torch.backends.cuda.matmul, "fp32_precision"),
        (torch.backends.cudnn, "fp32_precision"),
        (torch.backends.cudnn.rnn, "fp32_precision"),
    )
    if not all(hasattr(owner, name) for owner, name in controls):
        raise RuntimeError("PyTorch lacks per-backend FP32 precision controls")
    previous = [getattr(owner, name) for owner, name in controls]
    try:
        for owner, name in controls:
            setattr(owner, name, "ieee")
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        for (owner, name), value in zip(controls, previous, strict=True):
            setattr(owner, name, value)


def component_moments(
    targets: np.ndarray, predictions: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    """Build official centered per-component/target moments after finite checks and clipping."""
    if (
        targets.ndim != 3
        or targets.shape[-1] != 2
        or predictions.shape != targets.shape
        or selected.shape != targets.shape[:2]
    ):
        raise ValueError("unaligned component score arrays")
    result = np.zeros((len(targets), 2, 6), dtype=np.float64)
    for component in range(len(targets)):
        mask = selected[component]
        raw_y = targets[component, mask].astype(np.float64)
        raw_p = predictions[component, mask].astype(np.float64)
        if not mask.any() or not np.isfinite(raw_y).all() or not np.isfinite(raw_p).all():
            raise ValueError("empty or nonfinite selected component data")
        y = np.clip(raw_y.astype(np.float32), -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        p = np.clip(raw_p.astype(np.float32), -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        for target in range(2):
            result[component, target] = _centered_moments(y[:, target], p[:, target])
    return result


def bootstrap_multiplicities(component_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(component_ids)
    if (
        ids.ndim != 1
        or len(ids) != 16
        or not np.array_equal(ids, np.sort(ids))
        or len(np.unique(ids)) != 16
    ):
        raise ValueError("E483 bootstrap requires 16 unique numerically sorted component IDs")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    counts = np.zeros((BOOTSTRAP_COUNT, len(ids)), dtype=np.int16)
    for index in range(BOOTSTRAP_COUNT):
        sampled = rng.integers(0, len(ids), len(ids))
        counts[index] = np.bincount(sampled, minlength=len(ids))
    if not np.all(counts.sum(axis=1) == len(ids)):
        raise ValueError("invalid E483 bootstrap multiplicities")
    return counts


def pool_component_moments(moments: np.ndarray, multiplicities: np.ndarray) -> float:
    if moments.ndim != 3 or moments.shape[1:] != (2, 6) or multiplicities.shape != (len(moments),):
        raise ValueError("invalid E483 centered moments/multiplicity shapes")
    pooled = np.zeros((2, 6), dtype=np.float64)
    for index, count in enumerate(multiplicities):
        if count <= 0:
            continue
        contribution = moments[index].copy()
        contribution[:, [0, 3, 4, 5]] *= int(count)
        for target in range(2):
            pooled[target] = _merge_moments(pooled[target], contribution[target])
    return float(np.mean([_wp_from_moments(pooled[target]) for target in range(2)]))


def summarize_difference(draws: np.ndarray, point: float, a: str, b: str) -> dict[str, Any]:
    if (
        draws.shape != (BOOTSTRAP_COUNT,)
        or not np.isfinite(draws).all()
        or not math.isfinite(point)
    ):
        raise ValueError("invalid E483 contrast draws")
    return {
        "candidate": a,
        "reference": b,
        "mean_diff": float(point),
        "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "p_le_0": float(np.mean(draws <= 0)),
        "n_components": 16,
        "n_boot": BOOTSTRAP_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def registered_contrasts(
    score_by_arm: dict[str, float], draws_by_arm: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for beta in MOMENTA:
        candidate, reference = f"sine_beta{beta}", f"baseline_beta{beta}"
        key = f"sine_minus_baseline_beta{beta}"
        results[key] = {
            "family": "shape_at_momentum",
            "muon_beta1": beta,
            **summarize_difference(
                draws_by_arm[candidate] - draws_by_arm[reference],
                score_by_arm[candidate] - score_by_arm[reference],
                candidate,
                reference,
            ),
        }
    for shape in SHAPES:
        for beta in (0.8, 0.95):
            candidate, reference = f"{shape}_beta{beta}", f"{shape}_beta0.9"
            key = f"{shape}_beta{beta}_minus_beta0.9"
            results[key] = {
                "family": "momentum_within_shape",
                "shape": shape,
                **summarize_difference(
                    draws_by_arm[candidate] - draws_by_arm[reference],
                    score_by_arm[candidate] - score_by_arm[reference],
                    candidate,
                    reference,
                ),
            }
    for beta in (0.8, 0.95):
        a = (draws_by_arm[f"sine_beta{beta}"] - draws_by_arm[f"baseline_beta{beta}"]) - (
            draws_by_arm["sine_beta0.9"] - draws_by_arm["baseline_beta0.9"]
        )
        point = (score_by_arm[f"sine_beta{beta}"] - score_by_arm[f"baseline_beta{beta}"]) - (
            score_by_arm["sine_beta0.9"] - score_by_arm["baseline_beta0.9"]
        )
        key = f"shape_by_momentum_interaction_beta{beta}_vs0.9"
        results[key] = {
            "family": "shape_by_momentum_interaction",
            "muon_beta1": beta,
            **summarize_difference(
                a, point, f"(sine-baseline)_beta{beta}", "(sine-baseline)_beta0.9"
            ),
        }
    if len(results) != 9:
        raise AssertionError("E483 must define nine registered contrasts per seed")
    return results


def score_all(
    cells: dict[tuple[int, str, float], dict[str, Any]],
    test_nodes: list[dict],
    x: np.ndarray,
    targets: np.ndarray,
    selected: np.ndarray,
    run: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    component_ids = np.asarray([node["component"] for node in test_nodes], dtype=np.int64)
    if not np.array_equal(component_ids, np.sort(component_ids)):
        raise ValueError("test components are not in numeric order")
    counts = bootstrap_multiplicities(component_ids)
    score_result: dict[str, Any] = {}
    raw_predictions: dict[str, np.ndarray] = {}
    moments_by_seed: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}
    draws_by_seed: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}
    selected_counts = selected.sum(axis=1).astype(np.int64)
    for seed, shape, beta in ARMS:
        arm = cells[(seed, shape, beta)]
        name = f"seed{seed}__{shape}__beta{beta}"
        model = IndependentMomentumModel(shape == "sine")
        checkpoint = run / arm["checkpoint"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        prediction = predict_full(model, x, device)
        metric = global_wp(targets, prediction, selected)
        if metric.get("metric_id") != METRIC_ID or metric.get("selected_rows") != int(
            selected.sum()
        ):
            raise ValueError(f"official E483 metric metadata differs: {name}")
        raw_predictions[name] = prediction
        score_result[name] = {
            "seed": seed,
            "shape": shape,
            "muon_beta1": beta,
            "wp": float(metric["wp"]),
            "t0": float(metric["t0"]),
            "t1": float(metric["t1"]),
            "selected_rows": int(metric["selected_rows"]),
            "selected_rows_per_component": selected_counts.tolist(),
            "parameters": PARAMETERS,
            "updates": UPDATES,
            "clipped_updates": arm["clipped_updates"],
            "checkpoint_sha256": arm["checkpoint_sha256"],
        }
        arm_moments = component_moments(targets, prediction, selected)
        moments_by_seed[seed][f"{shape}_beta{beta}"] = arm_moments
        draws_by_seed[seed][f"{shape}_beta{beta}"] = np.asarray(
            [pool_component_moments(arm_moments, draw) for draw in counts], dtype=np.float64
        )
    contrasts: dict[str, dict[str, Any]] = {}
    contrast_draws: dict[str, np.ndarray] = {}
    for seed in SEEDS:
        per_seed = registered_contrasts(
            {key: score_result[f"seed{seed}__{key}"]["wp"] for key in moments_by_seed[seed]},
            draws_by_seed[seed],
        )
        if len(per_seed) != 9:
            raise AssertionError("E483 seed contrast count differs")
        contrasts[str(seed)] = per_seed
        for key, value in per_seed.items():
            arm_a, arm_b = value["candidate"], value["reference"]
            if value["family"] == "shape_by_momentum_interaction":
                beta = value["muon_beta1"]
                delta_draws = (
                    draws_by_seed[seed][f"sine_beta{beta}"]
                    - draws_by_seed[seed][f"baseline_beta{beta}"]
                ) - (draws_by_seed[seed]["sine_beta0.9"] - draws_by_seed[seed]["baseline_beta0.9"])
            else:
                delta_draws = draws_by_seed[seed][arm_a] - draws_by_seed[seed][arm_b]
            contrast_draws[f"seed{seed}__{key}"] = delta_draws
    if sum(map(len, contrasts.values())) != 27:
        raise AssertionError("E483 global contrast matrix must contain 27 effects")
    return {
        "kind": "independent E483 recurrent Muon momentum scorer",
        "metric_id": METRIC_ID,
        "scope": "three fixed new fit seeds; same 16 previously exposed components; descriptive only; no seed pooling",
        "fit_seeds": list(SEEDS),
        "arm_count": len(score_result),
        "arm_scores": score_result,
        "contrast_count": sum(map(len, contrasts.values())),
        "contrasts_by_seed": contrasts,
        "bootstrap": {
            "method": "official centered weighted-Pearson moments and shared component multiplicity draws",
            "n_boot": BOOTSTRAP_COUNT,
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": "numeric-sorted fixed component ID",
            "draws_shared_across_all_six_arms_per_seed": True,
            "no_seed_pooling": True,
            "component_ids": component_ids.tolist(),
            "multiplicity_sha256": hashlib.sha256(counts.astype("<i2").tobytes()).hexdigest(),
        },
    }, {
        "predictions": raw_predictions,
        "contrast_draws": contrast_draws,
        "arm_draws": draws_by_seed,
        "moments": moments_by_seed,
    }


def write_outputs(
    output: Path,
    result: dict[str, Any],
    export: dict[str, Any],
    targets: np.ndarray,
    selected: np.ndarray,
    need: np.ndarray,
    scored: np.ndarray,
    test_nodes: list[dict],
    counts: np.ndarray,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite E483 score output {output}")
    output.mkdir(parents=True)
    arm_names = [f"seed{seed}__{shape}__beta{beta}" for seed, shape, beta in ARMS]
    raw_targets_selected = targets[selected]
    predictions_selected = np.stack([export["predictions"][name][selected] for name in arm_names])
    selected_component_index = np.repeat(np.arange(len(test_nodes)), selected.sum(axis=1))
    row_indices = np.concatenate([np.flatnonzero(mask) for mask in selected])
    export_path = output / "selected_exports.npz"
    np.savez_compressed(
        export_path,
        arm_names=np.asarray(arm_names),
        targets_raw_selected=raw_targets_selected.astype(np.float64),
        predictions_raw_selected_by_arm=predictions_selected.astype(np.float32),
        component_ids=np.asarray([n["component"] for n in test_nodes], dtype=np.int64),
        row_groups=np.asarray([n["row_group"] for n in test_nodes], dtype=np.int64),
        selected_row_counts=selected.sum(axis=1).astype(np.int32),
        selected_component_index=selected_component_index.astype(np.int16),
        selected_row_index_within_component=row_indices.astype(np.int32),
        selected_masks_by_component=selected,
        need_masks_by_component=need,
        scored_masks_by_component=scored,
        bootstrap_multiplicities=counts,
        arm_bootstrap_wp=np.stack(
            [export["arm_draws"][seed][f"{shape}_beta{beta}"] for seed, shape, beta in ARMS]
        ),
        contrast_names=np.asarray(sorted(export["contrast_draws"])),
        contrast_bootstrap_diffs=np.stack(
            [export["contrast_draws"][key] for key in sorted(export["contrast_draws"])]
        ),
    )
    result["component_ids"] = [int(n["component"]) for n in test_nodes]
    result["row_groups"] = [int(n["row_group"]) for n in test_nodes]
    result["selected_rows_total"] = int(selected.sum())
    result["selected_exports_path"] = export_path.name
    result["selected_exports_sha256"] = sha256_file(export_path)
    (output / "scores.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--attempt-record", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--sealed-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--score-exposed-test",
        action="store_true",
        help="explicitly authorize reading/scoring the registered exposed test components",
    )
    args = parser.parse_args()
    run = args.run.resolve()
    spec_raw = json.loads(args.sealed_spec.read_text())
    fit_attempt = validate_attempt(spec_raw, args.attempt_record.resolve(), run)
    fit_report = json.loads((run / "fit_manifest.json").read_text())
    fit_nodes, test_nodes = canonical_cohort()
    cells, fit_provenance, mean, scale = validate_fit_manifest(run, fit_report, fit_nodes)
    verify_preflight()
    if not args.score_exposed_test:
        print(
            json.dumps(
                {
                    "mode": "fit_proof_validated_test_unread",
                    "fit_attempt_id": fit_attempt["attempt_id"],
                    "validated_arms": fit_provenance["validated_arms"],
                    "test_rows_loaded": False,
                    "test_rows_scored": False,
                    "fit_mean_sha256": hashlib.sha256(mean.astype("<f4").tobytes()).hexdigest(),
                    "fit_scale_sha256": hashlib.sha256(scale.astype("<f4").tobytes()).hexdigest(),
                    "preflight_sha256": sha256_file(PREFLIGHT_PATH),
                },
                indent=2,
            )
        )
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite E483 score output {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    x, targets, need, scored, selected = load_test_data(test_nodes, mean, scale)
    predictions, internal = score_all(
        cells, test_nodes, x, targets, selected, run, torch.device(args.device)
    )
    counts = bootstrap_multiplicities(
        np.asarray([n["component"] for n in test_nodes], dtype=np.int64)
    )
    predictions["fit_provenance"] = fit_provenance
    predictions["attempt_receipt"] = fit_attempt
    predictions["fit_manifest_sha256"] = sha256_file(run / "fit_manifest.json")
    predictions["cohort_sha256"] = COHORT_SHA256
    predictions["dataset_sha256"] = DATASET_SHA256
    predictions["normalizer"] = {
        "source": "independent derivation from only canonical 24 fit-role row groups",
        "fit_mean_sha256": hashlib.sha256(mean.astype("<f4").tobytes()).hexdigest(),
        "fit_scale_sha256": hashlib.sha256(scale.astype("<f4").tobytes()).hexdigest(),
        "identity_model_buffers_required": True,
    }
    predictions["source_sha256"] = {
        "grader": sha256_file(Path(__file__)),
        "gmetric": sha256_file(ROOT / "src/wnn/gmetric.py"),
        "runtime": runtime_identity(ROOT),
        "actor": EXPECTED_HASHES,
    }
    write_outputs(
        args.output.resolve(),
        predictions,
        internal,
        targets,
        selected,
        need,
        scored,
        test_nodes,
        counts,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "arms": 18,
                "contrasts": 27,
                "selected_rows": int(selected.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Independent E479 v2 scorer; held-out scoring waits for complete preflight.

The recurrent scan and model are reconstructed locally. The actor is consulted
only for frozen input-feature engineering constants and artifact identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
DATA_PATH = ROOT / "datasets/valid.parquet"
E470_PATH = ROOT / "research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py"
E474_PATH = ROOT / "research/campaign_2026-10-05/activation_placement/fit_activation_placement.py"
ACTOR_PATH = HERE / "fit_recurrent_activation_adaptive.py"
PREFLIGHT_PATH = HERE / "cpu_preflight_v4/fit_manifest.json"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_PATH.parent))
from fit_activation_optimizer import FEATURES, causal_features  # noqa: E402

from wnn.campaign_spec import digest as spec_digest  # noqa: E402
from wnn.campaign_spec import validate_spec  # noqa: E402
from wnn.evaluation import _score_component_moments  # noqa: E402
from wnn.gmetric import METRIC_CLIP, METRIC_ID, _centered_moments, global_wp  # noqa: E402

SEEDS = (271828, 271829)
OPTIMIZERS = ("adamw", "muon_adamw")
CONDITIONS = ("fixed0", "fixed01", "learned0", "learned01")
ARMS = tuple((s, o, c) for s in SEEDS for o in OPTIMIZERS for c in CONDITIONS)
INPUTS, HIDDEN, N_ROWS = 42, 64, 20_000
N_UPDATES, CHUNK_ROWS, N_BOOT = 474, 256, 2_000
TELEMETRY = (0, 46, 236, 473)
DATA_SHA = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
COHORT_SHA = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
ACTOR_SHA = "6a11025735bb8df7d88b08bdcf7dfd9d8879992a82c1c64d40b43ada7456c7bf"
PREFLIGHT_SHA = "92fcb33ea416527dc742911605da9059eb1d7c728a4a594ed9e5a65dae5bbb66"
E470_SHA = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E474_SHA = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
PREFLIGHT_SOURCE = {
    "fit_recurrent_activation_adaptive.py": ACTOR_SHA,
    "tests.py": "88ce5a39aa8c8a8af869689f5394ba1eae4225c008aaef61e6679de592994b17",
    "gpu_probe.py": "293fe14c00cc7f76d13daa44f9a3e9f99fbdb1b083a5a0045a020c99e1047f39",
}
RUNTIME_PATHS = {
    "scripts/run_campaign.py",
    "src/wnn/campaign.py",
    "src/wnn/campaign_local.py",
    "src/wnn/campaign_spec.py",
    "src/wnn/resources.py",
}
FIT_SEAL_NAME = "e479_learned_candidate_sixteen_fit_v2"
FIT_ATTEMPT_ID = "e479-learned-candidate-sixteen-fit-v2"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def verify_unit_success(attempt: dict) -> None:
    """Require systemd's durable unit result, in addition to runner state."""
    unit = attempt.get("unit_properties", {})
    require(
        unit.get("Result") == "success" and str(unit.get("ExecMainStatus")) == "0",
        "systemd unit did not report successful completion",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.jit.script
def scan_layer(
    x: Tensor,
    h0: Tensor,
    wi: Tensor,
    wh: Tensor,
    bi: Tensor,
    bh: Tensor,
    alpha: Tensor,
) -> tuple[Tensor, Tensor]:
    """Reset-after GRU (r,z,n ordering) with live tensor alpha per layer."""
    input_affine = F.linear(x, wi, bi)
    rows = torch.jit.annotate(list[Tensor], [])
    h = h0
    for t in range(x.size(1)):
        ir, iz, inn = input_affine[:, t, :].chunk(3, dim=1)
        hr, hz, hn = F.linear(h, wh, bh).chunk(3, dim=1)
        reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        u = torch.tanh(inn + reset * hn)
        candidate = u + alpha * torch.sin(math.pi * u)
        h = (1.0 - update) * candidate + update * h
        rows.append(h)
    return torch.stack(rows, dim=1), h


class IndependentAdaptiveRNN(nn.Module):
    """Checkpoint-compatible parameter layout; independently written forward."""

    def __init__(self, condition: str):
        super().__init__()
        require(condition in CONDITIONS, f"unregistered condition: {condition}")
        self.condition = condition
        self.register_buffer("mean", torch.zeros(INPUTS))
        self.register_buffer("std", torch.ones(INPUTS))
        self.inp = nn.Linear(INPUTS, HIDDEN)
        self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 2))
        if condition.startswith("learned"):
            initial = 0.0 if condition == "learned0" else math.atanh(0.5)
            self.raw_alpha = nn.Parameter(torch.full((2,), initial, dtype=torch.float32))
            self.register_buffer("fixed_alpha", torch.zeros(2), persistent=False)
        else:
            self.register_parameter("raw_alpha", None)
            value = 0.0 if condition == "fixed0" else 0.1
            self.register_buffer("fixed_alpha", torch.full((2,), value), persistent=False)

    def alphas(self) -> Tensor:
        if self.raw_alpha is None:
            return self.fixed_alpha
        return 0.2 * torch.tanh(self.raw_alpha)

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != INPUTS:
            raise ValueError("expected [batch,time,42] input")
        batch = x.shape[0]
        if state is None:
            state = x.new_zeros((2, batch, HIDDEN))
        elif state.shape != (2, batch, HIDDEN):
            raise ValueError("expected state shape [2,batch,64]")
        encoded = F.gelu(self.inp((x - self.mean) / self.std))
        layer_input = encoded
        finals = []
        alpha = self.alphas()
        for layer in range(2):
            layer_input, h = scan_layer(
                layer_input,
                state[layer],
                getattr(self.gru, f"weight_ih_l{layer}"),
                getattr(self.gru, f"weight_hh_l{layer}"),
                getattr(self.gru, f"bias_ih_l{layer}"),
                getattr(self.gru, f"bias_hh_l{layer}"),
                alpha[layer],
            )
            finals.append(h)
        prediction = self.head(torch.cat((layer_input, encoded), dim=-1))
        return prediction, torch.stack(finals)


def validate_receipts(run: Path, seal_path: Path, attempt_path: Path) -> tuple[dict, dict, dict]:
    seal, attempt = json.loads(seal_path.read_text()), json.loads(attempt_path.read_text())
    require(validate_spec(seal) == seal, "sealed specification is not canonical/valid")
    require(
        seal.get("kind") == "gpu" and seal.get("name") == FIT_SEAL_NAME,
        "fit seal identity or kind differs",
    )
    require(attempt.get("id") == FIT_ATTEMPT_ID, "wrong sealed fit attempt identity")
    require(
        attempt.get("spec") == seal and attempt.get("spec_sha256") == spec_digest(seal),
        "attempt does not bind exact sealed spec",
    )
    require(
        attempt.get("status") == "finished"
        and attempt.get("exit_code") == 0
        and attempt.get("failure") is None
        and attempt.get("teardown_verified") is True,
        "fit attempt did not finish successfully with teardown",
    )
    verify_unit_success(attempt)
    require(
        attempt.get("kind") == "gpu" and attempt.get("signals") == [],
        "runner kind/signals differ from successful unsignaled fit",
    )
    require(
        attempt.get("argv", [])[1:] == seal.get("argv", [])[1:], "runner argv differs from seal"
    )
    sealed_argv = seal.get("argv", [])
    require(
        len(sealed_argv) >= 2 and sealed_argv[1] == str(ACTOR_PATH.relative_to(ROOT)),
        "sealed argv does not execute the registered E479 trainer",
    )
    if "--output" in sealed_argv:
        require(
            Path(sealed_argv[sealed_argv.index("--output") + 1]).resolve() == run.resolve(),
            "requested run directory differs from sealed trainer output",
        )
    sealed_code = {row["path"]: row["sha256"] for row in seal.get("code", [])}
    require(
        sealed_code.get(str(ACTOR_PATH.relative_to(ROOT))) == ACTOR_SHA,
        "seal does not bind the expected E479 actor bytes",
    )
    require(
        sealed_code.get(str(E470_PATH.relative_to(ROOT))) == E470_SHA
        and sealed_code.get(str(E474_PATH.relative_to(ROOT))) == E474_SHA,
        "seal does not bind both frozen parent implementations",
    )
    require(
        set(attempt.get("runtime_sha256", {})) == RUNTIME_PATHS,
        "runner runtime source inventory is incomplete",
    )
    require(
        attempt["runtime_sha256"] == {p: sealed_code.get(p) for p in RUNTIME_PATHS},
        "runtime source hashes do not match seal",
    )
    for start_key, end_key in (
        ("initial_memory_events", "final_memory_events"),
        ("initial_parent_memory_events", "final_parent_memory_events"),
    ):
        start, end = attempt.get(start_key, {}), attempt.get(end_key, {})
        require(
            start.keys() == end.keys() and all(start[k] == end[k] for k in start),
            f"memory event counters changed: {start_key}",
        )
    require(
        all(
            attempt.get("final_memory_events", {}).get(k, 0) == 0
            for k in ("oom", "oom_group_kill", "oom_kill")
        ),
        "fit worker OOM recorded",
    )
    manifest_path = run / "fit_manifest.json"
    relative_manifest = str(manifest_path.resolve().relative_to(ROOT.resolve()))
    require(
        seal.get("artifacts") == [relative_manifest], "seal must declare only fit_manifest.json"
    )
    artifacts = attempt.get("artifacts", [])
    require(
        len(artifacts) == 1 and artifacts[0].get("path") == relative_manifest,
        "attempt must attest exactly fit_manifest.json",
    )
    artifact = artifacts[0]
    require(
        manifest_path.is_file()
        and manifest_path.stat().st_size == artifact.get("size")
        and sha256_file(manifest_path) == artifact.get("sha256"),
        "fit manifest bytes do not match runner artifact receipt",
    )
    for row in seal.get("code", []) + seal.get("inputs", []):
        path = ROOT / row["path"]
        require(
            path.is_file()
            and path.stat().st_size == row["size"]
            and sha256_file(path) == row["sha256"],
            f"sealed input changed: {row['path']}",
        )
    require(
        sha256_file(ACTOR_PATH) == ACTOR_SHA and sha256_file(PREFLIGHT_PATH) == PREFLIGHT_SHA,
        "E479 actor or CPU-preflight-v4 identity changed",
    )
    sealed_inputs = {row["path"]: row["sha256"] for row in seal.get("inputs", [])}
    require(
        sealed_inputs.get(str(PREFLIGHT_PATH.relative_to(ROOT))) == PREFLIGHT_SHA
        and sealed_inputs.get("datasets/valid.parquet") == DATA_SHA
        and sealed_inputs.get(str(COHORT_PATH.relative_to(ROOT))) == COHORT_SHA,
        "seal does not bind the dataset, cohort, and CPU-preflight-v4 receipt",
    )
    report = json.loads(manifest_path.read_text())
    require(
        report.get("status") == "all_16_fit_arms_complete_test_unread"
        and report.get("fit_only") is True
        and report.get("test_rows_loaded") is False,
        "fit report is incomplete or read test rows",
    )
    require(
        report.get("seeds") == list(SEEDS)
        and report.get("optimizers") == list(OPTIMIZERS)
        and report.get("conditions") == list(CONDITIONS)
        and report.get("planned_fit_count") == 16,
        "fit grid differs from registered 16-arm matrix",
    )
    require(
        report.get("dataset_sha256") == DATA_SHA and report.get("cohort_sha256") == COHORT_SHA,
        "fit dataset/cohort receipt differs",
    )
    require(
        report.get("parent_source_sha256") == {"e470": E470_SHA, "e474": E474_SHA},
        "parent source identity differs",
    )
    # The actor's report intentionally records only its own fit implementation inputs.
    sources = report.get("source_sha256", {})
    require(
        sources.get("fit_recurrent_activation_adaptive.py") == ACTOR_SHA,
        "fit report actor source hash differs",
    )
    require(
        report.get("updates_per_full_fit") == N_UPDATES
        and report.get("batch") == 4
        and report.get("chunk_rows") == CHUNK_ROWS
        and report.get("telemetry_steps_zero_based") == list(TELEMETRY),
        "fit cadence differs from registered recipe",
    )
    require(
        report.get("parameters") == {"fixed": 61_058, "learned": 61_060},
        "common fixed/learned parameter receipt differs",
    )
    optimizer = report.get("optimizer", {})
    require(
        optimizer.get("lr") == 0.001
        and optimizer.get("weight_decay_original_parameters") == 0.01
        and optimizer.get("learned_alpha_weight_decay") == 0.0
        and optimizer.get("betas") == [0.9, 0.999]
        and optimizer.get("schedule") == "constant"
        and optimizer.get("clip_norm") == 1.0
        and optimizer.get("muon_matrices")
        == ["gru.weight_hh_l0", "gru.weight_hh_l1", "gru.weight_ih_l0", "gru.weight_ih_l1"]
        and optimizer.get("muon_blocks_per_matrix") == 3,
        "optimizer recipe differs from E479 preregistration",
    )
    return seal, attempt, report


def get_cohort() -> tuple[list[dict], list[dict]]:
    require(sha256_file(COHORT_PATH) == COHORT_SHA, "frozen cohort hash changed")
    nodes = json.loads(COHORT_PATH.read_text())["cohort"]
    fit = [n for n in nodes if n["role"] == "fit"]
    test = [n for n in nodes if n["role"] == "test"]
    require(len(fit) == 24 and len(test) == 16, "expected 24 fit and 16 test components")
    require(
        len({n["component"] for n in fit}) == 24 and len({n["component"] for n in test}) == 16,
        "duplicate fit/test component",
    )
    require(
        not ({n["component"] for n in fit} & {n["component"] for n in test}),
        "fit/test components overlap",
    )
    require(len({n["row_group"] for n in nodes}) == 40, "duplicate cohort row group")
    return fit, test


def validate_common_cadence(report: dict) -> None:
    """The current actor stores batch/chunk once at manifest scope."""
    require(
        report.get("batch") == 4 and report.get("chunk_rows") == CHUNK_ROWS,
        "common manifest batch/chunk receipt differs",
    )


def validate_optional_arm_cadence(item: dict, key: tuple[int, str, str]) -> None:
    """Accept omitted per-arm cadence fields; reject contradictory ones."""
    require(
        item.get("batch", 4) == 4 and item.get("chunk_rows", CHUNK_ROWS) == CHUNK_ROWS,
        f"arm-level batch/chunk, when present, differs for {key}",
    )


def verify_arms(report: dict, run: Path, fit_nodes: list[dict]) -> dict[tuple[int, str, str], Path]:
    records = report.get("arms", [])
    require(len(records) == 16, "all 16 registered checkpoint records required")
    validate_common_cadence(report)
    groups = [n["row_group"] for n in fit_nodes]
    expected_keys = set(ARMS)
    result: dict[tuple[int, str, str], Path] = {}
    for item in records:
        key = (item.get("seed"), item.get("optimizer"), item.get("condition"))
        require(key in expected_keys and key not in result, f"unplanned or duplicate arm: {key}")
        permutation = np.arange(24)
        np.random.default_rng(key[0]).shuffle(permutation)
        order = [groups[int(i)] for i in permutation]
        require(item.get("fit_row_groups") == order, f"fit order differs for {key}")
        params = 61_060 if key[2].startswith("learned") else 61_058
        require(
            item.get("parameters") == params and item.get("updates") == N_UPDATES,
            f"parameter/update count differs for {key}",
        )
        # The current fit actor records B4/chunk256 once at manifest scope.
        validate_optional_arm_cadence(item, key)
        require(
            item.get("optimizer") in OPTIMIZERS and item.get("condition") in CONDITIONS,
            f"arm labels malformed: {key}",
        )
        telemetry = item.get("telemetry", [])
        require(
            [t.get("update_index_zero_based") for t in telemetry] == list(TELEMETRY),
            f"telemetry index mismatch: {key}",
        )
        for sample in telemetry:
            step = sample["update_index_zero_based"]
            expected_batch = order[(step // 79) * 4 : (step // 79) * 4 + 4]
            require(
                sample.get("fit_row_groups") == expected_batch
                and sample.get("chunk_start") == (step % 79) * CHUNK_ROWS,
                f"telemetry position differs: {key}/{step}",
            )
            require(
                sample.get("seed") == key[0]
                and sample.get("optimizer") == key[1]
                and sample.get("condition") == key[2],
                f"telemetry arm identity differs: {key}/{step}",
            )
            effective = np.asarray(sample.get("effective_alpha"), dtype=np.float64)
            require(
                effective.shape == (2,)
                and np.isfinite(effective).all()
                and np.max(np.abs(effective)) <= 0.2 + 1e-7,
                f"invalid candidate telemetry: {key}/{step}",
            )
            require(
                np.isfinite(sample.get("loss")) and np.isfinite(sample.get("state_abs_max")),
                f"nonfinite telemetry: {key}/{step}",
            )
            if key[2].startswith("learned"):
                require(
                    np.isfinite(sample.get("raw_alpha_gradient", [])).all()
                    and np.isfinite(sample.get("raw_alpha_update", [])).all(),
                    f"missing/nonfinite alpha telemetry: {key}/{step}",
                )
        checkpoint = f"seed{key[0]}_{key[1]}_{key[2]}.pt"
        path = run / checkpoint
        require(
            item.get("checkpoint") == checkpoint
            and path.is_file()
            and sha256_file(path) == item.get("checkpoint_sha256"),
            f"checkpoint identity mismatch: {key}",
        )
        result[key] = path
    require(set(result) == expected_keys, "one or more registered arms are absent")
    return result


def load_verified_checkpoints(
    checkpoints: dict[tuple[int, str, str], Path], device: torch.device | None = None
) -> dict[str, str]:
    """Strictly instantiate every checkpoint before any test-role row is read."""
    device = torch.device("cpu") if device is None else device
    receipts = {}
    for (seed, optimizer, condition), path in checkpoints.items():
        key = f"seed{seed}_{optimizer}_{condition}"
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = IndependentAdaptiveRNN(condition)
        expected = model.state_dict()
        require(
            isinstance(state, dict) and set(state) == set(expected),
            f"checkpoint state keys differ: {key}",
        )
        for name, value in state.items():
            require(
                torch.is_tensor(value)
                and value.shape == expected[name].shape
                and torch.isfinite(value).all().item(),
                f"invalid checkpoint tensor: {key}/{name}",
            )
        model.load_state_dict(state, strict=True)
        require(
            torch.equal(model.mean, torch.zeros_like(model.mean))
            and torch.equal(model.std, torch.ones_like(model.std)),
            f"checkpoint normalization buffers differ from identity: {key}",
        )
        require(
            sum(parameter.numel() for parameter in model.parameters())
            == (61_060 if condition.startswith("learned") else 61_058),
            f"checkpoint model parameter count differs: {key}",
        )
        if model.raw_alpha is not None:
            alpha = model.alphas().detach()
            require(
                torch.isfinite(model.raw_alpha).all().item()
                and torch.isfinite(alpha).all().item()
                and bool((alpha.abs() <= 0.2).all()),
                f"learned alpha checkpoint invalid: {key}",
            )
        model.to(device)
        receipts[key] = sha256_file(path)
        del model, state
    require(len(receipts) == 16, "not all 16 checkpoints loaded successfully")
    return receipts


def derive_normalizer(fit_nodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    parquet = pq.ParquetFile(DATA_PATH)
    rows = []
    for node in fit_nodes:
        table = parquet.read_row_group(node["row_group"], columns=list(FEATURES), use_threads=False)
        raw = np.column_stack([table[name].to_numpy() for name in FEATURES]).astype(np.float32)
        require(raw.shape == (N_ROWS, len(FEATURES)) and np.isfinite(raw).all(), "bad fit features")
        feat = causal_features(raw)
        require(feat.shape == (N_ROWS, INPUTS) and np.isfinite(feat).all(), "bad causal features")
        rows.append(feat)
    matrix = np.stack(rows)
    mean = matrix.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(matrix.std(axis=(0, 1), dtype=np.float64), 0.001).astype(np.float32)
    return mean, scale


def verify_normalizer_arrays(
    saved_mean: np.ndarray, saved_scale: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> None:
    """Fail closed unless the manifest normalizer exactly matches fit rows."""
    require(saved_mean.shape == mean.shape == (INPUTS,), "fit-only mean shape differs")
    require(saved_scale.shape == scale.shape == (INPUTS,), "fit-only scale shape differs")
    require(
        np.isfinite(saved_mean).all()
        and np.isfinite(saved_scale).all()
        and np.array_equal(saved_mean, mean)
        and np.array_equal(saved_scale, scale),
        "saved normalizer differs from independent fit-group recomputation",
    )


def verify_saved_normalizer(report: dict, mean: np.ndarray, scale: np.ndarray) -> None:
    normalization = report.get("fit_only_cohort", {}).get("normalization", {})
    saved_mean = np.asarray(normalization.get("fit_mean"), dtype=np.float32)
    saved_scale = np.asarray(normalization.get("fit_scale"), dtype=np.float32)
    verify_normalizer_arrays(saved_mean, saved_scale, mean, scale)


def require_fresh_outputs(output: Path) -> tuple[Path, Path]:
    """Resolve paths and reject either occupied output before source/data reads."""
    resolved = output.resolve()
    npz_path = resolved.with_name(resolved.stem + "_paired_components.npz")
    require(not resolved.exists() and not npz_path.exists(), "refusing to overwrite scorer outputs")
    return resolved, npz_path


def verify_fit_artifacts_before_test(
    run: Path, seal_path: Path, attempt_path: Path
) -> tuple[
    dict,
    dict,
    dict,
    list[dict],
    list[dict],
    dict[tuple[int, str, str], Path],
    np.ndarray,
    np.ndarray,
    dict[str, str],
]:
    """Verify receipt, fit-only cohort/normalizer, and load all models before test rows."""
    run = run.resolve()
    seal, attempt, report = validate_receipts(run, seal_path, attempt_path)
    fit_nodes, test_nodes = get_cohort()
    reported_fit_nodes = report.get("fit_only_cohort", {}).get("cohort", [])
    expected_fit_identity = [
        {
            key: node[key]
            for key in ("row_group", "seq_ix", "component", "label_class_diagnostic_only", "role")
        }
        for node in fit_nodes
    ]
    require(
        reported_fit_nodes == expected_fit_identity,
        "fit manifest frozen cohort differs from the sealed 24 fit groups",
    )
    checkpoints = verify_arms(report, run, fit_nodes)
    mean, scale = derive_normalizer(fit_nodes)
    verify_saved_normalizer(report, mean, scale)
    checkpoint_hashes = load_verified_checkpoints(checkpoints)
    return seal, attempt, report, fit_nodes, test_nodes, checkpoints, mean, scale, checkpoint_hashes


def read_test(node: dict, mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, ...]:
    table = pq.ParquetFile(DATA_PATH).read_row_group(
        node["row_group"],
        columns=[*FEATURES, "t0", "t1", "need_prediction", "is_scored"],
        use_threads=False,
    )
    require(table.num_rows == N_ROWS, f"test row count changed: {node['row_group']}")
    raw = np.column_stack([table[name].to_numpy() for name in FEATURES]).astype(np.float32)
    require(np.isfinite(raw).all(), f"nonfinite raw test feature: {node['row_group']}")
    features = causal_features(raw)
    require(
        features.shape == (N_ROWS, INPUTS) and np.isfinite(features).all(), "invalid test features"
    )
    x = np.clip((features - mean) / scale, -8, 8).astype(np.float32)
    y = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")]).astype(np.float32)
    need = table["need_prediction"].to_numpy().astype(bool)
    scored = table["is_scored"].to_numpy().astype(bool)
    mask = need & scored
    require(mask.any() and np.isfinite(y[mask]).all(), "selected raw labels missing/nonfinite")
    return x, y, need, scored, mask


def predict(model: IndependentAdaptiveRNN, x: np.ndarray, device: torch.device) -> np.ndarray:
    state = None
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, N_ROWS, CHUNK_ROWS):
            chunk = torch.from_numpy(x[start : start + CHUNK_ROWS][None]).to(device)
            pred, state = model(chunk, state)
            require(
                torch.isfinite(pred).all().item() and torch.isfinite(state).all().item(),
                f"nonfinite scorer output at chunk {start}",
            )
            output.append(pred[0].cpu().numpy())
    out = np.concatenate(output).astype(np.float32)
    require(out.shape == (N_ROWS, 2) and np.isfinite(out).all(), "invalid full prediction")
    return out


def moments(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    require(y.shape == pred.shape == (N_ROWS, 2) and mask.shape == (N_ROWS,), "bad score arrays")
    require(
        np.isfinite(y[mask]).all() and np.isfinite(pred[mask]).all(),
        "nonfinite raw selected labels/predictions before clipping",
    )
    yc, pc = (
        np.clip(y[mask], -METRIC_CLIP, METRIC_CLIP),
        np.clip(pred[mask], -METRIC_CLIP, METRIC_CLIP),
    )
    return np.stack(
        [
            _centered_moments(yc[:, j].astype(np.float64), pc[:, j].astype(np.float64))
            for j in range(2)
        ]
    )


def registered_contrasts() -> list[tuple[str, tuple[int, str, str], tuple[int, str, str], tuple]]:
    """Return 28 signed contrasts, including four paired difference-in-differences."""
    out = []
    for seed in SEEDS:
        for opt in OPTIMIZERS:
            for left, right, label in (
                ("learned0", "fixed0", "learned0_minus_fixed0"),
                ("learned01", "fixed01", "learned01_minus_fixed01"),
                ("fixed01", "fixed0", "fixed01_minus_fixed0"),
                ("learned01", "learned0", "learned01_minus_learned0"),
            ):
                out.append(
                    (f"seed{seed}:{opt}:{label}", (seed, opt, left), (seed, opt, right), (1, -1))
                )
        for cond in CONDITIONS:
            out.append(
                (
                    f"seed{seed}:{cond}:muon_minus_adamw",
                    (seed, "muon_adamw", cond),
                    (seed, "adamw", cond),
                    (1, -1),
                )
            )
        for start, label in (("0", "learned0_vs_fixed0"), ("01", "learned01_vs_fixed01")):
            out.append(
                (
                    f"seed{seed}:interaction:{label}",
                    (seed, "muon_adamw", f"learned{start}"),
                    (seed, "muon_adamw", f"fixed{start}"),
                    (seed, "adamw", f"learned{start}"),
                    (seed, "adamw", f"fixed{start}"),
                )
            )
    require(len(out) == 28, "contrast registry count error")
    return out


def self_check() -> dict:
    """Small synthetic-only checks; never opens validation rows or checkpoints."""
    torch.manual_seed(479)
    for condition, expected_count in (
        ("fixed0", 61_058),
        ("fixed01", 61_058),
        ("learned0", 61_060),
        ("learned01", 61_060),
    ):
        model = IndependentAdaptiveRNN(condition)
        require(
            sum(p.numel() for p in model.parameters()) == expected_count,
            f"parameter count differs for {condition}",
        )
        require(
            torch.equal(model.mean, torch.zeros_like(model.mean))
            and torch.equal(model.std, torch.ones_like(model.std)),
            "identity normalization buffer initialization changed",
        )
        if condition.startswith("learned"):
            expected_alpha = 0.0 if condition == "learned0" else 0.1
            require(
                torch.allclose(model.alphas(), torch.full((2,), expected_alpha), atol=1e-7, rtol=0),
                f"initial learned alpha differs: {condition}",
            )
    fixed = IndependentAdaptiveRNN("fixed0").eval()
    learned = IndependentAdaptiveRNN("learned0").eval()
    state = fixed.state_dict()
    missing, unexpected = learned.load_state_dict(state, strict=False)
    require(
        missing == ["raw_alpha"] and not unexpected, "baseline-to-learned identity transfer differs"
    )
    fixed_pos = IndependentAdaptiveRNN("fixed01").eval()
    learned_pos = IndependentAdaptiveRNN("learned01").eval()
    missing_pos, unexpected_pos = learned_pos.load_state_dict(fixed_pos.state_dict(), strict=False)
    require(
        missing_pos == ["raw_alpha"] and not unexpected_pos,
        "positive-alpha fixed-to-learned identity transfer differs",
    )
    require(
        torch.allclose(learned_pos.alphas(), fixed_pos.fixed_alpha, atol=1e-7, rtol=0),
        "fixed/learned positive initialization does not match",
    )
    torch.manual_seed(480)
    x = torch.randn(2, 29, INPUTS)
    with torch.no_grad():
        y_full, s_full = fixed(x)
        y0, s0 = fixed(x[:, :11])
        y1, s1 = fixed(x[:, 11:], s0)
        torch.testing.assert_close(torch.cat((y0, y1), 1), y_full, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(s1, s_full, atol=2e-6, rtol=2e-5)
        changed = x.clone()
        changed[:, 13:] += 3.0
        y_changed, _ = fixed(changed)
        torch.testing.assert_close(y_changed[:, :13], y_full[:, :13], atol=0, rtol=0)
        y_learned, s_learned = learned(x)
        torch.testing.assert_close(y_learned, y_full, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(s_learned, s_full, atol=2e-6, rtol=2e-5)
        y_fixed_pos, s_fixed_pos = fixed_pos(x)
        y_learned_pos, s_learned_pos = learned_pos(x)
        torch.testing.assert_close(y_learned_pos, y_fixed_pos, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(s_learned_pos, s_fixed_pos, atol=2e-6, rtol=2e-5)
    train_model = IndependentAdaptiveRNN("learned01").train()
    train_x = torch.randn(2, 9, INPUTS, requires_grad=True)
    pred, end = train_model(train_x)
    (pred.square().mean() + end.square().mean()).backward()
    require(
        train_model.raw_alpha.grad is not None
        and torch.isfinite(train_model.raw_alpha.grad).all().item(),
        "synthetic alpha gradient missing/nonfinite",
    )
    require(
        float(train_model.raw_alpha.grad.abs().sum()) > 0,
        "synthetic differentiable alpha path is disconnected",
    )
    require(
        bool((train_model.alphas().abs() < 0.2).all()), "learned alpha escaped its strict bound"
    )

    # Synthetic component moments exercise the registered shared-bootstrap path.
    rng = np.random.default_rng(481)
    # The scorer requires exactly 16 components; each fake component has a distinct target offset.
    fake_arms = {
        key: np.stack(
            [
                np.stack(
                    [
                        _centered_moments(
                            y := rng.normal(loc=(i + target) / 10, size=101),
                            y + rng.normal(0, 0.15, len(y)),
                        )
                        for target in range(2)
                    ]
                )
                for i in range(16)
            ]
        )
        for key in ARMS
    }
    fake_contrasts = bootstrap_all(fake_arms, np.arange(100, 116, dtype=np.int64))
    require(
        len(fake_contrasts) == 28
        and all(np.isfinite(row["difference"]) for row in fake_contrasts.values()),
        "synthetic registered paired bootstrap failed",
    )
    try:
        bootstrap_all(fake_arms, np.arange(116, 100, -1, dtype=np.int64))
    except ValueError:
        pass
    else:
        raise ValueError("bootstrap accepted unsorted component IDs")
    require(len(registered_contrasts()) == 28, "registered contrast count changed")
    return {
        "status": "synthetic_cpu_checks_passed",
        "real_data_opened": False,
        "real_checkpoint_loaded": False,
        "arms": len(ARMS),
        "contrasts": 28,
        "alpha_tensor_gradient": True,
        "identity_buffers": True,
        "causality_and_chunk_carry": True,
        "parameters_fixed": 61_058,
        "parameters_learned": 61_060,
    }


def pooled_wp(m: np.ndarray, counts: np.ndarray) -> float:
    return float(_score_component_moments(m, counts))


def contrast_values(
    arms: dict[tuple, np.ndarray], contrasts: list, counts_by_seed: dict[int, np.ndarray]
) -> dict[str, float]:
    values = {}
    for item in contrasts:
        name = item[0]
        seed = item[1][0]
        counts = counts_by_seed[seed]
        if len(item) == 4:
            _, a, b, _sign = item
            values[name] = pooled_wp(arms[a], counts) - pooled_wp(arms[b], counts)
        else:
            _, mu_l, mu_r, ad_l, ad_r = item
            values[name] = (
                pooled_wp(arms[mu_l], counts)
                - pooled_wp(arms[mu_r], counts)
                - pooled_wp(arms[ad_l], counts)
                + pooled_wp(arms[ad_r], counts)
            )
    return values


def bootstrap_all(arm_moments: dict[tuple, np.ndarray], components: np.ndarray) -> dict[str, dict]:
    """Same sorted-component bootstrap multiplicities jointly drive all contrasts."""
    contrasts = registered_contrasts()
    sorted_ids = np.sort(np.asarray(components, dtype=np.int64))
    require(
        np.array_equal(sorted_ids, components),
        "components must be numerically sorted before resampling",
    )
    result = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        draws = np.empty((N_BOOT, len(components)), dtype=np.int16)
        for i in range(N_BOOT):
            draws[i] = np.bincount(
                rng.integers(0, len(components), len(components)), minlength=len(components)
            )
        relevant = [c for c in contrasts if c[1][0] == seed]
        point = contrast_values(
            arm_moments, relevant, {seed: np.ones(len(components), dtype=np.int64)}
        )
        sampled = {c[0]: np.empty(N_BOOT, dtype=np.float64) for c in relevant}
        for i, counts in enumerate(draws):
            one = contrast_values(arm_moments, relevant, {seed: counts.astype(np.int64)})
            for name, value in one.items():
                sampled[name][i] = value
        for name, values in sampled.items():
            result[name] = {
                "difference": float(point[name]),
                "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
                "bootstrap_sd": float(values.std(ddof=1)),
                "p_le_0": float(np.mean(values <= 0)),
                "n_components": len(components),
                "n_boot": N_BOOT,
                "seed": seed,
                "joint_same_component_draws": True,
            }
    return result


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)  # exclusive create; never replace an existing result
    finally:
        tmp.unlink(missing_ok=True)


def score_run(
    run: Path, seal_path: Path, attempt_path: Path, output: Path, device: torch.device
) -> dict:
    output, output_npz = require_fresh_outputs(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    (seal, attempt, report, fit_nodes, test_nodes, checkpoints, mean, scale, checkpoint_hashes) = (
        verify_fit_artifacts_before_test(run, seal_path, attempt_path)
    )
    run = run.resolve()
    # First test-role Parquet read occurs only after every sealed checkpoint has
    # passed hash, state-layout, finite-tensor, identity-buffer and model-load checks.
    sequences = [read_test(node, mean, scale) for node in test_nodes]
    component_ids = np.asarray([node["component"] for node in test_nodes], dtype=np.int64)
    order = np.argsort(component_ids, kind="stable")
    require(len(np.unique(component_ids)) == 16, "duplicate held-out component")
    # The frozen cohort is normally already ordered; explicitly sort to bind every
    # row and every bootstrap draw to the deterministic numeric component order.
    test_nodes = [test_nodes[i] for i in order]
    sequences = [sequences[i] for i in order]
    component_ids = component_ids[order]
    row_groups = np.asarray([node["row_group"] for node in test_nodes], dtype=np.int32)
    full_need = np.stack([seq[2] for seq in sequences])
    full_scored = np.stack([seq[3] for seq in sequences])
    full_masks = np.stack([seq[4] for seq in sequences])
    counts = full_masks.sum(axis=1).astype(np.int32)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    targets = np.concatenate([seq[1][seq[4]] for seq in sequences]).astype(np.float32)
    selected_component_index = np.concatenate(
        [np.full(counts[i], i, dtype=np.int16) for i in range(16)]
    )
    selected_rows = np.concatenate([np.flatnonzero(seq[4]).astype(np.int32) for seq in sequences])
    arm_stats: dict[tuple, np.ndarray] = {}
    arm_scores: dict[str, dict] = {}
    arm_predictions: dict[tuple, np.ndarray] = {}
    prediction_hashes = {}
    for seed, optimizer, condition in ARMS:
        key = (seed, optimizer, condition)
        path = checkpoints[key]
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = IndependentAdaptiveRNN(condition)
        expected = model.state_dict()
        require(
            isinstance(state, dict) and set(state) == set(expected), f"state keys differ: {key}"
        )
        for name, value in state.items():
            require(
                torch.is_tensor(value)
                and value.shape == expected[name].shape
                and torch.isfinite(value).all().item(),
                f"invalid checkpoint tensor {key}/{name}",
            )
        model.load_state_dict(state, strict=True)
        require(
            torch.equal(model.mean, torch.zeros_like(model.mean))
            and torch.equal(model.std, torch.ones_like(model.std)),
            f"normalization buffers not identity: {key}",
        )
        pred_parts, per_component = [], []
        model.to(device)
        for x, y, _need, _scored, mask in sequences:
            pred = predict(model, x, device)
            require(np.isfinite(pred[mask]).all(), f"nonfinite raw selected predictions: {key}")
            pred_parts.append(pred[mask])
            per_component.append(moments(y, pred, mask))
        selected_prediction = np.concatenate(pred_parts).astype(np.float32)
        require(
            selected_prediction.shape == targets.shape and np.isfinite(selected_prediction).all(),
            f"selected prediction alignment differs: {key}",
        )
        official = global_wp(
            targets[None], selected_prediction[None], np.ones((1, len(targets)), dtype=bool)
        )
        component_moments = np.stack(per_component)
        pooled = pooled_wp(component_moments, np.ones(16, dtype=np.int64))
        require(
            np.isclose(official["wp"], pooled, atol=2e-7, rtol=0),
            f"official metric mismatch: {key}",
        )
        name = f"seed{seed}_{optimizer}_{condition}"
        arm_stats[key] = component_moments
        arm_predictions[key] = selected_prediction
        prediction_hashes[name] = hashlib.sha256(selected_prediction.tobytes()).hexdigest()
        arm_scores[name] = {
            "seed": seed,
            "optimizer": optimizer,
            "condition": condition,
            "global_wp": float(official["wp"]),
            "target_wp": {"t0": float(official["t0"]), "t1": float(official["t1"])},
            "selected_rows": len(targets),
            "selected_rows_per_component": counts.tolist(),
            "component_ids": component_ids.tolist(),
        }
        del model, state
    contrasts = bootstrap_all(arm_stats, component_ids)
    arm_keys = [f"seed{s}_{o}_{c}" for s, o, c in ARMS]
    arrays = {
        "component_ids": component_ids,
        "row_groups": row_groups,
        "selected_rows_per_component": counts,
        "selected_targets_raw": targets,
        "selected_raw_predictions_by_arm": np.stack([arm_predictions[k] for k in ARMS]),
        "registered_arm_keys": np.asarray(arm_keys),
        "selected_component_index": selected_component_index,
        "selected_row_index_within_component": selected_rows,
        "selected_component_offsets": offsets,
        "need_prediction_masks_by_component": full_need,
        "is_scored_masks_by_component": full_scored,
        "score_masks_by_component": full_masks,
        "seed_values": np.asarray(SEEDS, dtype=np.int64),
        "optimizer_values": np.asarray(OPTIMIZERS),
        "condition_values": np.asarray(CONDITIONS),
    }
    atomic_save_npz(output_npz, arrays)
    result = {
        "kind": "E479 independent adaptive recurrent grader",
        "status": "scored_exposed_components_descriptive_only",
        "metric_id": METRIC_ID,
        "provenance": {
            "seal_name": seal["name"],
            "seal_sha256": attempt["spec_sha256"],
            "attempt_id": attempt["id"],
            "fit_manifest_sha256": sha256_file(run / "fit_manifest.json"),
            "checkpoint_sha256_by_arm": checkpoint_hashes,
            "actor_sha256": ACTOR_SHA,
            "cpu_preflight_v4_sha256": PREFLIGHT_SHA,
        },
        "data": {
            "dataset_sha256": DATA_SHA,
            "cohort_sha256": COHORT_SHA,
            "component_ids_numeric_sorted": component_ids.tolist(),
            "row_groups": row_groups.tolist(),
            "selected_rows_per_component": counts.tolist(),
            "total_selected_rows": len(targets),
        },
        "scorer": {
            "source_sha256": sha256_file(Path(__file__)),
            "device": str(device),
            "forward": "independent reset-after GRU scan; live per-layer tensor alpha",
            "normalization": "independently recomputed from exactly 24 fit row groups; model buffers checked identity",
            "raw_selected_prediction_sha256_by_arm": prediction_hashes,
        },
        "arms": arm_scores,
        "contrasts": contrasts,
        "bootstrap": {
            "resampling_unit": "numeric-sorted test component",
            "n_boot": N_BOOT,
            "seed_per_fit_seed": "fit seed",
            "all_registered_contrasts_share_draws_within_seed": True,
        },
        "export": {
            "path": str(output_npz),
            "sha256": sha256_file(output_npz),
            "arrays": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()
            },
        },
        "scope": "16 arms on 16 previously exposed components; descriptive only; no pooled-seed inference or public forecast",
    }
    require(not output.exists(), "refusing to overwrite result receipt")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp_name, output)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=False)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--self-check-only", action="store_true")
    parser.add_argument(
        "--verify-fit-only",
        action="store_true",
        help="verify receipts/fit rows/normalizer/checkpoints without reading test rows",
    )
    args = parser.parse_args()
    if args.self_check_only:
        print(json.dumps(self_check(), indent=2))
        return
    if not all((args.run_dir, args.seal, args.attempt)):
        parser.error("verification/scoring requires --run-dir --seal --attempt")
    if args.verify_fit_only:
        if args.device != "cpu":
            raise RuntimeError("fit-only receipt verification is CPU-only")
        torch.set_num_threads(1)
        result = verify_fit_artifacts_before_test(args.run_dir, args.seal, args.attempt)
        print(
            json.dumps(
                {
                    "status": "all_fit_receipts_and_checkpoints_verified_test_rows_unread",
                    "arms": len(result[-1]),
                    "test_rows_read": False,
                    "fit_manifest_sha256": sha256_file(
                        args.run_dir.resolve() / "fit_manifest.json"
                    ),
                    "checkpoint_sha256_by_arm": result[-1],
                },
                indent=2,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required unless --verify-fit-only is set")
    if args.device != "cpu":
        raise RuntimeError("E479 grader is CPU-only by resource contract")
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = score_run(args.run_dir, args.seal, args.attempt, args.output, torch.device("cpu"))
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(args.output),
                "arms": 16,
                "contrasts": 28,
                "npz": result["export"]["path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

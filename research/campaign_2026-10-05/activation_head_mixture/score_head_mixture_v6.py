"""Independently score the immutable E477 twelve-arm head-mixture run.

The grader opens the exposed E470 test components only after all fit arms have
finished. It never chooses an arm or changes the fitted checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN_DEFAULT = HERE / "runs/twelve_v1"
SEAL_PATH = HERE / "e477_twelve_fit_v1.sealed.json"
ATTEMPT_PATH = (
    ROOT / "research/campaign_2026-10-04/runner/attempts/e477-head-mixture-twelve-fit-v1.json"
)
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
DATA_PATH = ROOT / "datasets/valid.parquet"
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))

from fit_activation_optimizer import FEATURES, causal_features  # noqa: E402

from wnn.campaign_spec import digest as spec_digest  # noqa: E402
from wnn.campaign_spec import validate_spec  # noqa: E402
from wnn.evaluation import _score_component_moments, paired_component_bootstrap  # noqa: E402
from wnn.gmetric import METRIC_CLIP, METRIC_ID, _centered_moments, global_wp  # noqa: E402

SEEDS = (271828, 271829)
ACTIVATIONS = ("gelu", "channel_split", "same_channel_mix")
DROPOUTS = (0.0, 0.1)
ARMS = tuple(
    (seed, activation, p) for seed in SEEDS for activation in ACTIVATIONS for p in DROPOUTS
)
BOOTSTRAPS = 2_000
BOOTSTRAP_CONTRASTS = (
    ("split_p0 - gelu_p0", "channel_split_p0", "gelu_p0"),
    ("mix_p0 - gelu_p0", "same_channel_mix_p0", "gelu_p0"),
    ("split_p0 - mix_p0", "channel_split_p0", "same_channel_mix_p0"),
    ("split_p01 - gelu_p01", "channel_split_p01", "gelu_p01"),
    ("mix_p01 - gelu_p01", "same_channel_mix_p01", "gelu_p01"),
    ("split_p01 - mix_p01", "channel_split_p01", "same_channel_mix_p01"),
)
TELEMETRY_STEPS = (0, 46, 236, 473)
HIDDEN = 64
FEATURE_COUNT = 42
ROWS_PER_SEQUENCE = 20_000
CHUNK_ROWS = 256
EXPECTED_PARAMS = 61_058


def sha256_file(path: Path) -> str:
    """Hash a file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Fail closed with a readable receipt validation error."""
    if not condition:
        raise ValueError(message)


def verify_sealed_run(run: Path) -> tuple[dict, dict, dict]:
    """Verify campaign seal, successful attempt, inputs, sources and checkpoints."""
    seal = json.loads(SEAL_PATH.read_text())
    attempt = json.loads(ATTEMPT_PATH.read_text())
    checked_seal = validate_spec(seal)
    require(checked_seal == seal, "campaign seal is not in canonical validated form")
    require(seal.get("name") == "e477_head_mixture_twelve_fit_v1", "wrong E477 seal")
    require(attempt.get("id") == "e477-head-mixture-twelve-fit-v1", "wrong runner attempt")
    require(
        attempt.get("status") == "finished" and attempt.get("exit_code") == 0,
        "fit attempt did not finish successfully",
    )
    unit = attempt.get("unit_properties", {})
    require(
        unit.get("Result") == "success" and str(unit.get("ExecMainStatus")) == "0",
        "systemd unit did not report successful completion",
    )
    require(
        attempt.get("failure") is None and attempt.get("teardown_verified") is True,
        "fit attempt failure or teardown is unresolved",
    )
    require(
        attempt.get("kind") == seal.get("kind") == "gpu" and attempt.get("signals") == [],
        "fit attempt kind or signal receipt differs",
    )
    require(attempt.get("spec") == seal, "attempt's sealed spec differs from current seal")
    require(attempt.get("spec_sha256") == spec_digest(seal), "attempt spec digest mismatch")
    require(
        attempt.get("argv", [])[1:] == seal.get("argv", [])[1:],
        "runner command differs from the sealed fit argv",
    )
    sealed_sources = {entry["path"]: entry["sha256"] for entry in seal["code"]}
    runtime_paths = {
        "scripts/run_campaign.py",
        "src/wnn/campaign.py",
        "src/wnn/campaign_local.py",
        "src/wnn/campaign_spec.py",
        "src/wnn/resources.py",
    }
    expected_runtime = {path: sealed_sources[path] for path in runtime_paths}
    require(
        attempt.get("runtime_sha256") == expected_runtime,
        "runner runtime source inventory differs from the sealed version",
    )
    for start_name, end_name in (
        ("initial_memory_events", "final_memory_events"),
        ("initial_parent_memory_events", "final_parent_memory_events"),
    ):
        start, end = attempt.get(start_name, {}), attempt.get(end_name, {})
        require(
            start.keys() == end.keys() and all(start[k] == end[k] for k in start),
            f"memory/OOM event counters changed during run: {start_name}",
        )
    require(
        all(
            attempt.get("final_memory_events", {}).get(key, 0) == 0
            for key in ("oom", "oom_group_kill", "oom_kill")
        ),
        "fit worker reported an OOM",
    )
    require(
        seal.get("artifacts") == [str((run / "fit_manifest.json").relative_to(ROOT))],
        "sealed fit artifact path differs",
    )

    manifest_path = run / "fit_manifest.json"
    artifact_records = attempt.get("artifacts", [])
    require(
        len(artifact_records) == 1
        and artifact_records[0].get("path") == str(manifest_path.relative_to(ROOT)),
        "runner must record exactly the fit manifest artifact",
    )
    record = artifact_records[0]
    require(
        manifest_path.is_file() and manifest_path.stat().st_size == record.get("size"),
        "fit manifest artifact is missing or has wrong size",
    )
    require(
        sha256_file(manifest_path) == record.get("sha256"), "fit manifest artifact hash mismatch"
    )
    require(
        record.get("size") == manifest_path.stat().st_size, "fit manifest artifact size mismatch"
    )
    manifest = json.loads(manifest_path.read_text())

    for entry in seal["code"]:
        path = ROOT / entry["path"]
        require(
            path.is_file() and path.stat().st_size == entry["size"],
            f"sealed source missing/size mismatch: {entry['path']}",
        )
        require(
            sha256_file(path) == entry["sha256"], f"sealed source hash mismatch: {entry['path']}"
        )
    for entry in seal["inputs"]:
        path = ROOT / entry["path"]
        require(
            path.is_file() and path.stat().st_size == entry["size"],
            f"sealed input missing/size mismatch: {entry['path']}",
        )
        require(
            sha256_file(path) == entry["sha256"], f"sealed input hash mismatch: {entry['path']}"
        )

    require(
        manifest.get("status") == "all_12_fit_arms_complete_test_unread",
        "fit manifest is incomplete or does not attest test-unread fitting",
    )
    require(
        manifest.get("fit_only") is True and manifest.get("test_rows_loaded") is False,
        "fit-only data boundary receipt failed",
    )
    require(manifest.get("seeds") == list(SEEDS), "fit seeds differ from fixed design")
    require(manifest.get("activations") == list(ACTIVATIONS), "activation inventory differs")
    require(manifest.get("dropout_probabilities") == list(DROPOUTS), "dropout inventory differs")
    require(
        manifest.get("parameters_per_arm") == EXPECTED_PARAMS, "parameter-count receipt differs"
    )
    require(manifest.get("updates_per_fit") == 474, "fixed update budget differs")
    require(
        manifest.get("architecture")
        == "E474 RNNModel H64 2-layer GRU; vary only head activation and post-activation dropout",
        "architecture receipt differs",
    )
    require(
        manifest.get("batch") == 4 and manifest.get("chunk_rows") == CHUNK_ROWS,
        "fit batch/chunk recipe differs",
    )
    require(
        manifest.get("telemetry_sampling", {}).get("step_indices_zero_based")
        == list(TELEMETRY_STEPS)
        and manifest.get("telemetry_sampling", {}).get("families") == ["inp", "gru", "head"],
        "telemetry recipe differs",
    )
    require(
        manifest.get("optimizer")
        == {
            "name": "AdamW",
            "lr": 0.001,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "schedule": "constant",
            "clip_norm": 1.0,
        },
        "optimizer recipe differs",
    )
    require(
        manifest.get("source_sha256")
        == {
            "fit_activation_head_mixture.py": "2d4b1b103dd31912fea7ab24a63e6ce176e532dca8d0dfa2cdb32aebd9e567d7",
            "tests.py": "876bd7bb397ef2f948fe0fe8f4bcd75b41bc6f67223892b0060ffe5b960cb778",
            "gpu_probe.py": "1275b980942943ef9982992d965587ffdd3dacc21d593e41d02fdf9bf282cdc4",
        },
        "fit manifest source receipt differs",
    )
    require(
        manifest.get("e470_source_sha256")
        == "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
        and manifest.get("e474_source_sha256")
        == "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068",
        "parent model source receipts differ",
    )
    require(
        manifest.get("precision", {}).get("device_math") == "FP32 IEEE"
        and manifest.get("precision", {}).get("autocast") is False,
        "fit precision receipt differs",
    )
    require(
        manifest.get("dataset_sha256")
        == "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
        "dataset identity differs",
    )
    require(manifest.get("cohort_sha256") == sha256_file(COHORT_PATH), "cohort identity differs")
    require(
        manifest.get("planned_contrasts_within_seed")
        == [
            "split_p0 - gelu_p0",
            "mix_p0 - gelu_p0",
            "split_p0 - mix_p0",
            "split_p01 - gelu_p01",
            "mix_p01 - gelu_p01",
            "split_p01 - mix_p01",
            "(mix_p01 - mix_p0) - (gelu_p01 - gelu_p0)",
        ],
        "fit manifest contrast set differs from preregistration",
    )
    return seal, attempt, manifest


class HeadNonlinearity(nn.Module):
    """Independent scorer definition of the parameter-free head intervention."""

    def __init__(self, activation: str, probability: float):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(probability)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        gelu = F.gelu(values, approximate="none")
        if self.activation == "gelu":
            activated = gelu
        elif self.activation == "channel_split":
            activated = torch.cat(
                (gelu[..., :32], F.leaky_relu(values[..., 32:], negative_slope=0.01)), dim=-1
            )
        elif self.activation == "same_channel_mix":
            activated = 0.5 * gelu + 0.5 * F.leaky_relu(values, negative_slope=0.01)
        else:
            raise ValueError(f"unregistered activation {self.activation}")
        return self.dropout(activated)


class ScoringRNN(nn.Module):
    """Standalone E474-shaped scoring model; it does not import actor forward code."""

    def __init__(self, activation: str, probability: float):
        super().__init__()
        self.register_buffer("mean", torch.zeros(FEATURE_COUNT))
        self.register_buffer("std", torch.ones(FEATURE_COUNT))
        self.inp = nn.Linear(FEATURE_COUNT, HIDDEN)
        self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(
            nn.Linear(2 * HIDDEN, HIDDEN),
            HeadNonlinearity(activation, probability),
            nn.Linear(HIDDEN, 2),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        z = F.gelu(self.inp((x - self.mean) / self.std))
        output, state = self.gru(z, state)
        return self.head(torch.cat((output, z), dim=-1)), state


def arm_key(activation: str, probability: float) -> str:
    """Return the fixed checkpoint/manifest arm key."""
    return f"{activation}_p{'0' if probability == 0 else '01'}"


def validate_arms(manifest: dict, run: Path, fit_nodes: list[dict]) -> dict[str, dict]:
    """Validate exact fit order, receipts, and checkpoint identity for all arms."""
    arms = manifest.get("arms", [])
    require(len(arms) == len(SEEDS) * len(ACTIVATIONS) * len(DROPOUTS), "not all 12 arms present")
    by_key = {}
    for arm in arms:
        seed, activation, probability = (
            arm.get("seed"),
            arm.get("activation"),
            arm.get("head_dropout"),
        )
        require(
            seed in SEEDS and activation in ACTIVATIONS and probability in DROPOUTS,
            "unregistered seed, activation, or dropout arm",
        )
        key = arm_key(activation, probability)
        combined = f"seed{seed}_{key}"
        require(combined not in by_key, f"duplicate fit arm {combined}")
        expected_order = [node["row_group"] for node in fit_nodes]
        permutation = np.arange(len(expected_order))
        np.random.default_rng(seed).shuffle(permutation)
        expected_order = [expected_order[int(i)] for i in permutation]
        require(arm.get("arm") == key, f"arm label mismatch: {combined}")
        require(
            arm.get("parameters") == EXPECTED_PARAMS and arm.get("updates") == 474,
            f"fit size/update receipt mismatch: {combined}",
        )
        require(arm.get("fit_row_groups") == expected_order, f"fit order mismatch: {combined}")
        telemetry = arm.get("telemetry", [])
        require(
            [item.get("update_index_zero_based") for item in telemetry] == list(TELEMETRY_STEPS),
            f"telemetry sampling indices mismatch: {combined}",
        )
        for sample in telemetry:
            step = sample["update_index_zero_based"]
            batch_start = (step // 79) * 4
            expected_batch = expected_order[batch_start : batch_start + 4]
            require(
                sample.get("fit_row_groups") == expected_batch,
                f"telemetry batch order mismatch: {combined}, step {step}",
            )
            require(
                sample.get("chunk_start") == (step % 79) * CHUNK_ROWS,
                f"telemetry chunk position mismatch: {combined}, step {step}",
            )
            for family in ("inp", "gru", "head"):
                for metric in ("preclip_gradient_rms", "parameter_update_rms"):
                    value = sample.get(metric, {}).get(family)
                    require(
                        isinstance(value, (int, float)) and np.isfinite(value) and value >= 0,
                        f"invalid telemetry {metric}/{family}: {combined}",
                    )
        checkpoint_name = arm.get("checkpoint")
        require(checkpoint_name == combined + ".pt", f"checkpoint name mismatch: {combined}")
        checkpoint = run / checkpoint_name
        require(checkpoint.is_file(), f"checkpoint missing: {combined}")
        checkpoint_hash = sha256_file(checkpoint)
        require(
            checkpoint_hash == arm.get("checkpoint_sha256"), f"checkpoint hash mismatch: {combined}"
        )
        by_key[combined] = {"record": arm, "path": checkpoint}
    expected_keys = {
        f"seed{seed}_{arm_key(activation, probability)}" for seed, activation, probability in ARMS
    }
    require(set(by_key) == expected_keys, "arm set contains missing/unplanned entries")
    return by_key


def load_test_sequences(cohort: dict, fit_report: dict) -> tuple[list[dict], list[dict]]:
    """Load source cohort and fit-only normalization; return test nodes and fit nodes."""
    source = json.loads(COHORT_PATH.read_text())
    copied = cohort.get("cohort")
    nodes = source["cohort"]
    fit_nodes = [node for node in nodes if node["role"] == "fit"]
    test_nodes = [node for node in nodes if node["role"] == "test"]
    require(len(fit_nodes) == 24 and len(test_nodes) == 16, "expected 24 fit and 16 test sequences")
    require(copied == fit_nodes, "fit-only cohort copy differs from the canonical 24 fit groups")
    require(
        len(copied) == 24 and all(node.get("role") == "fit" for node in copied),
        "fit-only cohort receipt contains non-fit groups or the wrong number of groups",
    )
    require(
        len({node["component"] for node in test_nodes}) == 16, "test components are not distinct"
    )
    require(
        not (
            {node["component"] for node in fit_nodes} & {node["component"] for node in test_nodes}
        ),
        "fit/test components overlap",
    )
    norm = cohort.get("normalization", {})
    mean = np.asarray(norm.get("fit_mean"), dtype=np.float32)
    scale = np.asarray(norm.get("fit_scale"), dtype=np.float32)
    require(
        mean.shape == (FEATURE_COUNT,) and scale.shape == (FEATURE_COUNT,),
        "fit-only normalization shape mismatch",
    )
    require(
        np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(),
        "invalid fit-only normalization",
    )
    return test_nodes, fit_nodes


def verify_fit_normalization(fit_nodes: list[dict], cohort: dict) -> None:
    """Recompute the saved normalizer from fit groups only, before reading test labels."""
    parquet = pq.ParquetFile(DATA_PATH)
    fit_features = []
    for node in fit_nodes:
        table = parquet.read_row_group(
            int(node["row_group"]), columns=list(FEATURES), use_threads=False
        )
        raw = np.column_stack([table[name].to_numpy() for name in FEATURES]).astype(np.float32)
        require(np.isfinite(raw).all(), f"nonfinite raw fit features in {node['row_group']}")
        features = causal_features(raw)
        require(
            features.shape == (ROWS_PER_SEQUENCE, FEATURE_COUNT) and np.isfinite(features).all(),
            f"invalid fit causal features in {node['row_group']}",
        )
        fit_features.append(features)
    matrix = np.stack(fit_features)
    mean = matrix.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(matrix.std(axis=(0, 1), dtype=np.float64), 0.001).astype(np.float32)
    saved_mean = np.asarray(cohort["normalization"]["fit_mean"], dtype=np.float32)
    saved_scale = np.asarray(cohort["normalization"]["fit_scale"], dtype=np.float32)
    require(
        np.array_equal(mean, saved_mean) and np.array_equal(scale, saved_scale),
        "saved input normalization is not exactly derived from fit groups",
    )


def read_test_sequence(
    node: dict, mean: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one component, build causal features, and validate raw selected labels."""
    table = pq.ParquetFile(DATA_PATH).read_row_group(
        int(node["row_group"]),
        columns=[*FEATURES, "t0", "t1", "need_prediction", "is_scored"],
        use_threads=False,
    )
    require(table.num_rows == ROWS_PER_SEQUENCE, f"row count mismatch for {node['row_group']}")
    raw_features = np.column_stack([table[name].to_numpy() for name in FEATURES]).astype(np.float32)
    require(
        np.isfinite(raw_features).all(), f"nonfinite raw features in row group {node['row_group']}"
    )
    x = causal_features(raw_features)
    require(
        x.shape == (ROWS_PER_SEQUENCE, FEATURE_COUNT) and np.isfinite(x).all(),
        f"causal feature shape/finite check failed for {node['row_group']}",
    )
    x = np.clip((x - mean) / scale, -8, 8).astype(np.float32)
    raw_y = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")]).astype(np.float32)
    mask = table["need_prediction"].to_numpy().astype(bool)
    mask &= table["is_scored"].to_numpy().astype(bool)
    require(mask.any(), f"no selected rows in test row group {node['row_group']}")
    require(np.isfinite(raw_y[mask]).all(), f"nonfinite selected raw labels in {node['row_group']}")
    # Keep full-row labels aligned to predictions and the full score mask; only
    # select and clip at the metric/export boundary.
    return x, raw_y, mask


def predict_one(model: ScoringRNN, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict with eval-mode dropout disabled and carried state across chunks."""
    model.eval()
    state = None
    pieces = []
    with torch.no_grad():
        for start in range(0, len(x), CHUNK_ROWS):
            chunk = torch.from_numpy(x[start : start + CHUNK_ROWS][None]).to(device=device)
            prediction, state = model(chunk, state)
            if not torch.isfinite(prediction).all():
                raise ValueError(f"nonfinite predictions before metric clipping in chunk {start}")
            pieces.append(prediction[0].cpu().numpy())
    output = np.concatenate(pieces, axis=0).astype(np.float32)
    require(
        output.shape == (ROWS_PER_SEQUENCE, 2) and np.isfinite(output).all(),
        "invalid full-sequence predictions",
    )
    return output


def score_sequence_moments(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return official centered moments for one component, checking raw values first."""
    require(
        y.shape == p.shape and y.shape == (ROWS_PER_SEQUENCE, 2),
        "sequence labels/predictions misalign",
    )
    require(mask.shape == (ROWS_PER_SEQUENCE,), "score mask shape mismatch")
    require(
        np.isfinite(y[mask]).all() and np.isfinite(p[mask]).all(),
        "selected raw targets or predictions are nonfinite",
    )
    clipped_y = np.clip(y[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    clipped_p = np.clip(p[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    return np.stack([_centered_moments(clipped_y[:, j], clipped_p[:, j]) for j in range(2)])


def score_target_moments(moments: np.ndarray, component_counts: np.ndarray, target: int) -> float:
    """Compute one target's component-pooled WP from stable centered moments."""
    weight = component_counts * moments[:, target, 0]
    total = weight.sum()
    if total <= 0:
        return 0.0
    active = np.flatnonzero(weight > 0)
    anchor_y, anchor_p = moments[active[0], target, 1:3]
    dy, dp = moments[:, target, 1] - anchor_y, moments[:, target, 2] - anchor_p
    mean_y = np.sum(weight * dy) / total
    mean_p = np.sum(weight * dp) / total
    vy = np.sum(component_counts * moments[:, target, 3] + weight * (dy - mean_y) ** 2)
    vp = np.sum(component_counts * moments[:, target, 4] + weight * (dp - mean_p) ** 2)
    cov = np.sum(component_counts * moments[:, target, 5] + weight * (dy - mean_y) * (dp - mean_p))
    if np.sqrt(vy / total) <= 1e-8 or np.sqrt(vp / total) <= 1e-8:
        return 0.0
    return float(np.clip(cov / np.sqrt(vy * vp), -1.0, 1.0))


def paired_target_contrast(
    arms: dict[str, np.ndarray],
    candidate: str,
    reference: str,
    target: int,
    n_boot: int,
    seed: int,
) -> dict:
    """Paired component bootstrap for one target, common component resamples."""
    a, b = arms[candidate], arms[reference]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    component_count = len(a)
    ones = np.ones(component_count, dtype=np.int64)
    point = score_target_moments(a, ones, target) - score_target_moments(b, ones, target)
    for index in range(n_boot):
        sampled = rng.integers(0, component_count, component_count)
        counts = np.bincount(sampled, minlength=component_count)
        deltas[index] = score_target_moments(a, counts, target) - score_target_moments(
            b, counts, target
        )
    return {
        "mean_diff": point,
        "ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
        "bootstrap_sd": float(deltas.std(ddof=1)),
        "p_le_0": float(np.mean(deltas <= 0)),
        "n_components": component_count,
        "n_boot": n_boot,
        "seed": seed,
    }


def paired_interaction(
    arms: dict[str, np.ndarray],
    high_a: str,
    low_a: str,
    high_b: str,
    low_b: str,
    n_boot: int,
    seed: int,
    target: int | None = None,
) -> dict:
    """Paired joint component bootstrap that recomputes all four pooled scores."""
    n = len(arms[high_a])
    require(all(len(arms[key]) == n for key in (low_a, high_b, low_b)), "interaction arms misalign")

    def score(key: str, counts: np.ndarray) -> float:
        return (
            _score_component_moments(arms[key], counts)
            if target is None
            else score_target_moments(arms[key], counts, target)
        )

    ones = np.ones(n, dtype=np.int64)
    point = (score(high_a, ones) - score(low_a, ones)) - (score(high_b, ones) - score(low_b, ones))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        deltas[index] = (score(high_a, counts) - score(low_a, counts)) - (
            score(high_b, counts) - score(low_b, counts)
        )
    return {
        "mean_diff": point,
        "ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
        "bootstrap_sd": float(deltas.std(ddof=1)),
        "p_le_0": float(np.mean(deltas <= 0)),
        "n_components": n,
        "n_boot": n_boot,
        "seed": seed,
        "joint_resample": True,
    }


def write_paired_export(
    path: Path,
    arm_keys: list[str],
    component_ids: list[int],
    row_groups: list[int],
    counts: list[int],
    masks: np.ndarray,
    raw_targets: np.ndarray,
    predictions: np.ndarray,
    row_indices: np.ndarray,
    sequence_indices: np.ndarray,
) -> str:
    """Atomically create the full paired raw selected-row export without overwrite."""
    expected_rows = int(sum(counts))
    require(masks.shape == (16, ROWS_PER_SEQUENCE), "paired export full-mask shape mismatch")
    require(raw_targets.shape == (expected_rows, 2), "paired export raw target shape mismatch")
    require(predictions.shape == (12, expected_rows, 2), "paired export prediction shape mismatch")
    require(row_indices.shape == (expected_rows,), "paired export row-index shape mismatch")
    require(sequence_indices.shape == (expected_rows,), "paired export sequence-index shape mismatch")
    require(len(arm_keys) == 12 and len(component_ids) == len(row_groups) == len(counts) == 16,
            "paired export arm/component metadata mismatch")
    require(np.isfinite(raw_targets).all(), "paired export has nonfinite raw selected targets")
    require(np.isfinite(predictions).all(), "paired export has nonfinite raw predictions")
    require(np.array_equal(masks.sum(axis=1), np.asarray(counts)), "paired export counts disagree with masks")
    require(np.array_equal(sequence_indices, np.repeat(np.arange(16), counts)),
            "paired export selected sequence indices are misaligned")
    require(np.array_equal(row_indices, np.concatenate([np.flatnonzero(mask) for mask in masks])),
            "paired export selected row indices are misaligned")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite paired export: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                arm_keys=np.asarray(arm_keys),
                component_ids=np.asarray(component_ids, dtype=np.int64),
                row_groups=np.asarray(row_groups, dtype=np.int64),
                selected_counts=np.asarray(counts, dtype=np.int64),
                full_masks=np.asarray(masks, dtype=np.bool_),
                raw_targets=np.asarray(raw_targets, dtype=np.float32),
                raw_predictions=np.asarray(predictions, dtype=np.float32),
                selected_row_indices=np.asarray(row_indices, dtype=np.int32),
                selected_sequence_indices=np.asarray(sequence_indices, dtype=np.int16),
            )
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes the completed file atomically and fails if another
        # process has already created the destination.
        os.link(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return sha256_file(path)


def evaluate(run: Path, device: torch.device, paired_export_path: Path) -> dict:
    """Validate receipts, score all twelve checkpoints, and compute fixed contrasts."""
    seal, attempt, manifest = verify_sealed_run(run)
    cohort_receipt = json.loads((run / "fit_only_cohort.json").read_text())
    # load_test_sequences validates this fit-only copy against the canonical
    # 24 fit-role rows, while sourcing test16 separately from the full cohort.
    test_nodes, fit_nodes = load_test_sequences(cohort_receipt, manifest)
    verify_fit_normalization(fit_nodes, cohort_receipt)
    by_key = validate_arms(manifest, run, fit_nodes)
    component_ids = [node["component"] for node in test_nodes]
    sequences = [
        read_test_sequence(
            node,
            np.asarray(cohort_receipt["normalization"]["fit_mean"], np.float32),
            np.asarray(cohort_receipt["normalization"]["fit_scale"], np.float32),
        )
        for node in test_nodes
    ]
    full_masks = np.stack([mask for _, _, mask in sequences])
    raw_selected_targets = np.concatenate([y[mask] for _, y, mask in sequences]).astype(np.float32)
    all_targets = np.concatenate(
        [np.clip(y[mask], -METRIC_CLIP, METRIC_CLIP) for _, y, mask in sequences]
    )
    counts = [int(mask.sum()) for _, _, mask in sequences]
    score_mask = np.ones(len(all_targets), dtype=bool)
    outputs: dict[str, dict] = {}
    moments_by_seed: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}
    raw_output_sha = {}
    selected_predictions_by_arm = []
    arm_keys = []
    with torch.no_grad():
        for seed, activation, probability in ARMS:
            key = f"seed{seed}_{arm_key(activation, probability)}"
            checkpoint = by_key[key]["path"]
            model = ScoringRNN(activation, probability)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            require(isinstance(state, dict), f"checkpoint is not a state dict: {key}")
            require(set(state) == set(model.state_dict()), f"checkpoint keys mismatch: {key}")
            for name, tensor in state.items():
                require(
                    torch.is_tensor(tensor)
                    and tuple(tensor.shape) == tuple(model.state_dict()[name].shape),
                    f"checkpoint tensor shape mismatch {key}/{name}",
                )
                require(
                    torch.isfinite(tensor).all().item(), f"nonfinite checkpoint tensor {key}/{name}"
                )
            require(
                torch.equal(state["mean"], torch.zeros(FEATURE_COUNT))
                and torch.equal(state["std"], torch.ones(FEATURE_COUNT)),
                f"checkpoint normalization buffers are not identity: {key}",
            )
            model.load_state_dict(state, strict=True)
            require(
                sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMS,
                f"scorer parameter count mismatch: {key}",
            )
            model.to(device).eval()
            require(
                torch.equal(model.mean.cpu(), torch.zeros(FEATURE_COUNT))
                and torch.equal(model.std.cpu(), torch.ones(FEATURE_COUNT)),
                f"loaded model normalization buffers are not identity: {key}",
            )
            predictions = []
            moment_rows = []
            for x, y, mask in sequences:
                pred = predict_one(model, x, device)
                predictions.append(pred[mask])
                moment_rows.append(score_sequence_moments(y, pred, mask))
            selected_predictions = np.concatenate(predictions).astype(np.float32)
            require(
                selected_predictions.shape == all_targets.shape, f"selected-row mismatch: {key}"
            )
            require(
                np.isfinite(selected_predictions).all(), f"nonfinite raw scored predictions: {key}"
            )
            official = global_wp(all_targets[None], selected_predictions[None], score_mask[None])
            moments = np.stack(moment_rows)
            pooled = _score_component_moments(moments, np.ones(len(moments), dtype=np.int64))
            require(
                np.isclose(official["wp"], pooled, atol=2e-7, rtol=0),
                f"centered component pool differs from official global score: {key}",
            )
            outputs[key] = {
                "seed": seed,
                "activation": activation,
                "head_dropout": probability,
                "global_wp": float(official["wp"]),
                "target_wp": {"t0": float(official["t0"]), "t1": float(official["t1"])},
                "selected_rows": int(len(all_targets)),
                "scored_rows_by_component": counts,
                "component_ids": component_ids,
            }
            selected_predictions_by_arm.append(selected_predictions.copy())
            arm_keys.append(key)
            moments_by_seed[seed][arm_key(activation, probability)] = moments
            raw_output_sha[key] = hashlib.sha256(selected_predictions.tobytes()).hexdigest()
            del model, state

    selected_row_indices = np.concatenate([np.flatnonzero(mask) for mask in full_masks]).astype(
        np.int32
    )
    selected_sequence_indices = np.repeat(np.arange(len(test_nodes)), counts).astype(np.int16)
    export_sha256 = write_paired_export(
        paired_export_path,
        arm_keys,
        component_ids,
        [int(node["row_group"]) for node in test_nodes],
        counts,
        full_masks,
        raw_selected_targets,
        np.stack(selected_predictions_by_arm),
        selected_row_indices,
        selected_sequence_indices,
    )

    contrasts: dict[str, dict] = {}
    bootstrap_order = np.argsort(np.asarray(component_ids), kind="stable")
    bootstrap_component_ids = np.asarray(component_ids)[bootstrap_order]
    for seed in SEEDS:
        # Match the sorted component order used internally by
        # paired_component_bootstrap so every contrast shares the same draws.
        arms = {
            name: moments[bootstrap_order]
            for name, moments in moments_by_seed[seed].items()
        }
        for name, candidate, reference in BOOTSTRAP_CONTRASTS:
            result = paired_component_bootstrap(
                arms[candidate],
                arms[reference],
                bootstrap_component_ids,
                n_boot=BOOTSTRAPS,
                seed=seed,
                centered=True,
            )
            result["candidate"] = candidate
            result["reference"] = reference
            result["family"] = "preregistered_primary"
            result["target_wp"] = {
                target: paired_target_contrast(arms, candidate, reference, target, BOOTSTRAPS, seed)
                for target in (0, 1)
            }
            contrasts[f"seed{seed}:{name}"] = result
        # Useful regularization baseline contrast: absent from the preregistered
        # seven, so it is retained and clearly marked secondary.
        gelu_dropout = paired_component_bootstrap(
            arms["gelu_p01"],
            arms["gelu_p0"],
            bootstrap_component_ids,
            n_boot=BOOTSTRAPS,
            seed=seed,
            centered=True,
        )
        gelu_dropout.update(
            {"candidate": "gelu_p01", "reference": "gelu_p0", "family": "secondary_descriptive"}
        )
        gelu_dropout["target_wp"] = {
            target: paired_target_contrast(arms, "gelu_p01", "gelu_p0", target, BOOTSTRAPS, seed)
            for target in (0, 1)
        }
        contrasts[f"seed{seed}:gelu_dropout_effect_secondary"] = gelu_dropout
        primary = paired_interaction(
            arms,
            "same_channel_mix_p01",
            "same_channel_mix_p0",
            "gelu_p01",
            "gelu_p0",
            BOOTSTRAPS,
            seed,
        )
        primary.update(
            {
                "family": "preregistered_primary",
                "formula": "(mix,p=0.1 - mix,p=0) - (GELU,p=0.1 - GELU,p=0)",
            }
        )
        primary["target_wp"] = {
            target: paired_interaction(
                arms,
                "same_channel_mix_p01",
                "same_channel_mix_p0",
                "gelu_p01",
                "gelu_p0",
                BOOTSTRAPS,
                seed,
                target,
            )
            for target in (0, 1)
        }
        contrasts[f"seed{seed}:mixture_x_dropout"] = primary
        secondary = paired_interaction(
            arms, "channel_split_p01", "channel_split_p0", "gelu_p01", "gelu_p0", BOOTSTRAPS, seed
        )
        secondary.update(
            {
                "family": "secondary_descriptive",
                "formula": "(split,p=0.1 - split,p=0) - (GELU,p=0.1 - GELU,p=0)",
            }
        )
        secondary["target_wp"] = {
            target: paired_interaction(
                arms,
                "channel_split_p01",
                "channel_split_p0",
                "gelu_p01",
                "gelu_p0",
                BOOTSTRAPS,
                seed,
                target,
            )
            for target in (0, 1)
        }
        contrasts[f"seed{seed}:split_x_dropout_secondary"] = secondary

    return {
        "kind": "E477 independent head-mixture scorer",
        "status": "scored_exposed_components_descriptive_only",
        "metric_id": METRIC_ID,
        "seal": {
            "name": seal["name"],
            "spec_sha256": attempt["spec_sha256"],
            "fit_attempt": attempt["id"],
            "fit_manifest_sha256": sha256_file(run / "fit_manifest.json"),
        },
        "data": {
            "validation_sha256": next(
                item["sha256"]
                for item in seal["inputs"]
                if item["path"] == "datasets/valid.parquet"
            ),
            "cohort_sha256": sha256_file(COHORT_PATH),
            "component_ids": component_ids,
            "sequence_row_groups": [n["row_group"] for n in test_nodes],
            "selected_rows_by_component": counts,
        },
        "scorer": {
            "source_sha256": sha256_file(Path(__file__)),
            "device": str(device),
            "model": "independent E474 H64 two-layer GRU forward; strict checkpoint load",
            "dropout": "model.eval() disables the post-head activation dropout",
            "raw_prediction_hashes": raw_output_sha,
        },
        "paired_export": {
            "path": str(paired_export_path.relative_to(ROOT)),
            "sha256": export_sha256,
            "arrays": [
                "arm_keys", "component_ids", "row_groups", "selected_counts", "full_masks",
                "raw_targets", "raw_predictions", "selected_row_indices",
                "selected_sequence_indices",
            ],
            "raw_values_checked_finite_before_metric_clipping": True,
        },
        "arms": outputs,
        "contrasts": contrasts,
        "inference_scope": "two fixed optimization seeds; 16 previously exposed components; no seed-generalization inference or winner selection",
        "bootstrap": {
            "resampling_unit": "component",
            "n_boot": BOOTSTRAPS,
            "seed_per_fit_seed": "fit seed",
            "interaction": "joint paired resampling; recompute all pooled arm scores",
        },
        "contrast_inventory": {
            "preregistered_primary_per_seed": 7,
            "secondary_gelu_dropout_effect_per_seed": 1,
            "secondary_split_interaction_per_seed": 1,
            "total_paired_results": 18,
            "note": "PREREG.md specifies 7 primaries; both requested dropout deltas are included, with split interaction and GELU dropout effect labeled secondary.",
        },
    }


def main() -> None:
    """Run the source-bound scorer without overwriting prior output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    paired_export_path = args.output.with_name("RESULT_paired_components.npz")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite scorer output: {args.output}")
    if paired_export_path.exists():
        raise FileExistsError(f"refusing to overwrite paired export: {paired_export_path}")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        from wnn.resources import configure_torch

        configure_torch()
        # Match the full IEEE FP32 policy recorded by all E477 fits.
        for owner, attribute in (
            (torch.backends, "fp32_precision"),
            (torch.backends.cuda.matmul, "fp32_precision"),
            (torch.backends.cudnn, "fp32_precision"),
            (torch.backends.cudnn.rnn, "fp32_precision"),
        ):
            if not hasattr(owner, attribute):
                raise RuntimeError("full IEEE FP32 backend controls unavailable")
            setattr(owner, attribute, "ieee")
    torch.set_num_threads(1)
    result = evaluate(args.run_dir, torch.device(args.device), paired_export_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "arms": len(result["arms"]),
                "contrasts": len(result["contrasts"]),
                "scorer_sha256": result["scorer"]["source_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

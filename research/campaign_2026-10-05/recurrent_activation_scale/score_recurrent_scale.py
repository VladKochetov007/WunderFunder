"""Independently score all six E478 recurrent-candidate fits.

This grader reconstructs the registered two-layer RNN parameter layout and
implements the reset-after-bias GRU recurrence separately from the fit actor.
Scores are exploratory on previously exposed components.
"""

from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN_DEFAULT = HERE / "runs/six_v1"
SEAL_PATH = HERE / "e478_six_fit_v1.sealed.json"
ATTEMPT_PATH = (
    ROOT / "research/campaign_2026-10-04/runner/attempts/e478-recurrent-scale-six-fit-v1.json"
)
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
DATA_PATH = ROOT / "datasets/valid.parquet"
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))

from fit_activation_optimizer import FEATURES, causal_features  # noqa: E402

from wnn.campaign_spec import digest as spec_digest  # noqa: E402
from wnn.evaluation import _score_component_moments, paired_component_bootstrap  # noqa: E402
from wnn.gmetric import METRIC_CLIP, METRIC_ID, _centered_moments, global_wp  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402

SEEDS = (271828, 271829, 271830)
ALPHAS = (0.0, 0.1)
HIDDEN = 64
INPUTS = 42
N_PARAMS = 61_058
N_UPDATES = 474
CHUNK_ROWS = 256
N_ROWS = 20_000
BOOTSTRAPS = 2_000
TELEMETRY_INDICES = (0, 46, 236, 473)
FIT_SOURCE = "3ccc2fcd0601f6a2d239bd47799561834ffd6e42517b2b2af603e03cd3b08d20"
TEST_SOURCE = "1a20253e28df94cf7debca215bc593989a4ea6c5b456c2d8a584b72d1f298d91"
PROBE_SOURCE = "e11b1dce6f21e24beb32174758af950e7642d7b1ae448fa60ec3e6763da1d75a"
E470_SOURCE = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E471_SOURCE = "998bceb024b5c29e81e73efc5d0ebe434d53c7356cfcb947f7ca36dd625426cf"
E474_SOURCE = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
EXPECTED_DATA = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
EXPECTED_COHORT = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"


def sha256_file(path: Path) -> str:
    """Hash a file with a fixed, bounded buffer."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    """Stop on a failed provenance or shape invariant."""
    if not condition:
        raise ValueError(message)


@torch.jit.script
def candidate_scan_independent(
    x: Tensor,
    h0: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor]:
    """Independent PyTorch GRU scan: reset-after hidden affine, r/z/n order."""
    input_affine = F.linear(x, weight_ih, bias_ih)
    output_rows = torch.jit.annotate(list[Tensor], [])
    h = h0
    for t in range(x.size(1)):
        ir, iz, inn = input_affine[:, t, :].chunk(3, dim=1)
        hr, hz, hn = F.linear(h, weight_hh, bias_hh).chunk(3, dim=1)
        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        u = torch.tanh(inn + reset * hn)
        proposed = u + alpha * torch.sin(math.pi * u)
        h = (1.0 - update) * proposed + update * h
        output_rows.append(h)
    return torch.stack(output_rows, dim=1), h


class IndependentScaleRNN(nn.Module):
    """Standalone scorer model with RNNModel-compatible state-dict layout."""

    def __init__(self, alpha: float):
        super().__init__()
        if alpha not in ALPHAS:
            raise ValueError(f"unregistered alpha {alpha}")
        self.alpha = float(alpha)
        self.register_buffer("mean", torch.zeros(INPUTS))
        self.register_buffer("std", torch.ones(INPUTS))
        self.inp = nn.Linear(INPUTS, HIDDEN)
        # This is only the parameter container; forward uses the independent scan.
        self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 2))

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != INPUTS:
            raise ValueError("input must have shape [batch,time,42]")
        b = x.shape[0]
        if state is None:
            state = x.new_zeros((2, b, HIDDEN))
        elif state.shape != (2, b, HIDDEN):
            raise ValueError("state must have shape [2,batch,64]")
        z = F.gelu(self.inp((x - self.mean) / self.std))
        layer_input = z
        final_states = []
        for layer in range(2):
            layer_input, h = candidate_scan_independent(
                layer_input,
                state[layer],
                getattr(self.gru, f"weight_ih_l{layer}"),
                getattr(self.gru, f"weight_hh_l{layer}"),
                getattr(self.gru, f"bias_ih_l{layer}"),
                getattr(self.gru, f"bias_hh_l{layer}"),
                self.alpha,
            )
            final_states.append(h)
        next_state = torch.stack(final_states, dim=0)
        prediction = self.head(torch.cat((layer_input, z), dim=-1))
        return prediction, next_state


def verify_receipts(run: Path) -> tuple[dict, dict, dict]:
    """Verify the single declared runner artifact, exact code/input seal and all fits."""
    seal = json.loads(SEAL_PATH.read_text())
    attempt = json.loads(ATTEMPT_PATH.read_text())
    require(seal.get("name") == "e478_recurrent_scale_six_fit_v1", "wrong E478 fit seal")
    require(attempt.get("id") == "e478-recurrent-scale-six-fit-v1", "wrong runner attempt")
    require(
        attempt.get("status") == "finished"
        and attempt.get("exit_code") == 0
        and attempt.get("failure") is None
        and attempt.get("teardown_verified") is True,
        "fit runner attempt is incomplete, failed, or not torn down",
    )
    require(
        attempt.get("kind") == "gpu" and attempt.get("signals") == [],
        "fit runner provider kind or signal receipt differs",
    )
    require(
        attempt.get("spec") == seal and attempt.get("spec_sha256") == spec_digest(seal),
        "attempt does not bind the current sealed specification",
    )
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
    require(
        attempt.get("runtime_sha256") == {path: sealed_sources[path] for path in runtime_paths},
        "runner runtime source inventory differs from sealed code",
    )
    for start_name, end_name in (
        ("initial_memory_events", "final_memory_events"),
        ("initial_parent_memory_events", "final_parent_memory_events"),
    ):
        start, end = attempt.get(start_name, {}), attempt.get(end_name, {})
        require(
            start.keys() == end.keys() and all(start[k] == end[k] for k in start),
            f"runner memory/OOM event counters changed: {start_name}",
        )
    require(
        all(
            attempt.get("final_memory_events", {}).get(key, 0) == 0
            for key in ("oom", "oom_group_kill", "oom_kill")
        ),
        "fit worker reports an OOM",
    )
    fit_path = run / "fit_manifest.json"
    require(
        seal.get("artifacts") == [str(fit_path.relative_to(ROOT))],
        "seal must declare only the fit manifest artifact",
    )
    artifact = attempt.get("artifacts", [])
    require(
        len(artifact) == 1 and artifact[0].get("path") == str(fit_path.relative_to(ROOT)),
        "runner does not attest exactly the fit manifest",
    )
    require(
        fit_path.is_file()
        and fit_path.stat().st_size == artifact[0].get("size")
        and sha256_file(fit_path) == artifact[0].get("sha256"),
        "fit manifest differs from the runner receipt",
    )
    for entry in seal["code"] + seal["inputs"]:
        path = ROOT / entry["path"]
        require(
            path.is_file() and path.stat().st_size == entry["size"],
            f"sealed dependency missing/size mismatch: {entry['path']}",
        )
        require(
            sha256_file(path) == entry["sha256"],
            f"sealed dependency hash mismatch: {entry['path']}",
        )
    report = json.loads(fit_path.read_text())
    require(
        report.get("status") == "all_6_fit_arms_complete_test_unread"
        and report.get("fit_only") is True
        and report.get("test_rows_loaded") is False,
        "fit manifest does not attest a complete fit-only run",
    )
    require(
        report.get("seeds") == list(SEEDS) and report.get("alphas") == list(ALPHAS),
        "fit seeds/alpha grid differs",
    )
    require(
        report.get("parameters_per_arm") == N_PARAMS and report.get("updates_per_fit") == N_UPDATES,
        "fit parameter count or update count differs",
    )
    require(
        report.get("dataset_sha256") == EXPECTED_DATA
        and report.get("cohort_sha256") == EXPECTED_COHORT,
        "fit data/cohort receipt differs",
    )
    require(
        report.get("parent_source_sha256")
        == {"e470": E470_SOURCE, "e471_scan": E471_SOURCE, "e474": E474_SOURCE},
        "parent model source receipt differs",
    )
    require(
        report.get("source_sha256")
        == {
            "fit_recurrent_activation_scale.py": FIT_SOURCE,
            "tests.py": TEST_SOURCE,
            "gpu_probe.py": PROBE_SOURCE,
        },
        "actor fit source receipt differs",
    )
    require(
        report.get("optimizer", {}).get("name") == "MuonAdamW"
        and report["optimizer"].get("muon_matrices")
        == sorted(["gru.weight_ih_l0", "gru.weight_hh_l0", "gru.weight_ih_l1", "gru.weight_hh_l1"])
        and report["optimizer"].get("muon_blocks_per_matrix") == 3
        and report["optimizer"].get("adamw_for") == "all other trainable parameters"
        and report["optimizer"].get("lr") == 0.001
        and report["optimizer"].get("weight_decay") == 0.01
        and report["optimizer"].get("betas") == [0.9, 0.999]
        and report["optimizer"].get("schedule") == "constant"
        and report["optimizer"].get("clip_norm") == 1.0,
        "optimizer partition receipt differs",
    )
    require(
        report.get("batch") == 4
        and report.get("chunk_rows") == CHUNK_ROWS
        and report.get("architecture")
        == "42->GELU linear projection->two H64 GRU layers->concat recurrent/embedding->GELU head",
        "architecture or training batch/chunk recipe differs",
    )
    require(
        report.get("telemetry_sampling", {}).get("step_indices_zero_based")
        == list(TELEMETRY_INDICES)
        and report.get("telemetry_sampling", {}).get("families") == ["inp", "gru", "head"],
        "telemetry sampling recipe differs",
    )
    require(
        report.get("precision", {}).get("device_math") == "FP32 IEEE"
        and report["precision"].get("autocast") is False,
        "fit precision receipt differs",
    )
    return seal, attempt, report


def get_cohort() -> tuple[list[dict], list[dict]]:
    """Read exactly the frozen source cohort, without depending on extra artifacts."""
    require(sha256_file(COHORT_PATH) == EXPECTED_COHORT, "source cohort identity differs")
    nodes = json.loads(COHORT_PATH.read_text())["cohort"]
    fit_nodes = [n for n in nodes if n["role"] == "fit"]
    test_nodes = [n for n in nodes if n["role"] == "test"]
    require(len(fit_nodes) == 24 and len(test_nodes) == 16, "expected 24 fit and 16 test sequences")
    fit_components = {n["component"] for n in fit_nodes}
    test_components = [n["component"] for n in test_nodes]
    require(
        not (fit_components & set(test_components)) and len(set(test_components)) == 16,
        "fit/test component identity overlap or duplicates",
    )
    require(len({n["row_group"] for n in nodes}) == 40, "duplicate frozen cohort row groups")
    return fit_nodes, test_nodes


def verify_arms(report: dict, run: Path, fit_nodes: list[dict]) -> dict[str, dict]:
    """Verify all checkpoint, true fit-order, and sampled telemetry receipts."""
    records = report.get("arms", [])
    require(len(records) == 6, "all six fixed fits are required")
    groups = [n["row_group"] for n in fit_nodes]
    by_key = {}
    for item in records:
        seed, alpha = item.get("seed"), item.get("alpha")
        require(seed in SEEDS and alpha in ALPHAS, "unplanned seed or candidate alpha")
        key = f"seed{seed}_alpha{alpha:.1f}"
        require(key not in by_key, f"duplicate fit cell {key}")
        permutation = np.arange(len(groups))
        np.random.default_rng(seed).shuffle(permutation)
        order = [groups[int(i)] for i in permutation]
        require(item.get("fit_row_groups") == order, f"actual shuffled fit order differs: {key}")
        require(
            item.get("parameters") == N_PARAMS and item.get("updates") == N_UPDATES,
            f"parameter or update count mismatch: {key}",
        )
        require(
            item.get("optimizer") == "MuonAdamW; Muon on four GRU matrices, AdamW on others",
            f"optimizer mismatch: {key}",
        )
        require(
            item.get("muon_matrices")
            == sorted(
                ["gru.weight_ih_l0", "gru.weight_hh_l0", "gru.weight_ih_l1", "gru.weight_hh_l1"]
            )
            and item.get("muon_gate_blocks") == 3,
            f"Muon partition mismatch: {key}",
        )
        telemetry = item.get("telemetry", [])
        require(
            [r.get("update_index_zero_based") for r in telemetry] == list(TELEMETRY_INDICES),
            f"telemetry indices mismatch: {key}",
        )
        for sample in telemetry:
            step = sample["update_index_zero_based"]
            expected_batch = order[(step // 79) * 4 : (step // 79) * 4 + 4]
            require(
                sample.get("fit_row_groups") == expected_batch
                and sample.get("chunk_start") == (step % 79) * CHUNK_ROWS,
                f"telemetry fit order/chunk mismatch: {key} step {step}",
            )
            for family in ("inp", "gru", "head"):
                for kind in ("preclip_gradient_rms", "parameter_update_rms"):
                    value = sample.get(kind, {}).get(family)
                    require(
                        isinstance(value, (int, float)) and np.isfinite(value) and value >= 0,
                        f"invalid {kind}/{family} telemetry in {key}",
                    )
        name = f"seed{seed}_alpha{alpha:.1f}.pt"
        require(item.get("checkpoint") == name, f"checkpoint label mismatch: {key}")
        path = run / name
        require(
            path.is_file() and sha256_file(path) == item.get("checkpoint_sha256"),
            f"checkpoint missing or hash mismatch: {key}",
        )
        by_key[key] = {"record": item, "path": path}
    expected = {f"seed{seed}_alpha{alpha:.1f}" for seed in SEEDS for alpha in ALPHAS}
    require(set(by_key) == expected, "fit matrix has omission/duplicate")
    return by_key


def derive_normalizer(fit_nodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Recompute exact normalization from the frozen 24 fit row groups only."""
    parquet = pq.ParquetFile(DATA_PATH)
    features = []
    for node in fit_nodes:
        table = parquet.read_row_group(node["row_group"], columns=list(FEATURES), use_threads=False)
        raw = np.column_stack([table[name].to_numpy() for name in FEATURES]).astype(np.float32)
        require(np.isfinite(raw).all(), f"nonfinite raw fit feature in {node['row_group']}")
        x = causal_features(raw)
        require(x.shape == (N_ROWS, INPUTS) and np.isfinite(x).all(), "invalid fit features")
        features.append(x)
    matrix = np.stack(features)
    mean = matrix.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(matrix.std(axis=(0, 1), dtype=np.float64), 0.001).astype(np.float32)
    return mean, scale


def read_sequence(
    node: dict, mean: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one raw test component, normalize from fit-only statistics, preserve masks."""
    table = pq.ParquetFile(DATA_PATH).read_row_group(
        int(node["row_group"]),
        columns=[*FEATURES, "t0", "t1", "need_prediction", "is_scored"],
        use_threads=False,
    )
    require(table.num_rows == N_ROWS, f"test sequence row count differs: {node['row_group']}")
    raw = np.column_stack([table[name].to_numpy() for name in FEATURES]).astype(np.float32)
    require(np.isfinite(raw).all(), f"nonfinite raw test features: {node['row_group']}")
    x = causal_features(raw)
    require(x.shape == (N_ROWS, INPUTS) and np.isfinite(x).all(), "invalid causal test features")
    x = np.clip((x - mean) / scale, -8, 8).astype(np.float32)
    y = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")]).astype(np.float32)
    mask = table["need_prediction"].to_numpy().astype(bool)
    mask &= table["is_scored"].to_numpy().astype(bool)
    require(
        mask.any() and np.isfinite(y[mask]).all(),
        f"selected raw labels missing/nonfinite: {node['row_group']}",
    )
    return x, y, mask


def predict_sequence(model: IndependentScaleRNN, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Score one sequence with state reset and exact state carry between chunks."""
    model.eval()
    state = None
    outputs = []
    with torch.no_grad():
        for start in range(0, N_ROWS, CHUNK_ROWS):
            chunk = torch.from_numpy(x[start : start + CHUNK_ROWS][None]).to(device=device)
            prediction, state = model(chunk, state)
            require(
                torch.isfinite(prediction).all().item() and torch.isfinite(state).all().item(),
                f"nonfinite scorer output or state at chunk {start}",
            )
            outputs.append(prediction[0].cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32)
    require(
        result.shape == (N_ROWS, 2) and np.isfinite(result).all(), "invalid full sequence output"
    )
    return result


def score_moments(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Make stable per-component moments after finite raw check and metric clipping."""
    require(
        y.shape == prediction.shape == (N_ROWS, 2) and mask.shape == (N_ROWS,),
        "sequence prediction/target/mask shape mismatch",
    )
    require(
        np.isfinite(y[mask]).all() and np.isfinite(prediction[mask]).all(),
        "nonfinite selected raw target/prediction before metric clipping",
    )
    yc = np.clip(y[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    pc = np.clip(prediction[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    return np.stack([_centered_moments(yc[:, j], pc[:, j]) for j in range(2)])


def _target_wp(moments: np.ndarray, counts: np.ndarray, target: int) -> float:
    """Stable pooled WP for a single target over resampled components."""
    w = counts * moments[:, target, 0]
    total = w.sum()
    if total <= 0:
        return 0.0
    active = np.flatnonzero(w > 0)
    ay, ap = moments[active[0], target, 1:3]
    dy, dp = moments[:, target, 1] - ay, moments[:, target, 2] - ap
    my, mp = np.sum(w * dy) / total, np.sum(w * dp) / total
    vy = np.sum(counts * moments[:, target, 3] + w * (dy - my) ** 2)
    vp = np.sum(counts * moments[:, target, 4] + w * (dp - mp) ** 2)
    cov = np.sum(counts * moments[:, target, 5] + w * (dy - my) * (dp - mp))
    if np.sqrt(vy / total) <= 1e-8 or np.sqrt(vp / total) <= 1e-8:
        return 0.0
    return float(np.clip(cov / np.sqrt(vy * vp), -1.0, 1.0))


def paired_target_bootstrap(
    a: np.ndarray, b: np.ndarray, target: int, n_boot: int, seed: int
) -> dict:
    """Paired component bootstrap interval for one target's WP difference."""
    n = len(a)
    require(a.shape == b.shape and a.shape[1:] == (2, 6), "target contrast moment shapes differ")
    ones = np.ones(n, dtype=np.int64)
    point = _target_wp(a, ones, target) - _target_wp(b, ones, target)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        deltas[i] = _target_wp(a, counts, target) - _target_wp(b, counts, target)
    return {
        "mean_diff": point,
        "ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
        "bootstrap_sd": float(deltas.std(ddof=1)),
        "p_le_0": float(np.mean(deltas <= 0)),
        "n_components": n,
        "n_boot": n_boot,
        "seed": seed,
    }


def self_check() -> dict:
    """CPU synthetic parity, bounds, causality and recurrent-state carry checks."""
    torch.manual_seed(98731)
    baseline = RNNModel(INPUTS, hidden=HIDDEN, layers=2, dropout=0.0, n_out=2).train()
    independent = IndependentScaleRNN(0.0).train()
    independent.load_state_dict(baseline.state_dict(), strict=True)
    require(set(independent.state_dict()) == set(baseline.state_dict()), "state keys differ")
    for name, value in baseline.state_dict().items():
        require(
            torch.equal(value, independent.state_dict()[name]), f"initial state differs: {name}"
        )
    raw_inputs = torch.randn(2, 18, INPUTS)
    synthetic_fit_mean = torch.linspace(-0.4, 0.4, INPUTS)
    synthetic_fit_scale = torch.linspace(0.7, 1.3, INPUTS)
    # E470 externally normalizes features; source RNNModel buffers stay
    # identity (zero/one) even when the fit-derived transform is nontrivial.
    inputs_a = ((raw_inputs - synthetic_fit_mean) / synthetic_fit_scale).requires_grad_(True)
    inputs_b = inputs_a.detach().clone().requires_grad_(True)
    ref_output, ref_state = baseline(inputs_a)
    got_output, got_state = independent(inputs_b)
    torch.testing.assert_close(got_output, ref_output, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(got_state, ref_state, atol=2e-5, rtol=2e-4)
    (ref_output.square().mean() + ref_state.square().mean()).backward()
    (got_output.square().mean() + got_state.square().mean()).backward()
    torch.testing.assert_close(inputs_b.grad, inputs_a.grad, atol=5e-5, rtol=5e-4)
    for (name_a, param_a), (name_b, param_b) in zip(
        baseline.named_parameters(), independent.named_parameters(), strict=True
    ):
        require(name_a == name_b, "parameter names differ")
        torch.testing.assert_close(param_b.grad, param_a.grad, atol=5e-5, rtol=5e-4)
    require(
        sum(p.numel() for p in independent.parameters()) == N_PARAMS, "parameter count mismatch"
    )

    # The two chunks must equal one full causal pass when final state is carried.
    independent.zero_grad(set_to_none=True)
    x = torch.randn(2, 31, INPUTS)
    with torch.no_grad():
        full, full_state = independent(x)
        first, carry = independent(x[:, :13])
        second, final_state = independent(x[:, 13:], carry)
        reset_second, reset_state = independent(x[:, 13:])
        torch.testing.assert_close(torch.cat((first, second), 1), full, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(final_state, full_state, atol=2e-6, rtol=2e-5)
        require(
            not torch.allclose(reset_second, second),
            "state reset/carry control did not distinguish",
        )
        changed_future = x.clone()
        changed_future[:, 17:] += 5.0
        changed, _ = independent(changed_future)
        torch.testing.assert_close(changed[:, :17], full[:, :17], atol=0, rtol=0)
        require(reset_state.shape == (2, 2, HIDDEN), "reset state shape mismatch")

    # Candidate interpolation remains bounded for recurrently generated states.
    candidate_model = IndependentScaleRNN(0.1).eval()
    with torch.no_grad():
        bounded_output, bounded_state = candidate_model(torch.randn(3, 64, INPUTS))
        require(
            torch.isfinite(bounded_output).all().item()
            and torch.isfinite(bounded_state).all().item(),
            "candidate synthetic forward is nonfinite",
        )
        require(float(bounded_state.abs().max()) <= 1.0, "candidate recurrent state escaped [-1,1]")
        u = torch.linspace(-1.0, 1.0, 1001)
        candidate = u + 0.1 * torch.sin(math.pi * u)
        require(
            float(candidate.min()) >= -1.000001 and float(candidate.max()) <= 1.000001,
            "candidate map violates its declared bound",
        )
    u_grad = torch.linspace(-1.0, 1.0, 1001, requires_grad=True)
    candidate_grad = u_grad + 0.1 * torch.sin(math.pi * u_grad)
    derivative = torch.autograd.grad(candidate_grad.sum(), u_grad)[0]
    require(
        torch.isfinite(derivative).all().item() and float(derivative.min()) > 0.0,
        "candidate derivative is nonfinite or nonmonotone",
    )
    candidate_train = IndependentScaleRNN(0.1).train()
    candidate_x = torch.randn(2, 11, INPUTS, requires_grad=True)
    candidate_y, candidate_state = candidate_train(candidate_x)
    candidate_loss = candidate_y.square().mean() + candidate_state.square().mean()
    candidate_loss.backward()
    require(torch.isfinite(candidate_loss).item(), "candidate synthetic loss is nonfinite")
    require(torch.isfinite(candidate_x.grad).all().item(), "candidate input gradient is nonfinite")
    require(
        all(
            p.grad is not None and torch.isfinite(p.grad).all().item()
            for p in candidate_train.parameters()
        ),
        "candidate parameter gradient is nonfinite",
    )
    return {
        "status": "synthetic_cpu_checks_passed",
        "baseline_parity": True,
        "chunk_carry_reset_causality": True,
        "candidate_bound": True,
        "external_fit_normalization_with_identity_model_buffers": True,
        "parameters": N_PARAMS,
        "state_shape": [2, 2, HIDDEN],
    }


def score_run(run: Path, device: torch.device, output_path: Path) -> dict:
    """Validate all fit receipts then score all six arms over 16 components."""
    seal, attempt, report = verify_receipts(run)
    fit_nodes, test_nodes = get_cohort()
    by_key = verify_arms(report, run, fit_nodes)
    mean, scale = derive_normalizer(fit_nodes)
    sequences = [read_sequence(node, mean, scale) for node in test_nodes]
    component_ids = [node["component"] for node in test_nodes]
    row_groups = [node["row_group"] for node in test_nodes]
    targets, masks, sequence_index, within_sequence_row = [], [], [], []
    for index, (_, y, mask) in enumerate(sequences):
        selected_rows = np.flatnonzero(mask)
        targets.append(y[mask])
        sequence_index.append(np.full(len(selected_rows), index, dtype=np.int16))
        within_sequence_row.append(selected_rows.astype(np.int32))
        masks.append(mask)
    raw_targets = np.concatenate(targets).astype(np.float32)
    counts = [int(mask.sum()) for mask in masks]
    require(sum(counts) == len(raw_targets), "test target segmentation mismatch")
    n_components = len(test_nodes)
    arm_moments: dict[int, dict[float, np.ndarray]] = {seed: {} for seed in SEEDS}
    scores: dict[str, dict] = {}
    raw_predictions: dict[str, np.ndarray] = {}
    arm_hashes = {}
    all_mask = np.ones(len(raw_targets), dtype=bool)
    for seed in SEEDS:
        for alpha in ALPHAS:
            key = f"seed{seed}_alpha{alpha:.1f}"
            state = torch.load(by_key[key]["path"], map_location="cpu", weights_only=True)
            model = IndependentScaleRNN(alpha)
            expected = model.state_dict()
            require(
                isinstance(state, dict) and set(state) == set(expected),
                f"checkpoint keys differ: {key}",
            )
            for name, value in state.items():
                require(
                    torch.is_tensor(value) and value.shape == expected[name].shape,
                    f"checkpoint tensor shape differs {key}/{name}",
                )
                require(
                    torch.isfinite(value).all().item(), f"nonfinite checkpoint tensor {key}/{name}"
                )
            model.load_state_dict(state, strict=True)
            require(
                sum(p.numel() for p in model.parameters()) == N_PARAMS,
                "scorer parameter count changed",
            )
            require(
                torch.equal(model.mean, torch.zeros_like(model.mean))
                and torch.equal(model.std, torch.ones_like(model.std)),
                f"checkpoint does not retain source RNNModel identity buffers: {key}",
            )
            model.to(device).eval()
            selected_predictions = []
            per_component_moments = []
            for x, y, mask in sequences:
                prediction = predict_sequence(model, x, device)
                require(
                    np.isfinite(prediction[mask]).all(),
                    f"nonfinite raw predictions before clipping: {key}",
                )
                selected_predictions.append(prediction[mask])
                per_component_moments.append(score_moments(y, prediction, mask))
            predictions = np.concatenate(selected_predictions).astype(np.float32)
            require(
                predictions.shape == raw_targets.shape and np.isfinite(predictions).all(),
                f"selected prediction order/count differs for {key}",
            )
            official = global_wp(raw_targets[None], predictions[None], all_mask[None])
            moments = np.stack(per_component_moments)
            pooled = _score_component_moments(moments, np.ones(n_components, dtype=np.int64))
            require(
                np.isclose(official["wp"], pooled, atol=2e-7, rtol=0),
                f"official pooled WP disagrees with centered component moments: {key}",
            )
            scores[key] = {
                "seed": seed,
                "alpha": alpha,
                "global_wp": float(official["wp"]),
                "target_wp": {"t0": float(official["t0"]), "t1": float(official["t1"])},
                "selected_rows": len(raw_targets),
                "selected_rows_per_component": counts,
                "component_ids": component_ids,
            }
            arm_moments[seed][alpha] = moments
            raw_predictions[key] = predictions
            arm_hashes[key] = hashlib.sha256(predictions.tobytes()).hexdigest()
            del model, state

    contrasts = {}
    for seed in SEEDS:
        zero, candidate = arm_moments[seed][0.0], arm_moments[seed][0.1]
        result = paired_component_bootstrap(
            candidate, zero, np.asarray(component_ids), n_boot=BOOTSTRAPS, seed=seed, centered=True
        )
        result["candidate"] = "alpha=+0.1"
        result["reference"] = "alpha=0"
        result["target_wp"] = {
            target: paired_target_bootstrap(candidate, zero, target, BOOTSTRAPS, seed)
            for target in (0, 1)
        }
        contrasts[f"seed{seed}:alpha_p10_minus_zero"] = result

    # A raw, selected-row export binds every prediction to its component and
    # within-sequence row; original score masks and registered row groups travel too.
    export_path = output_path.with_name(output_path.stem + "_paired_components.npz")
    require(
        not output_path.exists() and not export_path.exists(), "refusing to overwrite scorer output"
    )
    export_arrays = {
        "component_ids": np.asarray(component_ids, dtype=np.int64),
        "row_groups": np.asarray(row_groups, dtype=np.int32),
        "selected_rows_per_component": np.asarray(counts, dtype=np.int32),
        "selected_targets_raw": raw_targets,
        "score_masks_by_component": np.stack(masks).astype(bool),
        "sequence_index": np.concatenate(sequence_index),
        "row_index_within_sequence": np.concatenate(within_sequence_row),
        "alpha_values": np.asarray(ALPHAS, dtype=np.float32),
        "seed_values": np.asarray(SEEDS, dtype=np.int64),
        "predictions": np.stack(
            [raw_predictions[f"seed{seed}_alpha{alpha:.1f}"] for seed in SEEDS for alpha in ALPHAS]
        ),
        "registered_arm_keys": np.asarray(
            [f"seed{seed}_alpha{alpha:.1f}" for seed in SEEDS for alpha in ALPHAS]
        ),
    }
    for key, value in export_arrays.items():
        if value.dtype.kind in "f":
            require(np.isfinite(value).all(), f"nonfinite raw NPZ export field {key}")
    np.savez_compressed(export_path, **export_arrays)
    require(export_path.is_file(), "paired component export did not save")
    return {
        "kind": "E478 independent recurrent candidate scorer",
        "status": "scored_exposed_components_descriptive_only",
        "metric_id": METRIC_ID,
        "seal": {
            "name": seal["name"],
            "spec_sha256": attempt["spec_sha256"],
            "runner_attempt": attempt["id"],
            "fit_manifest_sha256": sha256_file(run / "fit_manifest.json"),
        },
        "data": {
            "validation_sha256": EXPECTED_DATA,
            "cohort_sha256": sha256_file(COHORT_PATH),
            "component_ids": component_ids,
            "row_groups": row_groups,
            "selected_rows_per_component": counts,
            "total_selected_rows": len(raw_targets),
        },
        "scorer": {
            "source_sha256": sha256_file(Path(__file__)),
            "device": str(device),
            "forward": "independent scripted reset-after-bias GRU equations, two stacked H64 layers",
            "normalization": "recomputed from exactly 24 frozen fit row groups",
            "raw_prediction_sha256_by_arm": arm_hashes,
        },
        "arms": scores,
        "contrasts": contrasts,
        "bootstrap": {
            "resampling_unit": "component",
            "n_boot": BOOTSTRAPS,
            "seed_per_fit_seed": "fit seed",
            "paired": True,
        },
        "export": {
            "path": str(export_path),
            "sha256": sha256_file(export_path),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in export_arrays.items()
            },
        },
        "scope": "three fixed fit seeds on 16 previously exposed components; no pooled-seed inference, winner selection, or public forecast",
    }


def main() -> None:
    """Run bounded self-check or source-bound scoring with a fresh output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    if args.self_check_only:
        print(json.dumps(self_check(), indent=2))
        return
    if args.output is None:
        parser.error("--output is required unless --self-check-only is set")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        from wnn.resources import configure_torch

        configure_torch()
        settings = (
            (torch.backends, "fp32_precision"),
            (torch.backends.cuda.matmul, "fp32_precision"),
            (torch.backends.cudnn, "fp32_precision"),
            (torch.backends.cudnn.rnn, "fp32_precision"),
        )
        for owner, attribute in settings:
            if not hasattr(owner, attribute):
                raise RuntimeError("full IEEE FP32 backend policy is unavailable")
            setattr(owner, attribute, "ieee")
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = score_run(args.run_dir, torch.device(args.device), args.output)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(args.output),
                "arms": len(result["arms"]),
                "contrasts": len(result["contrasts"]),
                "npz": result["export"]["path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Bounded CPU-only independent checks of paired-head screening code."""

from __future__ import annotations

import ast
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from campaign_pair_head_fit import ResidualHead
from campaign_pair_head_score import soft_global_wp

from wnn.gmetric import global_wp
from wnn.nn import RNNModel


def main() -> None:
    started = time.perf_counter()
    sources = (
        "campaign_pair_states.py",
        "campaign_pair_head_fit.py",
        "campaign_pair_head_score.py",
    )
    hashes_before = {
        name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() for name in sources
    }
    torch.set_num_threads(1)
    torch.manual_seed(998)
    result: dict = {}
    x = torch.randn(3, 117, 4)
    model = RNNModel(4, hidden=8, layers=2, n_out=7).eval()
    with torch.no_grad():
        manual, automatic = [], []
        hidden_manual = hidden_automatic = None
        for start in range(0, len(x[0]), 23):
            batch = x[:, start : start + 23]
            z = model.encode((batch - model.mean) / model.std)
            o, hidden_manual = model.gru(z, hidden_manual)
            manual.append(model.head(torch.cat([o, z], -1))[..., :2])
            prediction, hidden_automatic = model(batch, hidden_automatic)
            automatic.append(prediction[..., :2])
    error = float((torch.cat(manual, 1) - torch.cat(automatic, 1)).abs().max())
    assert error == 0
    result["manual_ordinary_gru_parity"] = {"max_abs_cpu_float32": error}

    state = torch.randn(7, 2, 8)
    parents = torch.randn(7, 2, 2)
    for kind in ("bilinear", "unary", "output_only"):
        head = ResidualHead(kind, 8, torch.zeros(2, 8), torch.ones(2, 8))
        assert torch.equal(head(state, parents), torch.zeros(7, 2))
    result["initial_residuals_are_zero"] = True

    # A genuinely additive control has zero mixed derivative in both members.
    unary = ResidualHead("unary", 1, torch.zeros(2, 1), torch.ones(2, 1)).double()
    with torch.no_grad():
        unary.left.weight.fill_(1)
        unary.right.weight.fill_(1)
        unary.left.bias.zero_()
        unary.right.bias.zero_()
        unary.out.weight.fill_(1)
    hidden = torch.zeros(1, 2, 1, dtype=torch.float64, requires_grad=True)
    first = torch.autograd.grad(
        unary(hidden, torch.zeros(1, 2, 2)).sum(), hidden, create_graph=True
    )[0]
    mixed = torch.autograd.grad(first[0, 0, 0], hidden)[0][0, 1, 0].item()
    result["unary_mixed_derivative"] = mixed

    y = np.tile(np.array([[-1, -1], [1, 1]], dtype=np.float32), (100, 1))[None]
    mask = np.ones(y.shape[:-1], dtype=bool)
    mask[:, :99] = False
    prediction = y.copy()
    mask[:, 100] = False
    prediction[:, 100] = np.inf
    try:
        value = soft_global_wp(y, prediction, mask, 0.02)
    except ValueError as error:
        result["soft_required_infinity"] = {"status": "rejected", "reason": str(error)}
    else:
        result["soft_required_infinity"] = {"status": "accepted", "score": value}
    try:
        value = soft_global_wp(y, y, mask, float("nan"))
    except ValueError as error:
        result["soft_nan_lambda"] = {"status": "rejected", "reason": str(error)}
    else:
        result["soft_nan_lambda"] = {"status": "accepted", "score": value}

    tree = ast.parse((ROOT / "scripts/campaign_pair_states.py").read_text())
    writer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(item, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "write_index"
                for target in item.targets
            )
            for item in node.body
        )
    )
    compiled = compile(
        ast.Module(body=[writer], type_ignores=[]), "reviewed_eval_write_index", "exec"
    )
    for chunk in (4000, 3343, 20000):
        groups = np.arange(19, dtype=np.int32) + 501
        saved_group = np.full(19 * 20000, -1, dtype=np.int32)
        saved_step = np.full_like(saved_group, -1)
        group_position = 0
        for batch_size in (8, 8, 3):
            batch_groups = groups[group_position : group_position + batch_size]
            group_position += batch_size
            for step in range(0, 20000, chunk):
                stop = min(step + chunk, 20000)
                scope = {
                    "args": SimpleNamespace(mode="eval"),
                    "np": np,
                    "group_position": group_position,
                    "x": np.zeros(batch_size),
                    "step": step,
                    "stop": stop,
                    "position": 0,
                    "end": 0,
                }
                exec(compiled, scope)
                index = scope["write_index"]
                saved_group[index] = np.repeat(batch_groups, stop - step)
                saved_step[index] = np.tile(np.arange(step, stop), batch_size)
        assert np.array_equal(
            saved_group.reshape(19, 20000), np.broadcast_to(groups[:, None], (19, 20000))
        )
        assert np.array_equal(
            saved_step.reshape(19, 20000), np.broadcast_to(np.arange(20000), (19, 20000))
        )
    result["reviewed_export_layout"] = {
        "groups": 19,
        "batches": [8, 8, 3],
        "chunks": [4000, 3343, 20000],
        "complete_exact_order": True,
    }

    rng = np.random.default_rng(815)
    y = rng.normal(size=(3, 211, 2)).astype(np.float32)
    prediction = (y + rng.normal(size=y.shape) * 0.5).astype(np.float32)
    mask = rng.random(y.shape[:-1]) < 0.4
    mask[:, :99] = False
    need = np.broadcast_to(np.arange(211)[None] >= 99, (3, 211))
    errors = []
    for lam in (0.0, 0.02, 0.05, 1.0):
        values = []
        for k in range(2):
            yy = np.clip(y[..., k], -2, 2).astype(np.longdouble)[need]
            pp = np.clip(prediction[..., k], -2, 2).astype(np.longdouble)[need]
            weights = np.abs(yy) * np.where(mask[need], 1.0, lam).astype(np.longdouble)
            yc = yy - np.sum(weights * yy) / weights.sum()
            pc = pp - np.sum(weights * pp) / weights.sum()
            values.append(
                float(
                    np.sum(weights * yc * pc)
                    / np.sqrt(np.sum(weights * yc * yc) * np.sum(weights * pc * pc))
                )
            )
        errors.append(abs(soft_global_wp(y, prediction, mask, lam) - np.mean(values)))
    assert max(errors) < 1e-12
    assert (
        abs(soft_global_wp(y, prediction, mask, 0.0) - global_wp(y, prediction, mask)["wp"]) < 1e-12
    )
    result["soft_metric_oracle_parity"] = {
        "lambdas": [0, 0.02, 0.05, 1],
        "max_abs_error": max(errors),
    }

    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as cache:
        row_index = {int(group): i for i, group in enumerate(cache["row_group"])}
        masks = cache["mask"]
    disk_rows = []
    for fold in range(4):
        path = ROOT / "research/campaign_2026-09-29/artifacts/splits" / f"fold{fold}.json"
        manifest = json.loads(path.read_text())
        fit_rows = sum(
            int(masks[row_index[int(group)], 99:].sum()) for group in manifest["fit"]["valid"]
        )
        eval_rows = len(manifest["evaluation"]["valid"]) * 20_000
        disk_rows.append(
            {
                "fold": fold,
                "fit_rows": fit_rows,
                "eval_rows": eval_rows,
                "all_export_bytes_at_hidden128": (fit_rows + eval_rows) * (4 * 128 + 29),
                "all_export_bytes_at_hidden256": (fit_rows + eval_rows) * (4 * 256 + 29),
            }
        )
    result["disk_exports"] = disk_rows
    result["hashes"] = {
        name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
        for name in (
            "campaign_pair_states.py",
            "campaign_pair_head_fit.py",
            "campaign_pair_head_score.py",
        )
    }
    result["hashes_before"] = hashes_before
    assert result["hashes"] == hashes_before, "source changed while checks were running"
    result["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["elapsed_seconds"] = time.perf_counter() - started
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

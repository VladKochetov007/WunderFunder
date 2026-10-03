from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[5]
OWN = Path(__file__).resolve().parent
ACTOR = OWN.parent / "objective"


def sha(path: Path) -> str:
    """Hash only bounded metadata, sources and summary artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actual_ast_checks() -> dict[str, object]:
    """Execute actual cached/TD/running elif chain, never an actor reimplementation."""
    tree = ast.parse((ROOT / "scripts/train_rnn.py").read_text())
    chain = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "teach is not None"
        and any(isinstance(k, ast.Name) and k.id == "norm_ema" for k in ast.walk(n))
    )
    branch_code = compile(
        ast.Module(body=[chain], type_ignores=[]), "live_trainer_cached_chain", "exec"
    )
    next_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "next_label_change"
    )
    mask_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "true_mask_weights"
    )
    scope = {"torch": torch}
    exec(
        compile(
            ast.Module(body=[next_fn, mask_fn], type_ignores=[]), "live_trainer_functions", "exec"
        ),
        scope,
    )
    y = torch.ones(1, 20, 2)
    y[:, :8, 0] = 3.0
    y[:, 8:, 0] = 4.0  # raw pair changes at8 even though clipped y0 stays2
    teach = torch.stack([torch.arange(20.0) / 3, -torch.arange(20.0) / 4], -1)[None]
    next_change = scope["next_label_change"](y)
    assert int(next_change[0, 0]) == 8
    valid = torch.linspace(0.1, 1.0, 20)[None]
    runs = []
    for mode in ("chunk", "running"):
        out = torch.zeros(1, 20, 2, requires_grad=True)
        sentinel = torch.tensor(17.0)
        env = {
            "torch": torch,
            "args": SimpleNamespace(
                distill_alpha_far=None,
                distill_alpha=0.5,
                distill_boot_h=8,
                td_alpha=0.0,
                loss_norm=mode,
                loss="wmse",
            ),
            "teach": teach,
            "y": y,
            "sl": slice(0, 20),
            "nxt_label": next_change,
            "main_valid": valid,
            "out": out,
            "norm_ema": sentinel,
        }
        exec(branch_code, env)
        yc = y.clamp(-2, 2)
        label = yc.clone()
        for t in range(20):
            if t + 8 < 20 and int(next_change[0, t]) > t + 8:
                label[0, t] = teach[0, t + 8].clamp(-2, 2)
        target = 0.5 * label + 0.5 * teach.clamp(-2, 2)
        w = yc.abs() * valid[..., None]
        expected = (w * (out - target).square()).sum() / w.sum()
        assert torch.equal(env["tgt"], target) and torch.equal(env["w"], w)
        assert torch.equal(env["loss"], expected) and env["norm_ema"] is sentinel
        runs.append(float(env["loss"].detach()))
        env["loss"].backward()
        torch.testing.assert_close(out.grad, 2 * w / w.sum() * (out.detach() - target))
        assert torch.equal(label[0, 0], yc[0, 0])  # strict boundary t0+8=firstchange8
        assert torch.equal(label[0, 8], teach[0, 16].clamp(-2, 2))
        assert torch.equal(label[0, 12], yc[0, 12])  # t+8 == T, so no bootstrap
    assert runs[0] == runs[1]
    average_assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "t" for t in n.targets)
        and "np.asarray" in ast.unparse(n)
        and "self._teach" in ast.unparse(n)
    )
    teacher_env = {
        "np": np,
        "j": 0,
        "self": SimpleNamespace(
            _teach=[
                (np.full((1, 2, 2), 6.0, np.float32), 1.0),
                (np.full((1, 2, 2), -2.0, np.float32), 1.0),
            ]
        ),
    }
    exec(
        compile(
            ast.Module(body=[average_assignment], type_ignores=[]), "live_teacher_average", "exec"
        ),
        teacher_env,
    )
    assert np.array_equal(np.clip(teacher_env["t"], -2, 2), np.full((2, 2), 2.0))
    assert not np.array_equal(np.clip(teacher_env["t"], -2, 2), np.zeros((2, 2)))
    ema_node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "args.ema_teacher > 0"
        and "consistency_scale" in ast.unparse(n)
    )
    f = torch.tensor([[[0.2, -0.1], [0.5, 1.0]]], requires_grad=True)
    v = torch.tensor([[0.5, 2.0]])
    u = torch.tensor([[[3.0, -4.0], [5.0, -6.0]]])
    env = {
        "torch": torch,
        "args": SimpleNamespace(ema_teacher=1.0, ema_metric_weight=False),
        "main_valid": v,
        "sl": slice(0, 2),
        "out": f,
        "out_ema": u,
        "loss": torch.tensor(0.0),
    }
    exec(compile(ast.Module(body=[ema_node], type_ignores=[]), "live_ema_clause", "exec"), env)
    env["loss"].backward()
    torch.testing.assert_close(f.grad, 2 * v[..., None] / v.sum() * (f.detach() - u))
    assert env["cw"].sum() == v.sum() and u.abs().max() > 2
    structural_tree = ast.parse((ROOT / "src/wnn/spread_aux.py").read_text())
    structural = next(
        n
        for n in structural_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "structural_loss"
    )
    structural_env = {"torch": torch, "Tensor": torch.Tensor, "F": F}
    exec(
        compile(ast.Module(body=[structural], type_ignores=[]), "live_structural", "exec"),
        structural_env,
    )
    logits = torch.arange(18.0).reshape(1, 2, 9).requires_grad_()
    labels = torch.tensor([[0, 1]])
    (0.3 * structural_env["structural_loss"](logits, labels, labels, labels, v, 0, 2)).backward()
    expected = torch.zeros_like(logits)
    for lo, hi in ((0, 2), (2, 5), (5, 9)):
        probabilities = logits.detach()[..., lo:hi].softmax(-1)
        onehot = F.one_hot(labels, hi - lo)
        expected[..., lo:hi] = 0.3 * v[..., None] / v.sum() * (probabilities - onehot)
    torch.testing.assert_close(logits.grad, expected)
    next_clause = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "args.aux_next > 0"
        and "binary_cross_entropy_with_logits" in ast.unparse(n)
    )
    next_out = torch.tensor([[[0.2, -0.1], [0.3, 0.4]]], requires_grad=True)
    next_weight = torch.tensor([[0.5, 0.0]])
    nx_rows, nx_up = torch.tensor([[1.0, 2.0]]), torch.tensor([[0.0, 1.0]])
    next_env = {
        "torch": torch,
        "F": F,
        "args": SimpleNamespace(aux_next=0.3),
        "out": next_out,
        "next_at": 0,
        "sl": slice(0, 2),
        "nw": next_weight,
        "nx_rows": nx_rows,
        "nx_up": nx_up,
        "kth": [],
        "loss": torch.tensor(0.0),
    }
    exec(
        compile(ast.Module(body=[next_clause], type_ignores=[]), "live_next_aux", "exec"), next_env
    )
    next_env["loss"].backward()
    expected_next = torch.stack(
        [
            0.6 * next_weight / next_weight.sum() * (next_out.detach()[..., 0] - (nx_rows - 3) / 2),
            0.3 * next_weight / next_weight.sum() * (next_out.detach()[..., 1].sigmoid() - nx_up),
        ],
        -1,
    )
    torch.testing.assert_close(next_out.grad, expected_next)
    # Mass is conserved before since but not guaranteed after nonuniform since.
    old = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    known = torch.tensor([[True, False], [False, False]])
    replaced = scope["true_mask_weights"](old, torch.ones_like(known), known, 0.03, True)
    torch.testing.assert_close(old.sum(), replaced.sum())
    since = torch.tensor([[1.0, 0.25], [1.0, 0.25]])
    assert not torch.isclose((old * since).sum(), (replaced * since).sum())
    return {
        "cached_chunk_loss": runs[0],
        "cached_running_loss": runs[1],
        "norm_ema_sentinel_unchanged": True,
        "strict_bootstrap8_raw_pair_boundaries": True,
        "original_clipped_label_weights": True,
        "unclipped_ema_and_row_broadcast_factor": True,
        "teacher_average_before_clipping": True,
        "structural_three_CE_coefficients_no_divide_by_three": True,
        "next_auxiliary_MSE_and_BCE_coefficients": True,
        "true_mask_mass_before_since": [float(old.sum()), float(replaced.sum())],
        "true_mask_mass_after_since": [float((old * since).sum()), float((replaced * since).sum())],
        "cached_chain_ast_sha256": hashlib.sha256(ast.dump(chain).encode()).hexdigest(),
    }


def summaries_checks() -> dict[str, object]:
    """Reconstruct decisive statistics from compact sufficient statistics only."""
    with np.load(ACTOR / "chunk_summaries.npz") as data:
        masses = data["masses"].reshape(8, 32, 20, 4).sum(1)
        gradients = data["gradient"].reshape(8, 32, 20, 2, 2, 3).sum(1)
        groups = data["groups"].copy()
    W = masses[..., 0].ravel()
    scale = W / W.mean()
    compact = json.loads((ACTOR / "compact_results.json").read_text())[0]
    sham_report = json.loads((ACTOR / "sham_results.json").read_text())
    result: dict[str, object] = {
        "primary_cv_W": float(W.std() / W.mean()),
        "scalars_mean": float(scale.mean()),
        "slots": len(W),
    }
    assert abs(result["primary_cv_W"] - compact["main_cv"]) < 1e-12
    for arm, name in enumerate(("zero_prediction", "cached_teacher_prediction")):
        numerator = gradients[:, :, arm].reshape(160, 6)
        sn = numerator / W[:, None]
        aligned = scale[:, None] * sn
        pooled = numerator / W.mean()
        assert np.allclose(aligned, pooled, atol=1e-15, rtol=1e-14)
        expectation = sn.mean(0) * scale.mean()
        finite_covariance = np.mean((scale - 1)[:, None] * sn, axis=0)
        assert np.allclose(aligned.mean(0) - expectation, finite_covariance, atol=1e-15)
        covariance_g = np.cov(sn, rowvar=False, bias=True)
        permutation_covariance = scale.var() * covariance_g / (len(W) - 1)
        shams = []
        norm_changes = []
        for seed in range(9000, 9032):
            permuted = np.random.default_rng(seed).permutation(scale)
            assert np.array_equal(np.sort(permuted), np.sort(scale))
            sham = permuted[:, None] * sn
            shams.append(sham.mean(0))
            norm_changes.append(
                float(
                    np.linalg.norm(np.linalg.norm(sham, axis=1) - np.linalg.norm(aligned, axis=1))
                )
            )
        shams = np.stack(shams)
        distance = np.linalg.norm(shams - aligned.mean(0), axis=1) / np.linalg.norm(aligned.mean(0))
        relative = float(
            np.linalg.norm(sn.mean(0) - aligned.mean(0)) / np.linalg.norm(aligned.mean(0))
        )
        quoted = compact["gradient_proxy"][name]
        assert abs(relative - quoted["mean_gradient_relative_difference"]) < 1e-12
        q = np.quantile(distance, [0.1, 0.5, 0.9])
        assert np.allclose(
            q, sham_report[name]["sham_to_aligned_relative_distance_q10_q50_q90"], atol=1e-12
        )
        result[name] = {
            "relative_sn_to_aligned_mean_difference": relative,
            "alignment_exact_max_abs": float(np.abs(aligned - pooled).max()),
            "exact_permutation_expected_mean": expectation.tolist(),
            "aligned_minus_expected_covariance": finite_covariance.tolist(),
            "permutation_mean_gradient_trace_variance_exact": float(
                np.trace(permutation_covariance)
            ),
            "finite_32_seed_sham_to_aligned_q10_q50_q90": q.tolist(),
            "realized_slot_gradient_norm_changes_minmax": [min(norm_changes), max(norm_changes)],
        }
    manifest = json.loads(
        (ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text()
    )
    lookup = {node["row_group"]: node["component"] for node in manifest["sequences"]["train"]}
    component_counts = Counter(lookup[int(g)] for g in groups)
    batches_of = defaultdict(set)
    for i, g in enumerate(groups):
        batches_of[lookup[int(g)]].add(i // 32)
    across = {c: sorted(b) for c, b in batches_of.items() if len(b) > 1}
    result["dependence"] = {
        "unique_components": len(component_counts),
        "components_spanning_multiple_LOBO_batches": len(across),
        "sample_sequences_in_spanning_components": sum(component_counts[c] for c in across),
        "max_component_sample_size": max(component_counts.values()),
    }
    actor_results = json.loads((ACTOR / "results.json").read_text())
    result["actor_predictability_decomposition_not_recomputed"] = actor_results["actual"][0][
        "predictability_logW_LOBO_R2"
    ]
    # Different aggregate gradients at a nonoptimum can share the same minimum.
    # H1=1*theta^2, H2=18*theta^2, W=[1,9]. Both normalized objectives minimize at0.
    toy_sn = np.array([2.0, 4.0])
    toy_aligned = np.array([0.4, 7.2])
    result["same_optimum_gradient_counterexample"] = {
        "SN_gradient_at_theta1": float(toy_sn.mean()),
        "pooled_gradient_at_theta1": float(toy_aligned.mean()),
        "both_exact_parameter_minimizers": 0.0,
        "scope": "positive quadratic coefficient counterexample; not a real-label fit or WP claim",
    }
    assert toy_sn.mean() != toy_aligned.mean()
    return result


def main() -> None:
    torch.set_num_threads(1)
    protected = {
        p.name: sha(p)
        for p in OWN.iterdir()
        if p.is_file() and not p.name.startswith("objective_review_")
    }
    actor_files = [
        "REPORT.md",
        "DERIVATION.md",
        "PREREG.json",
        "PREREG_SHAM.json",
        "results.json",
        "chunk_summaries.npz",
        "compact_results.json",
        "sham_results.json",
    ]
    frozen = {name: sha(ACTOR / name) for name in actor_files}
    source = sha(ROOT / "scripts/train_rnn.py")
    result = {
        "actual_AST": actual_ast_checks(),
        "compact_properties": summaries_checks(),
        "actor_sha256": frozen,
        "live_trainer_sha256": source,
        "prior_blind_I1_artifacts_sha256": protected,
    }
    assert {name: sha(ACTOR / name) for name in actor_files} == frozen
    assert sha(ROOT / "scripts/train_rnn.py") == source
    assert {name: sha(OWN / name) for name in protected} == protected
    (OWN / "objective_review_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

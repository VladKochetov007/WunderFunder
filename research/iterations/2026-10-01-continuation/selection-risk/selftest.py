"""Generated-only CPU smoke. Never opens datasets or checkpoint files."""
from pathlib import Path
import argparse
import ast
import json
import sys
import time
import numpy as np
from common import ROOT, HERE, PANELS, BINS, sha, select_records, role_indices, derangement, panel_positions
from gradient_core import sufficient, gradient


def main():
    start = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also exercise fixed sklearn analysis on generated arrays")
    args = ap.parse_args()
    for source in HERE.glob("*.py"):
        ast.parse(source.read_text())
    rng = np.random.default_rng(621311)
    h = rng.normal(size=(13, 128))
    w = rng.normal(size=(2, 128)) * .03
    b = np.array([.2, -.4])
    y = rng.normal(size=(13, 2)) * np.array([.1, 4.])
    q = rng.uniform(.01, .99, 13)
    a = np.arange(1, 14, dtype=float)
    p = h @ w.T + b
    n, d, _ = sufficient(h, p, y, q, a)
    actual = gradient(n, d)

    def literal_loss(parameters, probs=q):
        weights = parameters[:, :-1]
        bias = parameters[:, -1]
        numerator = denominator = 0.
        for i in range(len(h)):
            for k in range(2):
                clipped = min(2., max(-2., y[i, k]))
                weight = a[i] * probs[i] * abs(clipped)
                residual = np.dot(weights[k], h[i]) + bias[k] - clipped
                numerator += weight * residual * residual
                denominator += weight
        return numerator / max(denominator, 1e-8)

    params = np.concatenate([w, b[:, None]], axis=1)
    numeric = np.empty_like(params)
    for ix in np.ndindex(params.shape):
        plus, minus = params.copy(), params.copy()
        plus[ix] += 1e-6
        minus[ix] -= 1e-6
        numeric[ix] = (literal_loss(plus) - literal_loss(minus)) / 2e-6
    np.testing.assert_allclose(actual, numeric, atol=2e-9, rtol=2e-7)
    nz, dz, _ = sufficient(h, p, y, np.zeros(13), a)
    assert dz == 0 and np.array_equal(gradient(nz, dz), np.zeros((2, 129)))
    na, da, _ = sufficient(h[:6], p[:6], y[:6], q[:6], a[:6])
    nb, db, _ = sufficient(h[6:], p[6:], y[6:], q[6:], a[6:])
    np.testing.assert_allclose(gradient(na + nb, da + db), actual, atol=2e-15)

    cf = {"split": np.asarray(["valid"] * 128), "row_group": np.arange(128),
          "chain": np.arange(1000, 1128), "fold": np.repeat([0, 1], 64),
          "pure": np.ones(128, dtype=bool)}
    classes = np.tile(np.repeat(np.arange(4), 16), 2)
    records = select_records(cf, classes)
    assert records == select_records(cf, classes)
    seen = np.zeros(128, dtype=int)
    for fold in (0, 1):
        for rotation in range(4):
            roles = role_indices(records, fold, rotation)
            seen[roles["audit"]] += 1
            for role, ids in roles.items():
                donor = derangement(records, ids, fold, rotation, role)
                assert set(donor) == set(ids) and np.all(donor != ids)
    assert np.all(seen == 1)
    for record in records:
        for panel in PANELS:
            positions = np.asarray(record["panels"][panel]["positions"])
            assert np.all(positions >= BINS[:-1]) and np.all(positions < BINS[1:])
        assert len({record["panels"][p]["seed"] for p in PANELS}) == 3

    report = {"status": "PASS", "generated_only": True, "numerical_gradient_max_error": float(abs(actual-numeric).max()),
              "common_denominator_unequal_target_mass": True, "zero_weight": True, "aggregate_before_normalize": True,
              "roles_and_within_role_class_derangement": True, "independent_panel_seed_streams": True,
              "data_reads": 0, "checkpoint_reads": 0, "GPU_calls": 0, "WP_queries": 0}
    try:
        import torch
    except ImportError:
        report["autograd"] = "NOT_AVAILABLE_LOCAL"
    else:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(621311)
        assert not torch.cuda.is_initialized()
        parameter = torch.tensor(params, requires_grad=True, dtype=torch.float64)
        phi = torch.tensor(np.concatenate([h, np.ones((13, 1))], axis=1))
        clipped = torch.tensor(y).clamp(-2, 2)
        weights = torch.tensor(a*q)[:, None] * clipped.abs()
        loss = (weights * (phi @ parameter.T - clipped)**2).sum() / weights.sum().clamp_min(1e-8)
        loss.backward()
        np.testing.assert_allclose(actual, parameter.grad.numpy(), atol=5e-15)
        report["autograd"] = "PASS"
        sys.path.insert(0, str(ROOT / "src"))
        from wnn.nn import RNNModel
        from wnn.features import event_features_torch
        from extract_selected import extract_sequence
        model = RNNModel(211, 128, 2, 0., 23).eval()
        raw = torch.randn(1, 43, 112) * .1
        build = lambda x: torch.cat([x, event_features_torch(x)], dim=-1)
        forward = lambda model_, x, sl, state: model_(x[:, sl], state)
        positions = np.asarray([[0, 7, 17, 42], [1, 10, 30, 40], [0, 20, 40, 42]])
        result = extract_sequence(model, build, forward, raw, positions, chunk=7)
        captured = []
        hook = model.head[2].register_forward_pre_hook(lambda mod, arg: captured.append(arg[0]))
        with torch.no_grad():
            inputs = build(raw)
            prediction, _ = model(inputs)
            literal = torch.cat([(inputs - model.mean) / model.std, captured[0], prediction[..., :2]], dim=-1)[0].numpy()
        hook.remove()
        np.testing.assert_allclose(result, literal[positions], atol=3e-6, rtol=3e-6)
        report["generated_prefix_gather_max_error"] = float(abs(result-literal[positions]).max())
        assert not torch.cuda.is_initialized()

    print(json.dumps({"generated_primitives": "PASS", "seconds": time.monotonic()-start}), flush=True)
    if args.full:
        import fit_selectors as fs
        X = rng.normal(size=(128, 3, 256, 341)).astype(np.float32)
        Y = rng.normal(size=(128, 3, 256, 2)).astype(np.float32)
        M = (rng.random((128, 3, 256)) < 1 / (1 + np.exp(-(.5*X[..., 0] + .15*Y[..., 0]))))
        original = fs.design
        calls = []
        def audited_design(X_, Y_, ids, panel, kind, donors=None):
            at = len(calls)
            fold, rotation = divmod(at // 12, 4)
            role = ("fit", "calibration", "audit", "audit")[at % 4]
            expected_panel = (0, 0, 1, 2)[at % 4]
            assert panel == expected_panel and np.array_equal(ids, role_indices(records, fold, rotation)[role])
            calls.append((fold, rotation, role, panel, kind))
            return original(X_, Y_, ids, panel, kind, donors)
        fs.design = audited_design
        # qX must not access targets at all.
        ids = role_indices(records, 0, 0)["fit"]
        assert original(X, None, ids, 0, "X").shape == (8192, 341)
        result, stats = fs.analyze(X, Y, M, records, lambda v: print(json.dumps({"fold":v["fold"],"rotation":v["rotation"],"kind":v["kind"],"seconds":time.monotonic()-start}), flush=True))
        assert len(calls) == 96 and result["selector_fit_count"] == result["calibration_fit_count"] == 24
        assert all(np.isfinite(v).all() for v in stats.values())
        report["full_generated_orchestration"] = "PASS"
        report["fit_calibration_F_only_and_audit_AB_only"] = "PASS"
        report["generated_selector_fits"] = 24
        report["generated_calibration_fits"] = 24
        report["generated_forecast_fits"] = 0
    report["seconds"] = time.monotonic() - start
    report["sources"] = {path.name: sha(path) for path in HERE.glob("*.py")}
    (HERE / "selftest-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

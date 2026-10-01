"""Generated-only checks. No competition labels, predictions, raw arrays or OOF metrics."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

S = importlib.util.spec_from_file_location("analyze_errors", Path(__file__).with_name("analyze_errors.py"))
A = importlib.util.module_from_spec(S)
S.loader.exec_module(A)


def gram(Z, w):
    return Z.T @ (w[:, None]*Z)


class GeneratedChecks(unittest.TestCase):
    def test_metric_and_additive_components(self):
        rng = np.random.default_rng(17)
        y = rng.normal(size=(301, 2))*1.7
        p = .8*y+rng.normal(size=y.shape)
        par = A.metric_parameters(y, p)
        yc, pc = np.clip(y, -2, 2), np.clip(p, -2, 2)
        w = np.abs(yc)
        group = np.arange(len(y)) % 7
        total = 0
        for t in range(2):
            u = (yc[:, t]-par["mu_y"][t])/par["sy"][t]
            z = (pc[:, t]-par["mu_p"][t])/par["sp"][t]
            count, moments = A.group_moments(w[:, t], u, z, group, 8)
            records = [A.component_record(m, par["W"][t]) for m in moments]
            self.assertEqual(sum(count), len(y))
            self.assertEqual(records[-1]["budget"], 0)
            b = sum(r["budget"] for r in records)
            self.assertAlmostEqual(b, .5*(1-par["rho"][t]), places=12)
            total += b
        self.assertAlmostEqual(total, 1-par["score"], places=12)
        zero = A.metric_parameters(np.zeros_like(y), p)
        constant = A.metric_parameters(np.ones_like(y), p)
        self.assertEqual(zero["score"], 0)
        self.assertEqual(constant["score"], 0)
        self.assertFalse(zero["valid"].any())

    def test_q_gradient_and_identity(self):
        rng = np.random.default_rng(42)
        y = rng.uniform(-1.7, 1.7, 100)
        p = .3*y+rng.normal(0, .2, len(y))
        par = A.metric_parameters(y, p)
        W, sy, sp, rho = [float(par[k][0]) for k in ("W", "sy", "sp", "rho")]
        u, z = (y-par["mu_y"][0])/sy, (p-par["mu_p"][0])/sp
        q = u-rho*z
        w = np.abs(y)
        self.assertAlmostEqual(np.dot(w, q)/W, 0, places=13)
        self.assertAlmostEqual(np.dot(w, q*z)/W, 0, places=13)
        self.assertAlmostEqual(np.dot(w, q*q)/W, 1-rho*rho, places=13)
        for i in (3, 29, 82):
            pp, pm = p.copy(), p.copy()
            pp[i] += 1e-5
            pm[i] -= 1e-5
            derivative = (A.metric_parameters(y, pp)["score"]-A.metric_parameters(y, pm)["score"])/2e-5
            self.assertAlmostEqual(derivative, w[i]*q[i]/(W*sp), places=9)
        # Clipped raw response has zero derivative beyond the boundary.
        p[3] = 3
        pp, pm = p.copy(), p.copy()
        pp[3] += .01
        pm[3] -= .01
        self.assertEqual(A.metric_parameters(y, pp)["score"], A.metric_parameters(y, pm)["score"])

    def test_consensus_signed_cross_term(self):
        rng = np.random.default_rng(106)
        y = rng.normal(size=(800, 2))
        p = rng.normal(size=(800, 8, 2))+.7*y[:, None]
        metadata = np.zeros((800, 6), int)
        metadata[:, 5] = np.arange(800) % 4
        pars = [A.metric_parameters(y, p[:, m]) for m in range(8)]
        result = A.consensus_analysis(y, p, metadata, pars)
        for groups in result["groups"].values():
            for row in groups:
                self.assertAlmostEqual(row["consensus_energy"]+row["dispersion_energy"], row["mean_member_deficit"], places=12)
                self.assertAlmostEqual(row["consensus_energy"]+row["student_deviation_energy"]+row["student_signed_cross_term"], row["student_deficit"], places=12)

    def test_ridge_matches_augmented_weighted_least_squares(self):
        rng = np.random.default_rng(101)
        n = 400
        Z = np.column_stack([np.ones(n), rng.normal(size=(n, 4)), np.ones(n)*3])
        w = rng.uniform(.1, 2, n)
        alpha = np.array([.2, 0, .7, -.2, .1, 0])
        G = gram(Z, w)
        beta, info = A.fit_ridge(G, alpha, [1, 3, 4, 5])
        X = Z[:, [1, 3, 4]]
        mu = np.average(X, axis=0, weights=w)
        sd = np.sqrt(np.average((X-mu)**2, axis=0, weights=w))
        Xs = np.column_stack([np.ones(n), (X-mu)/sd])
        augmented_X = np.vstack([Xs*np.sqrt(w/w.sum())[:, None], np.column_stack([np.zeros(3), np.sqrt(A.RIDGE)*np.eye(3)])])
        augmented_y = np.r_[(Z@alpha)*np.sqrt(w/w.sum()), np.zeros(3)]
        manual = np.linalg.lstsq(augmented_X, augmented_y, rcond=None)[0]
        np.testing.assert_allclose(Z@beta, Xs@manual, rtol=1e-10, atol=1e-10)
        self.assertEqual(info["active_columns"], 3)

    def test_planted_available_signal_and_noise_controls(self):
        rng = np.random.default_rng(33)
        def sample(n, signal):
            p, x, noise = rng.normal(size=(3, n))
            y = .7*p+(.9*x if signal else 0)+rng.normal(size=n)*.6
            # These are generated abstract metric moments, no clipping needed here.
            return np.column_stack([np.ones(n), p, x, noise, y]), np.abs(y)+.01
        for signal in (True, False):
            fit, fw = sample(6000, signal)
            held, hw = sample(6000, signal)
            G, H = gram(fit, fw), gram(held, hw)
            alpha, qi = A.q_definition(G, 0, prediction_index=1, target_index=4)
            base, _ = A.fit_ridge(G, alpha, [1])
            causal, _ = A.fit_ridge(G, alpha, [1, 2])
            noisy, _ = A.fit_ridge(G, alpha, [1, 2, 3])
            baseline = float(alpha@H@alpha)
            skills = [1-float((alpha-b)@H@(alpha-b))/baseline for b in (base, causal, noisy)]
            if signal:
                self.assertGreater(skills[1]-skills[0], .5)
                self.assertLess(abs(skills[2]-skills[1]), .01)
            else:
                self.assertLess(abs(skills[2]), .02)
            # Changing heldout labels can change q/error but never fit coefficients.
            perturbed = held.copy()
            perturbed[:, -1] *= -4
            alpha2, _ = A.q_definition(G, 0, prediction_index=1, target_index=4)
            beta2, _ = A.fit_ridge(G, alpha2, [1, 2, 3])
            np.testing.assert_array_equal(noisy, beta2)
            np.testing.assert_allclose(held@noisy, perturbed@noisy, atol=0, rtol=0)
            direct = np.dot(hw, (held@alpha-held@noisy)**2)
            self.assertAlmostEqual(direct/baseline, float((alpha-noisy)@H@(alpha-noisy))/baseline, places=12)

    def test_chain_split_is_whole_and_order_independent(self):
        md = np.array([(f, c) for f in (0, 1) for c in range(16)])
        counts = np.eye(4, dtype=int)[np.arange(32) % 4]*50
        h, d = A.make_halves(md, counts, np.arange(4))
        order = np.random.default_rng(62).permutation(32)
        h2, d2 = A.make_halves(md[order], counts[order], np.arange(4))
        np.testing.assert_array_equal(h[order], h2)
        np.testing.assert_array_equal(d[order], d2)
        for f in (0, 1):
            for c in range(4):
                self.assertEqual(h[(md[:, 0] == f) & (d == c)].sum(), 2)

    def test_full_gram_pipeline_shapes_and_fixed_response(self):
        rng = np.random.default_rng(29)
        md = np.array([(f, c) for f in (0, 1) for c in range(8)])
        gs = []
        for f, c in md:
            p = rng.normal(size=(180, 2))*.3
            signal = rng.normal(size=(180, 2))*.3
            y = .7*p+signal+rng.normal(size=(180, 2))*.12
            Z = np.zeros((180, 261))
            Z[:, 0] = 1
            Z[:, 1:3] = p
            Z[:, 3:5] = abs(p)
            Z[:, 5:7] = p*p
            Z[:, 7] = p[:, 0]*p[:, 1]
            Z[:, 8:10] = signal
            Z[:, 259:] = y
            gs.append([gram(Z, abs(y[:, t])) for t in range(2)])
        cc = np.eye(2, dtype=int)[np.arange(16) % 2]*180
        result, arrays = A.crossfit_grams(np.asarray(gs), md, cc, np.arange(2), bootstrap=40)
        skill = result["summaries"]["pooled"]["skill_R2"]
        self.assertEqual(skill.shape, (4, 2))
        self.assertTrue(np.all(skill[1]-skill[0] > .5))
        np.testing.assert_allclose(skill[1], skill[2], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(skill[2], skill[3], rtol=1e-10, atol=1e-10)
        self.assertEqual(arrays["sse"].shape, (16, 2, 4))
        self.assertEqual(result["conditional_chain_bootstrap"]["intervals_95"]["skill_by_target"].shape, (4, 2, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)

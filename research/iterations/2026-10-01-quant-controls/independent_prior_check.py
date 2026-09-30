"""Independent scalar/count verification. Synthetic data only; no real artifacts read."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("reviewed_prior_control", HERE/"prior_control.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def manual_vectors(marks, pi, original_R, matched_R=None):
    """Literal scalar loops: no occurrence ranks, broadcasting shortcut or old online helper."""
    counts = [0]*16
    transitions = [[0]*16 for _ in range(16)]
    contexts = [0]*16
    previous = None
    result = []
    for mark in marks:
        row = []
        for j in range(16):
            f = (counts[j]+16*pi[j])/(sum(counts)+16)
            if previous is None:
                t = independent = static = matched = pi[j]
            else:
                t = (transitions[previous][j]+16*original_R[previous, j])/(contexts[previous]+16)
                independent = (transitions[previous][j]+16*pi[j])/(contexts[previous]+16)
                static = original_R[previous, j]
                matched = (transitions[previous][j]+16*matched_R[previous, j])/(contexts[previous]+16) if matched_R is not None else t
            row.append([pi[j], f, t, independent, static, matched])
        result.append(np.array(row).T)
        if previous is not None:
            transitions[previous][int(mark)] += 1
            contexts[previous] += 1
        counts[int(mark)] += 1
        previous = int(mark)
    return np.array(result).reshape(-1, 6, 16)


def manual_prior(sequences):
    frequencies = np.zeros(16)
    counts = np.zeros((16, 16))
    nonempty = 0
    for seq in sequences:
        if len(seq):
            nonempty += 1
            for category in seq:
                frequencies[int(category)] += 1/len(seq)
        if len(seq) > 1:
            for i in range(1, len(seq)):
                counts[int(seq[i-1]), int(seq[i])] += 1/(len(seq)-1)
    pi = np.array([(frequencies[j]+1/16)/(nonempty+1) for j in range(16)])
    R = np.array([[(counts[i, j]+pi[j])/(sum(counts[i])+1) for j in range(16)] for i in range(16)])
    return pi, R


def main():
    started = time.monotonic()
    pi = np.arange(1, 17, dtype=float); pi /= pi.sum()
    R = (np.arange(256).reshape(16, 16)*7 % 31)+1.
    R /= R.sum(1)[:, None]
    alternative = np.roll(R, 3, axis=1)
    marks = np.array([0, 0, 3, 4, 0, 3, 15, 15, 0, 12, 4, 3, 1, 1, 9, 2,
                      7, 7, 7, 0, 3, 15, 4, 12, 8, 9, 2, 0, 3, 4, 15, 1, 0])
    full = manual_vectors(marks, pi, R, alternative)
    np.testing.assert_allclose(full.sum(2), 1, atol=1e-14)
    assert (full > 0).all()
    observed = full[np.arange(len(marks)), :, marks]
    np.testing.assert_allclose(module.original_probabilities(marks, pi, R), observed[:, :5], atol=1e-14)
    np.testing.assert_allclose(module.block_probabilities(marks, pi, R, alternative), observed[:, [1, 2, 5]], atol=1e-14)
    # Reconstruct each full prediction vector by querying all hypothetical CURRENT outcomes.
    # Each probe shares exactly the same past, so matching verifies no target-dependent support shortcut.
    for t in (0, 1, 3, 15, 16, 31, 32):
        counterfactual = np.column_stack([module.original_probabilities(np.r_[marks[:t], j], pi, R)[-1] for j in range(16)])
        np.testing.assert_allclose(counterfactual, full[t, :5], atol=1e-14)
    suffix = marks.copy(); suffix[16:] = 13
    np.testing.assert_allclose(manual_vectors(suffix, pi, R)[:17, :5], full[:17, :5], atol=1e-14)
    for n in (0, 1, 16, 17, len(marks)):
        p = module.original_probabilities(marks[:n], pi, R)
        np.testing.assert_allclose(p, observed[:n, :5], atol=1e-14)
        loss = module.loss_record(p)
        np.testing.assert_allclose(np.array(loss["loss_sums"])[0], -np.log(observed[:n, :5]).sum(0), atol=1e-12)
        np.testing.assert_allclose(np.array(loss["loss_sums"])[0], np.array(loss["loss_sums"])[1:].sum(0), atol=1e-12)
    same = module.block_probabilities(marks, pi, R, R)
    np.testing.assert_array_equal(same[:, 1], same[:, 2])
    fits = [np.array([], dtype=int), np.array([14]), np.array([0, 1, 0]), np.array([0, 0, 1, 7, 7, 3])]
    expected_pi, expected_R = manual_prior(fits)
    actual_pi, actual_R = module.eda.priors(fits)
    np.testing.assert_allclose(actual_pi, expected_pi, atol=1e-14)
    np.testing.assert_allclose(actual_R, expected_R, atol=1e-14)
    # Class-independent arbitrary evaluation has no effect on any fit prior; matched construction
    # receives only explicit fitting arrays. Opposite-half orchestration remains separately code-reviewed.
    long = np.tile(marks, 9)
    for seed in module.SEEDS:
        shuffled = module.eda.permute(long, "block128", seed, 997)
        a = module.original_probabilities(long, pi, R)
        b = module.original_probabilities(shuffled, pi, R)
        np.testing.assert_allclose(-np.log(a[:, 1]).sum(), -np.log(b[:, 1]).sum(), atol=1e-11)
        for start in range(0, len(long), 128):
            np.testing.assert_array_equal(np.sort(long[start:start+128]), np.sort(shuffled[start:start+128]))
    result = {"status": "passed", "empirical_data_read": False, "tests": [
        "independent literal scalar probabilities for all five original and three block estimators",
        "full counterfactual category vectors and positive normalization",
        "current/future independence, reset/empty/singleton/prefix cases",
        "loss slice additivity and identical-prior identity",
        "independent sequence-weighted prior calculation including empty/singleton fits",
        "Dirichlet F exchangeability and final partial block count preservation"],
        "code_sha256": hashlib.sha256((HERE/"prior_control.py").read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wall_seconds": time.monotonic()-started,
        "limits": "Synthetic math check plus separate source review; not empirical replication or nuisance-fit uncertainty."}
    with (HERE/"independent-prior-check-result.json").open("x") as out:
        json.dump(result, out, indent=2); out.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

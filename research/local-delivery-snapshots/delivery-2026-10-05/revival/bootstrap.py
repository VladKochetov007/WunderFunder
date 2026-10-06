"""Paired chain bootstrap of fixed model artifacts, not training uncertainty."""
import hashlib
import json
import sys

def score(s):
    import numpy as np
    w, sy, sp, syy, spp, syp = np.moveaxis(s, -1, 0)
    cov = syp - sy * sp / w
    den = np.sqrt(np.maximum(syy - sy*sy/w, 0) * np.maximum(spp - sp*sp/w, 0))
    return np.mean(cov / np.maximum(den, 1e-30), axis=-1)

def main():
    import numpy as np
    src, dest = sys.argv[1:3]
    z = np.load(src)
    names = z["names"].tolist()
    ref = names.index(sys.argv[3] if len(sys.argv) > 3 else "raw_reference")
    report = {"input_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
              "reference": names[ref], "bootstrap_replicates": 2000, "results": {},
              "caveat": "Conditional fixed-artifact chain intervals; reused validation, estimated train mask, no selection or training-noise correction. Not a public-score estimate."}
    for split in ("valid", "train", "valid_clean", "train_clean"):
        a = z[split]
        chains, inv = np.unique(z[split + "_chains"], return_inverse=True)
        c = np.zeros((len(names), len(chains), 2, 6))
        for k in range(len(names)):
            np.add.at(c[k], inv, a[k])
        point = score(c.sum(1))
        rng = np.random.default_rng(20261005)
        draws = np.empty((2000, len(names)))
        for b in range(2000):
            ix = rng.integers(len(chains), size=len(chains))
            draws[b] = score(c[:, ix].sum(1))
        delta = draws - draws[:, ref, None]
        report["results"][split] = {"chains": len(chains), "sequences": a.shape[1], "arms": {
            n: {"wp": float(point[k]), "delta_vs_raw": float(point[k]-point[ref]),
                "conditional_ci95": np.quantile(delta[:, k], [.025, .975]).tolist(),
                "bootstrap_fraction_positive": float((delta[:, k] > 0).mean())} for k, n in enumerate(names)}}
    with open(dest, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__":
    main()

from __future__ import annotations

import json

import numpy as np
from review import AUTHOR, ROOT, features, oracle, predict, sha


def main() -> None:
    frozen = json.loads((ROOT / "frozen_sources.json").read_text())
    for name, expected in frozen.items():
        assert sha(ROOT / name) == expected
    with np.load(AUTHOR / "witness.npz") as z:
        sample = {k: z[k] for k in z.files}
    labels, _, lags = oracle(sample["mid"])
    assert np.array_equal(labels, sample["targets"]) and np.array_equal(lags, sample["lags"])
    maxerror = 0.0
    checks = 0
    with np.load(AUTHOR / "models.npz") as z:
        for arm in ["row", "reset", "history"]:
            model = {k: z[f"nominal_0_{arm}_{k}"] for k in ["mean", "scale", "beta", "intercept"]}
            original = predict(model, features(sample, arm))
            modified = {k: v.copy() for k, v in sample.items()}
            modified["values"][:, 49:] += 999
            modified["ages"][:, 49:] += 777
            modified["dt"][:, 49:] *= 31
            modified["updates"][:, 49:] = ~modified["updates"][:, 49:]
            actual = predict(model, features(modified, arm))
            error = float(np.abs(original[:, :49] - actual[:, :49]).max())
            assert error == 0.0
            maxerror = max(maxerror, error)
            checks += 1
            latent = {k: v.copy() for k, v in sample.items()}
            latent["mid"][:, 49:] += 100
            latent["factor"][:, 49:] *= -99
            changed, _, _ = oracle(latent["mid"])
            assert not np.array_equal(changed, labels)
            assert np.array_equal(original, predict(model, features(latent, arm)))
            checks += 1
    manifest = json.loads((AUTHOR / "PROVENANCE.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        assert sha(AUTHOR / name) == expected, name
    result = dict(
        passed=True,
        witness_target_lag_recreated=True,
        actual_saved_model_future_canaries=checks,
        max_prediction_canary_error=maxerror,
        author_artifacts_unchanged=len(manifest["files_sha256"]),
        frozen_review_source_hashes_verified=True,
    )
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

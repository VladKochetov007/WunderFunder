"""Synthetic known-answer and artifact-linkage checks for the independent scorer."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from campaign_score import (
    ROOT,
    artifact,
    file_sha256,
    mmap_stored_npz,
    moments,
    pooled,
    score_sequence,
    summary,
    validate_member,
)


def fails(action: object, description: str) -> None:
    """Require the specified invalid input to fail with a clear boundary error."""
    try:
        action()
    except ValueError:
        return
    raise AssertionError(description)


def main() -> None:
    """Test arithmetic independently, including complete provenance rejection."""
    rng = np.random.default_rng(481)
    y = rng.normal(size=(7, 257, 2)).astype(np.float32) * 3
    mask = rng.random(y.shape[:2]) < 0.7
    exact = {"perfect": [], "opposite": []}
    soft = {str(lam): {name: [] for name in exact} for lam in (0.02, 0.05)}
    blocks = {name: [] for name in exact}
    for yy, mm in zip(y, mask, strict=True):
        row = score_sequence(yy, mm, {"perfect": yy, "opposite": -yy})
        for name in exact:
            exact[name].append(row["moments"][name])
            blocks[name].append(row["blocks"][name])
            for lam in soft:
                soft[lam][name].append(row["soft"][lam][name])
    m = {name: np.stack(values) for name, values in exact.items()}
    s = {
        lam: {name: np.stack(values) for name, values in variants.items()}
        for lam, variants in soft.items()
    }
    design = {"comparisons": [["perfect", "opposite"]], "n_boot": 2000, "bootstrap_seed": 271828}
    result = summary(
        m, s, blocks, np.array([0, 0, 1, 1, 2, 3, 3]), np.array([1, 1, 2, 3, 4, 5, 5]), design
    )
    assert abs(result["exact_global_wp"]["perfect"]["wp"] - 1) < 1e-12
    assert abs(result["exact_global_wp"]["opposite"]["wp"] + 1) < 1e-12
    paired = result["paired_component_bootstrap"]["perfect_minus_opposite"]
    assert abs(paired["mean_diff"] - 2) < 1e-12
    assert paired["bootstrap_sd"] < 1e-12
    assert np.allclose(paired["ci95"], [2, 2], atol=1e-12)
    arbitrary = rng.normal(size=(257, 2)).astype(np.float32)
    observed = score_sequence(y[0], mask[0], {"random": arbitrary})["moments"]["random"]
    selection = mask[0] & (np.arange(257) >= 99)
    oracle = []
    for k in range(2):
        yy = np.clip(y[0, selection, k], -2, 2).astype(np.longdouble)
        pp = np.clip(arbitrary[selection, k], -2, 2).astype(np.longdouble)
        w = abs(yy)
        dy = yy - sum(w * yy) / sum(w)
        dp = pp - sum(w * pp) / sum(w)
        oracle.append(float(sum(w * dy * dp) / np.sqrt(sum(w * dy * dy) * sum(w * dp * dp))))
    assert abs(pooled(observed[None])["wp"] - np.mean(oracle)) < 1e-12
    for bad in (np.nan, np.inf, -np.inf):
        p = y[0].copy()
        p[100, 0] = bad
        fails(lambda p=p: score_sequence(y[0], mask[0], {"bad": p}), "raw nonfinite clipped away")
        fails(
            lambda bad=bad: moments(np.array([0.0]), np.array([bad]), np.array([1.0])),
            "zero-target weight hid nonfinite",
        )
    constant = score_sequence(
        np.ones((257, 2)), np.ones(257, dtype=bool), {"constant": np.ones((257, 2))}
    )
    assert pooled(constant["moments"]["constant"][None])["wp"] == 0
    with tempfile.TemporaryDirectory(dir=ROOT / "research/campaign_2026-09-30/audit") as tmp:
        base = Path(tmp)
        target_file = base / "targets.npz"
        targets = np.arange(40000, dtype=np.float32).reshape(1, 20000, 2)
        np.savez(
            target_file,
            y=targets,
            mask=np.ones((1, 20000), dtype=bool),
            row_group=np.array([0]),
            market_class=np.array([1]),
        )
        mapped = mmap_stored_npz(target_file)
        assert isinstance(mapped["y"], np.memmap) and np.array_equal(mapped["y"], targets)
        checkpoint = base / "model_ep0.pt"
        checkpoint.write_bytes(b"fixture checkpoint")
        prediction = base / "pred.npy"
        np.save(prediction, np.zeros((1, 20000, 2), dtype=np.float32))
        provenance = {
            "args": {
                "name": "fixture",
                "checkpoint_policy": "fixed",
                "checkpoint_epoch": 0,
                "no_eval_during_fit": True,
            },
            "selected_epoch": 0,
            "split_manifest_sha256": "split-hash",
            "fit_row_groups": {"train": [2], "valid": [3]},
            "evaluation_row_groups": [0],
            "checkpoint_sha256": {checkpoint.name: file_sha256(checkpoint)},
        }
        sidecar = {
            "run": "fixture",
            "split": "valid",
            "epoch": 0,
            "gate_mode": "soft",
            "row_groups": [0],
            "split_manifest_sha256": "split-hash",
            "checkpoint_sha256": file_sha256(checkpoint),
            "prediction_sha256": file_sha256(prediction),
        }
        job = {
            "name": "fixture",
            "status": "finished",
            "exit_code": 0,
            "kind": "gpu",
            "peak_observed_vram_bytes": 123,
        }
        identities = {}
        for key, path, value in [
            ("prediction", prediction, None),
            ("checkpoint", checkpoint, None),
            ("provenance", base / "provenance.json", provenance),
            ("sidecar", base / "sidecar.json", sidecar),
            ("job", base / "job.json", job),
        ]:
            if value is not None:
                path.write_text(json.dumps(value))
            identities[key] = {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}

        def assert_fit(train: list[int], valid: list[int]) -> None:
            assert train == [2] and valid == [3]

        split = SimpleNamespace(
            sha256="split-hash", evaluation_groups=[0], assert_fit_groups=assert_fit
        )
        manifest = {"fit": {"train": [2], "valid": [3]}}
        declared = {
            "run": "fixture",
            "epoch": 0,
            "fit_source": "train+valid",
            "checkpoint": str(checkpoint.relative_to(ROOT)),
        }
        validate_member(declared, identities, split, manifest, [0])
        for key, value in [
            ("checkpoint_sha256", "wrong-hash"),
            ("row_groups", [1]),
            ("epoch", 1),
            ("split_manifest_sha256", "other-split"),
        ]:
            bad = sidecar | {key: value}
            path = base / "sidecar.json"
            path.write_text(json.dumps(bad))
            identities["sidecar"]["sha256"] = file_sha256(path)
            fails(
                lambda: validate_member(declared, identities, split, manifest, [0]),
                f"sidecar {key} accepted",
            )
        path.write_text(json.dumps(sidecar))
        identities["sidecar"]["sha256"] = file_sha256(path)
        checkpoint.write_bytes(b"changed checkpoint bytes")
        fails(
            lambda: validate_member(declared, identities, split, manifest, [0]),
            "changed checkpoint accepted",
        )
        fails(
            lambda: artifact({"path": str(target_file.relative_to(ROOT)), "sha256": "wrong"}),
            "target hash change accepted",
        )
    output = {
        "status": "pass",
        "known_perfect_wp": 1.0,
        "known_opposite_wp": -1.0,
        "known_paired_delta": 2.0,
        "known_bootstrap_sd": paired["bootstrap_sd"],
        "longdouble_mask_clip_oracle": True,
        "nonfinite_before_clip": True,
        "stored_npz_zero_copy_memmap": True,
        "checkpoint_prediction_sidecar_split_row_epoch_hash_failures_rejected": True,
        "candidate_data_scored": False,
        "source_sha256": file_sha256(Path(__file__).with_name("campaign_score.py")),
        "test_source_sha256": file_sha256(Path(__file__)),
    }
    destination = Path(sys.argv[1])
    with destination.open("x") as stream:
        stream.write(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()

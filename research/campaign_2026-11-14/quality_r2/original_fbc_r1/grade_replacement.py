"""Score the predeclared TL3C slot replacement using raw aligned predictions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE = Path("/home/vlad/development/wunder/research_tmp/chrono_root/cache")
sys.path[:0] = [
    str(ROOT / "src"),
    str(OUT.parent / "blend_audit_r1"),
    str(OUT.parent.parent / "local_r1"),
]
from audit_blends import component_layout, score_from_sums  # noqa: E402
from grade_submissions import predictions, score  # noqa: E402

from wnn.gmetric import wp_sums  # noqa: E402


def digest(path: Path) -> str:
    """Hash one persisted prediction or report without loading its entire contents."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def native(folder: Path, groups: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Require a completed exact full-valid native replay and align source row groups."""
    report = json.loads((folder / "report.json").read_text())
    if (
        not report.get("validation_completed")
        or report["data_sha256"]
        != "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
    ):
        raise ValueError("native replay incomplete or data identity changed")
    rows = report["row_groups"]
    if len(rows) != 1873 or len(set(rows)) != 1873 or set(rows) != set(map(int, groups)):
        raise ValueError("native coverage mismatch")
    path = folder / "predictions.npy"
    actual_hash = digest(path)
    if actual_hash != report["predictions"]["sha256"]:
        raise ValueError("native prediction identity mismatch")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (1873, 20000, 2) or array.dtype != np.float32:
        raise ValueError("native array layout changed")
    position = {row: i for i, row in enumerate(rows)}
    aligned = np.concatenate(
        [array[position[int(g)]][m] for g, m in zip(groups, mask, strict=True)]
    )
    if not np.isfinite(aligned).all():
        raise ValueError("nonfinite native predictions")
    return aligned, {
        "prediction_sha256": actual_hash,
        "report_sha256": digest(folder / "report.json"),
        "recorded_wp": report["metrics"]["wp"],
        "timing": report["timing"],
        "artifact": report["artifact"],
    }


def main() -> None:
    """Write a proxy comparison and component interval; never approve a package."""
    with np.load(CACHE / "valid_targets.npz") as archive:
        groups, mask, y = archive["row_group"], archive["mask"], archive["y"]
    if len(groups) != 1873 or int(mask.sum()) != 3528273:
        raise ValueError("official scored membership changed")
    selected_y = y[mask]
    tl3c, baseline_source = native(
        ROOT / "submissions/tl3c_pr49_20261029/blend-int8-full-v3", groups, mask
    )
    fbc, fbc_source = native(OUT / "full_valid", groups, mask)
    package = json.loads((OUT / "replacement_receipt.json").read_text())
    isolated = json.loads((OUT / "receipt.json").read_text())
    if (
        baseline_source["artifact"]["sha256"] != package["source_artifacts"]["tl3c"]["sha256"]
        or fbc_source["artifact"]["sha256"] != isolated["artifact"]["sha256"]
    ):
        raise ValueError("native package identities differ from the frozen source packages")
    kd90_path = CACHE / "pred_l14_kd90_s3_valid.npy"
    kd90_metadata = json.loads(kd90_path.with_suffix(".json").read_text())
    kd90_hash = digest(kd90_path)
    if (
        kd90_metadata.get("run") != "l14_kd90_s3"
        or kd90_metadata.get("epoch") != 3
        or kd90_metadata.get("checkpoint_sha256") != package["members"][0]["checkpoint"]["sha256"]
        or kd90_metadata.get("prediction_sha256") != kd90_hash
        or set(kd90_metadata["row_groups"]) != set(map(int, groups))
    ):
        raise ValueError("KD90 prediction/checkpoint identity differs from the packaged member")
    kd90 = predictions("l14_kd90_s3", groups, mask)
    if not np.isfinite(kd90).all():
        raise ValueError("nonfinite KD90 predictions")
    replacement = (tl3c.astype(np.float64) - 0.34 * fbc + 0.34 * kd90).astype(np.float32)
    scores = {
        name: score(selected_y, value)["wp"]
        for name, value in [
            ("tl3c", tl3c),
            ("original_fbc1k", fbc),
            ("kd90", kd90),
            ("replacement_proxy", replacement),
        ]
    }
    if (
        abs(scores["tl3c"] - baseline_source["recorded_wp"]) > 1e-8
        or abs(scores["original_fbc1k"] - fbc_source["recorded_wp"]) > 1e-8
    ):
        raise ValueError("native score reconstruction differs")
    components, row_component, _ = component_layout(groups)
    index = {component: i for i, component in enumerate(components)}
    sufficient = np.zeros((2, len(components), 2, 6), dtype=np.float64)
    offset = 0
    for g, m in zip(groups, mask, strict=True):
        count = int(m.sum())
        sl = slice(offset, offset + count)
        for k, pred in enumerate((tl3c, replacement)):
            sufficient[k, index[row_component[int(g)]]] += wp_sums(selected_y[sl].T, pred[sl].T)
        offset += count
    for k, name in enumerate(("tl3c", "replacement_proxy")):
        if abs(score_from_sums(sufficient[k].sum(axis=0)) - scores[name]) > 1e-8:
            raise ValueError("component metric reconstruction differs")
    rng = np.random.default_rng(20261114)
    boot = []
    for _ in range(2000):
        sample = rng.integers(0, len(components), len(components))
        boot.append(
            score_from_sums(sufficient[1, sample].sum(axis=0))
            - score_from_sums(sufficient[0, sample].sum(axis=0))
        )
    delta = scores["replacement_proxy"] - scores["tl3c"]
    result = {
        "status": "proxy_improved_requires_native" if delta > 0 else "proxy_not_improved",
        "scores": scores,
        "delta_vs_tl3c": delta,
        "paired_component_interval95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "components": len(components),
        "bootstrap_replicates": 2000,
        "coefficients": [0.34, 0.33, 0.33],
        "fits": 0,
        "sources": {
            "tl3c": baseline_source,
            "original_fbc1k": fbc_source,
            "kd90_prediction_sha256": digest(CACHE / "pred_l14_kd90_s3_valid.npy"),
        },
        "limits": [
            "Exposed full-valid, conditional fixed-prediction component uncertainty only.",
            "Native TL3C subtraction plus Torch KD90 is a proxy; actual ZIP INT8 quality/runtime still required.",
            "Original TL3C fit-membership equality with the local split is not established.",
        ],
    }
    (OUT / "REPLACEMENT_PROXY_R1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "scores", "delta_vs_tl3c", "paired_component_interval95")
            }
        )
    )


if __name__ == "__main__":
    main()

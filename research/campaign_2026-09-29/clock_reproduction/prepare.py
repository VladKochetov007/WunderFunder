"""Freeze a fresh component-disjoint clock reproduction panel from metadata alone."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SEED = "clock-reproduction-fold2-20260929-v1"
FIT_COUNT = 160
EVALUATION_COUNT = 32


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_order(pair: tuple[int, int]) -> bytes:
    return hashlib.sha256(f"{SEED}:{pair[0]}:{pair[1]}".encode()).digest()


def main() -> None:
    path = ROOT / "cache/chain_folds.npz"
    with np.load(path) as metadata:
        split = metadata["split"]
        group = metadata["row_group"]
        component = metadata["chain"]
        fold = metadata["fold"]
    prior = set()
    for name in ("sample_manifest.json", "preregistration_v2.json"):
        previous = json.loads((HERE.parent / "provided_decode" / name).read_text())
        prior.update(previous["evaluation_components"])
    valid = sorted(
        [(int(g), int(c)) for g, c in zip(group[(split == "valid") & (fold == 2)],
                                                component[(split == "valid") & (fold == 2)], strict=True)
         if int(c) not in prior],
        key=hash_order,
    )
    evaluation: list[tuple[int, int]] = []
    seen: set[int] = set()
    for pair in valid:
        if pair[1] not in seen:
            evaluation.append(pair)
            seen.add(pair[1])
        if len(evaluation) >= EVALUATION_COUNT:
            break
    if len(evaluation) != EVALUATION_COUNT:
        raise ValueError("Insufficient fresh fold-2 components")
    fit = sorted(
        [(int(g), int(c)) for g, c in zip(group[split == "train"],
                                                component[split == "train"], strict=True)
         if int(c) not in seen and int(c) not in prior],
        key=hash_order,
    )[:FIT_COUNT]
    if len(fit) != FIT_COUNT or {c for _, c in fit} & seen:
        raise ValueError("Fit/evaluation component check failed")
    manifest = {
        "version": "clock-reproduction-v1",
        "seed": SEED,
        "chain_metadata_sha256": hash_file(path),
        "fit_train_groups": [g for g, _ in fit],
        "fit_components": [c for _, c in fit],
        "evaluation_valid_groups": [g for g, _ in evaluation],
        "evaluation_components": [c for _, c in evaluation],
        "prior_evaluation_components_excluded": sorted(prior),
        "selected_using_metadata_only": True,
        "primary_regime": 3,
        "train_fit": "class3 target lattice; modal spread from TRAIN labels; ask-move votes from unclamped TRAIN target jumps",
        "minimum_spread_support": 3,
        "minimum_spread_confidence": 0.90,
        "minimum_move_support": 3,
        "minimum_move_confidence": 0.80,
        "code_event": "any changed i0_p0..i0_p21 representation field",
        "quote_event": "known decoded nonzero ask or bid change",
        "unknown_handling": "invalidate all horizons spanning an unrecognized transition or initial spread",
        "evaluation_labels_read_after_feature_only_gate": True,
        "feature_only_gate": "class3 TRAIN-map quote4 eligibility >=1000 required rows and >=5% of panel required rows",
        "primary_target_rule": "nearest-even rounded mean of next four decoded quote-event midpoints, common future reference",
        "matched_control": "nearest-even rounded mean of next four code-event midpoints, same eligible scored rows",
        "auxiliary_horizons": [8, 24],
        "auxiliary_label": "clipped 0.65687 times future midpoint mean minus current midpoint",
        "auxiliary_comparison": "code events versus nonzero decoded quote events on identical eligible scored rows, weighted by |clipped target t0|+|clipped target t1|",
        "promotion_gate": "Recommend one batch32 GPU screen only if clock correction addresses >=25% target-weighted scored class3 mass and quote4 rule gains >=0.05 absolute exact-pair fraction over code4 matched control; identify failures/unknown mass and no causal gain claim",
        "no_external_data": True,
        "no_gpu_run": True,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    destination = HERE / "preregistration.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(destination), "sha256": hash_file(destination),
                      "fit": len(fit), "evaluation": len(evaluation), "prior_excluded": len(prior)}))


if __name__ == "__main__":
    main()

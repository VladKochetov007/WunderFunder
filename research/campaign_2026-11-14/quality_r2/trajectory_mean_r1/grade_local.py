"""Grade the four predeclared local endpoints and conditional seed5 triggers."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
CACHE = Path("/home/vlad/development/wunder/research_tmp/chrono_root/cache")
RUNS = CACHE.parent / "artifacts/runs"
SPLIT = ROOT / "research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json"
SPLIT_SHA256 = "6d23a1b76f66d8001dac805d2b218a3b139e4eebe555f0c9b96fe1263ce186f0"
sys.path[:0] = [str(ROOT / "src"), str(OUT.parent.parent / "local_r1")]
from grade_submissions import predictions, score  # noqa: E402

ARMS = {
    "kd90": ("q14s_local_kd90_s3", "q14s_swa23_kd90_s3"),
    "kd100": ("q14s_local_kd100_s3", "q14s_swa23_kd100_s3"),
}


def digest(path: Path) -> str:
    """Stream a persisted prediction's identity."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def validate_identity(
    run: str, metadata: dict, provenance: dict, *, checkpoint_file: str = "model_ep3.pt"
) -> None:
    """Bind source membership and checkpoint bytes before any quality score is used."""
    assert digest(SPLIT) == SPLIT_SHA256
    split = json.loads(SPLIT.read_text())
    assert metadata["split"] == "valid"
    assert metadata["campaign_split_manifest_sha256"] == SPLIT_SHA256
    assert metadata["split_manifest_sha256"] == provenance["split_manifest_sha256"] == SPLIT_SHA256
    groups = np.asarray(metadata["row_groups"], dtype="<i8")
    assert metadata["evaluation_membership_sha256"] == hashlib.sha256(groups.tobytes()).hexdigest()
    assert len(groups) == len(set(groups)) and set(groups) == set(split["evaluation"]["valid"])
    args = provenance["args"]
    assert args["name"] == run and args["train_source"] == "train"
    assert args["include_heldout_valid"] is False
    assert args["split_manifest_sha256"] == SPLIT_SHA256
    assert set(provenance["fit_row_groups"]) == set(split["fit"])
    for source, expected in split["fit"].items():
        actual = provenance["fit_row_groups"][source]
        assert len(actual) == len(set(actual)) == len(expected) and set(actual) == set(expected)
    assert checkpoint_file in ("model_ep3.pt", "model.pt")
    assert metadata["checkpoint_file"] == checkpoint_file
    assert digest(RUNS / run / checkpoint_file) == metadata["checkpoint_sha256"]
    assert metadata["checkpoint_sha256"] == provenance["checkpoint_sha256"][checkpoint_file]
    if "derivation" in provenance:
        receipt = json.loads((OUT / f"{run}.json").read_text())
        derivation = provenance["derivation"]
        assert receipt["derivation"] == derivation
        assert receipt["checkpoint_sha256"] == provenance["checkpoint_sha256"]
        parent = RUNS / derivation["parent_run"]
        assert digest(parent / "provenance.json") == derivation["parent_provenance_sha256"]
        for epoch, expected in zip(
            derivation["parent_epochs"], derivation["parent_checkpoint_sha256"], strict=True
        ):
            assert digest(parent / f"model_ep{epoch}.pt") == expected


def main() -> None:
    """Keep seed3 evidence distinct from two-seed confirmation and final promotion."""
    for pair in ARMS.values():
        for run in pair:
            for extension in ("json", "npy"):
                if not (CACHE / f"pred_{run}_valid.{extension}").is_file():
                    raise FileNotFoundError(f"incomplete local export: {run}")
    with np.load(CACHE / "valid_targets.npz") as data:
        groups, mask, y = data["row_group"], data["mask"], data["y"]
    assert len(groups) == 1873 and int(mask.sum()) == 3528273
    selected_y = y[mask]
    scores, sources = {}, {}
    for pair in ARMS.values():
        for run in pair:
            path = CACHE / f"pred_{run}_valid.npy"
            metadata = json.loads(path.with_suffix(".json").read_text())
            provenance = json.loads((RUNS / run / "provenance.json").read_text())
            validate_identity(run, metadata, provenance)
            actual = digest(path)
            assert metadata["run"] == run and metadata["epoch"] == 3
            assert metadata["inference_precision"] == "fp32" and metadata["feature_only"] is True
            assert metadata["campaign_export_scope"] == "full-valid-feature-only"
            assert metadata["prediction_sha256"] == actual
            assert metadata["checkpoint_sha256"] == provenance["checkpoint_sha256"]["model_ep3.pt"]
            assert metadata["run_provenance_sha256"] == digest(RUNS / run / "provenance.json")
            assert len(metadata["row_groups"]) == 1873 and set(metadata["row_groups"]) == set(
                map(int, groups)
            )
            scores[run] = float(score(selected_y, predictions(run, groups, mask))["wp"])
            sources[run] = {
                "prediction_sha256": actual,
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "provenance_sha256": metadata["run_provenance_sha256"],
                "derivation": provenance.get("derivation"),
            }
    for raw, _ in ARMS.values():
        history = json.loads((RUNS / raw / "history.json").read_text())
        assert len(history) == 4 and abs(scores[raw] - history[3]["weighted_pearson"]) <= 1e-5
    pure_delta = scores[ARMS["kd100"][0]] - scores[ARMS["kd90"][0]]
    best_raw = max(scores[pair[0]] for pair in ARMS.values())
    gains = {arm: scores[pair[1]] - scores[pair[0]] for arm, pair in ARMS.items()}
    useful_means = [
        arm for arm, pair in ARMS.items() if gains[arm] >= 0.0002 and scores[pair[1]] > best_raw
    ]
    confirm = set()
    if pure_delta >= 0.0003:
        confirm.update(("q14s_local_kd90_s5", "q14s_local_kd100_s5"))
    for arm in useful_means:
        confirm.add("q14s_local_kd90_s5")
        if arm == "kd100":
            confirm.add("q14s_local_kd100_s5")
    result = {
        "status": "complete_exposed_local_seed3_grade",
        "scores": scores,
        "pure_teacher_minus_kd90": pure_delta,
        "swa23_minus_raw": gains,
        "pure_teacher_confirmation_trigger": pure_delta >= 0.0003,
        "useful_mean_arms": useful_means,
        "proposed_confirmation_runs": sorted(confirm),
        "confirmation_is_launched": False,
        "resource_admission_required": True,
        "final_refit_admitted": False,
        "source_inputs": sources,
        "scored_rows": int(mask.sum()),
        "metric": "global_wp_clip2_v1; independent reference checked",
        "rule_sha256": digest(OUT.parent / "LOCAL_CONFIRMATION_R3.json"),
        "scope": "One child seed, full valid repeatedly exposed; no final production or public improvement claim.",
    }
    (OUT / "LOCAL_ENDPOINTS_R1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "scores",
                    "pure_teacher_minus_kd90",
                    "swa23_minus_raw",
                    "proposed_confirmation_runs",
                )
            }
        )
    )


if __name__ == "__main__":
    main()

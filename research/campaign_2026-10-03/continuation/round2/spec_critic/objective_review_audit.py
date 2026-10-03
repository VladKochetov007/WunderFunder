from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OWN = Path(__file__).resolve().parent
ACTOR = OWN.parent / "objective"


def digest(path: Path) -> str:
    """Hash bounded frozen artifacts and source files only."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    review = json.loads((OWN / "objective_review_results.json").read_text())
    for name, expected in review["actor_sha256"].items():
        assert digest(ACTOR / name) == expected
    for name, expected in review["prior_blind_I1_artifacts_sha256"].items():
        assert digest(OWN / name) == expected
    actor_result = json.loads((ACTOR / "results.json").read_text())
    primary, sham = digest(ACTOR / "PREREG.json"), digest(ACTOR / "PREREG_SHAM.json")
    assert actor_result["prereg_sha256"] == primary
    assert sham in json.dumps(actor_result["preregistrations_sha256"])
    provenance_path = ROOT / "artifacts/runs/c20261003_combo_s3_f0/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    fields = {
        name: provenance["args"].get(name)
        for name in (
            "loss",
            "loss_norm",
            "distill_alpha",
            "distill_boot_h",
            "ema_teacher",
            "ema_metric_weight",
            "aux_struct",
            "aux_next",
            "chunk",
            "prop_power",
            "prop_eps",
            "valid_true_mask",
            "since_weight",
            "since_floor",
            "optim",
            "spread_aux_mode",
            "spread_cache_root",
            "aux_next_k",
            "class_weight",
            "episode_weight",
        )
    }
    assert fields["distill_alpha"] == 0.5 and fields["distill_boot_h"] == 8
    assert fields["ema_teacher"] == 1 and not fields["ema_metric_weight"]
    assert fields["aux_struct"] == fields["aux_next"] == 0.3
    timeline_names = [
        "PREREG.json",
        "logs/stream_01.log",
        "logs/analyze_01.log",
        "PREREG_SHAM.json",
        "logs/sham_01.log",
    ]
    times = {name: (ACTOR / name).stat().st_mtime_ns for name in timeline_names}
    assert list(times.values()) == sorted(times.values())
    artifact_list = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (ACTOR / "artifact_hashes.txt").read_text().splitlines()
        if line.strip()
    }
    for name, expected in review["actor_sha256"].items():
        actor_relative = str((ACTOR / name).relative_to(ROOT))
        if actor_relative in artifact_list:
            assert artifact_list[actor_relative] == expected
    output = {
        "prior_blind_I1_files_unchanged": True,
        "actor_frozen_hashes_match": True,
        "primary_prereg_hash_matches_results": True,
        "sham_prereg_hash_matches_results": True,
        "filesystem_timeline_monotone": True,
        "timeline_mtime_ns": times,
        "timeline_limit": "file mtimes and recorded hashes support order, not a tamper-proof preregistration service",
        "actual_combo_provenance_sha256": digest(provenance_path),
        "actual_combo_args": fields,
        "actual_combo_fit_counts": {
            s: len(groups) for s, groups in provenance["fit_row_groups"].items()
        },
        "review_source_sha256": {
            name: digest(ROOT / name)
            for name in ("scripts/train_rnn.py", "src/wnn/spread_aux.py", "src/wnn/features.py")
        },
    }
    (OWN / "objective_review_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

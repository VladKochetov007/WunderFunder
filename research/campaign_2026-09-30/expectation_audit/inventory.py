from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DESTINATION = Path(__file__).with_name("source_inventory.json")
ARCHIVE_NAMES = {"ab18": "ab18_wmse_struct"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory() -> dict[str, object]:
    """Inspect small existing metadata and ZIPs without targets or predictions."""
    sources = ROOT / "research/campaign_2026-09-29/artifacts"
    reports = {
        scenario: json.loads((sources / f"nested_public_calibration_{scenario}.json").read_text())
        for scenario in ("pair", "single")
    }
    pair = reports["pair"]
    archives: dict[str, object] = {}
    for name, public in zip(
        pair["submission_order"], pair["public_scores_user_reported"], strict=True
    ):
        candidates = list((ROOT / "submissions" / ARCHIVE_NAMES.get(name, name)).glob("*.zip"))
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous or missing archive for {name}: {candidates}")
        path = candidates[0]
        archive_hash = digest(path.read_bytes())
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Invalid archive: {path}")
            entries = archive.namelist()
            solution = archive.read("solution.py").decode()
            members = (
                json.loads(archive.read("members.json")) if "members.json" in entries else None
            )
            archives[name] = {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": archive_hash,
                "crc_integrity": "pass",
                "public_user_reported": public,
                "public_association": "user-reported filename; portal-uploaded bytes not independently observed",
                "calibration_prediction_spec": pair["submission_prediction_specs"][name],
                "embedded_members": members,
                "payload_sha256": {
                    item: digest(archive.read(item)) for item in entries if not item.endswith("/")
                },
                "v2_member0_branch_detected": 'members = [{**members[0], "coef": 1.0}]' in solution,
                "recorded_archive_hash_matches": (
                    archive_hash == pair["local_archive_sha256"][name]
                    if name in pair["local_archive_sha256"]
                    else None
                ),
                "source_label_scope": "manifest declarations and static solution source; cached-prediction-to-weight parity not replayed",
            }
    summaries = {}
    for scenario, report in reports.items():
        family = json.loads((sources / f"nested_public_family_holdout_{scenario}.json").read_text())
        summaries[scenario] = {
            key: report[key]
            for key in (
                "nested_loso_rmse",
                "nested_loso_mae",
                "nested_loso_bias",
                "nested_loso_error_range",
                "full_selected_variant",
                "full_variant_loso_rmse",
                "full_slope",
                "full_intercept",
                "candidate_forecasts_exploratory",
            )
        } | {
            key: family[key]
            for key in ("family_heldout_rmse", "family_heldout_mae", "family_heldout_bias")
        }
    paths = [
        *(sources / f"nested_public_calibration_{s}.json" for s in reports),
        *(sources / f"nested_public_family_holdout_{s}.json" for s in reports),
        ROOT / "scripts/lb_calibration.py",
        ROOT / "scripts/campaign_public_calibration.py",
        ROOT / "scripts/campaign_calibration_audit.py",
        ROOT / "scripts/holdout_compare.py",
        ROOT / "scripts/decide.py",
        ROOT / "research/artifacts/lb_calibration.json",
        ROOT / "research/experiments.jsonl",
        ROOT / "research/campaign_2026-09-30/RESUME_CONTRACT.md",
        ROOT / "research/campaign_2026-09-30/audit/reference_fourfold_design.json",
        Path(__file__),
    ]
    return {
        "scope": "Existing metadata and archive reads only; no target, scoring-mask, prediction-array, checkpoint, fitting, or GPU access",
        "archives": archives,
        "reported_calibration_summaries": summaries,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in paths},
    }


if __name__ == "__main__":
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    result = inventory()
    with DESTINATION.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"archives_verified": len(result["archives"]), "output": str(DESTINATION)}))

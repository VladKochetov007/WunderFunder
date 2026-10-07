from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_distill_cache import _estimated_cache_bytes, _selected_main_variants

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "research/campaign_2026-10-06/bandit_continuation"


def test_e512_plan_binds_all_own_teachers_and_pending_e510_pair() -> None:
    plan_path = CAMPAIGN / "r1_family_loss/E512_COHORT_PLAN.json"
    plan = json.loads(plan_path.read_text())
    source_families = json.loads((CAMPAIGN / "SOURCE_FAMILIES.json").read_text())
    sources = {item["run"]: item for item in plan["source_families"]}

    assert len(sources) == 8
    assert sum(item["coefficient"] for item in plan["source_families"]) == 0.8125
    for source in source_families["families"]:
        actual = sources[source["run"]]
        assert actual["family"] == source["family"]
        assert actual["epoch"] == source["epoch_zero_based"]
        assert actual["coefficient"] == source["main_coefficient"]
        if source["status"] == "existing_audited":
            assert actual["checkpoint_sha256"] == source["checkpoint_sha256"]
        else:
            assert actual["checkpoint_sha256"] is None
            assert actual["prediction_status"] == "pending_e510_terminal_output"

    assert plan["cache"]["output_bytes"] == _estimated_cache_bytes(
        11_940, _selected_main_variants(["source"])
    )
    assert plan["profile"]["no_fit"] is True
    assert plan["production"]["fit_count"] == 4


def test_teacher_asset_manifest_is_small_and_matches_plan_pin() -> None:
    asset_root = CAMPAIGN / "r1_family_loss/teacher_assets_v1"
    manifest_path = asset_root / "assets-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    plan = json.loads((CAMPAIGN / "r1_family_loss/E512_COHORT_PLAN.json").read_text())
    manifest = json.loads(manifest_bytes)

    assert (
        hashlib.sha256(manifest_bytes).hexdigest()
        == plan["basis"]["teacher_assets_manifest_sha256"]
    )
    assert manifest["dataset"] == plan["basis"]["teacher_assets_dataset"]
    assert len(manifest["files"]) == 18
    assert sum(item["size"] for item in manifest["files"]) < 20_000_000
    assert all(item["filename"].endswith(".wnn") for item in manifest["files"])
    assert {Path(path).name for path in asset_root.iterdir()} == {
        "assets-manifest.json",
        "dataset-metadata.json",
        *(item["filename"] for item in manifest["files"]),
    }

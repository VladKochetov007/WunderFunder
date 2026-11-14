"""Check receiver regressions with explicit fixtures, never synthetic quality evidence."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from collect_results import PACKET, CollectionError, check_provenance, collect, expected_specs


def main() -> None:
    """Exercise summary presence and source-qualified membership against producer metadata."""
    name = "q14s_a50_s3"
    spec = expected_specs()[name]
    actual = Path(
        "/home/vlad/development/wunder/research_tmp/chrono_root/artifacts/runs/l14_kd50_s3/provenance.json"
    )
    fixture = json.loads(actual.read_text())
    fixture["args"]["name"] = name
    fixture["args"]["split_manifest"] = (
        "/kaggle/working/experiment/research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json"
    )
    hashes = {
        f"artifacts/runs/{name}/{f}": {"sha256": fixture["checkpoint_sha256"][f]}
        for f in ("model.pt", "model_ep3.pt")
    }
    with tempfile.TemporaryDirectory(
        dir="/home/vlad/development/wunder/research_tmp", prefix="q14-receiver-check-"
    ) as folder:
        root = Path(folder)
        (root / "summary.json").write_text(
            json.dumps({"incident": None, "finished": [], "not_started": []})
        )
        report, code = collect(root)
        assert code == 2 and report["status"] == "incomplete" and report["scores"] is None
        path = root / "provenance.json"
        path.write_text(json.dumps(fixture))
        check_provenance(path, name, spec, hashes)
        bad = copy.deepcopy(fixture)
        bad["fit_row_groups"]["train"][0] = 999999
        path.write_text(json.dumps(bad))
        try:
            check_provenance(path, name, spec, hashes)
        except CollectionError:
            pass
        else:
            raise AssertionError("wrong same-count train membership was accepted")
        (root / "summary.json").write_text("[]")
        report, code = collect(root)
        assert code == 1 and report["status"] == "failed" and report["scores"] is None
    result = {
        "status": "passed",
        "scope": "fixtures only; not cloud results",
        "checks": [
            "present empty summary returns incomplete",
            "source-qualified exact membership accepted despite 1822 cross-file numeric collisions",
            "wrong same-count train membership rejected",
            "non-object summary rejected",
        ],
        "remaining": "actual six-worker completed path and strict optimizer state checks await downloaded outputs",
    }
    (PACKET / "COLLECTION_PREFLIGHT_R2.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

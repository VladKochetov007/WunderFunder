from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OWN = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    """Hash bounded source/checkpoint/diagnostic files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result = json.loads((OWN / "stream_equivalence_results.json").read_text())
    sample = json.loads((OWN / "stream_equivalence_sample.json").read_text())
    for name, expected in result["source_sha256"].items():
        assert digest(ROOT / name) == expected
    for name, expected in sample["protected_artifacts_sha256"].items():
        assert digest(OWN / name) == expected
    checkpoint = ROOT / "artifacts/runs/c20261001_d256_s3_f0/model_ep3.pt"
    assert digest(checkpoint) == result["checkpoint_sha256"]
    provenance_path = checkpoint.with_name("provenance.json")
    provenance = json.loads(provenance_path.read_text())
    assert provenance["split_manifest_sha256"] == sample["split_manifest_sha256"]
    assert provenance["fit_row_groups"] == sample["checkpoint_provenance"]["fit_row_groups"]
    audit = {
        "protected_original_artifacts_unchanged": True,
        "source_and_checkpoint_hashes_unchanged": True,
        "baseline_provenance_json_sha256": digest(provenance_path),
        "baseline_provenance_membership_matches_checkpoint": True,
        "artifact_sha256": {
            path.name: digest(path)
            for path in sorted(OWN.glob("stream_equivalence_*"))
            if path.is_file()
            and path.name not in {"stream_equivalence_audit.json", "stream_equivalence_audit.log"}
        },
    }
    (OWN / "stream_equivalence_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        "Original blind artifacts, sample membership, real baseline provenance, source and checkpoint hashes verified."
    )


if __name__ == "__main__":
    main()

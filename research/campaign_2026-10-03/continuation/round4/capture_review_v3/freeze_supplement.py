from __future__ import annotations

import json
from pathlib import Path

from verify_inputs import digest

OWN = Path(__file__).resolve().parent


def main() -> None:
    """Bind an additive specification without changing the terminal v3 manifest."""
    original = json.loads((OWN / "HASHES.json").read_text())
    assert all(digest(OWN / name) == value for name, value in original.items())
    names = ("MINIMAL_V4_SPEC.md", "freeze_supplement.py", "HASHES.json", "REPORT.md", "STATE.json")
    result = {
        "kind": "additive_minimal_spec_NOT_receipt",
        "original_final_manifest_verified": True,
        "original_review_files_unchanged": True,
        "sha256": {name: digest(OWN / name) for name in names},
        "verdict": "BLOCKED_TORCH_RNG_ACCEPTED_STATE_NORMALIZATION",
        "new_adversary_tests": 0,
        "GPU_launched": False,
        "receipt_written": False,
    }
    (OWN / "SUPPLEMENT_HASHES.json").write_text(json.dumps(result, indent=2) + "\n")
    assert all(digest(OWN / name) == value for name, value in result["sha256"].items())
    print(json.dumps(result))


if __name__ == "__main__":
    main()

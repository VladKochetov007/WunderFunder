"""Read-only verify author and prior independent frozen artifact identities."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
BUILDER = OWN.parent / "live_capture"
LOGGER = logging.getLogger(__name__)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def checks(entries: dict[str, str], root: Path) -> dict:
    return {name: {"expected": expected, "actual": digest(root / name), "matches": digest(root / name) == expected}
            for name, expected in entries.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    author = {}
    for line in (BUILDER / "protected_hashes.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        author[name] = expected
    artifacts = json.loads((BUILDER / "artifact_hashes.json").read_text())
    blind = OWN.parent / "capture_critic"
    lineage = OWN.parent / "teacher_lineage"
    prior_blind = json.loads((blind / "HASHES.json").read_text())["own"]
    prior_lineage = json.loads((lineage / "HASHES.json").read_text())
    source = json.loads((BUILDER / "source_binding.json").read_text())
    results = {"author_protected": checks(author, ROOT),
               "author_artifacts": checks(artifacts, BUILDER),
               "blind_preserved": checks(prior_blind, blind),
               "lineage_preserved": checks(prior_lineage, lineage),
               "source_bound": checks(source, ROOT)}
    result = {"groups": results, "all_match": all(v["matches"] for group in results.values() for v in group.values()),
              "counts": {name: len(group) for name, group in results.items()},
              "manifest_hashes": {str(p.relative_to(ROOT)): digest(p) for p in
                                  (BUILDER / "protected_hashes.sha256", BUILDER / "artifact_hashes.json", blind / "HASHES.json", lineage / "HASHES.json")}}
    (OWN / "protected_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("PROTECTED %s", json.dumps({k: v for k, v in result.items() if k != "groups"}))
    if not result["all_match"]:
        raise RuntimeError("Protected artifact hash mismatch")


if __name__ == "__main__":
    main()

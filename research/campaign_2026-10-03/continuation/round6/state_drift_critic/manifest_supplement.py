"""Complete only the blocked blind-premortem manifest; preserve frozen files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "HASHES_SUPPLEMENT.json"
ADDENDUM = HERE / "MANIFEST_ADDENDUM.txt"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if MANIFEST.exists() or ADDENDUM.exists():
        raise RuntimeError("Supplement already exists; refusing to overwrite")
    protected = [HERE / name for name in
                 ("REPORT.txt", "STATE.json", "RESOURCE_FAILURE.txt", "COST.json",
                  "PREREG.json", "results.json", "run.log")]
    before = {p.name: digest(p) for p in protected}
    results = json.loads((HERE / "results.json").read_text())
    if digest(HERE / "PREREG.json") != results["prereg_sha256_before_tests"]:
        raise RuntimeError("Preregistration changed since toy execution")
    ADDENDUM.write_text(
        "METADATA-ONLY MANIFEST COMPLETION SUPPLEMENT\n"
        "The controller authorized this retry after diagnosing and clearing the\n"
        "shared admission barrier caused by a command-description predicate.\n"
        "The previous exit-75 failure remains recorded in RESOURCE_FAILURE.txt,\n"
        "REPORT.txt, STATE.json and COST.json; none was edited.\n"
        "This successful command only checked the existing preregistration hash\n"
        "and hashed files in this critic directory. No toy adversary was rerun;\n"
        "no source outside this directory, Carver code, capture, panel, data or\n"
        "actual measurement values were read. No GPU or fits.\n"
        "HASHES_SUPPLEMENT.json is the completed SHA-256 artifact manifest.\n"
        "It includes this addendum and supplement script, excludes itself, and\n"
        "retains the already-recorded source hashes without rereading sources.\n"
        "One additional metadata-only CPU body through run_cpu.sh reviewer:\n"
        "one core, 2 GiB, 900 seconds; exit success.\n"
        "Blind premortem and manifest are frozen. Await controller request for\n"
        "state-drift builder review; actual measurements remain unreviewed.\n"
    )
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p != MANIFEST)
    hashes = {p.name: digest(p) for p in files}
    if before != {p.name: digest(p) for p in protected}:
        raise RuntimeError("Protected artifact changed")
    payload = {
        "phase": "blind_premortem_manifest_frozen",
        "scope": "metadata-only supplement; prior frozen files unchanged",
        "algorithm": "sha256",
        "manifest_self_excluded": True,
        "files": hashes,
        "recorded_source_hashes_not_reread": results["source_hashes"],
        "prereg_hash_matches_pretest": True,
        "protected_artifacts_unchanged": before,
        "toy_reruns": 0,
        "candidate_code_reads": 0,
        "actual_measurement_reads": 0,
        "data_reads": 0,
        "gpu_calls": 0,
    }
    with MANIFEST.open("x") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    # Verify the frozen manifest against the final artifact bytes.
    saved = json.loads(MANIFEST.read_text())
    if any(digest(HERE / name) != value for name, value in saved["files"].items()):
        raise RuntimeError("Manifest verification failed")
    print("SUCCESS: supplemental manifest frozen and verified; protected files unchanged.")


if __name__ == "__main__":
    main()

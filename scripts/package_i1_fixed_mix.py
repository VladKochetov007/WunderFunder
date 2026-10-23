from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from build_i1_history_native import source_hashes

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/campaign_2026-10-21/next_cycle_r1/i1/submission_r1"


def sha256_file(path: Path) -> str:
    """Hash a checkpoint or library without loading tensors or native code."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_native(native: Path) -> None:
    """Require a source-bound Linux build and all three intact library payloads."""
    receipt = json.loads((native / "NATIVE_BUILD.json").read_text())
    if receipt.get("schema") != "wnn.i1-native-build.v1":
        raise ValueError("unsupported native build receipt")
    if receipt.get("platform") != "linux/amd64" or not receipt["static_abi"]["passed"]:
        raise ValueError("native build did not pass the static Linux ABI gate")
    if receipt.get("source_hashes") != source_hashes():
        raise ValueError("native build is stale relative to this checkout; rebuild it")
    expected = {f"{level}/wnn_native.abi3.so" for level in ("v2", "avx2", "v3")}
    if set(receipt["library_hashes"]) != expected:
        raise ValueError("native build must bind all three expected libraries")
    for name, digest in receipt["library_hashes"].items():
        if sha256_file(native / name) != digest:
            raise ValueError(f"native library hash mismatch: {name}")


def verify_members(runs: Path, members: list[dict[str, object]]) -> list[str]:
    """Accept only the exact two graded checkpoints and their run provenance."""
    result = []
    for member in members:
        run = str(member["run"])
        if Path(run).name != run:
            raise ValueError("run names must be single path components")
        checkpoint = (runs / run / "model_ep2.pt").resolve(strict=True)
        if not checkpoint.is_relative_to(runs):
            raise ValueError("checkpoint path escapes the run directory")
        if sha256_file(checkpoint) != member["checkpoint_sha256"]:
            raise ValueError(f"{run}: checkpoint differs from the fixed panel grade")
        provenance = json.loads((checkpoint.parent / "provenance.json").read_text())
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError(f"{run}: empty or malformed provenance")
        checkpoint_hashes = provenance.get("checkpoint_sha256")
        if (
            not isinstance(checkpoint_hashes, dict)
            or checkpoint_hashes.get("model_ep2.pt") != member["checkpoint_sha256"]
        ):
            raise ValueError(f"{run}: provenance does not bind the selected checkpoint")
        result.append(f"{run}@ep2:{member['coefficient']}")
    return result


def main() -> None:
    """Export a fixed FP32 pair to a ZIP without inference, tests or fitting.

    Run this on a machine with the checkpoint files and the already provisioned
    research Python environment. A successful build is not an inference release
    gate: exact-ZIP streaming parity and one-CPU runtime remain separate work.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("f0_s1001", "f2_s1002"), required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--native-from", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=ROOT / "submissions")
    args = parser.parse_args()
    manifest_path = PACKET / "FIXED_CANDIDATES.json"
    manifest = json.loads(manifest_path.read_text())
    grade_path = PACKET.parent / "runtime/GRADE_R1.json"
    if sha256_file(grade_path) != manifest["grade_sha256"]:
        raise ValueError("the source grade changed; do not silently retarget this release")
    candidate = manifest["candidates"][args.cell]
    members = candidate["members"]
    if [m["coefficient"] for m in members] != [0.9, 0.1]:
        raise ValueError("this packet only packages the fixed 90%B/10%C pair")
    grade = json.loads(grade_path.read_text())["folds"][str(candidate["fold"])]
    for arm, member in zip(("B", "C"), members, strict=True):
        expected = grade["prediction_identity_and_npy_coverage"][arm]
        if (
            member["run"] != expected["run"]
            or member["checkpoint_sha256"] != expected["checkpoint_sha256"]
        ):
            raise ValueError(f"{arm}: candidate is not the checkpoint pinned in the grade")
    runs = args.runs_dir.resolve(strict=True)
    fixed_members = verify_members(runs, members)
    native = args.native_from.resolve(strict=True)
    verify_native(native)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/package_submission.py"),
            candidate["name"],
            *fixed_members,
            "--runs-dir",
            str(runs),
            "--native-from",
            str(native),
            "--out-root",
            str(args.out_root.resolve()),
            "--precision",
            "f32",
            "--require-provenance",
            "--normalized-coefficients",
            "--evidence-file",
            str(manifest_path),
            "--evidence-file",
            str(grade_path),
            "--evidence-file",
            str(native / "NATIVE_BUILD.json"),
            "--evidence-note",
            "Fixed 90B/10C epoch2 exploratory screen; exposed fold; "
            "no public forecast; no inference, parity or timing checks during packaging.",
            "--max-zip-bytes",
            "19999999",
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )


if __name__ == "__main__":
    main()

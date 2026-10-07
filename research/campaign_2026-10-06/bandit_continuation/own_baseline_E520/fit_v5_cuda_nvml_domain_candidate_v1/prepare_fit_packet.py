"""Build the E520 fit packet only from exact independent review/admission receipts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import textwrap
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from fit_guards import (
    EXPERIMENT_ID,
    REQUIRED_ARTIFACTS,
    artifact_hashes,
    sha256_bytes,
    sha256_file,
    validate_controller_admission,
    validate_review,
    validate_training_target_semantics,
)

PACKET_ID = "vladkochetov07/e520-scratch-pair-fit-v5-memory-domains"
KAGGLE_TITLE = "E520 Scratch Pair Fit V5 Memory Domains"
KAGGLE_DATASETS = [
    "vladkochetov07/wunder-funder",
    "vladkochetov07/wnn-c20261004-fold0-assets",
]
GRADER_SPEC_REPOSITORY_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/"
    "scratch_pair_fit_candidate_v1/E520_GRADER_SPEC.json"
)
GRADER_SCRIPT_REPOSITORY_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/"
    "scratch_pair_fit_candidate_v1/evaluate_e520_pair.py"
)
REVIEW_NAME = "external_review.json"
ADMISSION_NAME = "controller_admission.json"
META_NAME = "packet_meta.json"
BUNDLE_NAME = "fit_kernel_bundle.py"
BUNDLE_PAYLOAD_FILES = tuple(sorted((*REQUIRED_ARTIFACTS, REVIEW_NAME, ADMISSION_NAME, META_NAME)))


def render_fit_kernel_bundle(payloads: dict[str, bytes]) -> str:
    """Render a deterministic, self-contained wrapper for the reviewed packet."""
    for name in payloads:
        relative = PurePosixPath(name)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"unsafe E520 bundle path: {name}")
    if set(payloads) != set(BUNDLE_PAYLOAD_FILES):
        missing = sorted(set(BUNDLE_PAYLOAD_FILES) - set(payloads))
        extra = sorted(set(payloads) - set(BUNDLE_PAYLOAD_FILES))
        raise ValueError(f"bundle payload mismatch: missing={missing}, extra={extra}")
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, content in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compresslevel=9)
    archive_bytes = archive_buffer.getvalue()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    file_hashes = {
        name: hashlib.sha256(content).hexdigest() for name, content in sorted(payloads.items())
    }
    encoded = base64.b64encode(archive_bytes).decode("ascii")
    encoded_lines = "\n".join(f"    {line!r}" for line in textwrap.wrap(encoded, 100))
    return f'''"""Self-contained transport wrapper for the reviewed E520 fit packet."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WNN_LOADER_WORKERS"] = "0"
os.environ["WNN_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PAYLOAD_ZIP_SHA256 = {archive_sha!r}
PAYLOAD_FILE_SHA256 = {file_hashes!r}
PAYLOAD_B64 = (
{encoded_lines}
)


def main() -> None:
    """Verify and extract the reviewed packet, then invoke its gated worker."""
    archive_bytes = base64.b64decode(PAYLOAD_B64, validate=True)
    if hashlib.sha256(archive_bytes).hexdigest() != PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded E520 packet archive digest mismatch")
    packet_root = Path("/kaggle/working/e520_fit_packet_v5")
    packet_root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        names = sorted(item.filename for item in archive.infolist())
        if names != sorted(PAYLOAD_FILE_SHA256):
            raise RuntimeError("embedded E520 packet member set differs from the reviewed map")
        for item in archive.infolist():
            relative = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                raise RuntimeError(f"unsafe embedded packet path: {{item.filename}}")
            destination = packet_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            with temporary.open("xb") as stream:
                stream.write(archive.read(item))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
    for name, expected in PAYLOAD_FILE_SHA256.items():
        digest = hashlib.sha256()
        with (packet_root / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"embedded E520 file hash mismatch: {{name}}")
    sys.path.insert(0, str(packet_root))
    from fit_worker import run_fit

    result = run_fit(packet_root)
    print(
        f"E520 fit status={{result['status']}} elapsed_seconds={{result['elapsed_seconds']:.1f}}",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


def review_targets(packet_root: Path) -> dict[str, str]:
    """Hash reviewed fit/grader artifacts without assembling a packet."""
    return artifact_hashes(packet_root)


def _load_receipt(path: Path, expected_sha256: str, label: str) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    if sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"{label} bytes differ from the externally published receipt hash")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, value


def build_packet(
    packet_root: Path,
    output_dir: Path,
    review_path: Path,
    review_sha256: str,
    admission_path: Path,
    admission_sha256: str,
) -> dict[str, Any]:
    """Stage a Kaggle script only after exact review, profile, quota, and fit admission."""
    config = json.loads((packet_root / "FIT_STAGE_CONFIG.json").read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("packet_id") != PACKET_ID:
        raise ValueError("fit config experiment or packet ID differs from this candidate")
    validate_training_target_semantics(config, packet_root)
    hashes = review_targets(packet_root)
    if hashes["source_archive.tar"] != config["source"]["archive_sha256"]:
        raise ValueError("pinned trainer source archive hash differs from the frozen config")
    grader = config.get("grader_artifacts", {})
    if (
        grader.get("spec_path") != GRADER_SPEC_REPOSITORY_PATH
        or grader.get("spec_sha256") != hashes["E520_GRADER_SPEC.json"]
        or grader.get("script_path") != GRADER_SCRIPT_REPOSITORY_PATH
        or grader.get("script_sha256") != hashes["evaluate_e520_pair.py"]
        or config.get("evaluation", {}).get("E520_independent_grader_script_sha256")
        != hashes["evaluate_e520_pair.py"]
    ):
        raise ValueError("fit config does not bind the exact frozen E520 grader source and spec")
    review_bytes, review = _load_receipt(review_path, review_sha256, "independent review")
    validate_review(review_bytes, review_sha256, hashes)
    admission_bytes, admission = _load_receipt(
        admission_path, admission_sha256, "controller admission"
    )
    validate_controller_admission(
        admission_bytes,
        admission_sha256,
        review_sha256=review_sha256,
        hashes=hashes,
        config=config,
    )
    if (
        admission.get("evaluation_gate", {}).get("grader_script_sha256")
        != config["grader_artifacts"]["script_sha256"]
    ):
        raise ValueError("controller admission does not bind the reviewed E520 grader")
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite packet output: {output_dir}")
    partial_dir.parent.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir()
    try:
        for relative in REQUIRED_ARTIFACTS:
            destination = partial_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(packet_root / relative, destination)
        shutil.copyfile(review_path, partial_dir / REVIEW_NAME)
        shutil.copyfile(admission_path, partial_dir / ADMISSION_NAME)
        metadata = {
            "packet_id": PACKET_ID,
            "experiment_id": EXPERIMENT_ID,
            "fit_enabled": True,
            "score_enabled": False,
            "reviewed_hashes": hashes,
            "artifact_hashes": hashes,
            "external_review_sha256": review_sha256,
            "controller_admission_sha256": admission_sha256,
            "profile_receipt_sha256": admission["profile_gate"]["receipt_sha256"],
            "grader_script_sha256": config["grader_artifacts"]["script_sha256"],
            "anchor_report_sha256": config["evaluation"]["anchor_evaluation_report_sha256"],
            "worker_wall_cap_seconds": admission["live_admission"]["worker_wall_cap_seconds"],
            "fit_order": ["seed3", "seed4"],
            "parallel_fits": False,
            "retry_count": 0,
        }
        (partial_dir / META_NAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        kaggle_meta = {
            "id": PACKET_ID,
            "title": KAGGLE_TITLE,
            "code_file": BUNDLE_NAME,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "dataset_sources": KAGGLE_DATASETS,
            "competition_sources": [],
        }
        (partial_dir / "kernel-metadata.json").write_text(
            json.dumps(kaggle_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        payloads = {name: (partial_dir / name).read_bytes() for name in BUNDLE_PAYLOAD_FILES}
        bundle_path = partial_dir / BUNDLE_NAME
        bundle_path.write_text(render_fit_kernel_bundle(payloads), encoding="utf-8")
        bundle_sha = sha256_file(bundle_path)
        os.replace(partial_dir, output_dir)
        final_bundle = output_dir / BUNDLE_NAME
        return {
            "packet_id": PACKET_ID,
            "packet_dir": str(output_dir),
            "reviewed_hashes": hashes,
            "bundle_path": str(final_bundle),
            "bundle_sha256": bundle_sha,
            "bundle_payload_files": list(BUNDLE_PAYLOAD_FILES),
            "kernel_metadata_sha256": sha256_file(output_dir / "kernel-metadata.json"),
            "provider_calls": False,
        }
    except Exception:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise


def main() -> int:
    """Print exact source hashes or stage only with independently pinned receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review-targets-only", action="store_true")
    mode.add_argument("--build", action="store_true")
    parser.add_argument("--packet-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--review-receipt", type=Path)
    parser.add_argument("--review-sha256")
    parser.add_argument("--controller-admission", type=Path)
    parser.add_argument("--controller-admission-sha256")
    args = parser.parse_args()
    if args.review_targets_only:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "review-targets-only-no-packet-created",
                    "reviewed_hashes": review_targets(args.packet_root),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    required = (
        args.output_dir,
        args.review_receipt,
        args.review_sha256,
        args.controller_admission,
        args.controller_admission_sha256,
    )
    if any(value is None for value in required):
        parser.error(
            "--build requires --output-dir, exact --review-receipt/--review-sha256, "
            "and --controller-admission/--controller-admission-sha256"
        )
    result = build_packet(
        args.packet_root,
        args.output_dir,
        args.review_receipt,
        args.review_sha256,
        args.controller_admission,
        args.controller_admission_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - packet staging CLI
    raise SystemExit(main())

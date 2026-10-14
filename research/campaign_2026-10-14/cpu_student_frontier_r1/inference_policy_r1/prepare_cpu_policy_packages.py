"""Build fixed B/C/D derivatives from the original xteam_h4 archive."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import zipfile
from pathlib import Path

ARCHIVE_SHA256 = "f1635d14c0e1b47848b5caeb2ec4ad9362a85800f1b2e2df80702127d3867b27"
BASE_SOURCE_REVISION = "7800d30473e3810e12d9ba22596f9fd6df5626ca"
NATIVE_SOURCE_FILES = (
    "native/Cargo.toml",
    "native/Cargo.lock",
    "native/src/python.rs",
    "native/src/rnn_fast.rs",
    "native/src/vnni.rs",
    "native/examples/bench_int8.rs",
)
BUILD_COMMAND = [
    "cargo",
    "build",
    "--release",
    "--manifest-path",
    "native/Cargo.toml",
    "--features",
    "extension-module",
]
POLICIES = {
    "B": "q8vnni256",
    "C": "q8vnni_blocked",
    "D": "q8vnni_fused",
}
REBUILT_CONTROL = ("A_rebuilt", "q8vnni")
NATIVE_FILES = (
    "v2/wnn_native.abi3.so",
    "avx2/wnn_native.abi3.so",
    "v3/wnn_native.abi3.so",
)
ASSERTION = """\
if PRECISION != _PRECISION_FILE.read_text().strip():
    raise RuntimeError("WNN_PRECISION cannot override the packaged CPU policy")
"""
MODEL_ASSERTION = """\
        if self.model.precision != PRECISION:
            raise RuntimeError(
                f"selected CPU policy mismatch: requested {PRECISION!r}, "
                f"got {self.model.precision!r}"
            )
"""


def digest_bytes(data: bytes) -> str:
    """Return a SHA-256 digest for one package member."""
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    """Hash a file without loading the whole payload into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def patched_solution(source: bytes, policy: str) -> bytes:
    """Add startup checks while retaining every scoring and prediction statement."""
    text = source.decode("utf-8")
    precision_line = 'PRECISION = os.environ.get("WNN_PRECISION") or (\n'
    precision_end = (
        '    _PRECISION_FILE.read_text().strip() if _PRECISION_FILE.is_file() else "i16")\n'
    )
    if precision_line not in text or precision_end not in text:
        raise ValueError("original solution precision loader changed")
    text = text.replace(precision_end, precision_end + ASSERTION, 1)
    model_anchor = "        # The scorer calls model.predict(data_point); bind it straight to the native method.\n"
    if model_anchor not in text:
        raise ValueError("original solution model callback changed")
    text = text.replace(model_anchor, MODEL_ASSERTION + model_anchor, 1)
    return text.encode("utf-8")


def parse_args() -> argparse.Namespace:
    """Parse immutable source, compiled native library and local package output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write byte-checked policy variants and a rebuilt-A control."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if digest_file(args.archive) != ARCHIVE_SHA256:
        raise ValueError("original archive is not the frozen xteam_h4 source")
    if not args.library.is_file():
        raise FileNotFoundError(args.library)
    native_payload = args.library.read_bytes()
    if not native_payload.startswith(b"\x7fELF"):
        raise ValueError("native output is not an ELF shared library")
    args.out.mkdir(parents=True, exist_ok=True)
    variant_root = args.out / "extracted"
    variant_root.mkdir(exist_ok=True)
    native_hash = digest_bytes(native_payload)
    package_records = []
    with zipfile.ZipFile(args.archive) as source:
        infos = source.infolist()
        names = {info.filename for info in infos}
        if not set(NATIVE_FILES).issubset(names):
            raise ValueError("original xteam_h4 archive lacks expected native libraries")
        if "precision.txt" not in names or "solution.py" not in names:
            raise ValueError("original xteam_h4 archive lacks callback metadata")
        if source.read("precision.txt").decode("utf-8").strip() != "q8vnni":
            raise ValueError("frozen original xteam_h4 precision is not q8vnni")
        build_variants = (REBUILT_CONTROL, *POLICIES.items())
        for label, policy in build_variants:
            archive_path = args.out / f"xteam_h4_{label}_{policy}.zip"
            extract_path = variant_root / label
            if archive_path.exists() or extract_path.exists():
                raise FileExistsError(archive_path if archive_path.exists() else extract_path)
            changed_hashes = {}
            with zipfile.ZipFile(
                archive_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as target:
                for info in infos:
                    original = source.read(info.filename)
                    if info.filename == "precision.txt":
                        payload = f"{policy}\n".encode()
                    elif info.filename == "solution.py":
                        payload = patched_solution(original, policy)
                    elif info.filename in NATIVE_FILES:
                        payload = native_payload
                    else:
                        payload = original
                    # A ZipInfo object is mutable: _open_to_write overwrites its
                    # sizes and offset. Keep the source archive's index intact
                    # across the three separately written policy variants.
                    target.writestr(copy.copy(info), payload, compress_type=info.compress_type)
                    if payload != original:
                        changed_hashes[info.filename] = {
                            "source_sha256": digest_bytes(original),
                            "package_sha256": digest_bytes(payload),
                            "package_bytes": len(payload),
                        }
            with zipfile.ZipFile(archive_path) as prepared:
                prepared.extractall(extract_path)
                if set(prepared.namelist()) != names:
                    raise ValueError(f"package file inventory changed for policy {policy}")
            package_records.append(
                {
                    "submission_order": (
                        ["A", "B", "C", "D", "A"].index(label) if label in POLICIES else None
                    ),
                    "label": label,
                    "policy": policy,
                    "archive_path": str(archive_path),
                    "archive_sha256": digest_file(archive_path),
                    "archive_bytes": archive_path.stat().st_size,
                    "extracted_path": str(extract_path),
                    "source_archive_sha256": ARCHIVE_SHA256,
                    "native_library_sha256": native_hash,
                    "changed_files": changed_hashes,
                    "fixed": [
                        "weights and member order",
                        "coefficients",
                        "preprocessing callback",
                        "GRU architecture",
                        "Q8 codes/scales and activation quantization",
                    ],
                }
            )
    packages_by_label = {record["label"]: record for record in package_records}
    submission_order = []
    for label in ("A", "B", "C", "D", "A"):
        if label == "A":
            submission_order.append(
                {
                    "label": "A",
                    "policy": "q8vnni",
                    "archive_path": str(args.archive.resolve()),
                    "archive_sha256": ARCHIVE_SHA256,
                    "archive_bytes": args.archive.stat().st_size,
                }
            )
        else:
            variant = packages_by_label[label]
            submission_order.append(
                {
                    "label": label,
                    "policy": variant["policy"],
                    "archive_path": variant["archive_path"],
                    "archive_sha256": variant["archive_sha256"],
                    "archive_bytes": variant["archive_bytes"],
                }
            )
    receipt = {
        "format": "wnn.cpu-policy-packages",
        "version": 1,
        "source_archive": str(args.archive.resolve()),
        "source_archive_sha256": ARCHIVE_SHA256,
        "native_source_base_revision": BASE_SOURCE_REVISION,
        "native_source_files_sha256": {
            name: digest_file(Path(name)) for name in NATIVE_SOURCE_FILES
        },
        "campaign_scripts_sha256": {
            name: digest_file(Path(__file__).resolve().parent / name)
            for name in (
                "prepare_cpu_policy_packages.py",
                "benchmark_cpu_policy.py",
                "export_xteam_probe_data.py",
            )
        },
        "native_build_command": BUILD_COMMAND,
        "native_build_constraints": {
            "cargo_build_jobs": 1,
            "cpu_quota": "100%",
            "memory_max_bytes": 1_500_000_000,
            "swap_bytes": 0,
        },
        "native_library": str(args.library.resolve()),
        "native_library_sha256": native_hash,
        "native_library_bytes": len(native_payload),
        "package_policy_order": ["A", "B", "C", "D", "A"],
        "rebuilt_a_control": {
            "label": REBUILT_CONTROL[0],
            "policy": REBUILT_CONTROL[1],
            "purpose": "compare the original callback archive with a newly written ZIP using the rebuilt native library and policy assertions",
            "submission_order": None,
        },
        "submission_order": submission_order,
        "variants": package_records,
        "upload_status": "prepared locally; user owns later upload",
    }
    args.manifest.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    logging.info("wrote %s", args.manifest)


if __name__ == "__main__":
    main()

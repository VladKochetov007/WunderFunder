from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V10_DIR = ROOT / "preparation/first_cell_evaluation_score_admission_v10"
OUT_DIR = Path(__file__).resolve().parent
V10_BODY = V10_DIR / "REMOTE_SUBMISSION_V10.py"
V10_BINDING = V10_DIR / "SCORE_ADMISSION_SOURCE_BINDING_V10.json"
V11_BODY = OUT_DIR / "REMOTE_SUBMISSION_V11.py"
V11_BINDING = OUT_DIR / "SCORE_ADMISSION_SOURCE_BINDING_V11.json"
V10_BODY_BYTES = 187_989
V10_BODY_SHA256 = "24787cf037328baabebe19d4eb6ba5c452740d2b5648dbf478544e3ce33cbd53"
V10_BINDING_BYTES = 5_602
V10_BINDING_SHA256 = "9767f437d545ee9a6b448a6606be972ea6a7766ea3efebf24cbbb346df56a42e"

OLD_READ_LOOP = (
    "        while block := os.read(descriptor, 1 << 20):\n            hasher.update(block)\n"
)
NEW_READ_LOOP = (
    "        bytes_read = 0\n"
    "        while True:\n"
    "            block = os.read(descriptor, min(1 << 20, cap - bytes_read + 1))\n"
    "            if not block:\n"
    "                break\n"
    "            bytes_read += len(block)\n"
    '            require(bytes_read <= cap, f"source file exceeded cap while hashing: {path}")\n'
    "            hasher.update(block)\n"
)
HASH_STABILITY_PREFIX = (
    "        after = os.fstat(descriptor)\n"
    "    finally:\n"
    "        os.close(descriptor)\n"
    "    require(\n"
    "        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)\n"
)
HASH_STABILITY_WITH_COUNT = (
    "        after = os.fstat(descriptor)\n"
    "    finally:\n"
    "        os.close(descriptor)\n"
    '    require(bytes_read == after.st_size, f"source bytes read differ from stable file size: {path}")\n'
    "    require(\n"
    "        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)\n"
)
OLD_FILE_HASH_CALL = "        observed_sha, info = file_hash(path, MAX_SOURCE_BYTES)\n"
NEW_FILE_HASH_CALL = (
    "        live_size = path.stat().st_size\n"
    '        require(live_size == expected_bytes, f"source size differs from pinned manifest: {relative}")\n'
    "        remaining_source_bytes = MAX_SOURCE_AGGREGATE_BYTES - total_bytes\n"
    "        observed_sha, info = file_hash(\n"
    "            path, min(expected_bytes, remaining_source_bytes)\n"
    "        )\n"
)
INVENTORY_LINE = "    inventory = inventory_exact_source_files(SOURCE_ROOT, seen)\n"
PREHASH_SIZE_SUM = (
    "    expected_total_bytes = sum(size for _path, size, _sha in ordered_rows)\n"
    "    require(\n"
    "        expected_total_bytes <= MAX_SOURCE_AGGREGATE_BYTES,\n"
    '        "canonical source manifest aggregate exceeds cap before hashing",\n'
    "    )\n"
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one {label}, found {count}")
    return text.replace(old, new, 1)


def write_create_only(path: Path, data: bytes) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def main() -> None:
    source = V10_BODY.read_bytes()
    if len(source) != V10_BODY_BYTES or digest(source) != V10_BODY_SHA256:
        raise SystemExit("frozen V10 body pin mismatch")
    text = source.decode("utf-8")
    text = replace_once(text, OLD_READ_LOOP, NEW_READ_LOOP, "file-hash loop")
    text = replace_once(
        text, HASH_STABILITY_PREFIX, HASH_STABILITY_WITH_COUNT, "hash stability check"
    )
    text = replace_once(text, OLD_FILE_HASH_CALL, NEW_FILE_HASH_CALL, "source-tree file hash call")
    text = replace_once(
        text, INVENTORY_LINE, PREHASH_SIZE_SUM + INVENTORY_LINE, "source inventory gate"
    )
    body = text.encode("utf-8")
    if V11_BODY.exists() or V11_BINDING.exists():
        raise SystemExit("V11 outputs already exist; refusing overwrite")

    raw_binding = V10_BINDING.read_bytes()
    if len(raw_binding) != V10_BINDING_BYTES or digest(raw_binding) != V10_BINDING_SHA256:
        raise SystemExit("frozen V10 binding pin mismatch")
    binding = json.loads(raw_binding)
    binding["schema"] = "fg19.first-cell-score-admission-source-binding.v11"
    binding["scope"] = (
        "Exact V10 score-admission source with pre-summed strict source-manifest file sizes, "
        "pre-open live-size checks, and a remaining-aggregate-bounded incremental hash loop. "
        "V10 launch-receipt directory fsyncs, all six payload pins, worker identity, and other gates remain unchanged."
    )
    binding["rendered_source"] = {
        "bytes": len(body),
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v11/REMOTE_SUBMISSION_V11.py",
        "sha256": digest(body),
    }
    binding_raw = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode()
    write_create_only(V11_BODY, body)
    write_create_only(V11_BINDING, binding_raw)
    print(f"body_bytes={len(body)} body_sha256={digest(body)}")
    print(f"binding_bytes={len(binding_raw)} binding_sha256={digest(binding_raw)}")


if __name__ == "__main__":
    main()

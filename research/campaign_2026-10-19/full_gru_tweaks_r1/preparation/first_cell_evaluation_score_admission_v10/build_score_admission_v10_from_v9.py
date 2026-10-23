from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V9_DIR = ROOT / "preparation/first_cell_evaluation_score_admission_v9"
OUT_DIR = Path(__file__).resolve().parent
V9_BODY = V9_DIR / "REMOTE_SUBMISSION_V9.py"
V9_BINDING = V9_DIR / "SCORE_ADMISSION_SOURCE_BINDING_V9.json"
V10_BODY = OUT_DIR / "REMOTE_SUBMISSION_V10.py"
V10_BINDING = OUT_DIR / "SCORE_ADMISSION_SOURCE_BINDING_V10.json"
V9_BODY_BYTES = 187_751
V9_BODY_SHA256 = "039d52fc297ddfe7b537baab905dbb23933df780f2a815d0ed48aa862c7bd662"
V9_BINDING_BYTES = 5_516
V9_BINDING_SHA256 = "2798dc1cee7279425fc4d69c9ef5adcedd8736f3a619392abe40fdab4d448e68"

BLOCK4 = (
    "    directory_fd = os.open(\n"
    '        SCORE_D, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)\n'
    "    )\n"
    "    try:\n"
    "        os.fsync(directory_fd)\n"
    "    finally:\n"
    "        os.close(directory_fd)\n"
)
BLOCK8 = "".join(
    "    " + line if line.strip() else line for line in BLOCK4.splitlines(keepends=True)
)
MAIN_POST = "    launch = submit_once(admission)\n" + BLOCK4
MAIN_LAUNCH = "    launch = submit_once(admission)\n"
UNRESOLVED_END = (
    "        )\n"
    "        raise RuntimeError(\n"
    '            "score invocation ID unresolved; durable unknown outcome retained, never retry"\n'
    "        )\n"
)
UNRESOLVED_WITH_FSYNC = (
    "        )\n"
    + BLOCK8
    + "        raise RuntimeError(\n"
    + (
        '            "score invocation ID unresolved; durable unknown outcome retained, never retry"\n'
        "        )\n"
    )
)
LAUNCH_WRITE = '    write_json_exclusive(SCORE_D / "LAUNCH.json", launch)\n'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_create_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    source = V9_BODY.read_bytes()
    if len(source) != V9_BODY_BYTES or digest(source) != V9_BODY_SHA256:
        raise SystemExit("frozen V9 body pin mismatch")
    text = source.decode("utf-8")
    if text.count(MAIN_POST) != 1:
        raise SystemExit("V9 main post-submit fsync block is not unique")
    text = text.replace(MAIN_POST, MAIN_LAUNCH, 1)
    if text.count(UNRESOLVED_END) != 1:
        raise SystemExit("V9 unresolved-receipt branch is not unique")
    text = text.replace(UNRESOLVED_END, UNRESOLVED_WITH_FSYNC, 1)
    if text.count(LAUNCH_WRITE) != 1:
        raise SystemExit("V9 successful-receipt write is not unique")
    text = text.replace(LAUNCH_WRITE, LAUNCH_WRITE + BLOCK4, 1)
    body = text.encode("utf-8")
    if V10_BODY.exists() or V10_BINDING.exists():
        raise SystemExit("V10 outputs already exist; refusing overwrite")
    write_create_only(V10_BODY, body)

    raw_binding = V9_BINDING.read_bytes()
    if len(raw_binding) != V9_BINDING_BYTES or digest(raw_binding) != V9_BINDING_SHA256:
        raise SystemExit("frozen V9 binding pin mismatch")
    binding = json.loads(raw_binding)
    binding["schema"] = "fg19.first-cell-score-admission-source-binding.v10"
    binding["scope"] = (
        "Exact V9 score-admission source with the post-submit SCORE_D directory fsync moved "
        "to immediately after durable LAUNCH.json creation and added after durable "
        "LAUNCH_UNRESOLVED.json creation before the unknown-outcome raise. The original "
        "pre-submit fsync remains; all six payload pins, worker identity, and gates are unchanged."
    )
    binding["rendered_source"] = {
        "bytes": len(body),
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_admission_v10/REMOTE_SUBMISSION_V10.py",
        "sha256": digest(body),
    }
    binding_bytes = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode()
    write_create_only(V10_BINDING, binding_bytes)
    print(f"body_bytes={len(body)} body_sha256={digest(body)}")
    print(f"binding_bytes={len(binding_bytes)} binding_sha256={digest(binding_bytes)}")


if __name__ == "__main__":
    main()

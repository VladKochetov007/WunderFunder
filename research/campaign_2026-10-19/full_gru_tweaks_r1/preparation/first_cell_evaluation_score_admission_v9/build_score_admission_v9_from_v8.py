"""Create the V9 score stdin body by one exact post-submit directory fsync insertion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPOSITORY = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
CAMPAIGN = REPOSITORY / "research/campaign_2026-10-19/full_gru_tweaks_r1"
V8_DIRECTORY = CAMPAIGN / "preparation/first_cell_evaluation_score_admission_v8"
V9_DIRECTORY = CAMPAIGN / "preparation/first_cell_evaluation_score_admission_v9"
V8_BODY = V8_DIRECTORY / "REMOTE_SUBMISSION_TEMPLATE_V8.py"
V8_BINDING = V8_DIRECTORY / "SCORE_ADMISSION_SOURCE_BINDING_V8.json"
V9_BODY = V9_DIRECTORY / "REMOTE_SUBMISSION_V9.py"
V9_BINDING = V9_DIRECTORY / "SCORE_ADMISSION_SOURCE_BINDING_V9.json"
V8_BODY_PIN = (
    187_541,
    "cbb339e6225152e12fb044ce60a6c7cf3f7d58d00f395790097ea7a2e037fd11",
)
V8_BINDING_PIN = (
    5_422,
    "b23d12daa1cb1a4d43f84507e1799ef045c0a9c105a5cf04bf3374005482b9e2",
)
POST_SUBMIT_DIRSYNC = b"""    directory_fd = os.open(
        SCORE_D, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
"""
LAUNCH_MARKER = b"    launch = submit_once(admission)\n"
NEXT_LINE = b'    admission_raw, _ = stable_file(SCORE_D / "ADMISSION.json", 1_000_000)\n'


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    if not V9_DIRECTORY.is_dir() or V9_DIRECTORY.is_symlink():
        raise ValueError("new V9 output directory is absent or redirected")
    if any(path.exists() or path.is_symlink() for path in (V9_BODY, V9_BINDING)):
        raise FileExistsError("V9 output already exists; do not overwrite or retry")
    old_body = V8_BODY.read_bytes()
    old_binding_raw = V8_BINDING.read_bytes()
    if (len(old_body), sha256(old_body)) != V8_BODY_PIN:
        raise ValueError("frozen successful V8 body pin differs")
    if (len(old_binding_raw), sha256(old_binding_raw)) != V8_BINDING_PIN:
        raise ValueError("frozen successful V8 binding pin differs")
    if old_body.count(LAUNCH_MARKER + NEXT_LINE) != 1:
        raise ValueError("expected one exact V8 post-submit launch/reply boundary")
    new_body = old_body.replace(
        LAUNCH_MARKER + NEXT_LINE,
        LAUNCH_MARKER + POST_SUBMIT_DIRSYNC + NEXT_LINE,
        1,
    )
    if new_body.count(POST_SUBMIT_DIRSYNC) != 2:
        raise ValueError("V9 must contain the original pre-submit fsync plus one post-submit fsync")

    binding = json.loads(old_binding_raw)
    binding["schema"] = "fg19.first-cell-score-admission-source-binding.v9"
    binding["scope"] = (
        "Exact V8 score-admission source with one additional SCORE_D directory fsync after the "
        "successful submission receipt is written and before durable LAUNCH reread/four-key reply. "
        "All six payload pins, worker identity, and gates remain unchanged."
    )
    binding["rendered_source"] = {
        "path": V9_BODY.relative_to(REPOSITORY).as_posix(),
        "bytes": len(new_body),
        "sha256": sha256(new_body),
    }
    binding_raw = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8")

    _write_exclusive(V9_BODY, new_body)
    directory_fd = os.open(
        V9_DIRECTORY,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    _write_exclusive(V9_BINDING, binding_raw)
    directory_fd = os.open(
        V9_DIRECTORY,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(
        json.dumps(
            {
                "body_bytes": len(new_body),
                "body_sha256": sha256(new_body),
                "body_path": V9_BODY.relative_to(REPOSITORY).as_posix(),
                "binding_bytes": len(binding_raw),
                "binding_sha256": sha256(binding_raw),
                "binding_path": V9_BINDING.relative_to(REPOSITORY).as_posix(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

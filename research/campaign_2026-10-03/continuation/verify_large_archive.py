"""Exercise streamed creation and paused restore of an opaque600MiB state fixture."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime
from pathlib import Path

from restore_progress_artifacts import restore
from save_progress_parts import HashingReader

STATE = Path(__file__).resolve().parent
RELATIVE = "research/campaign_2026-10-03/continuation/round4/archive_fixture/OPAQUE_TEST_ONLY.pt"


def checksum(path: Path) -> str:
    """Hash a file with bounded buffers."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    """Verify a larger-than512MiB single member without loading/executing its bytes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path)
    options = parser.parse_args()
    base = (
        options.resume.resolve()
        if options.resume
        else STATE / "review" / datetime.now().strftime("large_state_probe_%H%M%S")
    )
    if base.parent != (STATE / "review").resolve():
        raise ValueError("fixture must stay under controller review directory")
    if not options.resume:
        base.mkdir(parents=True, exist_ok=False)
    source_path = base / "opaque_test_only.bin"
    block = bytes(1024**2)
    if not options.resume:
        with source_path.open("xb") as output:
            for _ in range(600):
                output.write(block)
    archive = base / "progress-artifacts.tar"
    if not options.resume:
        write_fixture_archive(source_path, archive)
    destination = base / "paused_stage"
    destination.mkdir(exist_ok=bool(options.resume))
    (destination / "_recovery").mkdir(exist_ok=True)
    for relative in (
        "research/campaign_2026-10-03/continuation/RESOURCE_FAILURE",
        "research/campaign_2026-10-01/QUEUE_PAUSE",
        "research/campaign_2026-09-29/STOP_NEW_JOBS",
    ):
        marker = destination / relative
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("archive fixture only; stay paused\n")
    result = restore(archive, checksum(archive), destination)
    if checksum(source_path) != checksum(destination / RELATIVE):
        raise ValueError("restored opaque bytes differ")
    report = {
        **result,
        "test": "single600MiBopaque state archive; no Torch deserialize",
        "fixture": str(base),
        "source_vs_restored_hash": "PASS",
        "scope": "streamed tar member and paused companion restore; not scientific replay",
        "retained_initial_fixture_failure": "missing _recovery directory after byte extraction; retry reuses and verifies unchanged source/archive/staged bytes"
        if options.resume
        else None,
    }
    (STATE / "ARCHIVE_STREAMING_CHECK.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def write_fixture_archive(source_path: Path, archive: Path) -> None:
    """Create the fixture using exactly the streaming member writer."""
    with tarfile.open(archive, "w") as bundle:
        info = tarfile.TarInfo(RELATIVE)
        info.size, info.mode = source_path.stat().st_size, 0o644
        with source_path.open("rb") as source:
            reader = HashingReader(source)
            bundle.addfile(info, reader)
        if reader.bytes_read != info.size:
            raise ValueError("stream byte count mismatch")
        manifest = {
            "members": [
                {"path": RELATIVE, "bytes": info.size, "sha256": reader.checksum.hexdigest()}
            ],
            "evidence_kind": "opaque archive fixture, NOT scientific training state",
        }
        body = json.dumps(manifest).encode()
        metadata = tarfile.TarInfo("progress_manifest.json")
        metadata.size, metadata.mode = len(body), 0o644
        bundle.addfile(metadata, io.BytesIO(body))


if __name__ == "__main__":
    main()

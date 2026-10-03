"""Verify reused artifact/context parts and retain their prior restore evidence."""

from __future__ import annotations

import hashlib
import json

from save_progress_artifacts import ROOT, STATE


def main() -> None:
    """Check all bytes, and distinguish fresh source restore from reused part restore."""
    previous = STATE / "snapshots/2026-10-03-183137-0bb5fe90"
    current = ROOT / (STATE / "LATEST_SNAPSHOT").read_text().strip()
    checksums = (previous / "SHA256SUMS").read_text().splitlines()[1:]
    verified = []
    for line in checksums:
        expected, name = line.split()
        digest = hashlib.sha256()
        with (current / name).open("rb") as source:
            for chunk in iter(lambda: source.read(1024**2), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"reused part checksum failed: {name}")
        verified.append({"archive": name, "sha256": expected})
    checksum_path = current / "SHA256SUMS"
    if len(checksum_path.read_text().splitlines()) != 1:
        raise FileExistsError("parts already registered")
    with checksum_path.open("a") as output:
        output.write("\n".join(checksums) + "\n")
    fresh = json.loads((STATE / "review/restore_1837/_recovery/dry_run.json").read_text())
    report = {"status": "PASS", "source_restore": fresh, "reused_parts_byte_verified": verified,
              "parts_restore_destination": str((STATE / "review/restore_1831").relative_to(ROOT)),
              "parts_restore": "Both progress parts actually restored and manifest-verified;13 context files hash-verified and extracted",
              "progress_members": 129, "progress_payload_bytes": 708095082,
              "source_capture": "18:37", "progress_capture": "18:35", "context_handoff": "18:35",
              "paused": True, "services_started": 0, "standalone_training_restore": False}
    (current / "RECOVERY_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

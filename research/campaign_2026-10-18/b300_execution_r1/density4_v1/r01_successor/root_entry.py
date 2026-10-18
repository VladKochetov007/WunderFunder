"""Root-owned staged entry: verify, preview, await a bound gate, then adopt."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


def main() -> None:
    """Make the real unit observable before any controller freeze or fit claim."""
    directory = Path(__file__).resolve().parent
    authority_path = Path(sys.argv[1])
    authority_sha = sys.argv[2]
    control_sha = sys.argv[3]
    files = json.loads((directory / "CONTROL_MANIFEST.json").read_text())["files"]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != control_sha:
        raise ValueError("Root entry control identity changed")
    for name, digest in files.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Root entry member changed: " + name)
    if hashlib.sha256(authority_path.read_bytes()).hexdigest() != authority_sha:
        raise ValueError("Root authority identity changed")
    sys.path.insert(0, str(directory))
    from authority import load_authority, resource_baseline, validate_live_resources
    from durability import write_json_fsync
    from lifecycle import freeze_old_controller, prepare_state
    from successor import _preview
    from successor_runner import _frozen_modules, run_successor

    authority = load_authority(authority_path, require_execute=True)
    preview = _preview(authority)
    _frozen_modules(Path(authority["packet_root"]))
    baseline = resource_baseline(Path(authority["resource_preflight_path"]))
    admission = authority_path.parent
    limit = time.monotonic() + 300
    while True:
        try:
            resources = validate_live_resources(
                Path(authority["live_resource_receipt_path"]), baseline
            )
            break
        except (OSError, ValueError, TimeoutError):
            if time.monotonic() >= limit:
                raise
            time.sleep(1)
    write_json_fsync(
        admission / "READ_ONLY_READY.json",
        dict(
            authority_sha256=authority_sha,
            control_bundle_sha256=control_sha,
            preview=preview,
            live_resource_observed_unix=resources["observed_unix"],
            no_controller_signal_sent=True,
        ),
    )
    gate = admission / "START_GATE.json"
    while not gate.exists():
        if time.monotonic() >= limit:
            raise TimeoutError("Root start gate did not arrive; old controller untouched")
        time.sleep(1)
    if json.loads(gate.read_text()) != dict(authority_sha256=authority_sha, execute=True):
        raise ValueError("Start gate does not match the exact reviewed authority")
    validate_live_resources(Path(authority["live_resource_receipt_path"]), baseline)
    prepare_state(authority)
    freeze_old_controller(authority)
    run_successor(authority)


if __name__ == "__main__":
    main()

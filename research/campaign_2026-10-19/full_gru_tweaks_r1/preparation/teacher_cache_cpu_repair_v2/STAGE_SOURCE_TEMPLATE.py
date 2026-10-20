"""Create a new immutable published code snapshot; no model or data payload work."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

PAYLOAD = json.loads(base64.b64decode("__PAYLOAD_B64__", validate=True))
B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")


def main() -> None:
    """Check source bytes, empty campaign and bounded headroom before create-only staging."""
    if sys.flags.optimize != 0:
        raise RuntimeError("normal Python mode required")
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (128_000_000, 128_000_000))
    commit = PAYLOAD["source_commit"]
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    source = B / f"source-fg19-cache-{commit[:8]}"
    assert PAYLOAD["source_root"] == str(source)
    assert B.is_dir() and not B.is_symlink() and not os.path.lexists(source)
    assert not os.path.lexists(B / "runtime/STOP_ADMISSION.json")
    files = PAYLOAD["files"]
    assert len(files) == 167 and sum(v["bytes"] for v in files.values()) < 8_000_000
    manifest_raw = base64.b64decode(PAYLOAD["source_manifest_b64"], validate=True)
    assert hashlib.sha256(manifest_raw).hexdigest() == PAYLOAD["source_manifest_sha256"]
    manifest = json.loads(manifest_raw)
    assert manifest["source_commit"] == commit and manifest["source_root"] == str(source)
    assert {item["path"]: (item["bytes"], item["sha256"]) for item in manifest["files"]} == {
        name: (item["bytes"], item["sha256"]) for name, item in files.items()
    }
    decoded = {}
    for name, item in files.items():
        rel = Path(name)
        assert not rel.is_absolute() and ".." not in rel.parts
        assert rel.parts[0] in ["src", "scripts", "pyproject.toml", "uv.lock"]
        raw = base64.b64decode(item["base64"], validate=True)
        assert len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"]
        decoded[name] = raw
    fields = ["ControlGroup", "CPUQuotaPerSecUSec", "MemoryHigh", "MemoryMax", "MemorySwapMax"]
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            "--all",
            "wnn-campaign.slice",
            *sum((["-p", k] for k in fields), []),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    parent = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert parent["ControlGroup"] not in ["", "/"]
    assert [parent[k] for k in fields[1:]] == ["2s", "18000000000", "20000000000", "0"]
    group = Path("/sys/fs/cgroup") / parent["ControlGroup"].lstrip("/")
    assert group.name == "wnn-campaign.slice"
    paths = [group / "cgroup.procs", *group.rglob("cgroup.procs")]
    assert len(paths) < 128
    populated = {str(p): v for p in paths if (v := p.read_text().strip())}
    assert not populated
    current = int((group / "memory.current").read_text())
    assert current + 128_000_000 < 17_000_000_000
    disk = os.statvfs(B)
    assert disk.f_bavail * disk.f_frsize > 30 * (1 << 30)
    mount = subprocess.run(
        ["findmnt", "-T", str(B), "-n", "-o", "FSTYPE"], capture_output=True, text=True, timeout=10
    )
    assert mount.returncode == 0 and mount.stdout.strip() not in ["", "tmpfs", "ramfs"]
    assert not os.path.lexists(source)
    source.mkdir(mode=0o700)
    for name, raw in decoded.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(raw)
    assert {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()} == set(
        files
    )
    for name, item in files.items():
        path = source / name
        assert not path.is_symlink() and path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    print(
        json.dumps(
            {
                "source_commit": commit,
                "source_root": str(source),
                "source_files": len(files),
                "source_manifest_sha256": PAYLOAD["source_manifest_sha256"],
                "parent": parent,
                "parent_memory_current_before_bytes": current,
                "populated": populated,
                "scope": "Published code bytes only; create-only new source root. No existing source, data/model/array read, unit mutation or native scientific job.",
            }
        )
    )


if __name__ == "__main__":
    main()

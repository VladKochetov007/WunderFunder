"""Advise clean cache pages for four accepted, stopped, owned prediction files only."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import resource
import stat
import subprocess
from pathlib import Path

ITEMS = json.loads(
    base64.b64decode(
        "W3sicGF0aCI6Ii9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvcmVzZWFyY2hfdG1wL2ZnMTlfdDNfZml0X3ByZWRpY3Rpb25zX3MzX3M0X3M1X3YxL3ByZWRfZmcxOV90M19oMzg0eDJfZTNfczNfdjFfdHJhaW4ubnB5IiwibWV0YWRhdGFfcGF0aCI6Ii9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvcmVzZWFyY2hfdG1wL2ZnMTlfdDNfZml0X3ByZWRpY3Rpb25zX3MzX3M0X3M1X3YxL3ByZWRfZmcxOV90M19oMzg0eDJfZTNfczNfdjFfdHJhaW4uanNvbiIsIm1ldGFkYXRhX3NoYTI1NiI6IjQ5MTc2N2M4YTkxNGNmOTQ1ZDAyODRkMjk1MDZlNDFjYjc0ZTI5NzdjN2IyYjc4YmFlYzUyNGVhYmFlODU5ZjYiLCJwcm9kdWNlcl9zaGEyNTYiOiIyMThlM2M0MWU5MWZmNDVlNjkxYjg1ZTY5OTk0MTI4NDFjYTA1YzY4YWViZGZjMjVjMWM3NjU5YjBjYzZiZWM1IiwiZXhwZWN0ZWRfc3RhdCI6eyJwYXRoIjoiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV90M19maXRfcHJlZGljdGlvbnNfczNfczRfczVfdjEvcHJlZF9mZzE5X3QzX2gzODR4Ml9lM19zM192MV90cmFpbi5ucHkiLCJieXRlcyI6MTY4NTYwMDEyOCwiZGV2aWNlIjo2NDUxMiwiaW5vZGUiOjczOTY3Nzc5OSwibXRpbWVfbnMiOjE3OTI0NDk1NTE0MDM1MzI0MTksInBhcnRpYWxfYWJzZW50Ijp0cnVlLCJwYXlsb2FkX3JlYWRfb3JfaGFzaGVkX2J5X2NhcHR1cmUiOmZhbHNlfX0seyJwYXRoIjoiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV90M19maXRfcHJlZGljdGlvbnNfczNfczRfczVfdjEvcHJlZF9mZzE5X3QzX2gzODR4Ml9lM19zM192MV92YWxpZC5ucHkiLCJtZXRhZGF0YV9wYXRoIjoiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV90M19maXRfcHJlZGljdGlvbnNfczNfczRfczVfdjEvcHJlZF9mZzE5X3QzX2gzODR4Ml9lM19zM192MV92YWxpZC5qc29uIiwibWV0YWRhdGFfc2hhMjU2IjoiNTVkZGM5OWZkN2FhMDZiYTNhOWIzMDk0MjFiM2Y0NjQyYzgyNmFiZjBhOTE4Y2E5NjE2YjgyMDIwYjE5YmU5YiIsInByb2R1Y2VyX3NoYTI1NiI6IjBiYTk4ZTZkZjlkNTlmYjNjMzAxMzg4YWIwNmZmMWQzNDg3NTk0NmIyODA5ZDJiZTZlZmIyNTFjMWZhMGJhZWYiLCJleHBlY3RlZF9zdGF0Ijp7InBhdGgiOiIvaG9tZS9ncmluY2hlbmtvZWcvY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLTIwMjYxMDE4L3Jlc2VhcmNoX3RtcC9mZzE5X3QzX2ZpdF9wcmVkaWN0aW9uc19zM19zNF9zNV92MS9wcmVkX2ZnMTlfdDNfaDM4NHgyX2UzX3MzX3YxX3ZhbGlkLm5weSIsImJ5dGVzIjoyMjQ4MDAxMjgsImRldmljZSI6NjQ1MTIsImlub2RlIjo3Mzk2Nzc4MDUsIm10aW1lX25zIjoxNzkyNDUwODIyOTEyMjIwNzkxLCJwYXJ0aWFsX2Fic2VudCI6dHJ1ZSwicGF5bG9hZF9yZWFkX29yX2hhc2hlZF9ieV9jYXB0dXJlIjpmYWxzZX19LHsicGF0aCI6Ii9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvcmVzZWFyY2hfdG1wL2ZnMTlfdDNfZml0X3ByZWRpY3Rpb25zX3MzX3M0X3M1X3YxL3ByZWRfZmcxOV90M19oMzg0eDJfZTNfczRfdjFfdHJhaW4ubnB5IiwibWV0YWRhdGFfcGF0aCI6Ii9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvcmVzZWFyY2hfdG1wL2ZnMTlfdDNfZml0X3ByZWRpY3Rpb25zX3MzX3M0X3M1X3YxL3ByZWRfZmcxOV90M19oMzg0eDJfZTNfczRfdjFfdHJhaW4uanNvbiIsIm1ldGFkYXRhX3NoYTI1NiI6IjJhOGM4M2YxZmE0MGVjMmUwMGVmMGJiMWZhNmU2MmIxMTVhZjM0MGJjZDg0MTI4ZGE2MTE1OTZlMTEwN2M5Y2QiLCJwcm9kdWNlcl9zaGEyNTYiOiI0M2ZkYzJkMzg0YTg4NDZhNWI2YzM3MzY3OWE1ZjBmMGU0ZWYyNGM1ZDU5MjVhYTJiODQ5MDUwMTZkMGU1YzkzIiwiZXhwZWN0ZWRfc3RhdCI6eyJwYXRoIjoiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV90M19maXRfcHJlZGljdGlvbnNfczNfczRfczVfdjEvcHJlZF9mZzE5X3QzX2gzODR4Ml9lM19zNF92MV90cmFpbi5ucHkiLCJieXRlcyI6MTY4NTYwMDEyOCwiZGV2aWNlIjo2NDUxMiwiaW5vZGUiOjczOTY3NzgxMSwibXRpbWVfbnMiOjE3OTI0NTI4MTkzNzc0MzMzMjUsInBhcnRpYWxfYWJzZW50Ijp0cnVlLCJwYXlsb2FkX3JlYWRfb3JfaGFzaGVkX2J5X2NhcHR1cmUiOmZhbHNlfX0seyJwYXRoIjoiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV90M19maXRfcHJlZGljdGlvbnNfczNfczRfczVfdjEvcHJlZF9mZzE5X3QzX2gzODR4Ml9lM19zNF92MV92YWxpZC5ucHkiLCJtZXRhZGF0YV9wYXRoIjoiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV90M19maXRfcHJlZGljdGlvbnNfczNfczRfczVfdjEvcHJlZF9mZzE5X3QzX2gzODR4Ml9lM19zNF92MV92YWxpZC5qc29uIiwibWV0YWRhdGFfc2hhMjU2IjoiZTIzNGVjY2UxYmI1YTU4M2FkZDJmZjk3MzcyZmJkZTQ4MmQxZTY0ZjdiZmVlNmQzOWFiYzUzOGFlNTE4OTBmMSIsInByb2R1Y2VyX3NoYTI1NiI6IjljNmNjMDQzZWFhYmFmZjBhZGI5ZjFkYzU1MTMzYzM0MTg5MTRhMWY3Y2EzMWUzMzM1OWUwN2Q2OGIzNzhlNGIiLCJleHBlY3RlZF9zdGF0Ijp7InBhdGgiOiIvaG9tZS9ncmluY2hlbmtvZWcvY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLTIwMjYxMDE4L3Jlc2VhcmNoX3RtcC9mZzE5X3QzX2ZpdF9wcmVkaWN0aW9uc19zM19zNF9zNV92MS9wcmVkX2ZnMTlfdDNfaDM4NHgyX2UzX3M0X3YxX3ZhbGlkLm5weSIsImJ5dGVzIjoyMjQ4MDAxMjgsImRldmljZSI6NjQ1MTIsImlub2RlIjo3Mzk2Nzc4MTcsIm10aW1lX25zIjoxNzkyNDU0MzIzNTIzODk2NTM5LCJwYXJ0aWFsX2Fic2VudCI6dHJ1ZSwicGF5bG9hZF9yZWFkX29yX2hhc2hlZF9ieV9jYXB0dXJlIjpmYWxzZX19XQ==",
        validate=True,
    )
)
ROOT = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
PREDICTIONS = ROOT / "research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1"


def parent_snapshot() -> dict[str, object]:
    """Observe the explicit campaign parent without any host-root fallback."""
    fields = [
        "ControlGroup",
        "MemoryCurrent",
        "MemoryHigh",
        "MemoryMax",
        "MemorySwapMax",
        "CPUQuotaPerSecUSec",
    ]
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            "--all",
            "wnn-campaign.slice",
            *sum((["-p", field] for field in fields), []),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    unit = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert unit["ControlGroup"] not in ("", "/")
    assert (
        unit["MemoryHigh"],
        unit["MemoryMax"],
        unit["MemorySwapMax"],
        unit["CPUQuotaPerSecUSec"],
    ) == ("18000000000", "20000000000", "0", "2s")
    group = Path("/sys/fs/cgroup") / unit["ControlGroup"].lstrip("/")
    assert group.name == "wnn-campaign.slice"
    paths = [group / "cgroup.procs", *group.rglob("cgroup.procs")]
    assert len(paths) <= 128
    populated = {str(path): value for path in paths if (value := path.read_text().strip())}
    assert not populated
    return {
        "unit": unit,
        "populated": populated,
        "memory_current": int((group / "memory.current").read_text()),
        "events": {
            name: (group / name).read_text() for name in ("memory.events", "memory.events.local")
        },
    }


def identity(value: os.stat_result) -> dict[str, int]:
    """Record file metadata without reading prediction bytes."""
    return {
        "bytes": value.st_size,
        "device": value.st_dev,
        "inode": value.st_ino,
        "mtime_ns": value.st_mtime_ns,
    }


def main() -> None:
    """Reclaim only when the next 8GB export reservation cannot meet the 17GB target."""
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (128_000_000, 128_000_000))
    assert callable(getattr(os, "posix_fadvise", None)) and hasattr(os, "POSIX_FADV_DONTNEED")
    assert len(ITEMS) == 4 and PREDICTIONS.is_dir() and not PREDICTIONS.is_symlink()
    before = parent_snapshot()
    if before["memory_current"] + 8_000_000_000 < 17_000_000_000:
        print(
            json.dumps(
                {
                    "utc": dt.datetime.now(dt.UTC).isoformat(),
                    "status": "NO_HANDOFF_NEEDED",
                    "before": before,
                    "advised": [],
                    "limits_changed": False,
                }
            )
        )
        return
    assert before["memory_current"] + 128_000_000 < 17_000_000_000
    records = []
    opened = []
    try:
        for item in ITEMS:
            path = Path(item["path"])
            metadata = Path(item["metadata_path"])
            assert path.parent == metadata.parent == PREDICTIONS and path.suffix == ".npy"
            assert not path.is_symlink() and not metadata.is_symlink()
            assert metadata.stat().st_size < 1_000_000
            raw = metadata.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == item["metadata_sha256"]
            assert json.loads(raw)["prediction_sha256"] == item["producer_sha256"]
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            opened.append((fd, path, os.fstat(fd)))
            info = opened[-1][2]
            assert stat.S_ISREG(info.st_mode) and identity(info) == {
                key: item["expected_stat"][key] for key in ("bytes", "device", "inode", "mtime_ns")
            }
        for fd, path, info in opened:
            os.posix_fadvise(fd, 0, info.st_size, os.POSIX_FADV_DONTNEED)
            assert identity(os.fstat(fd)) == identity(info)
            records.append(
                {
                    "path": str(path),
                    "stat_before_after": identity(info),
                    "advice": "POSIX_FADV_DONTNEED",
                    "offset": 0,
                    "length": info.st_size,
                    "payload_read_or_hashed": False,
                }
            )
    finally:
        for fd, _, _ in opened:
            os.close(fd)
    after = parent_snapshot()
    status = (
        "HEADROOM_READY"
        if after["memory_current"] + 8_000_000_000 < 17_000_000_000
        else "HEADROOM_HOLD"
    )
    print(
        json.dumps(
            {
                "utc": dt.datetime.now(dt.UTC).isoformat(),
                "status": status,
                "before": before,
                "after": after,
                "advised": records,
                "limits_changed": False,
                "prediction_payload_read_or_hashed": False,
                "scope": "Kernel advice for clean pages of four immutable accepted own files only; no deletion, write, global cache drop, model/data read, old-unit poll or inference replay. Observed before/after samples are not a guaranteed reclamation amount.",
            }
        )
    )


if __name__ == "__main__":
    main()

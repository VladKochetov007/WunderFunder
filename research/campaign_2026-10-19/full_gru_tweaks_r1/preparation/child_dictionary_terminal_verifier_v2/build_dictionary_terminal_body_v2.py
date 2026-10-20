from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_BYTES = 558_636
SOURCE_SHA256 = "f49a618e3102b52756c68ec049630019360add5fb0ddb26da704af7515846f91"
REMOVE_LINE = b"    body_raw, body_stat = stable_file(Path(__file__), 250_000)\n"
OLD_ADMISSION_LINE = (
    b'        "rendered_body": {"bytes": len(body_raw), "sha256": digest(body_raw), **body_stat},\n'
)
NEW_ADMISSION_BLOCK = b"""        "rendered_body_authority": {
            "schema": "fg19.root-source-review-raw-transport-authority.v1",
            "owner": "root",
            "source_review": source_review_pin,
        "source_review_scope": "checker bytes only; exact V2 body review remains a root prelaunch gate",
            "root_preparation": root_preparation_pin,
            "raw_transport": {
            "owner": "root",
            "method": "exact V2 body bytes supplied on stdin",
            "external_binding_required": "root binds raw transport bytes to reviewed V2 body",
                "worker_self_read": False,
                "worker_body_sha256_claimed": False,
            },
        },
"""


def sha256(raw: bytes) -> str:
    """Return the lowercase SHA-256 digest for a byte string."""
    return hashlib.sha256(raw).hexdigest()


def transform(source: bytes) -> bytes:
    """Apply the two approved body-only fixes to the frozen V1 rendering."""
    if len(source) != SOURCE_BYTES or sha256(source) != SOURCE_SHA256:
        raise ValueError("V1 rendered body differs from the frozen source pin")
    if source.count(REMOVE_LINE) != 1 or source.count(OLD_ADMISSION_LINE) != 1:
        raise ValueError("expected exactly one self-read and one body-hash admission line")
    result = source.replace(REMOVE_LINE, b"", 1).replace(OLD_ADMISSION_LINE, NEW_ADMISSION_BLOCK, 1)
    if result == source:
        raise ValueError("V2 body transformation did not produce distinct bytes")
    return result


def main() -> int:
    """Create or verify the exact local V2 body from the frozen V1 body."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    expected = transform(args.source.read_bytes())
    if args.check:
        actual = args.output.read_bytes()
        if actual != expected:
            raise SystemExit("V2 body differs from the exact two-patch derivation")
    else:
        with args.output.open("xb") as stream:
            stream.write(expected)
    print(
        json.dumps(
            {
                "source_bytes": SOURCE_BYTES,
                "source_sha256": SOURCE_SHA256,
                "output_bytes": len(expected),
                "output_sha256": sha256(expected),
                "patches": 2,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

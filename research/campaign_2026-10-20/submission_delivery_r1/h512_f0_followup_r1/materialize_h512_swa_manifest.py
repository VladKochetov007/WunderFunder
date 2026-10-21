"""Materialize the predeclared SWA manifest from accepted F0 receipts."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PACKET = Path("research/campaign_2026-10-20/submission_delivery_r1/h512_f0_followup_r1")
TEMPLATE_SHA256 = "284c4ead1f78546f13883c054019356964fc9ae0f6e00e3b55a9eea3aec07fa7"
RUN = "r01_h512_f0_s1003_local_cp5_r1"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
PLAN_SHA256 = "51d797d1d568d98b0d9553c934721aab2826ba5d4d61b96a24e496ce514c7ec9"
SCREEN_SHA256 = "f6b3b954cd2253a9eac10032c9a053f7abcd55e359c41c2062ed3bdeff2920d8"
MAX_JSON_BYTES = 1_000_000


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_object(root: Path, relative: str) -> tuple[dict[str, Any], str]:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"input escapes project root: {relative}")
    raw = path.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"metadata exceeds {MAX_JSON_BYTES} bytes: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def materialize(root: Path) -> dict[str, Any]:
    """Verify terminal/provenance/checkpoint hashes and write one manifest."""
    root = root.resolve(strict=True)
    template, template_sha = _read_object(
        root, str(PACKET / "SWA345_SOURCE_MANIFEST_TEMPLATE_R1.json")
    )
    if template_sha != TEMPLATE_SHA256:
        raise ValueError("SWA345 source template differs from its pinned bytes")

    fit_item = template["fit_terminal_receipt"]
    fit, fit_sha = _read_object(root, fit_item["path"])
    if (
        fit.get("schema") != "wnn.h512-fit-terminal.v1"
        or fit.get("status") != "FIT_COMPLETE_ACCEPTED"
        or fit.get("run_name") != RUN
        or fit.get("source_execution_commit") != COMMIT
        or fit.get("plan_sha256") != PLAN_SHA256
        or fit.get("screen_sha256") != SCREEN_SHA256
    ):
        raise ValueError("terminal receipt is not the accepted pinned F0 fit")

    provenance_item = template["source_provenance"]
    provenance, provenance_sha = _read_object(root, provenance_item["path"])
    if (
        fit.get("provenance_sha256") != provenance_sha
        or provenance.get("args", {}).get("name") != RUN
    ):
        raise ValueError("source provenance differs from the accepted fit receipt")

    receipt_hashes = fit.get("checkpoint_sha256", {})
    source_hashes = provenance.get("checkpoint_sha256", {})
    for item in template["checkpoints"]:
        path = (root / item["path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"checkpoint escapes project root: {item['path']}")
        digest = _hash_file(path)
        name = path.name
        if receipt_hashes.get(name) != digest or source_hashes.get(name) != digest:
            raise ValueError(f"checkpoint bytes are not bound by both receipts: {name}")
        item["sha256"] = digest

    output = root / PACKET / "SWA345_SOURCE_MANIFEST_R1.json"
    template["status"] = "READY_FOR_DERIVATION"
    template["fit_terminal_receipt"]["sha256"] = fit_sha
    template["source_provenance"]["sha256"] = provenance_sha
    template["materialized_from_template_sha256"] = template_sha
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(template, indent=2, allow_nan=False) + "\n")
    return {"path": str(output), "sha256": _hash_file(output), "status": template["status"]}


def main() -> None:
    """Create the post-fit, hash-bound derivation manifest once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(materialize(args.root), sort_keys=True))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

"""Restore the two exact archived propensity files, without loading tensors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

HELPER_SHA = "85d954ab833c988d2a4a4d919d68bde049b75c3aa4061b8e0bb9f4eb347e7272"
FILES = (
    ("model.pt", 165623, "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"),
    ("provenance.json", 21626, "f6d6cf68db53c728556c1750e869e65d5d4813d622399aec768809a47985e812"),
)
REPO = "hft-kagglers/wnn-challenge2-model-archive"
REVISION = "167452c72a8e994b280446d2fca93f7c08e16f66"
PREFIX = "artifacts/runs/c20260929_prop_f0"


def main() -> None:
    """Stream two size/SHA-pinned files through the reviewed redirect helper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    raw = args.helper.read_bytes()
    if len(raw) > 20000 or hashlib.sha256(raw).hexdigest() != HELPER_SHA:
        raise ValueError("restoration helper identity mismatch")
    spec = importlib.util.spec_from_file_location("frozen_restore", args.helper)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load frozen helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    args.destination.mkdir(parents=True, exist_ok=False)
    token = (Path.home() / ".cache/huggingface/token").read_text().strip()
    opener = urllib.request.build_opener(helper.SafeRedirect())
    restored = []
    for name, size, digest in FILES:
        target = args.destination / PREFIX / name
        target.parent.mkdir(parents=True, exist_ok=True)
        hub_path = f"files/{PREFIX}/{name}"
        request = urllib.request.Request(
            f"https://huggingface.co/{REPO}/resolve/{REVISION}/{hub_path}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with opener.open(request, timeout=25) as response:
            helper.copy_verified(response, target, size, digest)
        restored.append(
            {"path": str(target), "hub_path": hub_path, "bytes": size, "sha256": digest}
        )
    receipt = {
        "schema": "quality_gap.propensity_restore.v1",
        "observed_utc": datetime.now(UTC).isoformat(),
        "status": "size_and_sha_verified_no_tensor_load",
        "repo_id": REPO,
        "revision": REVISION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": HELPER_SHA,
        "files": restored,
        "total_bytes": sum(item[1] for item in FILES),
        "credential_copied": False,
    }
    with (args.destination / "RESTORE_RECEIPT.json").open("x") as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": receipt["status"], "total_bytes": receipt["total_bytes"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        failure = {"status": "failed", "exception_type": type(error).__name__}
        if isinstance(error, urllib.error.HTTPError):
            failure["http_status"] = str(error.code)
        print(json.dumps(failure))
        raise SystemExit(1) from None

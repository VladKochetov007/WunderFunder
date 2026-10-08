from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import worker


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = Path(os.environ["WNN_OUTPUT_DIR"])
    result_path = output / "result.json"
    result = (
        json.loads(result_path.read_text())
        if result_path.is_file()
        else {
            "schema_version": 1,
            "status": "failed_before_worker",
            "attempt_id": os.environ["WNN_ATTEMPT_ID"],
            "packet_sha256": os.environ["WNN_PACKET_SHA256"],
        }
    )
    result["worker_exit_code"] = int(os.environ["WNN_WORKER_EXIT"])
    image_path = output / "image_preflight.json"
    if not image_path.is_file():
        image_path.write_text(
            json.dumps({"status": "not_reached", "worker_exit_code": result["worker_exit_code"]})
            + "\n"
        )
    worker.finalize(output, result)
    expected = json.loads(os.environ["WNN_EXPECTED_ARTIFACTS"])
    for name in expected:
        if not (output / name).is_file():
            raise FileNotFoundError("required terminal artifact missing")
    files = [
        {"path": name, "size_bytes": (output / name).stat().st_size, "sha256": sha(output / name)}
        for name in expected
    ]
    manifest = {
        "schema_version": 1,
        "attempt_id": os.environ["WNN_ATTEMPT_ID"],
        "packet_sha256": os.environ["WNN_PACKET_SHA256"],
        "worker_terminal": True,
        "files": files,
    }
    temporary = output / "artifact_manifest.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "artifact_manifest.json")


if __name__ == "__main__":
    main()

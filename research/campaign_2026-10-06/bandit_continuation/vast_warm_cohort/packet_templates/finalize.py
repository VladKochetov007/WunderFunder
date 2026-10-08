"""Write result.json and artifact_bundle.tar for one warm queue smoke packet.

The bundle contains queue records, step logs, grade JSON, the small fold-0
checkpoint, prediction sidecars and the validation receipts. The train-split
prediction cache and the training-state files are excluded by construction.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path

RUN_NAME = "smoke_c1v2_s3_f0"
GRADE_RELATIVE = "artifacts/smoke/grade_smoke_c1v2_s3_f0.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _queue_records(queue: Path) -> list[dict[str, object]]:
    results = queue / "results.jsonl"
    if not results.is_file():
        return []
    return [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    """Build the terminal result and the bounded artifact bundle."""
    output = Path(os.environ["WNN_OUTPUT_DIR"])
    spec = json.loads(Path("worker_spec.json").read_text(encoding="utf-8"))
    queue = Path("queue")
    project = Path("project")
    records = _queue_records(queue)
    succeeded = [r for r in records if r.get("status") == "succeeded"]
    grade_path = project / GRADE_RELATIVE
    grade = None
    if grade_path.is_file():
        grade = json.loads(grade_path.read_text(encoding="utf-8"))
    checkpoint = project / "artifacts" / "runs" / RUN_NAME / "model_ep0.pt"
    prediction = project / "cache" / f"pred_{RUN_NAME}_valid.npy"
    result = {
        "schema_version": 1,
        "attempt_id": spec["attempt_id"],
        "packet_sha256": os.environ.get("WNN_PACKET_SHA256"),
        "status": "delivered_smoke_graded" if succeeded and grade else "failed",
        "smoke": True,
        "scientific_claim": False,
        "heldout_claim": False,
        "grade_performed": bool(grade),
        "queue_records": records,
        "grade_json_path": GRADE_RELATIVE if grade else None,
        "grade": grade,
        "checkpoint": (
            {"path": str(checkpoint.relative_to(project)), "sha256": _sha256(checkpoint),
             "size_bytes": checkpoint.stat().st_size}
            if checkpoint.is_file() else None
        ),
        "prediction_valid": (
            {"path": str(prediction.relative_to(project)), "sha256": _sha256(prediction),
             "size_bytes": prediction.stat().st_size}
            if prediction.is_file() else None
        ),
        "worker_exit_code": int(os.environ.get("WNN_WORKER_EXIT", "-1")),
    }
    temporary = output / "result.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "result.json")

    bundle = output / "artifact_bundle.tar"
    members: list[tuple[Path, str]] = []
    for name in ("input_receipt.json", "image_preflight.json", "result.json"):
        if (output / name).is_file():
            members.append((output / name, name))
    if queue.is_dir():
        for path in sorted(queue.rglob("*")):
            if path.is_file():
                members.append((path, f"queue/{path.relative_to(queue)}"))
    for path in sorted((project / "artifacts" / "runs" / RUN_NAME).glob("*")):
        if path.is_file() and path.suffix in {".json", ".pt"} and path.name != "model.pt":
            members.append((path, f"project/artifacts/runs/{RUN_NAME}/{path.name}"))
    if grade_path.is_file():
        members.append((grade_path, f"project/{GRADE_RELATIVE}"))
    for path in sorted((project / "cache").glob(f"pred_{RUN_NAME}_valid.*")):
        if path.is_file():
            members.append((path, f"project/cache/{path.name}"))
    with tarfile.open(bundle, mode="w") as archive:
        for path, arcname in members:
            archive.add(path, arcname=arcname, recursive=False)
    return 0 if result["status"] == "delivered_smoke_graded" else 1


if __name__ == "__main__":
    raise SystemExit(main())

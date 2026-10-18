#!/usr/bin/env python3
"""Write sanitized metadata for one exact DataSphere job through SDK 0.10.0."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def write_json(path: Path, value: dict[str, object]) -> None:
    """Atomically write a private, credential-free provider metadata receipt."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    """Fetch exact job and output-file metadata without exposing signed URLs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--log-dir", required=True, type=Path)
    args = parser.parse_args()
    from datasphere.api import jobs_pb2 as jobs
    from datasphere.sdk import SDK
    from datasphere.version import version

    if version != "0.10.0":
        raise SystemExit("unsupported DataSphere SDK version")
    args.log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    sdk = SDK(profile=args.profile, log_level="ERROR", log_dir=str(args.log_dir))
    try:
        job = sdk.client.get(args.job_id)
        if job.id != args.job_id or job.project_id != args.project_id or job.name != args.job_name:
            raise ValueError("provider job identity mismatch")
        files = []
        for item in job.output_files:
            files.append(
                {
                    "path": item.desc.path,
                    "variable": item.desc.var,
                    "size_bytes": int(item.size_bytes),
                    "sha256": item.sha256,
                }
            )
        errors = []
        for error in job.file_errors:
            if error.HasField("output_file_desc"):
                errors.append(
                    {
                        "path": error.output_file_desc.path,
                        "variable": error.output_file_desc.var,
                        "type": jobs.FileUploadError.ErrorType.Name(error.type),
                    }
                )
        payload: dict[str, object] = {
            "id": job.id,
            "project_id": job.project_id,
            "name": job.name,
            "status": jobs.JobStatus.Name(job.status),
            "output_files": files,
            "output_file_errors": errors,
        }
        write_json(args.output, payload)
    except Exception as error:
        # Do not write SDK exception text; it may contain request or auth details.
        print(f"provider metadata query failed ({type(error).__name__})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

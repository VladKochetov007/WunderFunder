from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    sources = {
        name: Path("/home/vlad/Downloads") / f"{name}.zip"
        for name in ("xteam_h4", "pair_ftt2", "e6a", "p_c3zero", "p_c3only")
    }
    sources.update(
        {
            "C": ROOT / "submissions/10-12-best-r1/h390_s3_s4_q8_z4half_20261012.zip",
            "E": ROOT / "submissions/10-12-best-r2/E_c75_plus_z5_q8_20261012.zip",
        }
    )
    target = HERE / "INVENTORY.json"
    if target.exists():
        raise FileExistsError(target)
    assets = ROOT / "artifacts/team_models_20261013"
    assets.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, source in sources.items():
        archive_path = assets / f"{name}.zip"
        if not archive_path.exists():
            try:
                os.link(source, archive_path)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(source, archive_path)
        if digest(source) != digest(archive_path):
            raise ValueError("archive source mismatch")
        package = assets / name
        package.mkdir(exist_ok=True)
        members = {}
        with zipfile.ZipFile(archive_path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 50_000_000:
                raise ValueError("expanded archive exceeds inspected size bound")
            for item in archive.infolist():
                path = PurePosixPath(item.filename)
                if path.is_absolute() or ".." in path.parts or item.file_size > 6_000_000:
                    raise ValueError("unexpected archive member")
                if item.is_dir():
                    continue
                payload = archive.read(item)
                out = package / path
                out.parent.mkdir(exist_ok=True, parents=True)
                if out.exists() and out.read_bytes() != payload:
                    raise ValueError("existing extracted source changed")
                if not out.exists():
                    out.write_bytes(payload)
                record = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
                if path.suffix == ".npz":
                    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                        if sum(i.file_size for i in inner.infolist()) > 10_000_000:
                            raise ValueError("nested tensor payload too large")
                    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                        tensors = {}
                        trunk = hashlib.sha256()
                        for key in sorted(arrays.files):
                            array = arrays[key]
                            if not np.isfinite(array).all():
                                raise ValueError("nonfinite weight")
                            tensors[key] = {
                                "shape": list(array.shape),
                                "dtype": array.dtype.str,
                                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                            }
                            if not key.startswith("head."):
                                trunk.update(key.encode())
                                trunk.update(array.dtype.str.encode())
                                trunk.update(str(array.shape).encode())
                                trunk.update(array.tobytes())
                        record["tensors"] = tensors
                        record["trunk_sha256"] = trunk.hexdigest()
                members[item.filename] = record
        records[name] = {
            "source": str(source),
            "archive": str(archive_path),
            "package": str(package),
            "sha256": digest(archive_path),
            "bytes": archive_path.stat().st_size,
            "files": members,
            "members": json.loads((package / "members.json").read_text()),
            "scope": "final fitting or unknown overlap; diagnostic only",
        }
    remote = subprocess.run(
        ["git", "ls-remote", "--heads", "origin"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    value = {
        "created_utc": datetime.now(UTC).isoformat(),
        "packages": records,
        "remote_heads": remote.stdout.splitlines(),
        "archive_payloads_in_git": False,
        "data": {
            "path": "datasets/valid.parquet",
            "sha256": digest(ROOT / "datasets/valid.parquet"),
            "bytes": (ROOT / "datasets/valid.parquet").stat().st_size,
        },
    }
    target.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({k: {a: v[a] for a in ("sha256", "bytes")} for k, v in records.items()}))


if __name__ == "__main__":
    main()

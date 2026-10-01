"""Package closed campaign evidence and its 52 runs without model execution.

Run only after final reports and runtime receipts stop changing. The full Git
bundle must already exist in the delivery directory and name the current HEAD.
The archive is a research backup, not the <=20 MB competition submission.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, required=True)
    args = parser.parse_args()
    delivery = args.delivery.resolve()
    bundle = delivery / "research-source.bundle"
    output = delivery / "research-overnight.tar.gz"
    manifest_path = delivery / "research-archive.manifest.json"
    receipt_path = delivery / "research-archive.receipt.json"
    temp = output.with_suffix(".gz.incomplete")
    for path in (output, manifest_path, receipt_path, temp):
        assert not path.exists(), f"refusing overwrite: {path}"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    heads = subprocess.check_output(["git", "bundle", "list-heads", str(bundle)], text=True)
    assert f"{head} HEAD" in heads.splitlines(), "bundle must contain current HEAD"
    # A verify inside the populated source repository also accepts incremental
    # bundles. Prove this backup restores from an empty repository instead.
    with tempfile.TemporaryDirectory(prefix="wunder-bundle-restore-") as empty:
        subprocess.run(["git", "init", "--bare", "--quiet", empty], check=True)
        subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=empty, check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-c", "transfer.fsckObjects=true", "fetch", "--quiet",
                        str(bundle), "HEAD"], cwd=empty, check=True)
        restored = subprocess.check_output(["git", "rev-parse", "FETCH_HEAD"],
                                           cwd=empty, text=True).strip()
        assert restored == head
    sources = {}

    def include(source, member):
        source = Path(source)
        assert source.is_file() and not source.is_symlink(), str(source)
        assert member not in sources and not member.startswith("/")
        assert ".." not in Path(member).parts
        sources[member] = source

    for directory, subdirs, files in os.walk(HERE, followlinks=False):
        subdirs[:] = sorted(name for name in subdirs if name != "__pycache__")
        assert all(not (Path(directory) / name).is_symlink() for name in subdirs)
        for name in sorted(files):
            if name == ".DS_Store" or name.endswith(".pyc"):
                continue
            path = Path(directory) / name
            include(path, "WunderFunder-creative/" + str(path.relative_to(ROOT)))
    runs = json.loads((HERE / "completed-fit-archive-52.json").read_text())
    assert runs["run_count"] == 52 and len(runs["files"]) == 224
    for record in runs["files"]:
        path = ROOT / record["path"]
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"]
        include(path, "WunderFunder-creative/" + record["path"])
    include(bundle, "research-source.bundle")
    include(delivery / "RESTORE.md", "RESTORE.md")
    for relative in ["README.md", "manifest.json", "candV1_ft2.zip", "baseline3_fixed.zip",
                     "experimental/quant3_fixed.zip"]:
        include(delivery / relative, "delivery-2026-10-01/" + relative)
    records = [{"path": name, "bytes": source.stat().st_size, "sha256": digest(source)}
               for name, source in sorted(sources.items())]
    manifest = {
        "schema": "wunder-overnight-research-archive-v1", "source_commit": head,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "forecast_runs": 52, "run_files": 224, "files": records,
        "not_a_competition_submission": True,
        "excluded": ["organizer dataset bodies", "large remote prediction caches", "Python bytecode"],
        "remote_workspace": "beleriand:/home/grinchenkoeg/WunderFunder-creative",
        "restore_instructions": "RESTORE.md",
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    with tarfile.open(temp, "w:gz", compresslevel=6, dereference=True) as archive:
        for record in records:
            archive.add(sources[record["path"]], arcname=record["path"], recursive=False)
        info = tarfile.TarInfo("archive-manifest.json")
        info.size = len(manifest_bytes)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest_bytes))
    expected = {record["path"]: record for record in records}
    verified = set()
    with tarfile.open(temp, "r:gz") as archive:
        for member in archive:
            assert member.isfile() and member.name not in verified
            stream = archive.extractfile(member)
            if member.name == "archive-manifest.json":
                assert stream.read() == manifest_bytes
            else:
                record = expected[member.name]
                value = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    value.update(chunk)
                assert member.size == record["bytes"] and value.hexdigest() == record["sha256"]
            verified.add(member.name)
    assert verified == set(expected) | {"archive-manifest.json"}
    manifest_path.write_bytes(manifest_bytes)
    os.replace(temp, output)
    receipt = {"status": "PASS_ARCHIVE_BYTES_VERIFIED", "source_commit": head,
               "archive_file": output.name, "archive_bytes": output.stat().st_size,
               "archive_sha256": digest(output), "manifest_sha256": digest(manifest_path),
               "file_count_excluding_manifest": len(records),
               "uncompressed_file_bytes": sum(r["bytes"] for r in records),
               "forecast_runs": 52, "run_files": 224,
               "standalone_bundle_empty_repository_verify_and_fetch": True,
               "competition_model_execution": False, "WP_queries": 0}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

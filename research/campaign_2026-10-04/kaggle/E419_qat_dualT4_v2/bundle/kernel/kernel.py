from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

BUNDLE_SLUG = 'wnn-e419-qat-dualt4-20261004-v2'
EXPECTED_ARCHIVE_SHA256 = '818b90818aed37b7a89bdefbfa13cc2f8f708eb95046fb587e682a4de96f73c3'
EXPECTED_EXTERNAL_MOUNTS = {'artifacts/runs/c20260929_prop_f0/model.pt': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7.wnn'}, 'artifacts/runs/c20260929_since_s3_f0/model.pt': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'd0b3bf32497f22a7fa940ff47b21bbc4ea2414d9f955731d7fbc3e33f2c9350b.wnn'}, 'artifacts/runs/c20260929_since_s3_f0/provenance.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '52438c2c9d1a551251aee7a18cc1cca5fce2b9aed55bc558d1adafcae29c166a.wnn'}, 'artifacts/runs/c20260929_since_s4_f0/model.pt': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '25119fefe36f20fb93c8bcafb0748215e24ab956f6944f9ace34d629d36e0d2f.wnn'}, 'artifacts/runs/c20260929_since_s4_f0/provenance.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '5ec060b96a202bdd13dc7ec6b03bf1b878ea70811b99d82844a99ce164d8a99c.wnn'}, 'cache/chain_folds.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f.wnn'}, 'cache/chains.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766.wnn'}, 'cache/pred_c20260929_since_s3_f0_train.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'da6b0b23fb0f6a063e7ab59d2bd59e4a045f3ba764dd23246c7386a42683e81f.wnn'}, 'cache/pred_c20260929_since_s3_f0_train.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '2e74e44773d3d6d3e7c48c245a71f281b1659e6f6592ea2a9a2e4f96d37d19dc.wnn'}, 'cache/pred_c20260929_since_s3_f0_valid.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '9ba5a0105e0d5265415cc621e50d90148211c85427a540f0ade06e01bb1665ca.wnn'}, 'cache/pred_c20260929_since_s3_f0_valid.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '7a01c040849d9455016dee51a1f9c82c1079be085a0e6be23c99c23bddb3fba9.wnn'}, 'cache/pred_c20260929_since_s4_f0_train.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '978d949b9d7464c2706051bd351db6c1bbabe4ca3d32256afca3212b5439f5fe.wnn'}, 'cache/pred_c20260929_since_s4_f0_train.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '819ab8ef0b1303b289af3475e173da8e9e7ff404ca7a8d8e7e10a3d443648241.wnn'}, 'cache/pred_c20260929_since_s4_f0_valid.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '5f1aae3212e4316d94752b171692a8b8c5fb7d8bd9e141924aa8ddd03e708a07.wnn'}, 'cache/pred_c20260929_since_s4_f0_valid.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '6aa7b150097ebdf43ddbea8c3a52f8d705ebd2dbd01470e3c45b7eb2dbcfc430.wnn'}, 'cache/propcache_c20260929_prop_f0_train.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'b90e502506c4c93903057f5e8f15485f8e3af9148024055ecf77110f1f0155f9.wnn'}, 'cache/propcache_c20260929_prop_f0_train.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '811ce4328be739b039bc62ba1f696386e59b168fbe3733be81783a41417da174.wnn'}, 'cache/propcache_c20260929_prop_f0_valid.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '775ee227b15d0b5510c9eeee5012b23d674fed227a343e4581344568eacfb82a.wnn'}, 'cache/propcache_c20260929_prop_f0_valid.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '9f3e09f04b9fadfd142d4ecbaba8b4d17d6d1ae04ea40e2cb8d4b64ecd0a718c.wnn'}, 'cache/train_targets.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9.wnn'}, 'cache/valid_targets.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955.wnn'}, 'datasets/train.parquet': 'train.parquet', 'datasets/valid.parquet': 'valid.parquet', 'datasets/valid_mask.parquet': 'valid_mask.parquet', 'research/campaign_2026-09-29/artifacts/splits/fold0.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d.wnn'}, 'research/campaign_2026-10-04/portable_inputs.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c.wnn'}}


def release_mounts(root: Path, mounts: dict) -> None:
    """Unlink only external input symlinks before Kaggle captures working outputs."""
    for logical in mounts:
        target = root / logical
        if target.is_symlink():
            target.unlink()


def main() -> None:
    """Extract a verified private bundle and mount hashed raw data without copies."""
    matches = list(Path("/kaggle/input").glob(f"**/{BUNDLE_SLUG}/bundle-manifest.json"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one named private bundle")
    manifest = json.loads(matches[0].read_text())
    if manifest.get("private") is not True:
        raise ValueError("Bundle is not marked private")
    if manifest.get("external_mounts") != EXPECTED_EXTERNAL_MOUNTS:
        raise ValueError("External mappings differ from the controller-bound kernel")
    archive = matches[0].parent / "bundle.wnn"
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    if (
        digest.hexdigest() != manifest["archive_sha256"]
        or digest.hexdigest() != EXPECTED_ARCHIVE_SHA256
    ):
        raise ValueError("Private bundle archive identity mismatch")
    root = Path("/kaggle/working/experiment")
    if shutil.disk_usage(root.parent).free < manifest["expanded_bytes"] + 2 * 1024**3:
        raise ValueError("Insufficient working disk to extract bundle with reserve")
    root.mkdir(exist_ok=False)
    allowed = {item["path"] for item in manifest["files"]} | set(manifest["spec_paths"])
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("Archive path escapes experiment directory")
            if not member.isfile() or member.name not in allowed or member.name in seen:
                raise ValueError("Unsupported, undeclared or duplicate archive member")
            seen.add(member.name)
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("Archive member has no file content")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, 8 * 1024**2)
    for logical, mapping in manifest["external_mounts"].items():
        path = PurePosixPath(logical)
        if path.is_absolute() or ".." in path.parts or logical not in allowed:
            raise ValueError("Unsafe external input mapping")
        dataset = (
            mapping["dataset"] if isinstance(mapping, dict) else "vladkochetov07/wunder-funder"
        )
        filename = mapping["filename"] if isinstance(mapping, dict) else mapping
        slug = dataset.split("/")[1]
        candidates = list(Path("/kaggle/input").glob(f"**/{slug}/{filename}"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one authorized input: {dataset}/{filename}")
        target = root / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(candidates[0])
    command = [
        sys.executable,
        str(root / "scripts/kaggle_gpu_runner.py"),
        "--root",
        str(root),
        "--output",
        "/kaggle/working/results",
    ]
    for name in manifest["spec_paths"]:
        command.extend(["--spec", str(root / name)])
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        subprocess.run(command, check=True)
    finally:
        release_mounts(root, manifest["external_mounts"])


if __name__ == "__main__":
    main()

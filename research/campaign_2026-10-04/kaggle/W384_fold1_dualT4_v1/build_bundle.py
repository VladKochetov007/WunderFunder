from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_bundle  # noqa: E402
import kaggle_common  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold1-assets"
RAW_DATASET = "vladkochetov07/wunder-funder"
SLUG = "wnn-w384-fold1-dualt4-20261004-v1"


def build() -> None:
    asset_manifest = json.loads(
        (ROOT / "artifacts/kaggle_assets/wnn-c20261004-fold1-assets/assets-manifest.json").read_text()
    )
    assets = {item["path"]: item for item in asset_manifest["files"]}
    e413 = json.loads(
        (ROOT / "research/campaign_2026-10-04/kaggle/E413_volume_v1/bundle/bundle-manifest.json").read_text()
    )
    raw = {item["path"] for item in e413["files"] if item["path"].startswith("datasets/")}
    specs = sorted((HERE / "specs").glob("*.sealed.json"))
    paths = {item["path"] for path in specs for item in json.loads(path.read_text())["inputs"]}
    external = []
    for path in sorted(paths):
        if path in assets:
            external.append(f"{path}={ASSET_DATASET}:{assets[path]['filename']}")
        elif path in raw:
            external.append(f"{path}={path.rsplit('/', 1)[1]}")
    external_paths = {entry.split("=", 1)[0] for entry in external}
    original_verify = kaggle_common.verify_files

    def verify_small(root: Path, descriptors: list[dict]) -> None:
        original_verify(root, [item for item in descriptors if item["path"] not in external_paths])

    kaggle_common.verify_files = verify_small
    kaggle_bundle.verify_files = verify_small
    out = HERE / "bundle"
    kaggle_bundle.build_bundle(
        ROOT,
        specs,
        out,
        f"vladkochetov07/{SLUG}",
        external,
        [ASSET_DATASET],
    )

    kernel = out / "kernel/kernel.py"
    source = kernel.read_text()
    start = source.index("    command = [\n")
    end = source.index('    os.environ["PYTHONDONTWRITEBYTECODE"]', start)
    source = source[:start] + (
        "    command = [sys.executable, str(root / "
        f'"{HERE.relative_to(ROOT).as_posix()}/phase_runner.py")' 
        "]\n"
    ) + source[end:]
    kernel.write_text(source)
    metadata = json.loads((out / "kernel/kernel-metadata.json").read_text())
    metadata["enable_gpu"] = True
    metadata["enable_tpu"] = False
    metadata["enable_internet"] = False
    metadata["machine_shape"] = "NvidiaTeslaT4"
    (out / "kernel/kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "archive_sha256": json.loads((out / "bundle-manifest.json").read_text())["archive_sha256"],
        "external_mounts": len(external),
        "kernel_runner": "phase_runner.py",
        "private": True,
        "pushed": False,
    }, indent=2))


if __name__ == "__main__":
    build()

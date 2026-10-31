"""Install task-local build tools from the checksum-pinned official Rust manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
import tomllib
import urllib.request
from pathlib import Path

MANIFEST_SHA = "aaa177def36e01d539bee6bde95295230b9ce378f81057845db8d0ebe97898ee"
MIRROR = "https://rsproxy.cn"
HOST = "x86_64-unknown-linux-gnu"


def fetch(url: str, path: Path, expected: str) -> None:
    """Accept a mirror payload only when its official published digest matches."""
    if not path.exists():
        with urllib.request.urlopen(url, timeout=90) as source, path.open("xb") as output:
            while block := source.read(1 << 20):
                output.write(block)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"official checksum mismatch for {path.name}")


def main() -> None:
    """Use an alternate transport for official bytes; leave global tools untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    root = args.runtime.resolve(strict=True)
    tools = root / "tools"
    manifest_path = tools / "channel-rust-1.94.0.toml"
    fetch(MIRROR + "/dist/channel-rust-1.94.0.toml", manifest_path, MANIFEST_SHA)
    manifest = tomllib.loads(manifest_path.read_text())
    components = []
    for name in ("rustc", "rust-std", "cargo"):
        record = manifest["pkg"][name]["target"][HOST]
        official = record["xz_url"]
        if not official.startswith("https://static.rust-lang.org/dist/"):
            raise ValueError("unexpected official component origin")
        archive = tools / Path(official).name
        fetch(
            official.replace("https://static.rust-lang.org", MIRROR, 1), archive, record["xz_hash"]
        )
        stage = tools / ("unpacked-" + name)
        stage.mkdir(exist_ok=False)
        with tarfile.open(archive) as source:
            names = {Path(member.name).parts[0] for member in source.getmembers()}
            if len(names) != 1:
                raise ValueError("unexpected Rust component archive layout")
            source.extractall(stage, filter="data")
        subprocess.run(
            [
                "bash",
                str(stage / names.pop() / "install.sh"),
                "--prefix=" + str(tools / "rust"),
                "--disable-ldconfig",
            ],
            check=True,
            timeout=180,
        )
        components.append({"name": name, "official_url": official, "sha256": record["xz_hash"]})
    env = dict(
        os.environ,
        UV_PYTHON_INSTALL_DIR=str(tools / "python"),
        UV_CACHE_DIR=str(tools / "uv-cache"),
    )
    build = tools / "build-env"
    subprocess.run(
        [str(tools / "bin/uv"), "venv", "--python", "3.11", str(build)],
        env=env,
        check=True,
        timeout=90,
    )
    subprocess.run(
        [
            str(tools / "bin/uv"),
            "pip",
            "install",
            "--python",
            str(build / "bin/python"),
            "maturin>=1.8,<2",
            "numpy>=1.26,<3",
        ],
        env=env,
        check=True,
        timeout=240,
    )
    versions = subprocess.check_output([str(tools / "rust/bin/rustc"), "--version"], text=True)
    receipt = {
        "schema": "wnn.final-submit-userland-tooling.v1",
        "finished_epoch": time.time(),
        "official_manifest_sha256": MANIFEST_SHA,
        "transport": MIRROR,
        "components": components,
        "rustc": versions.strip(),
        "python": str(build / "bin/python"),
        "global_changes": False,
    }
    with (root / "receipts/userland-tooling-r2.json").open("x") as output:
        json.dump(receipt, output, indent=2)
    print("USERLAND_TOOLING_READY", versions.strip(), flush=True)


if __name__ == "__main__":
    main()

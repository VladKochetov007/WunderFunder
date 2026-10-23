from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVELS = {
    "v2": "-C target-cpu=x86-64-v2",
    "avx2": "-C target-cpu=x86-64 -C target-feature=+avx2,+fma",
    "v3": "-C target-cpu=x86-64-v3",
}


def source_hashes() -> dict[str, str]:
    """Bind a native build to its Rust sources, lockfile and build definition."""
    paths = [ROOT / "native" / name for name in ("Cargo.toml", "Cargo.lock")]
    paths += sorted((ROOT / "native" / "src").rglob("*.rs"))
    paths += [
        ROOT / "submission" / "Dockerfile.build",
        ROOT / "scripts" / "check_submission_abi.py",
        Path(__file__).resolve(),
    ]
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def run(command: list[str]) -> None:
    """Run a build command; propagate failure without starting any model code."""
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    """Compile Linux x86-64 libraries and inspect ELF metadata, without inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="wnn-i1-history-build:20261023")
    parser.add_argument("--skip-image-build", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    if not args.skip_image_build:
        run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "-t",
                args.image,
                "-f",
                str(ROOT / "submission" / "Dockerfile.build"),
                str(ROOT / "submission"),
            ]
        )
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True
    ).strip()
    for level, flags in LEVELS.items():
        run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "--cpus",
                "1",
                "--memory",
                "2g",
                "--memory-swap",
                "2g",
                "-e",
                "CARGO_BUILD_JOBS=1",
                "-e",
                "CARGO_TARGET_DIR=/out/target",
                "-e",
                f"RUSTFLAGS={flags}",
                "-v",
                f"{ROOT / 'native'}:/src:ro",
                "-v",
                f"{output}:/out",
                args.image,
                "maturin",
                "build",
                "--locked",
                "--release",
                "--features",
                "extension-module",
                "-m",
                "/src/Cargo.toml",
                "-o",
                f"/out/wheel/{level}",
            ]
        )
        wheels = list((output / "wheel" / level).glob("wnn_native-*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"expected exactly one wheel for {level}")
        with zipfile.ZipFile(wheels[0]) as wheel:
            libraries = [n for n in wheel.namelist() if n.endswith("wnn_native.abi3.so")]
            if len(libraries) != 1:
                raise ValueError(f"expected exactly one abi3 library for {level}")
            destination = output / level / "wnn_native.abi3.so"
            destination.parent.mkdir()
            destination.write_bytes(wheel.read(libraries[0]))
    if sources != source_hashes():
        raise ValueError("native sources changed during compilation; rebuild in a new directory")
    # Run only the stdlib/readelf inspector. It never imports the native library.
    run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "256m",
            "--memory-swap",
            "256m",
            "-v",
            f"{ROOT / 'scripts' / 'check_submission_abi.py'}:/inspect.py:ro",
            "-v",
            f"{output}:/out",
            *[
                item
                for level in LEVELS
                for item in ("-v", f"{output / level}:/artifact/{level}:ro")
            ],
            args.image,
            "python",
            "/inspect.py",
            "/artifact",
            "--output",
            "/out/STATIC_ABI.json",
        ]
    )
    abi = json.loads((output / "STATIC_ABI.json").read_text())
    library_hashes = {
        f"{level}/wnn_native.abi3.so": hashlib.sha256(
            (output / level / "wnn_native.abi3.so").read_bytes()
        ).hexdigest()
        for level in LEVELS
    }
    receipt = {
        "schema": "wnn.i1-native-build.v1",
        "image_id": image_id,
        "platform": "linux/amd64",
        "source_hashes": sources,
        "library_hashes": library_hashes,
        "static_abi": abi,
        "inference_executed": False,
        "tests_executed": False,
        "training_executed": False,
        "runtime_and_parity": "not_tested",
    }
    (output / "NATIVE_BUILD.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

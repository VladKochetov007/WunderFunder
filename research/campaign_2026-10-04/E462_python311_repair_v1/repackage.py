from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "research/campaign_2026-10-04/E462_public_q8_build_v2"
ZIP_INPUTS = {
    "q8r": (
        "q8r/E462_public_q8_build_v2q8r.zip",
        "7edb26d8903bafa2f3c38bf31b1a5d0dad1fd6f79e1f4d5a0ea13eaf3ca044fd",
    ),
    "q8x": (
        "q8x/submission.zip",
        "b435c04c0b047c72aeaca93a0be5aa2436a885d3abce5da483bd7fa43bfa87e7",
    ),
}


def sha256(path: Path) -> str:
    """Identify exact frozen source or package bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Replace native binaries only, update provenance, and gate the new ZIPs."""
    spec = importlib.util.spec_from_file_location("abi", ROOT / "scripts/check_submission_abi.py")
    if spec is None or spec.loader is None:
        raise ImportError("missing static ELF checker")
    abi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abi)
    frozen = ROOT / "research/campaign_2026-10-04/E416_native_q8_build/native"
    source_hashes = {}
    for source in sorted(frozen.rglob("*")):
        if source.is_file():
            relative = source.relative_to(frozen)
            if sha256(source) != sha256(OUT / "native_source" / relative):
                raise ValueError(f"frozen numerical source changed: {relative}")
            source_hashes[str(relative)] = sha256(source)
    libraries = {}
    for level in ("v2", "v3"):
        wheels = list((OUT / "wheels" / level).glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"expected exactly one {level} wheel")
        with zipfile.ZipFile(wheels[0]) as wheel:
            names = [name for name in wheel.namelist() if name.endswith(".so")]
            if len(names) != 1:
                raise ValueError(f"unexpected native members in {wheels[0]}")
            libraries[level] = wheel.read(names[0])
    report = {
        "scope": "packaging-only repair; frozen numerical source and predictor inputs",
        "frozen_source_hashes": source_hashes,
        "build_image_id": subprocess.run(
            ["docker", "image", "inspect", "wnn-native-build", "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "build_features": "python,pyo3/extension-module",
        "build_python": "3.11.16",
        "variants": {},
        "manual_ready": False,
        "remaining_gate": "offline Python 3.11 callbacks and existing competition runtime gate",
    }
    for label, (filename, expected_hash) in ZIP_INPUTS.items():
        original = ORIGINAL / filename
        if sha256(original) != expected_hash:
            raise ValueError(f"original failed artifact changed: {label}")
        package = OUT / label / "pkg"
        package.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(original) as archive:
            archive.extractall(package)
            unchanged = {
                name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()
                if not name.endswith("/")
                and not name.endswith(".so")
                and name != "E462_PROVENANCE.json"
            }
        for level, binary in libraries.items():
            (package / level / "wnn_native.abi3.so").write_bytes(binary)
        provenance_path = package / "E462_PROVENANCE.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["native_library_sha256"] = {
            level: hashlib.sha256(binary).hexdigest() for level, binary in libraries.items()
        }
        provenance["abi_repair"] = {
            "original_zip_sha256": expected_hash,
            "frozen_source": "research/campaign_2026-10-04/E416_native_q8_build/native",
            "features": report["build_features"],
            "numerical_source_unchanged": True,
        }
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        for name, expected in unchanged.items():
            if sha256(package / name) != expected:
                raise ValueError(f"predictor payload changed: {label}/{name}")
        repaired = OUT / label / f"E462_{label}_python311_repaired_v1.zip"
        with zipfile.ZipFile(repaired, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, str(path.relative_to(package)))
        static = abi.inspect_artifact(repaired)
        if not static["passed"]:
            raise ValueError(f"rebuilt library fails static scorer compatibility: {static}")
        report["variants"][label] = {
            "zip": str(repaired.relative_to(ROOT)),
            "sha256": sha256(repaired),
            "unchanged_payload_hashes": unchanged,
            "static_abi": static,
        }
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({label: result["sha256"] for label, result in report["variants"].items()}))


if __name__ == "__main__":
    main()

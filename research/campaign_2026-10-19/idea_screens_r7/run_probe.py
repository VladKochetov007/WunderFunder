"""Hash-gated CLI for the fixed R7 panel experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

DESIGN_SHA256 = "de55e24aa88bce4129116fdd15505cc7db5aa4afb07e806ac09b54aac56820d3"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
SOURCE_REVISION = "a266b036"
R7_REQUIRED_FILES = {
    "idea_screens_r7/DESIGN.md",
    "idea_screens_r7/__init__.py",
    "idea_screens_r7/model_math.py",
    "idea_screens_r7/model_training.py",
    "idea_screens_r7/panel.py",
    "idea_screens_r7/probe_execution.py",
    "idea_screens_r7/result_payload.py",
    "idea_screens_r7/run_probe.py",
    "idea_screens_r7/runtime_support.py",
    "idea_screens_r7/test_probe.py",
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json",
}

RUNTIME_IMPORT_PATHS = {
    "idea_screens_r7": "idea_screens_r7/__init__.py",
    "idea_screens_r7.model_math": "idea_screens_r7/model_math.py",
    "idea_screens_r7.model_training": "idea_screens_r7/model_training.py",
    "idea_screens_r7.panel": "idea_screens_r7/panel.py",
    "idea_screens_r7.probe_execution": "idea_screens_r7/probe_execution.py",
    "idea_screens_r7.result_payload": "idea_screens_r7/result_payload.py",
    "idea_screens_r7.runtime_support": "idea_screens_r7/runtime_support.py",
    "idea_screens_r1": "idea_screens_r1/__init__.py",
    "idea_screens_r1.eda_mechanisms": "idea_screens_r1/eda_mechanisms.py",
    "idea_screens_r1.constants": "idea_screens_r1/constants.py",
    "idea_screens_r1.input_diagnostics": "idea_screens_r1/input_diagnostics.py",
    "idea_screens_r1.sample_data": "idea_screens_r1/sample_data.py",
    "idea_screens_r1.feature_views": "idea_screens_r1/feature_views.py",
    "idea_screens_r1.target_operator": "idea_screens_r1/target_operator.py",
    "idea_screens_r2": "idea_screens_r2/__init__.py",
    "idea_screens_r2.views": "idea_screens_r2/views.py",
    "idea_screens_r3": "idea_screens_r3/__init__.py",
    "idea_screens_r3.probe_helpers": "idea_screens_r3/probe_helpers.py",
    "wnn": "canonical:wnn/__init__.py",
    "wnn.data": "canonical:wnn/data.py",
    "wnn.gmetric": "canonical:wnn/gmetric.py",
    "wnn.nn": "canonical:wnn/nn.py",
    "wnn.numeric_ple": "canonical:wnn/numeric_ple.py",
    "wnn.event_content": "canonical:wnn/event_content.py",
}


def sha256_file(path: Path) -> str:
    """Hash one bounded file in streaming blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(key: str, support_root: Path, source_root: Path) -> Path:
    """Resolve a safe frozen-source key in the staged portable bundle."""
    relative_text = key.partition(":")[2] if key.startswith("canonical:") else key
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source path in preparation manifest: {key}")
    if key.startswith("canonical:"):
        if relative.parts[:1] != ("wnn",):
            raise ValueError(f"invalid canonical source key: {key}")
        return source_root / "src" / relative
    if len(relative.parts) < 2:
        raise ValueError(f"invalid staged source key: {key}")
    return support_root / relative


def _verify_manifest(
    manifest_path: Path,
    expected_sha256: str,
    support_root: Path,
    source_root: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    """Verify reviewed design, every pinned staged/canonical byte and its digest."""
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("R7 preparation manifest SHA256 differs")
    manifest = json.loads(manifest_path.read_bytes())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("R7 preparation manifest has no frozen file pins")
    checked: dict[str, str] = {}
    for key, expected in files.items():
        if not isinstance(key, str) or not isinstance(expected, str):
            raise ValueError("invalid preparation manifest file entry")
        actual = sha256_file(_source_path(key, support_root, source_root))
        if actual != expected:
            raise ValueError(f"R7 source pin differs: {key}")
        checked[key] = actual
    if checked.get("idea_screens_r7/DESIGN.md") != DESIGN_SHA256:
        raise ValueError("preparation manifest is not bound to reviewed R7 design")
    missing_r7_files = R7_REQUIRED_FILES - checked.keys()
    if missing_r7_files:
        raise ValueError(
            f"preparation manifest omits required bundle code, test, or metadata pins: {sorted(missing_r7_files)}"
        )
    for key in RUNTIME_IMPORT_PATHS.values():
        if key not in checked:
            raise ValueError(f"preparation manifest omits runtime import source: {key}")
    return manifest, checked


def _parse_args() -> argparse.Namespace:
    """Parse the four required paths/identities for the bounded R7 CLI."""
    parser = argparse.ArgumentParser(description="Fixed R7 CPU mean/variance objective screen")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    return parser.parse_args()


def _verify_runtime_imports(
    support_root: Path, source_root: Path, pins: dict[str, str]
) -> dict[str, str]:
    """Ensure Python imported precisely the staged/pinned runtime files."""
    inventory: dict[str, str] = {}
    for module_name, pin_key in RUNTIME_IMPORT_PATHS.items():
        module = sys.modules.get(module_name)
        if module is None or not getattr(module, "__file__", None):
            raise RuntimeError(f"runtime source was not imported: {module_name}")
        path = Path(module.__file__).resolve()
        if path != _source_path(pin_key, support_root, source_root).resolve():
            raise RuntimeError(f"runtime import does not resolve to pinned source: {module_name}")
        if pins.get(pin_key) is None:
            raise RuntimeError(f"runtime source pin disappeared: {pin_key}")
        inventory[module_name] = str(path)
    return inventory


def _run(args: argparse.Namespace) -> None:
    """Verify static identities before importing data/model execution helpers."""
    sample_path = args.sample.resolve()
    output = args.output.resolve()
    if sample_path.parent.name != "research_tmp":
        raise ValueError("R7 sample must be a direct child of physical B/research_tmp")
    base = sample_path.parent.parent
    source_root = base / f"source-{SOURCE_REVISION}"
    if not source_root.is_dir():
        raise FileNotFoundError(f"canonical source snapshot is missing: {source_root}")
    if output.parent != base / "research_tmp":
        raise ValueError("R7 output must be a direct child of the sample scratch directory")
    if args.expected_sample_sha256 != SAMPLE_SHA256:
        raise ValueError("CLI sample SHA256 differs from the frozen R3 panel")

    package_dir = Path(__file__).resolve().parent
    support_root = package_dir.parent
    manifest_path = package_dir / "PREPARATION_MANIFEST.json"
    _, pins = _verify_manifest(
        manifest_path,
        args.expected_preparation_manifest_sha256,
        support_root,
        source_root,
    )
    if sha256_file(sample_path / "SAMPLE.json") != SAMPLE_SHA256:
        raise ValueError("R3 accepted SAMPLE.json SHA256 differs")
    if sha256_file(sample_path / "MEMBERSHIP.json") != MEMBERSHIP_SHA256:
        raise ValueError("R3 accepted MEMBERSHIP.json SHA256 differs")

    executor = importlib.import_module(".probe_execution", package=__package__)
    inventory = _verify_runtime_imports(support_root, source_root, pins)
    executor.execute(
        sample_path=sample_path,
        output=output,
        base=base,
        preparation_sha256=args.expected_preparation_manifest_sha256,
        identities={
            "design_sha256": DESIGN_SHA256,
            "sample_sha256": SAMPLE_SHA256,
            "membership_sha256": MEMBERSHIP_SHA256,
            "source_revision": SOURCE_REVISION,
        },
        source_pins=pins,
        runtime_imports=inventory,
    )


def main() -> None:
    """Run exactly the fixed R7 pair from hash-bound B paths."""
    _run(_parse_args())


if __name__ == "__main__":
    main()

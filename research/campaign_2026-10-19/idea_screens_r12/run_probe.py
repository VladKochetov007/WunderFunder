"""Hash-gated CLI and cold-import smoke entrypoint for the fixed R12 pair."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

DESIGN_SHA256 = "a1694748779414d2a53b89d7db3b9db791fdcd7cb55aaf87c360cf1746750d00"
REVIEW_SHA256 = "07ad8e4b749a03bcbad27913bd16d28808991a24e1df26bbe1ccfc0704d85af8"
HISTORY_SHA256 = "840dccb65445702d7e86ccf327b56a7cd51a8e972f92a1939e6d270d0c751d5a"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
SOURCE_REVISION = "a266b036"

RUNTIME_IMPORT_PATHS = {
    "idea_screens_r12": "idea_screens_r12/__init__.py",
    "idea_screens_r12.run_probe": "idea_screens_r12/run_probe.py",
    "idea_screens_r12.probe_execution": "idea_screens_r12/probe_execution.py",
    "idea_screens_r12.result_payload": "idea_screens_r12/result_payload.py",
    "idea_screens_r12.training_pair": "idea_screens_r12/training_pair.py",
    "idea_screens_r7": "idea_screens_r7/__init__.py",
    "idea_screens_r7.model_math": "idea_screens_r7/model_math.py",
    "idea_screens_r7.panel": "idea_screens_r7/panel.py",
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
    "wnn.event_content": "canonical:wnn/event_content.py",
    "wnn.numeric_ple": "canonical:wnn/numeric_ple.py",
}
REQUIRED_FILES = {
    "idea_screens_r12/DESIGN.md",
    "idea_screens_r12/DESIGN_REVIEW.json",
    "idea_screens_r12/SELECTION_HISTORY.json",
    "idea_screens_r12/__init__.py",
    "idea_screens_r12/run_probe.py",
    "idea_screens_r12/probe_execution.py",
    "idea_screens_r12/result_payload.py",
    "idea_screens_r12/training_pair.py",
    "idea_screens_r12/test_probe.py",
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json",
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_MEMBERSHIP.json",
}


def sha256_file(path: Path) -> str:
    """Hash one file using bounded-memory streaming reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(key: str, support_root: Path, source_root: Path) -> Path:
    """Resolve a relative staged or canonical source key without traversal."""
    canonical = key.startswith("canonical:")
    relative = Path(key.partition(":")[2] if canonical else key)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source path in preparation manifest: {key}")
    if canonical:
        if relative.parts[:1] != ("wnn",):
            raise ValueError(f"invalid canonical source key: {key}")
        return source_root / "src" / relative
    if len(relative.parts) < 2:
        raise ValueError(f"invalid staged source key: {key}")
    return support_root / relative


def _verify_manifest(
    path: Path, expected_sha256: str, support_root: Path, source_root: Path
) -> tuple[dict[str, object], dict[str, str]]:
    """Verify the actual manifest bytes, every source pin, and reviewed identities."""
    if sha256_file(path) != expected_sha256:
        raise ValueError("R12 preparation manifest SHA256 differs")
    manifest = json.loads(path.read_bytes())
    if manifest.get("schema") != "nf19.r12_preparation.v1":
        raise ValueError("unexpected R12 preparation manifest schema")
    files, runtime_py = manifest.get("files"), manifest.get("runtime_py")
    if not isinstance(files, dict) or not isinstance(runtime_py, dict) or not files:
        raise ValueError("R12 manifest requires file pins and runtime_py closure")
    if runtime_py != RUNTIME_IMPORT_PATHS:
        raise ValueError("R12 runtime import map differs from the reviewed closure")
    checked: dict[str, str] = {}
    for key, expected in files.items():
        if not isinstance(key, str) or not isinstance(expected, str):
            raise ValueError("invalid R12 file pin entry")
        actual = sha256_file(_source_path(key, support_root, source_root))
        if actual != expected:
            raise ValueError(f"R12 source pin differs: {key}")
        checked[key] = actual
    missing = REQUIRED_FILES - checked.keys()
    if missing:
        raise ValueError(f"R12 preparation manifest omits required files: {sorted(missing)}")
    expected_pins = {
        "idea_screens_r12/DESIGN.md": DESIGN_SHA256,
        "idea_screens_r12/DESIGN_REVIEW.json": REVIEW_SHA256,
        "idea_screens_r12/SELECTION_HISTORY.json": HISTORY_SHA256,
        "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json": SAMPLE_SHA256,
        "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_MEMBERSHIP.json": MEMBERSHIP_SHA256,
    }
    if any(checked.get(key) != value for key, value in expected_pins.items()):
        raise ValueError("R12 manifest is not bound to the reviewed design or accepted panel")
    for key in RUNTIME_IMPORT_PATHS.values():
        if key not in checked:
            raise ValueError(f"R12 manifest omits runtime source: {key}")
    return manifest, checked


def _cli_module(modules: dict[str, ModuleType]) -> ModuleType:
    """Resolve the exact named or python -m CLI module without importing a duplicate."""
    module = modules.get("idea_screens_r12.run_probe")
    if module is None:
        module = modules.get("__main__")
        spec = getattr(module, "__spec__", None) if module is not None else None
        if getattr(spec, "name", None) != "idea_screens_r12.run_probe":
            raise RuntimeError("R12 CLI must be invoked as python -m idea_screens_r12.run_probe")
    expected_file = Path(__file__).resolve()
    if Path(str(getattr(module, "__file__", ""))).resolve() != expected_file:
        raise RuntimeError("R12 CLI module path does not match this pinned entrypoint")
    return module


def _verify_runtime_imports(
    support_root: Path, source_root: Path, pins: dict[str, str]
) -> dict[str, str]:
    """Confirm all imported modules resolve to the pinned source files."""
    inventory: dict[str, str] = {}
    for name, key in RUNTIME_IMPORT_PATHS.items():
        module = sys.modules.get(name)
        if module is None and name == "idea_screens_r12.run_probe":
            module = _cli_module(sys.modules)
        if module is None or not getattr(module, "__file__", None):
            raise RuntimeError(f"R12 runtime source was not imported: {name}")
        path = Path(module.__file__).resolve()
        if path != _source_path(key, support_root, source_root).resolve():
            raise RuntimeError(f"R12 runtime import differs from pinned source: {name}")
        if key not in pins:
            raise RuntimeError(f"R12 source pin disappeared: {key}")
        inventory[name] = str(path)
    return inventory


def smoke_imports(support_root: Path, source_root: Path, pins: dict[str, str]) -> dict[str, str]:
    """Import and verify the complete runtime closure without reading data or checkpoints."""
    importlib.import_module(".probe_execution", package="idea_screens_r12")
    return _verify_runtime_imports(support_root, source_root, pins)


def _parse_args() -> argparse.Namespace:
    """Parse the fixed R12 panel and output paths plus expected byte identities."""
    parser = argparse.ArgumentParser(description="Fixed R12 CPU full-fit objective comparison")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    """Validate paths and frozen bytes before importing the data execution module."""
    sample_path, output = args.sample.resolve(), args.output.resolve()
    if sample_path.parent.name != "research_tmp" or output.parent != sample_path.parent:
        raise ValueError("R12 sample and new output must be direct children of B/research_tmp")
    base = sample_path.parent.parent
    source_root = base / f"source-{SOURCE_REVISION}"
    if not source_root.is_dir() or args.expected_sample_sha256 != SAMPLE_SHA256:
        raise ValueError("R12 canonical source snapshot or accepted panel identity differs")
    package_dir = Path(__file__).resolve().parent
    support_root = package_dir.parent
    manifest_path = package_dir / "PREPARATION_MANIFEST.json"
    _, pins = _verify_manifest(
        manifest_path, args.expected_preparation_manifest_sha256, support_root, source_root
    )
    if sha256_file(sample_path / "SAMPLE.json") != SAMPLE_SHA256:
        raise ValueError("R3 SAMPLE.json differs from the accepted panel")
    if sha256_file(sample_path / "MEMBERSHIP.json") != MEMBERSHIP_SHA256:
        raise ValueError("R3 MEMBERSHIP.json differs from the accepted panel")
    _cli_module(sys.modules)
    executor = importlib.import_module(".probe_execution", package="idea_screens_r12")
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
    """Require exact entrypoint identity before parsing or executing the R12 CLI."""
    _cli_module(sys.modules)
    _run(_parse_args())


if __name__ == "__main__":
    main()

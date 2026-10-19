"""Hash-gated R14 command entry and B-shaped source-only import smoke."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

from .runtime_support import sha256_file

DESIGN_SHA256 = "bde80d4260c0d84fe7e1b2da8229f75e7c0b3f775c8a976f65b13dc67e929d79"
REVIEW_SHA256 = "b72c67bbb57ad43e6a6063b38a013401205668489afb2ca9d40df87f02210a85"
HISTORY_SHA256 = "8aabd5810cb4a3b9cbcd721718c935f2a71e76dc5ebbdbe4bcc33e803405d8d5"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
SOURCE_REVISION = "a266b036"

RUNTIME_IMPORT_PATHS = {
    "idea_screens_r14": "idea_screens_r14/__init__.py",
    "idea_screens_r14.run_probe": "idea_screens_r14/run_probe.py",
    "idea_screens_r14.probe_execution": "idea_screens_r14/probe_execution.py",
    "idea_screens_r14.history_features": "idea_screens_r14/history_features.py",
    "idea_screens_r14.linear_math": "idea_screens_r14/linear_math.py",
    "idea_screens_r14.panel_data": "idea_screens_r14/panel_data.py",
    "idea_screens_r14.score_endpoints": "idea_screens_r14/score_endpoints.py",
    "idea_screens_r14.result_payload": "idea_screens_r14/result_payload.py",
    "idea_screens_r14.runtime_support": "idea_screens_r14/runtime_support.py",
    "idea_screens_r1": "idea_screens_r1/__init__.py",
    "idea_screens_r1.eda_mechanisms": "idea_screens_r1/eda_mechanisms.py",
    "idea_screens_r1.constants": "idea_screens_r1/constants.py",
    "idea_screens_r1.input_diagnostics": "idea_screens_r1/input_diagnostics.py",
    "idea_screens_r1.target_operator": "idea_screens_r1/target_operator.py",
    "idea_screens_r1.sample_data": "idea_screens_r1/sample_data.py",
    "idea_screens_r3": "idea_screens_r3/__init__.py",
    "idea_screens_r3.probe_helpers": "idea_screens_r3/probe_helpers.py",
    "wnn": "canonical:wnn/__init__.py",
    "wnn.data": "canonical:wnn/data.py",
    "wnn.gmetric": "canonical:wnn/gmetric.py",
}
REQUIRED_FILES = {
    "idea_screens_r14/DESIGN.md",
    "idea_screens_r14/DESIGN_REVIEW.json",
    "idea_screens_r14/SELECTION_HISTORY.json",
    "idea_screens_r14/__init__.py",
    "idea_screens_r14/run_probe.py",
    "idea_screens_r14/probe_execution.py",
    "idea_screens_r14/history_features.py",
    "idea_screens_r14/linear_math.py",
    "idea_screens_r14/panel_data.py",
    "idea_screens_r14/score_endpoints.py",
    "idea_screens_r14/result_payload.py",
    "idea_screens_r14/runtime_support.py",
    "idea_screens_r14/test_probe.py",
    "idea_screens_r14/TEST_RECEIPT.json",
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json",
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_MEMBERSHIP.json",
    "canonical:wnn/data.py",
    "canonical:wnn/gmetric.py",
    "canonical:wnn/__init__.py",
}


def _source_path(key: str, support_root: Path, source_root: Path) -> Path:
    """Resolve only safe staged or canonical source paths."""
    canonical = key.startswith("canonical:")
    relative = Path(key.partition(":")[2] if canonical else key)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe R14 source path: {key}")
    if canonical:
        if relative.parts[:1] != ("wnn",):
            raise ValueError(f"invalid R14 canonical key: {key}")
        return source_root / "src" / relative
    if len(relative.parts) < 2:
        raise ValueError(f"invalid R14 staged key: {key}")
    return support_root / relative


def _verify_manifest(
    path: Path, expected_sha256: str, support_root: Path, source_root: Path
) -> dict[str, str]:
    """Check the exact manifest, all pins, and frozen source identities."""
    if sha256_file(path) != expected_sha256:
        raise ValueError("R14 preparation manifest SHA256 differs")
    manifest = json.loads(path.read_bytes())
    if manifest.get("schema") != "nf19.r14_preparation.v1":
        raise ValueError("unexpected R14 preparation manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("runtime_py") != RUNTIME_IMPORT_PATHS:
        raise ValueError("R14 manifest must bind the reviewed runtime module map")
    checked: dict[str, str] = {}
    for key, expected in files.items():
        if not isinstance(key, str) or not isinstance(expected, str):
            raise ValueError("invalid R14 source pin")
        checked[key] = sha256_file(_source_path(key, support_root, source_root))
        if checked[key] != expected:
            raise ValueError(f"R14 source pin differs: {key}")
    if REQUIRED_FILES - checked.keys():
        raise ValueError(f"R14 manifest omits files: {sorted(REQUIRED_FILES - checked.keys())}")
    for key, digest in {
        "idea_screens_r14/DESIGN.md": DESIGN_SHA256,
        "idea_screens_r14/DESIGN_REVIEW.json": REVIEW_SHA256,
        "idea_screens_r14/SELECTION_HISTORY.json": HISTORY_SHA256,
        "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json": SAMPLE_SHA256,
        "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_MEMBERSHIP.json": MEMBERSHIP_SHA256,
        "canonical:wnn/data.py": "a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00",
        "canonical:wnn/gmetric.py": "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa",
        "canonical:wnn/__init__.py": "2dca10ee7e2e8c1139bed2165b322d2077fdbeac8095428e8c796efe47ea66c6",
    }.items():
        if checked.get(key) != digest:
            raise ValueError(f"R14 fixed source identity differs: {key}")
    if not set(RUNTIME_IMPORT_PATHS.values()).issubset(checked):
        raise ValueError("R14 manifest omits a runtime module source")
    return checked


def _cli_module(modules: dict[str, ModuleType]) -> ModuleType:
    """Require this exact pinned module or its valid python -m instance."""
    module = modules.get("idea_screens_r14.run_probe")
    if module is None:
        module = modules.get("__main__")
        spec = getattr(module, "__spec__", None) if module is not None else None
        if getattr(spec, "name", None) != "idea_screens_r14.run_probe":
            raise RuntimeError("invoke R14 with python -m idea_screens_r14.run_probe")
    if Path(str(getattr(module, "__file__", ""))).resolve() != Path(__file__).resolve():
        raise RuntimeError("R14 CLI module is not the pinned entrypoint")
    return module


def _verify_runtime_imports(
    support_root: Path, source_root: Path, pins: dict[str, str]
) -> dict[str, str]:
    """Require the observed package-prefixed import set to match all 20 pins."""
    prefixes = ("idea_screens_r14", "idea_screens_r1", "idea_screens_r3", "wnn")
    expected_names = set(RUNTIME_IMPORT_PATHS)
    actual_names = {
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    if "idea_screens_r14.run_probe" not in actual_names:
        _cli_module(sys.modules)
        actual_names.add("idea_screens_r14.run_probe")
    if actual_names != expected_names:
        missing, extra = expected_names - actual_names, actual_names - expected_names
        raise RuntimeError(
            f"R14 module closure differs: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    inventory: dict[str, str] = {}
    for name, key in RUNTIME_IMPORT_PATHS.items():
        module = sys.modules.get(name)
        if module is None and name == "idea_screens_r14.run_probe":
            module = _cli_module(sys.modules)
        if module is None or not getattr(module, "__file__", None):
            raise RuntimeError(f"R14 runtime import absent: {name}")
        path = Path(module.__file__).resolve()
        if path != _source_path(key, support_root, source_root).resolve() or key not in pins:
            raise RuntimeError(f"R14 runtime source differs from its pin: {name}")
        inventory[name] = str(path)
    return inventory


def _load_runtime_closure() -> None:
    """Import reused sample and metric helpers after source verification."""
    importlib.import_module("idea_screens_r1.sample_data")
    importlib.import_module("idea_screens_r3.probe_helpers")


def smoke_imports(
    support_root: Path, source_root: Path, manifest_path: Path, expected_sha256: str
) -> dict[str, str]:
    """Verify and import the full source closure without sample/model payload reads."""
    pins = _verify_manifest(manifest_path, expected_sha256, support_root, source_root)
    importlib.import_module("idea_screens_r14.probe_execution")
    _load_runtime_closure()
    return _verify_runtime_imports(support_root, source_root, pins)


def _parse_args() -> argparse.Namespace:
    """Parse fixed sample/output and manifest identity arguments."""
    parser = argparse.ArgumentParser(description="Fixed R14 analytic coordinate screen")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    """Validate identities and sources before loading panel payloads."""
    sample_path, output = args.sample.resolve(), args.output.resolve()
    if sample_path.parent.name != "research_tmp" or output.parent != sample_path.parent:
        raise ValueError("R14 sample and output must be direct children of B/research_tmp")
    base = sample_path.parent.parent
    source_root = base / f"source-{SOURCE_REVISION}"
    if args.expected_sample_sha256 != SAMPLE_SHA256 or not source_root.is_dir():
        raise ValueError("R14 source snapshot or accepted sample identity differs")
    support_root = Path(__file__).resolve().parent.parent
    package_dir = Path(__file__).resolve().parent
    manifest_path = package_dir / "PREPARATION_MANIFEST.json"
    pins = _verify_manifest(
        manifest_path,
        args.expected_preparation_manifest_sha256,
        support_root,
        source_root,
    )
    if sha256_file(sample_path / "SAMPLE.json") != SAMPLE_SHA256:
        raise ValueError("R14 accepted sample manifest differs")
    if sha256_file(sample_path / "MEMBERSHIP.json") != MEMBERSHIP_SHA256:
        raise ValueError("R14 accepted membership manifest differs")
    _cli_module(sys.modules)
    executor = importlib.import_module("idea_screens_r14.probe_execution")
    _load_runtime_closure()
    inventory = _verify_runtime_imports(support_root, source_root, pins)
    executor.execute(
        sample_path=sample_path,
        output=output,
        base=base,
        preparation_sha256=args.expected_preparation_manifest_sha256,
        source_pins=pins,
        runtime_imports=inventory,
    )


def main() -> None:
    """Verify the exact CLI module before parsing options or touching data."""
    _cli_module(sys.modules)
    _run(_parse_args())


if __name__ == "__main__":
    main()

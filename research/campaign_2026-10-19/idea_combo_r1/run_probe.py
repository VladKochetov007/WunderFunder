"""Hash-gated source-only smoke and fresh-process CLI for combo R1."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

DESIGN_SHA256 = "9f2c714bd09f8a4bf31507774326671ff2c279274b1af76810870262a7cf7360"
REVIEW_SHA256 = "6ede3784643fa5bb53cbf9ca001a147f4a5e0a059c7e557cb8ef97b82610e840"
HISTORY_SHA256 = "31d1f3f1edf0d18148ba9f49b273aa1cb8bb9c9edc47822101356d692205e924"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
REFERENCE_RESULT_SHA256 = "ab149a3fcf9d58c1221300ddca3eba356a16181fe5239ea1b3d7646bfe9099c3"
SOURCE_REVISION = "a266b036"
PACKAGE = "idea_combo_r1"

RUNTIME_IMPORT_PATHS = {
    "idea_combo_r1": "idea_combo_r1/__init__.py",
    "idea_combo_r1.run_probe": "idea_combo_r1/run_probe.py",
    "idea_combo_r1.probe_execution": "idea_combo_r1/probe_execution.py",
    "idea_combo_r1.reference": "idea_combo_r1/reference.py",
    "idea_combo_r1.panel": "idea_combo_r1/panel.py",
    "idea_combo_r1.training": "idea_combo_r1/training.py",
    "idea_combo_r1.scoring": "idea_combo_r1/scoring.py",
    "idea_combo_r1.runtime_support": "idea_combo_r1/runtime_support.py",
    "idea_screens_r1": "idea_screens_r1/__init__.py",
    "idea_screens_r1.constants": "idea_screens_r1/constants.py",
    "idea_screens_r1.eda_mechanisms": "idea_screens_r1/eda_mechanisms.py",
    "idea_screens_r1.input_diagnostics": "idea_screens_r1/input_diagnostics.py",
    "idea_screens_r1.sample_data": "idea_screens_r1/sample_data.py",
    "idea_screens_r1.target_operator": "idea_screens_r1/target_operator.py",
    "idea_screens_r15": "idea_screens_r15/__init__.py",
    "idea_screens_r15.panel": "idea_screens_r15/panel.py",
    "idea_screens_r15.pipeline_reports": "idea_screens_r15/pipeline_reports.py",
    "idea_screens_r15.representation": "idea_screens_r15/representation.py",
    "idea_screens_r15.runtime_support": "idea_screens_r15/runtime_support.py",
    "idea_screens_r15.training": "idea_screens_r15/training.py",
    "idea_screens_r3": "idea_screens_r3/__init__.py",
    "idea_screens_r3.probe_helpers": "idea_screens_r3/probe_helpers.py",
    "idea_screens_r7": "idea_screens_r7/__init__.py",
    "idea_screens_r7.model_math": "idea_screens_r7/model_math.py",
    "idea_screens_r7.model_training": "idea_screens_r7/model_training.py",
    "wnn": "canonical:wnn/__init__.py",
    "wnn.data": "canonical:wnn/data.py",
    "wnn.event_content": "canonical:wnn/event_content.py",
    "wnn.gmetric": "canonical:wnn/gmetric.py",
    "wnn.nn": "canonical:wnn/nn.py",
    "wnn.numeric_ple": "canonical:wnn/numeric_ple.py",
}

REQUIRED_FILES = {
    "idea_combo_r1/DESIGN.md",
    "idea_combo_r1/DESIGN_REVIEW.json",
    "idea_combo_r1/SELECTION_HISTORY.json",
    "idea_combo_r1/REFERENCE_R15_RESULT.json",
    "idea_combo_r1/test_combo.py",
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json",
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_MEMBERSHIP.json",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(key: str, support_root: Path, source_root: Path) -> Path:
    """Resolve only support-source or canonical source-relative pin keys."""
    canonical = key.startswith("canonical:")
    text = key.partition(":")[2] if canonical else key
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe combo source path: {key}")
    if canonical:
        if relative.parts[:1] != ("wnn",):
            raise ValueError(f"invalid combo canonical source key: {key}")
        return source_root / "src" / relative
    if len(relative.parts) < 2:
        raise ValueError(f"invalid combo staged source key: {key}")
    return support_root / relative


def _verify_manifest(
    path: Path, expected_sha256: str, support_root: Path, source_root: Path
) -> dict[str, str]:
    """Verify every declared pin and fixed design/data/reference identities."""
    if _sha256_file(path) != expected_sha256:
        raise ValueError("combo preparation manifest SHA256 differs")
    manifest = json.loads(path.read_bytes())
    files = manifest.get("files")
    if manifest.get("schema") != "nf19.combo_preparation.v1" or not isinstance(files, dict):
        raise ValueError("unexpected combo preparation manifest schema")
    if manifest.get("runtime_py") != RUNTIME_IMPORT_PATHS:
        raise ValueError("combo runtime module map differs")
    checked: dict[str, str] = {}
    for key, expected in files.items():
        if not isinstance(key, str) or not isinstance(expected, str):
            raise ValueError("invalid combo source pin entry")
        checked[key] = _sha256_file(_source_path(key, support_root, source_root))
        if checked[key] != expected:
            raise ValueError(f"combo source byte pin differs: {key}")
    missing = (REQUIRED_FILES | set(RUNTIME_IMPORT_PATHS.values())) - checked.keys()
    if missing:
        raise ValueError(f"combo preparation manifest omits: {sorted(missing)}")
    fixed = {
        "idea_combo_r1/DESIGN.md": DESIGN_SHA256,
        "idea_combo_r1/DESIGN_REVIEW.json": REVIEW_SHA256,
        "idea_combo_r1/SELECTION_HISTORY.json": HISTORY_SHA256,
        "idea_combo_r1/REFERENCE_R15_RESULT.json": REFERENCE_RESULT_SHA256,
        "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json": SAMPLE_SHA256,
        "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_MEMBERSHIP.json": MEMBERSHIP_SHA256,
        "canonical:wnn/data.py": "a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00",
        "canonical:wnn/gmetric.py": "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa",
        "canonical:wnn/nn.py": "7fca0e21fc0da2a40e2a39bed069a871b404adf6a0e41a49dcf24ddcab878d48",
    }
    for key, digest in fixed.items():
        if checked.get(key) != digest:
            raise ValueError(f"combo fixed identity differs: {key}")
    return checked


def _cli_module(modules: dict[str, ModuleType]) -> ModuleType:
    """Bind to this exact pinned CLI whether imported or invoked using -m."""
    module = modules.get(f"{PACKAGE}.run_probe")
    if module is None:
        module = modules.get("__main__")
        spec = getattr(module, "__spec__", None) if module is not None else None
        if getattr(spec, "name", None) != f"{PACKAGE}.run_probe":
            raise RuntimeError("invoke the combo runner with python -m idea_combo_r1.run_probe")
    if Path(str(getattr(module, "__file__", ""))).resolve() != Path(__file__).resolve():
        raise RuntimeError("combo CLI module is not the source-pinned entrypoint")
    return module


def _load_runtime_closure() -> None:
    """Import the declared scientific runtime closure without payload access."""
    for name in RUNTIME_IMPORT_PATHS:
        if name != f"{PACKAGE}.run_probe":
            importlib.import_module(name)


def _verify_runtime_imports(
    support_root: Path, source_root: Path, pins: dict[str, str]
) -> dict[str, str]:
    """Require the exact owned-package module set and resolved source paths."""
    prefixes = (
        "idea_combo_r1",
        "idea_screens_r1",
        "idea_screens_r15",
        "idea_screens_r3",
        "idea_screens_r7",
        "wnn",
    )
    expected_names = set(RUNTIME_IMPORT_PATHS)
    actual_names = {
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }
    cli = _cli_module(sys.modules)
    if f"{PACKAGE}.run_probe" not in actual_names:
        actual_names.add(f"{PACKAGE}.run_probe")
    if actual_names != expected_names:
        raise RuntimeError(
            f"combo source import closure differs: missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    inventory: dict[str, str] = {}
    for name, key in RUNTIME_IMPORT_PATHS.items():
        module = sys.modules.get(name)
        if name == f"{PACKAGE}.run_probe" and module is None:
            module = cli
        if module is None or not getattr(module, "__file__", None) or key not in pins:
            raise RuntimeError(f"combo runtime module has no pinned path: {name}")
        actual = Path(module.__file__).resolve()
        expected = _source_path(key, support_root, source_root).resolve()
        if actual != expected:
            raise RuntimeError(f"combo runtime module path differs: {name}")
        inventory[name] = str(actual)
    return inventory


def smoke_imports(
    support_root: Path, source_root: Path, manifest_path: Path, expected_sha256: str
) -> dict[str, str]:
    """Source-pin and import-smoke the full closure without reading payloads."""
    pins = _verify_manifest(manifest_path, expected_sha256, support_root, source_root)
    _cli_module(sys.modules)
    _load_runtime_closure()
    return _verify_runtime_imports(support_root, source_root, pins)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One fixed R15 sparse+EMA / R7 NLL combo fit")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    """Check all fixed identities and source pins before importing the executor."""
    sample_path, output = args.sample.resolve(), args.output.resolve()
    if sample_path.parent.name != "research_tmp" or output.parent != sample_path.parent:
        raise ValueError("combo sample and output must be direct children of B/research_tmp")
    base = sample_path.parent.parent
    support_root = Path(__file__).resolve().parent.parent
    package_dir = Path(__file__).resolve().parent
    source_root = base / f"source-{SOURCE_REVISION}"
    if args.expected_sample_sha256 != SAMPLE_SHA256 or not source_root.is_dir():
        raise ValueError("combo accepted sample identity or canonical source snapshot differs")
    manifest_path = package_dir / "PREPARATION_MANIFEST.json"
    pins = _verify_manifest(
        manifest_path, args.expected_preparation_manifest_sha256, support_root, source_root
    )
    if _sha256_file(sample_path / "SAMPLE.json") != SAMPLE_SHA256:
        raise ValueError("combo accepted sample manifest hash differs")
    if _sha256_file(sample_path / "MEMBERSHIP.json") != MEMBERSHIP_SHA256:
        raise ValueError("combo accepted membership manifest hash differs")
    _cli_module(sys.modules)
    _load_runtime_closure()
    inventory = _verify_runtime_imports(support_root, source_root, pins)
    cpu = importlib.import_module("idea_screens_r15.runtime_support").configure_cpu_runtime()
    if cpu["torch_threads"] != 1 or cpu["torch_interop_threads"] != 1:
        raise RuntimeError("combo requires one CPU thread and one interop thread")
    executor = importlib.import_module("idea_combo_r1.probe_execution")
    executor.execute(
        sample_path=sample_path,
        output=output,
        support_root=support_root,
        preparation_sha256=args.expected_preparation_manifest_sha256,
        source_pins=pins,
        runtime_imports=inventory,
        reference_result_path=package_dir / "REFERENCE_R15_RESULT.json",
        runtime_state=cpu,
    )


def main() -> None:
    """Validate the exact CLI identity before parsing even --help."""
    _cli_module(sys.modules)
    _run(_parse_args())


if __name__ == "__main__":
    main()

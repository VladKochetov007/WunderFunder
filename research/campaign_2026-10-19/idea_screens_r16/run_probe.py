"""Hash-gated cold CLI and exact source-only import smoke for R16."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

from .runtime_support import sha256_file

DESIGN_SHA256 = "7b0d01571b61e83362fa53a10bf4e76d0ba2d7bf0b2f6c7f9faa75ea27186696"
REVIEW_SHA256 = "952bd7561443d4240058e001fa978e6c47d7db642cd1b919c135ef3b17f46809"
CLARIFICATION_SHA256 = "821392a930258144df368158bd76411dd8f8a6f2c6f9e65e254923b42fa95be6"
HISTORY_SHA256 = "e19205a38248207ebfd601d460fec0426304ed3daa1439e16c22112058151fd1"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"

RUNTIME_IMPORT_PATHS = {
    "idea_screens_r16": "idea_screens_r16/__init__.py",
    "idea_screens_r16.run_probe": "idea_screens_r16/run_probe.py",
    "idea_screens_r16.runtime_support": "idea_screens_r16/runtime_support.py",
    "idea_screens_r16.probe_execution": "idea_screens_r16/probe_execution.py",
    "idea_screens_r16.panel": "idea_screens_r16/panel.py",
    "idea_screens_r16.estimator": "idea_screens_r16/estimator.py",
    "idea_screens_r16.linear_ops": "idea_screens_r16/linear_ops.py",
    "idea_screens_r10": "idea_screens_r10/__init__.py",
    "idea_screens_r10.panel_io": "idea_screens_r10/panel_io.py",
}
REQUIRED_FILES = {
    "idea_screens_r16/DESIGN.md",
    "idea_screens_r16/DESIGN_CLARIFICATION.json",
    "idea_screens_r16/DESIGN_REVIEW.json",
    "idea_screens_r16/SELECTION_HISTORY.json",
    "idea_screens_r16/test_probe.py",
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json",
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_MEMBERSHIP.json",
    *RUNTIME_IMPORT_PATHS.values(),
}
FIXED_FILE_HASHES = {
    "idea_screens_r16/DESIGN.md": DESIGN_SHA256,
    "idea_screens_r16/DESIGN_REVIEW.json": REVIEW_SHA256,
    "idea_screens_r16/DESIGN_CLARIFICATION.json": CLARIFICATION_SHA256,
    "idea_screens_r16/SELECTION_HISTORY.json": HISTORY_SHA256,
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json": SAMPLE_SHA256,
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_MEMBERSHIP.json": MEMBERSHIP_SHA256,
    "idea_screens_r10/panel_io.py": "3d2c68f07d169896218932c46fa3767e6a074065a99a744d455741046acf91ba",
    "idea_screens_r10/__init__.py": "44dae8f66420e4572bf8bbbea1de974dd3d63427bf79700c010439f9d8f0eb13",
}


def _source_path(key: str, support_root: Path) -> Path:
    """Resolve one safe path below the staged support source root."""
    relative = Path(key)
    root = support_root.resolve()
    candidate = (root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) < 2
        or not candidate.is_relative_to(root)
    ):
        raise ValueError(f"unsafe R16 source path: {key}")
    return candidate


def _verify_manifest(
    manifest_path: Path, expected_sha256: str, support_root: Path
) -> dict[str, str]:
    """Verify every staged byte, fixed metadata identity and the runtime closure."""
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("R16 preparation manifest SHA256 differs")
    manifest = json.loads(manifest_path.read_bytes())
    files = manifest.get("files")
    if manifest.get("schema") != "nf19.r16_preparation.v1" or not isinstance(files, dict):
        raise ValueError("unexpected R16 preparation manifest schema")
    if manifest.get("runtime_py") != RUNTIME_IMPORT_PATHS:
        raise ValueError("R16 manifest runtime module map differs")
    checked: dict[str, str] = {}
    for key, expected in files.items():
        if not isinstance(key, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("invalid R16 source pin")
        actual = sha256_file(_source_path(key, support_root))
        if actual != expected:
            raise ValueError(f"R16 source pin differs: {key}")
        checked[key] = actual
    missing = REQUIRED_FILES - checked.keys()
    if missing:
        raise ValueError(f"R16 preparation manifest omits sources: {sorted(missing)}")
    for key, expected in FIXED_FILE_HASHES.items():
        if checked.get(key) != expected:
            raise ValueError(f"R16 accepted source identity differs: {key}")
    return checked


def _cli_module(modules: dict[str, ModuleType]) -> ModuleType:
    """Require the pinned import or its exact `python -m` entrypoint."""
    module = modules.get("idea_screens_r16.run_probe")
    if module is None:
        module = modules.get("__main__")
        spec = getattr(module, "__spec__", None) if module is not None else None
        if getattr(spec, "name", None) != "idea_screens_r16.run_probe":
            raise RuntimeError("invoke R16 with python -m idea_screens_r16.run_probe")
    if Path(str(getattr(module, "__file__", ""))).resolve() != Path(__file__).resolve():
        raise RuntimeError("R16 CLI module is not the pinned entrypoint")
    return module


def _load_runtime_closure() -> None:
    """Import the complete analysis closure without opening panel payloads."""
    for name in ("idea_screens_r16.probe_execution", "idea_screens_r10.panel_io"):
        importlib.import_module(name)


def _require_numpy_only_process() -> None:
    """Fail closed if Torch or PyArrow is already present in this process."""
    if "torch" in sys.modules or "pyarrow" in sys.modules:
        raise RuntimeError("R16 requires a NumPy-only process with no Torch or PyArrow import")


def _verify_runtime_imports(support_root: Path, pins: dict[str, str]) -> dict[str, str]:
    """Require exact observed package imports and staged file identities."""
    _require_numpy_only_process()
    expected_names = set(RUNTIME_IMPORT_PATHS)
    prefixes = ("idea_screens_r16", "idea_screens_r10")
    actual_names = {
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    cli = _cli_module(sys.modules)
    if "idea_screens_r16.run_probe" not in actual_names:
        actual_names.add("idea_screens_r16.run_probe")
    if actual_names != expected_names:
        missing, extra = expected_names - actual_names, actual_names - expected_names
        raise RuntimeError(
            f"R16 import closure differs: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    inventory: dict[str, str] = {}
    for name, key in RUNTIME_IMPORT_PATHS.items():
        module = sys.modules.get(name)
        if name == "idea_screens_r16.run_probe" and module is None:
            module = cli
        if module is None or not getattr(module, "__file__", None):
            raise RuntimeError(f"R16 runtime import is absent: {name}")
        actual_path = Path(module.__file__).resolve()
        expected_path = _source_path(key, support_root).resolve()
        if actual_path != expected_path or key not in pins:
            raise RuntimeError(f"R16 runtime module path differs from its pin: {name}")
        inventory[name] = str(actual_path)
    return inventory


def smoke_imports(
    support_root: Path, manifest_path: Path, expected_manifest_sha256: str
) -> dict[str, str]:
    """Verify exact staged imports/source without reading any sample arrays."""
    _require_numpy_only_process()
    pins = _verify_manifest(manifest_path, expected_manifest_sha256, support_root)
    _load_runtime_closure()
    inventory = _verify_runtime_imports(support_root, pins)
    _require_numpy_only_process()
    return inventory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed R16 NumPy input-innovation screen")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    """Verify exact identities and source bytes before opening fit x/mask arrays."""
    sample_dir, output = args.sample.resolve(), args.output.resolve()
    if sample_dir.parent.name != "research_tmp" or output.parent != sample_dir.parent:
        raise ValueError("R16 sample and output must be direct children of B/research_tmp")
    if args.expected_sample_sha256 != SAMPLE_SHA256:
        raise ValueError("R16 accepted sample identity differs")
    support_root = Path(__file__).resolve().parent.parent
    manifest_path = Path(__file__).resolve().parent / "PREPARATION_MANIFEST.json"
    pins = _verify_manifest(
        manifest_path,
        args.expected_preparation_manifest_sha256,
        support_root,
    )
    if sha256_file(sample_dir / "SAMPLE.json") != SAMPLE_SHA256:
        raise ValueError("R16 accepted sample manifest SHA256 differs")
    if sha256_file(sample_dir / "MEMBERSHIP.json") != MEMBERSHIP_SHA256:
        raise ValueError("R16 accepted membership manifest SHA256 differs")
    _cli_module(sys.modules)
    _load_runtime_closure()
    inventory = _verify_runtime_imports(support_root, pins)
    executor = importlib.import_module("idea_screens_r16.probe_execution")
    executor.execute(
        sample_dir=sample_dir,
        output=output,
        expected_sample_sha256=args.expected_sample_sha256,
        preparation_manifest_sha256=args.expected_preparation_manifest_sha256,
        source_pins=pins,
        runtime_imports=inventory,
    )


def main() -> None:
    """Check CLI module identity before parsing any options."""
    _cli_module(sys.modules)
    _run(_parse_args())


if __name__ == "__main__":
    main()

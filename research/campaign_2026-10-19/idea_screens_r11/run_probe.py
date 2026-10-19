"""Pre-import source gates and CLI for the fixed R11 sampler comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

DESIGN_SHA256 = "15860112cd051995253120596cb57acf8bbbfe2493d76c1245c9494631decf5c"
INITIAL_DESIGN_SHA256 = "47b117730f817547600f16e1a5079842f507df6c74df8f157e79e9be8ff472d9"
REVIEW_SHA256 = "b35632e30d84678bfb2f8769523460e02667d27430a9670a267a1b869974a0a0"
SELECTION_SHA256 = "5482e3f5f506c143ea1a19af5f333077e58c39b7bad58c93bedc648832375978"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
SAMPLE_FIXTURE_SHA256 = SAMPLE_SHA256
SOURCE_REVISION = "a266b036"

RUNTIME_IMPORT_PATHS = {
    "idea_screens_r11": "idea_screens_r11/__init__.py",
    "idea_screens_r11.importance_math": "idea_screens_r11/importance_math.py",
    "idea_screens_r11.importance_training": "idea_screens_r11/importance_training.py",
    "idea_screens_r11.probe_execution": "idea_screens_r11/probe_execution.py",
    "idea_screens_r11.result_payload": "idea_screens_r11/result_payload.py",
    "idea_screens_r11.run_probe": "idea_screens_r11/run_probe.py",
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
    "idea_screens_r7": "idea_screens_r7/__init__.py",
    "idea_screens_r7.model_math": "idea_screens_r7/model_math.py",
    "idea_screens_r7.panel": "idea_screens_r7/panel.py",
    "idea_screens_r7.runtime_support": "idea_screens_r7/runtime_support.py",
    "wnn": "canonical:wnn/__init__.py",
    "wnn.data": "canonical:wnn/data.py",
    "wnn.gmetric": "canonical:wnn/gmetric.py",
}

REQUIRED_FILES = {
    "idea_screens_r11/DESIGN.md",
    "idea_screens_r11/DESIGN_INITIAL.md",
    "idea_screens_r11/DESIGN_REVIEW.json",
    "idea_screens_r11/SELECTION_HISTORY.json",
    "idea_screens_r11/__init__.py",
    "idea_screens_r11/importance_math.py",
    "idea_screens_r11/importance_training.py",
    "idea_screens_r11/probe_execution.py",
    "idea_screens_r11/result_payload.py",
    "idea_screens_r11/run_probe.py",
    "idea_screens_r11/test_probe.py",
    "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json",
}


def sha256_file(path: Path) -> str:
    """Hash one frozen source or sample metadata file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(key: str, support_root: Path, source_root: Path) -> Path:
    """Resolve a safe local bundle path or canonical WNN source path."""
    relative_text = key.partition(":")[2] if key.startswith("canonical:") else key
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source key in preparation manifest: {key}")
    if key.startswith("canonical:"):
        if relative.parts[:1] != ("wnn",):
            raise ValueError(f"canonical source key is outside wnn/: {key}")
        return source_root / "src" / relative
    if len(relative.parts) < 2:
        raise ValueError(f"local source key must include a staged directory: {key}")
    return support_root / relative


def verify_manifest(
    manifest_path: Path,
    expected_sha256: str,
    support_root: Path,
    source_root: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    """Verify exact manifest bytes and every required source pin before imports."""
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("R11 preparation manifest SHA256 differs")
    manifest = json.loads(manifest_path.read_bytes())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("R11 preparation manifest has no frozen file pins")
    checked: dict[str, str] = {}
    for key, expected in files.items():
        if not isinstance(key, str) or not isinstance(expected, str):
            raise ValueError("invalid preparation manifest file entry")
        actual = sha256_file(source_path(key, support_root, source_root))
        if actual != expected:
            raise ValueError(f"R11 frozen source pin differs: {key}")
        checked[key] = actual
    required = REQUIRED_FILES | set(RUNTIME_IMPORT_PATHS.values())
    if required - checked.keys():
        raise ValueError(
            f"preparation manifest omits required files: {sorted(required - checked.keys())}"
        )
    expected_identities = {
        "idea_screens_r11/DESIGN.md": DESIGN_SHA256,
        "idea_screens_r11/DESIGN_INITIAL.md": INITIAL_DESIGN_SHA256,
        "idea_screens_r11/DESIGN_REVIEW.json": REVIEW_SHA256,
        "idea_screens_r11/SELECTION_HISTORY.json": SELECTION_SHA256,
        "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json": SAMPLE_FIXTURE_SHA256,
    }
    for key, expected in expected_identities.items():
        if checked.get(key) != expected:
            raise ValueError(f"R11 manifest is not bound to the reviewed identity: {key}")
    return manifest, checked


def verify_runtime_imports(
    support_root: Path, source_root: Path, pins: dict[str, str]
) -> dict[str, str]:
    """Require imported runtime modules to resolve to their reviewed source files."""
    inventory: dict[str, str] = {}
    for module_name, source_key in RUNTIME_IMPORT_PATHS.items():
        module = sys.modules.get(module_name)
        expected = source_path(source_key, support_root, source_root).resolve()
        if module is None and module_name == "idea_screens_r11.run_probe":
            candidate = sys.modules.get("__main__")
            if _is_exact_cli_main(candidate, expected):
                module = candidate
        if module is None or not getattr(module, "__file__", None):
            raise RuntimeError(f"R11 runtime source was not imported: {module_name}")
        resolved = Path(module.__file__).resolve()
        if resolved != expected or source_key not in pins:
            raise RuntimeError(f"R11 runtime import differs from pinned source: {module_name}")
        inventory[module_name] = str(resolved)
    return inventory


def _is_exact_cli_main(module: object, expected_path: Path) -> bool:
    """Recognize only the exact package module executed through ``python -m``."""
    if not isinstance(module, ModuleType):
        return False
    spec = getattr(module, "__spec__", None)
    if getattr(spec, "name", None) != "idea_screens_r11.run_probe":
        return False
    module_file = getattr(module, "__file__", None)
    return bool(module_file) and Path(module_file).resolve() == expected_path


def _require_exact_cli_main() -> None:
    """Fail closed unless this CLI is executing as its pinned package module."""
    expected = Path(__file__).resolve()
    if not _is_exact_cli_main(sys.modules.get("__main__"), expected):
        raise RuntimeError("R11 CLI must run as the pinned python -m package entrypoint")


def smoke_imports(support_root: Path, source_root: Path, pins: dict[str, str]) -> dict[str, str]:
    """Import and verify the complete R11 analysis closure without reading data."""
    importlib.import_module(".probe_execution", package=__package__)
    return verify_runtime_imports(support_root, source_root, pins)


def _parse_args() -> argparse.Namespace:
    """Parse only the sample/output paths and expected content identities."""
    parser = argparse.ArgumentParser(
        description="Fixed-step R11 CPU importance-sampling comparison"
    )
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    """Verify all static identities before importing or mapping analysis data."""
    sample_path = args.sample.resolve()
    output = args.output.resolve()
    if sample_path.parent.name != "research_tmp":
        raise ValueError("R11 sample must be a direct child of physical B/research_tmp")
    base = sample_path.parent.parent
    source_root = base / f"source-{SOURCE_REVISION}"
    if not source_root.is_dir():
        raise FileNotFoundError(f"canonical source snapshot is missing: {source_root}")
    if output.parent != sample_path.parent or output.exists():
        raise ValueError("R11 output must be a new direct child of the sample scratch directory")
    if args.expected_sample_sha256 != SAMPLE_SHA256:
        raise ValueError("CLI sample SHA256 differs from accepted R3 panel")

    package_dir = Path(__file__).resolve().parent
    support_root = package_dir.parent
    manifest_path = package_dir / "PREPARATION_MANIFEST.json"
    _, pins = verify_manifest(
        manifest_path,
        args.expected_preparation_manifest_sha256,
        support_root,
        source_root,
    )
    if sha256_file(sample_path / "SAMPLE.json") != SAMPLE_SHA256:
        raise ValueError("accepted R3 SAMPLE.json SHA256 differs")
    if sha256_file(sample_path / "MEMBERSHIP.json") != MEMBERSHIP_SHA256:
        raise ValueError("accepted R3 MEMBERSHIP.json SHA256 differs")

    executor = importlib.import_module(".probe_execution", package=__package__)
    inventory = verify_runtime_imports(support_root, source_root, pins)
    identities = {
        "design_sha256": DESIGN_SHA256,
        "design_initial_sha256": INITIAL_DESIGN_SHA256,
        "design_review_sha256": REVIEW_SHA256,
        "selection_history_sha256": SELECTION_SHA256,
        "sample_sha256": SAMPLE_SHA256,
        "membership_sha256": MEMBERSHIP_SHA256,
        "source_revision": SOURCE_REVISION,
    }
    return executor.execute(
        sample_path=sample_path,
        output=output,
        base=base,
        preparation_sha256=args.expected_preparation_manifest_sha256,
        identities=identities,
        source_pins=pins,
        runtime_imports=inventory,
    )


def main() -> None:
    """Run exactly the fixed R11 sampler pair from hash-bound B paths."""
    _require_exact_cli_main()
    run(_parse_args())


if __name__ == "__main__":
    main()

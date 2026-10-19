"""Hash-gated CPU-only entry point for the single R8 control arm."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

DESIGN_SHA256 = "e9f36bd9ce16a0dae88c02b33dabc28c6dbec2185efc606afe11dd2ce135cbb9"
DESIGN_REVIEW_SHA256 = "89127dae450962bc3d35fea71857910d15badd30fedfdccf001f848841b4238b"
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
R7_DESIGN_SHA256 = "de55e24aa88bce4129116fdd15505cc7db5aa4afb07e806ac09b54aac56820d3"
R7_PREPARATION_SHA256 = "9b941a2180e9012b3011a716d07dde40b0272a04e34a69fcaaa8afc011dd90a4"
R7_RESULT_SHA256 = "dd36e82a51672a9b9e2ab694dd92fcd827a2445d607927548200d333d1f4b9e0"
R7_TEST_RECEIPT_SHA256 = "9ea887cd5d7cd7e115c0639971d6c0ae43ad9b3ff7075dcdbd4db58d9580d40a"
SOURCE_REVISION = "a266b036"
TORCH_VERSION = "2.14.0+cu130"
NUMPY_VERSION = "2.2.6"
R7_MANIFEST_KEY = "idea_screens_r7/PREPARATION_MANIFEST.json"
R7_RESULT_KEY = "idea_screens_r7/records/probe_v1/terminal_capture/RESULT.json"
R7_TEST_RECEIPT_KEY = "idea_screens_r7/TEST_RECEIPT.json"
R3_FIXTURE_KEY = "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json"
R8_RUNTIME_IMPORT_PATHS = {
    "idea_screens_r8": "idea_screens_r8/__init__.py",
    "idea_screens_r8.model": "idea_screens_r8/model.py",
    "idea_screens_r8.scoring": "idea_screens_r8/scoring.py",
    "idea_screens_r8.reporting": "idea_screens_r8/reporting.py",
    "idea_screens_r8.execution": "idea_screens_r8/execution.py",
    "idea_screens_r8.run_probe": "idea_screens_r8/run_probe.py",
}
R8_REQUIRED_FILES = {
    "idea_screens_r8/DESIGN.md",
    "idea_screens_r8/DESIGN_REVIEW.json",
    "idea_screens_r8/__init__.py",
    "idea_screens_r8/execution.py",
    "idea_screens_r8/model.py",
    "idea_screens_r8/reporting.py",
    "idea_screens_r8/run_probe.py",
    "idea_screens_r8/scoring.py",
    "idea_screens_r8/test_probe.py",
    R7_MANIFEST_KEY,
    R7_RESULT_KEY,
    R7_TEST_RECEIPT_KEY,
    R3_FIXTURE_KEY,
}


def sha256_file(path: Path) -> str:
    """Hash one file in bounded streaming blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(key: str, support_root: Path, source_root: Path) -> Path:
    """Resolve safe campaign or canonical source keys in the staged bundle."""
    relative_text = key.partition(":")[2] if key.startswith("canonical:") else key
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe R8 manifest path: {key}")
    if key.startswith("canonical:"):
        if relative.parts[:1] != ("wnn",):
            raise ValueError(f"invalid canonical R8 source path: {key}")
        return source_root / "src" / relative
    if len(relative.parts) < 2:
        raise ValueError(f"invalid R8 source path: {key}")
    return support_root / relative


def verify_manifest(
    manifest_path: Path,
    expected_sha256: str,
    support_root: Path,
    source_root: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    """Check every declared source pin and bind R8 to its reviewed evidence."""
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("R8 preparation manifest SHA256 differs")
    manifest = json.loads(manifest_path.read_bytes())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("R8 preparation manifest has no source pins")
    checked: dict[str, str] = {}
    for key, expected in files.items():
        if not isinstance(key, str) or not isinstance(expected, str):
            raise ValueError("invalid R8 source pin entry")
        actual = sha256_file(_source_path(key, support_root, source_root))
        if actual != expected:
            raise ValueError(f"R8 pinned source differs: {key}")
        checked[key] = actual
    missing = R8_REQUIRED_FILES - checked.keys()
    if missing:
        raise ValueError(f"R8 preparation manifest omits required sources: {sorted(missing)}")
    if checked.get("idea_screens_r8/DESIGN.md") != DESIGN_SHA256:
        raise ValueError("R8 preparation manifest is not bound to the frozen design")
    if checked.get("idea_screens_r8/DESIGN_REVIEW.json") != DESIGN_REVIEW_SHA256:
        raise ValueError("R8 design review pin differs")
    if checked.get(R7_MANIFEST_KEY) != R7_PREPARATION_SHA256:
        raise ValueError("frozen R7 preparation manifest identity differs")
    if checked.get(R7_RESULT_KEY) != R7_RESULT_SHA256:
        raise ValueError("stored R7 result identity differs")
    if checked.get(R7_TEST_RECEIPT_KEY) != R7_TEST_RECEIPT_SHA256:
        raise ValueError("frozen R7 test receipt identity differs")
    for key in R8_RUNTIME_IMPORT_PATHS.values():
        if key not in checked:
            raise ValueError(f"R8 manifest omits runtime import source: {key}")

    r7_manifest = json.loads(_source_path(R7_MANIFEST_KEY, support_root, source_root).read_bytes())
    r7_files = r7_manifest.get("files")
    if not isinstance(r7_files, dict) or not r7_files:
        raise ValueError("stored R7 manifest has no source pins")
    for key, digest in r7_files.items():
        if checked.get(key) != digest:
            raise ValueError(f"R8 does not preserve frozen R7 source identity: {key}")
    return manifest, checked


def validate_runtime_versions(numpy_version: str, torch_version: str) -> None:
    """Fail closed unless the exact reviewed B Python runtime is present."""
    if numpy_version != NUMPY_VERSION or torch_version != TORCH_VERSION:
        raise RuntimeError(
            f"R8 runtime requires NumPy {NUMPY_VERSION} and Torch {TORCH_VERSION}; "
            f"found NumPy {numpy_version} and Torch {torch_version}"
        )


def _reference_from_result(result_path: Path, result_sha256: str) -> dict[str, object]:
    """Extract only compact scalar R7 evidence after its exact file hash passes."""
    if sha256_file(result_path) != result_sha256 or result_sha256 != R7_RESULT_SHA256:
        raise ValueError("stored compact R7 result SHA256 differs")
    result = json.loads(result_path.read_bytes())
    if (
        result.get("schema") != "nf19.r7_mean_variance.result.v1"
        or result.get("status") != "completed"
        or result.get("sample_sha256") != SAMPLE_SHA256
        or result.get("membership_sha256") != MEMBERSHIP_SHA256
    ):
        raise ValueError("stored R7 compact result has unexpected identity or status")
    initial = result.get("initialization_sha256", {}).get("initial")
    final = result.get("final_evaluation", {})
    if final.get("metric_id") != "global_wp_clip2_v1" or final.get("rows") != 59_436:
        raise ValueError("stored R7 final score evidence is not the accepted endpoint")
    arms = final.get("arms", {})
    if set(arms) != {"control", "heteroskedastic"}:
        raise ValueError("stored R7 result lacks its matched two-arm endpoint")
    control = arms["control"]
    heteroskedastic = arms["heteroskedastic"]
    if any(
        len(arm.get("component_drop_one_scores", {})) != 32 for arm in (control, heteroskedastic)
    ):
        raise ValueError("stored R7 drop-one evidence does not cover all 32 components")
    return {
        "result_sha256": result_sha256,
        "initial_fingerprint": initial,
        "control_wp": float(control["official_metric"]["wp"]),
        "heteroskedastic_wp": float(heteroskedastic["official_metric"]["wp"]),
        "control_drop_one": control["component_drop_one_scores"],
        "heteroskedastic_drop_one": heteroskedastic["component_drop_one_scores"],
    }


def _runtime_imports(support_root: Path, source_root: Path, pins: dict[str, str]) -> dict[str, str]:
    """Import and path-check every R7/R8 runtime module before panel reads."""
    r7_cli = importlib.import_module("idea_screens_r7.run_probe")
    module_paths = {
        **r7_cli.RUNTIME_IMPORT_PATHS,
        "idea_screens_r7.run_probe": "idea_screens_r7/run_probe.py",
        **R8_RUNTIME_IMPORT_PATHS,
    }
    inventory: dict[str, str] = {}
    for name, key in module_paths.items():
        module = sys.modules.get(name)
        if module is None:
            module = importlib.import_module(name)
        if name == "idea_screens_r8.run_probe":
            imported = (
                Path(module.__file__).resolve() if module.__file__ else Path(__file__).resolve()
            )
        else:
            imported = Path(module.__file__).resolve() if module.__file__ else None
        expected = _source_path(key, support_root, source_root).resolve()
        if imported != expected or pins.get(key) is None:
            raise RuntimeError(f"R8 runtime import did not resolve to pinned source: {name}")
        inventory[name] = str(imported)
    for name, key in r7_cli.RUNTIME_IMPORT_PATHS.items():
        if name not in inventory or pins.get(key) is None:
            raise RuntimeError(f"R8 import closure omitted frozen R7 module: {name}")
    return inventory


def _parse_args() -> argparse.Namespace:
    """Parse paths and exact sample/manifest/R7 result identities."""
    parser = argparse.ArgumentParser(description="One fixed R8 constant-variance CPU arm")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    parser.add_argument("--expected-r7-result-sha256", required=True)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    """Verify all bytes and runtime identities before any sample array access."""
    sample_path = args.sample.resolve()
    output = args.output.resolve()
    if sample_path.parent.name != "research_tmp":
        raise ValueError("R8 sample must be a direct child of physical B/research_tmp")
    base = sample_path.parent.parent
    source_root = base / f"source-{SOURCE_REVISION}"
    if not source_root.is_dir():
        raise FileNotFoundError(f"canonical source snapshot is missing: {source_root}")
    if output.parent != base / "research_tmp" or output.exists():
        raise ValueError("R8 output must be an absent direct child of B/research_tmp")
    if args.expected_sample_sha256 != SAMPLE_SHA256:
        raise ValueError("CLI sample identity differs from the accepted R3 panel")
    if args.expected_r7_result_sha256 != R7_RESULT_SHA256:
        raise ValueError("CLI stored R7 compact result identity differs")

    package_dir = Path(__file__).resolve().parent
    support_root = package_dir.parent
    manifest_path = package_dir / "PREPARATION_MANIFEST.json"
    _, pins = verify_manifest(
        manifest_path,
        args.expected_preparation_manifest_sha256,
        support_root,
        source_root,
    )

    result_path = _source_path(R7_RESULT_KEY, support_root, source_root)
    reference = _reference_from_result(result_path, R7_RESULT_SHA256)

    import numpy as np
    import torch

    validate_runtime_versions(np.__version__, str(torch.__version__))
    sample_metadata = {
        "sample_sha256": sha256_file(sample_path / "SAMPLE.json"),
        "membership_sha256": sha256_file(sample_path / "MEMBERSHIP.json"),
    }
    if sample_metadata["sample_sha256"] != SAMPLE_SHA256:
        raise ValueError("accepted R3 SAMPLE.json identity differs")
    if sample_metadata["membership_sha256"] != MEMBERSHIP_SHA256:
        raise ValueError("accepted R3 MEMBERSHIP.json identity differs")

    fixture = json.loads(_source_path(R3_FIXTURE_KEY, support_root, source_root).read_bytes())
    if fixture.get("schema") != "idea_screens_r3.sample.v2":
        raise ValueError("pinned R3 source metadata fixture has an unexpected schema")
    if fixture.get("selection", {}).get("selected_component_counts") != {
        "fit_full64": 64,
        "fit_nested_low32": 32,
        "eval32": 32,
    }:
        raise ValueError("pinned R3 metadata fixture does not describe the accepted panel")

    runtime_imports = _runtime_imports(support_root, source_root, pins)
    executor = importlib.import_module(".execution", package=__package__)
    identities = {
        "design_sha256": DESIGN_SHA256,
        "design_review_sha256": DESIGN_REVIEW_SHA256,
        "sample_sha256": SAMPLE_SHA256,
        "membership_sha256": MEMBERSHIP_SHA256,
        "stored_r7_result_sha256": R7_RESULT_SHA256,
        "source_revision": SOURCE_REVISION,
    }
    executor.execute(
        sample_path=sample_path,
        output=output,
        scratch=base / "research_tmp",
        preparation_sha256=args.expected_preparation_manifest_sha256,
        identities=identities,
        source_pins=pins,
        runtime_imports=runtime_imports,
        reference=reference,
        sample_metadata=sample_metadata,
    )


def main() -> None:
    """Run the one new R8 endpoint without re-running either R7 arm."""
    _run(_parse_args())


if __name__ == "__main__":
    main()

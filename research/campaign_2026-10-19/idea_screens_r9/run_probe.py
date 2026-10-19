"""Hash-gated CLI and read-only import smoke for the fixed R9 diagnostic."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

from .source_manifest import (
    CHECKPOINT_BYTES,
    CHECKPOINT_SHA256,
    DESIGN_REVIEW_SHA256,
    DESIGN_SHA256,
    MEMBERSHIP_SHA256,
    NUMPY_VERSION,
    RUNTIME_IMPORT_PATHS,
    SAMPLE_SHA256,
    SOURCE_SNAPSHOT,
    TORCH_VERSION,
    sha256_file,
    source_path,
    verify_manifest,
)


def _check_module_paths(
    bundle_root: Path, source_root: Path, pins: dict[str, str]
) -> dict[str, str]:
    """Require every runtime package import to resolve to its pinned staged file."""
    observed: dict[str, str] = {}
    for name, key in RUNTIME_IMPORT_PATHS.items():
        module = sys.modules.get(name)
        if module is None:
            module = importlib.import_module(name)
        loaded_path = Path(module.__file__).resolve() if module.__file__ else None
        expected_path = source_path(key, bundle_root, source_root).resolve()
        if loaded_path != expected_path or key not in pins:
            raise RuntimeError(f"R9 import resolved outside its pinned source: {name}")
        observed[name] = str(loaded_path)
    return observed


def smoke_imports(bundle_root: Path, source_root: Path, pins: dict[str, str]) -> dict[str, object]:
    """Import the complete R9/R6/R1/WNN closure without opening arrays or a checkpoint.

    This entry point is safe for a root-owned reader-sentinel smoke in an isolated
    process. It exercises R6's dynamic R1 digest helper but never calls a reader.
    """
    if "wnn" in sys.modules:
        raise RuntimeError("R9 import smoke requires a fresh interpreter before WNN import")
    _configure_cpu_environment()
    sys.path.insert(0, str(bundle_root))
    sys.path.insert(0, str(source_root / "src"))
    importlib.import_module("idea_screens_r9.execution")
    execution_host = importlib.import_module("idea_screens_r6.execution_host")
    torch, runtime = execution_host._runtime_modules(source_root, pins)
    model_contract = importlib.import_module("idea_screens_r6.model_contract")
    sample_panel = importlib.import_module("idea_screens_r6.sample_panel")
    importlib.import_module("idea_screens_r1.sample_data")
    importlib.import_module("idea_screens_r9.history_math")
    importlib.import_module("idea_screens_r9.history_probe")
    importlib.import_module("idea_screens_r9.state_replay")
    importlib.import_module("idea_screens_r9.window_probe")
    importlib.import_module("idea_screens_r9.run_probe")
    helper = sample_panel._r1_helpers()
    if not callable(helper.digest) or not hasattr(model_contract, "load_frozen_model"):
        raise RuntimeError("R9 source smoke omitted required frozen R6 helpers")
    imported = _check_module_paths(bundle_root, source_root, pins)
    runtime["canonical_inventory_after_r9_imports"] = execution_host._verify_runtime_inventory(
        source_root, pins
    )
    runtime["torch_version"] = str(torch.__version__)
    runtime["reader_calls"] = 0
    return {"runtime_import_paths": imported, "runtime": runtime, "model_or_data_loaded": False}


def _configure_cpu_environment() -> None:
    """Fail if CUDA is visible and pin native threadpool environment to one."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise RuntimeError("CUDA must be hidden for the R9 CPU-only diagnostic")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _parse_args() -> argparse.Namespace:
    """Parse fixed data, checkpoint, output, and preparation identities."""
    parser = argparse.ArgumentParser(description="Fixed CPU no-update GRU history diagnostic")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    """Verify every input and source before loading the checkpoint or sample arrays."""
    sample_path = args.sample.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    if (
        sample_path.parent.name != "research_tmp"
        or output.parent != sample_path.parent.parent / "research_tmp"
    ):
        raise ValueError("R9 sample and output must be siblings under physical B/research_tmp")
    base, scratch = sample_path.parent.parent, sample_path.parent.parent / "research_tmp"
    source_root = base / f"source-{SOURCE_SNAPSHOT}"
    if not source_root.is_dir():
        raise FileNotFoundError(f"canonical source snapshot is missing: {source_root}")
    if args.expected_sample_sha256 != SAMPLE_SHA256:
        raise ValueError("CLI sample identity differs from accepted R3 panel")
    if args.expected_checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("CLI checkpoint identity differs from frozen R01 model")
    if args.expected_preparation_manifest_sha256 != sha256_file(
        Path(__file__).resolve().parent / "PREPARATION_MANIFEST.json"
    ):
        raise ValueError("CLI preparation manifest identity differs")
    if (
        checkpoint_path.stat().st_size != CHECKPOINT_BYTES
        or sha256_file(checkpoint_path) != CHECKPOINT_SHA256
    ):
        raise ValueError("full-SHA-bound R01 checkpoint differs")
    if (
        sha256_file(sample_path / "SAMPLE.json") != SAMPLE_SHA256
        or sha256_file(sample_path / "MEMBERSHIP.json") != MEMBERSHIP_SHA256
    ):
        raise ValueError("accepted R3 sample metadata differs")

    package_dir = Path(__file__).resolve().parent
    bundle_root = package_dir.parent
    manifest, pins = verify_manifest(
        package_dir / "PREPARATION_MANIFEST.json",
        args.expected_preparation_manifest_sha256,
        bundle_root,
        source_root,
    )
    execution_host = importlib.import_module("idea_screens_r6.execution_host")
    execution_host._check_physical_output(output, scratch)
    _configure_cpu_environment()
    runtime_imports = smoke_imports(bundle_root, source_root, pins)
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    if numpy.__version__ != NUMPY_VERSION or str(torch.__version__) != TORCH_VERSION:
        raise RuntimeError(
            f"R9 requires NumPy {NUMPY_VERSION} and Torch {TORCH_VERSION}; "
            f"found NumPy {numpy.__version__} and Torch {torch.__version__}"
        )
    identities = {
        "design_sha256": DESIGN_SHA256,
        "design_review_sha256": DESIGN_REVIEW_SHA256,
        "sample_sha256": SAMPLE_SHA256,
        "membership_sha256": MEMBERSHIP_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_snapshot": SOURCE_SNAPSHOT,
    }
    executor = importlib.import_module("idea_screens_r9.execution")
    executor.execute(
        sample_path=sample_path,
        checkpoint_path=checkpoint_path,
        output=output,
        base=base,
        scratch=scratch,
        source_root=source_root,
        pins=pins,
        preparation_sha256=args.expected_preparation_manifest_sha256,
        identities=identities,
        runtime={
            **runtime_imports["runtime"],
            "runtime_import_paths": runtime_imports["runtime_import_paths"],
            "preparation_manifest_sha256": args.expected_preparation_manifest_sha256,
            "source_pin_count": len(pins),
            "runtime_import_closure_verified": True,
        },
    )


def main() -> None:
    """Run one fixed R9 CPU diagnostic after strict source and payload gates."""
    _run(_parse_args())


if __name__ == "__main__":
    main()

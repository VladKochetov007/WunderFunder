"""Profile standalone FastRnn member cores from a frozen final full-mix ZIP.

This benchmarks native ModelStepper.run, not the competition callback. For tiedlead
roles it intentionally removes the tied/lead tensors and residual, so those numbers
are pure GRU-core baselines. The callback's W18-only/full-mix timing must be measured
separately to estimate total tied-feature and glue overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

ROLE_ORDER = ("w18", "tiedlead_s5", "tiedlead_s6", "source4_lstm")
TIED_ONLY_PREFIXES = ("tied.", "lead_out.", "tied_out.")
MAX_PACKAGE_BYTES = 20_000_000


def sha256_file(path: Path) -> str:
    """Hash a regular package or output file in bounded memory."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name: str) -> Path:
    """Validate a package member path and return its filesystem-relative form."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"unsafe package member path: {name}")
    return Path(*relative.parts)


def extract_package(package_zip: Path, destination: Path) -> None:
    """Extract a CRC-checked package after rejecting symlinks and traversal."""
    with zipfile.ZipFile(package_zip) as archive:
        if archive.testzip() is not None:
            raise ValueError("package ZIP failed CRC validation")
        for info in archive.infolist():
            relative = safe_relative(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"package contains a symlink: {info.filename}")
            target = destination / relative
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        output.write(block)


def load_solution(package_root: Path) -> Any:
    """Import the exact callback module so it selects the packaged native library."""
    solution_path = package_root / "solution.py"
    spec = importlib.util.spec_from_file_location("final_fullmix_benchmark_solution", solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import package solution: {solution_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_weights(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load a receipt-bound NPZ without pickle support."""
    import numpy as np

    if sha256_file(path) != expected_sha256:
        raise ValueError(f"member weight bytes differ from manifest: {path.name}")
    with np.load(path, allow_pickle=False) as tensors:
        arrays = {name: tensors[name] for name in tensors.files}
    if not arrays or any(array.dtype != np.float32 for array in arrays.values()):
        raise ValueError(f"member weights must be nonempty float32 arrays: {path.name}")
    return arrays


def benchmark_member(
    native: Any,
    weights: dict[str, Any],
    *,
    role: str,
    precision: str,
    fast_activations: bool,
    steps: int,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Measure repeated standalone Rust FastRnn loops for one member."""
    import numpy as np

    core_weights = {
        name: array
        for name, array in weights.items()
        if not (role.startswith("tiedlead_") and name.startswith(TIED_ONLY_PREFIXES))
    }
    expected_width = {"w18": 211, "tiedlead_s5": 263, "tiedlead_s6": 263, "source4_lstm": 112}[role]
    if core_weights["mean"].shape != (expected_width,) or core_weights["std"].shape != (expected_width,):
        raise ValueError(f"{role} raw recurrent tensors have an unexpected feature width")
    stepper = native.ModelStepper(core_weights, fast_activations, precision)
    inputs = np.random.default_rng(seed).normal(
        loc=0.0, scale=0.2, size=(steps, expected_width)
    ).astype(np.float32)
    if stepper.cell != ("lstm" if role == "source4_lstm" else "gru"):
        raise ValueError(f"{role} expected a different recurrent cell, got {stepper.cell}")

    sample_ns: list[int] = []
    output_checksum = 0.0
    for index in range(warmups + repeats):
        stepper.reset()
        start = time.perf_counter_ns()
        output = stepper.run(inputs)
        elapsed_ns = time.perf_counter_ns() - start
        if not np.isfinite(output).all():
            raise FloatingPointError(f"{role} native core produced non-finite output")
        output_checksum += float(output[-1].sum())
        if index >= warmups:
            sample_ns.append(elapsed_ns)
    per_step = [sample / steps for sample in sample_ns]
    return {
        "role": role,
        "kind": "standalone_fast_rnn_core_no_tiedlead_residual",
        "cell": stepper.cell,
        "effective_precision": stepper.precision,
        "fast_activations": fast_activations,
        "input_width": expected_width,
        "steps_per_sample": steps,
        "warmup_samples": warmups,
        "measured_samples": repeats,
        "sample_elapsed_ns": sample_ns,
        "median_ns_per_step": statistics.median(per_step),
        "mean_ns_per_step": statistics.fmean(per_step),
        "output_checksum": output_checksum,
    }


def main() -> None:
    """Extract package weights and emit independent member-core CPU timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-zip", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    if not 1 <= args.steps <= 20_000 or not 0 <= args.warmups <= 5 or not 1 <= args.repeats <= 10:
        raise ValueError("benchmark bounds: steps 1..20000, warmups 0..5, repeats 1..10")
    package_zip = args.package_zip.resolve(strict=True)
    if package_zip.stat().st_size > MAX_PACKAGE_BYTES:
        raise ValueError(f"package ZIP exceeds {MAX_PACKAGE_BYTES} bytes")
    output = args.output_json.resolve()
    if output.exists() or output == package_zip:
        raise FileExistsError("benchmark output is create-only and must differ from package input")
    output.parent.mkdir(parents=True, exist_ok=True)
    package_sha256 = sha256_file(package_zip)

    import numpy as np

    with tempfile.TemporaryDirectory(prefix="fullmix-member-bench-") as temporary:
        package_root = Path(temporary) / "package"
        package_root.mkdir()
        extract_package(package_zip, package_root)
        solution = load_solution(package_root)
        manifest = json.loads((package_root / "members.json").read_text(encoding="utf-8"))
        if manifest.get("serving_policy") != "full_mix" or tuple(
            member.get("role") for member in manifest.get("members", ())
        ) != ROLE_ORDER:
            raise ValueError("benchmark input is not the four-role full_mix package")
        native = solution.wnn_native
        rows = []
        for role_index, role in enumerate(ROLE_ORDER):
            member = manifest["members"][role_index]
            if member.get("role") != role:
                raise ValueError("package member order changed")
            arrays = load_weights(package_root / member["weights"], member["weights_sha256"])
            rows.append(
                benchmark_member(
                    native,
                    arrays,
                    role=role,
                    precision=manifest["precision"],
                    fast_activations=manifest["fast_activations"],
                    steps=args.steps,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    seed=args.seed + role_index,
                )
            )

        result = {
            "schema": "wnn.final-fullmix-standalone-member-core-benchmark.r1",
            "status": "completed",
            "package": {
                "path": str(package_zip),
                "sha256": package_sha256,
                "precision": manifest["precision"],
                "fast_activations": manifest["fast_activations"],
                "native_module": str(Path(solution.wnn_native.__file__).resolve()),
                "native_isa_directory": solution._LEVEL,
            },
            "runtime": {
                "python": sys.version,
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "blas_openmp_threads": 1,
                "numpy": np.__version__,
            },
            "measurement_scope": (
                "Standalone ModelStepper.run on exact package member weights. W18 and LSTM measure their recurrent member cores. "
                "Tiedlead entries omit tied/lead tensors and encoder residual, so they are GRU-core baselines only; "
                "use separate FinalTl3Predictor full-mix versus W18-only wall timing for total non-W18 overhead."
            ),
            "members": rows,
        }
    with output.open("xb") as stream:
        stream.write(json.dumps(result, allow_nan=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    print(
        json.dumps(
            {"status": result["status"], "output": str(output), "package_sha256": package_sha256},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

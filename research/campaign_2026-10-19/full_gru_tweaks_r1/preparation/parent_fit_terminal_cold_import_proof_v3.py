"""Record a cold import/read-sentinel proof for the parent checker V2."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pathlib
import subprocess
import sys
from types import ModuleType

CAMPAIGN = pathlib.Path(__file__).resolve().parent.parent
MANIFEST_PATH = CAMPAIGN / "preparation/teacher_cache_cpu_repair_v2/SOURCE_MANIFEST.json"
SOURCE = pathlib.Path("/Users/nb/Documents/wunder/WunderFunder-strong-gru-tweaks-20261019")
OUTPUT = CAMPAIGN / "preparation/PARENT_FIT_TERMINAL_COLD_IMPORT_V3.json"
MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
S5_HELPER_SHA = "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5"
SUPPORT_SHA = "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409"
RNG_DIAGNOSTIC_SHA = "0bb2a87b9796872d5e26666ea7398bca69b7f0a0ce27bb0895714b5ed654598b"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _module_pins(
    modules: dict[str, ModuleType], source_root: pathlib.Path, manifest: dict[str, object]
) -> list[dict[str, str]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("source manifest file list is missing")
    expected = {
        row["path"]: row["sha256"]
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("path"), str)
        and isinstance(row.get("sha256"), str)
    }
    pins: list[dict[str, str]] = []
    for name, module in sorted(modules.items()):
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        path = pathlib.Path(origin).resolve(strict=True)
        relative = path.relative_to(source_root).as_posix()
        actual = _sha256(path)
        if expected.get(relative) != actual:
            raise ValueError(f"runtime module differs from source manifest: {relative}")
        pins.append({"module": name, "path": relative, "sha256": actual})
    return pins


def main() -> int:
    """Import only the source-bound checker and persist module/read sentinels."""
    prep = CAMPAIGN / "preparation"
    sys.path.insert(0, str(prep))
    sys.path.insert(0, str(SOURCE / "src"))
    import numpy as np
    import torch

    calls: list[str] = []
    original_numpy_load = np.load
    original_numpy_memmap = np.memmap
    original_torch_load = torch.load
    np.load = lambda *args, **kwargs: (
        calls.append("numpy.load"),
        original_numpy_load(*args, **kwargs),
    )[1]
    np.memmap = lambda *args, **kwargs: (
        calls.append("numpy.memmap"),
        original_numpy_memmap(*args, **kwargs),
    )[1]
    torch.load = lambda *args, **kwargs: (
        calls.append("torch.load"),
        original_torch_load(*args, **kwargs),
    )[1]

    def audit(event: str, args: tuple[object, ...]) -> None:
        if event == "open" and args:
            value = str(args[0])
            if (
                "/data/" in value
                or "/research_tmp/" in value
                or value.endswith((".parquet", ".npy", ".pt"))
            ):
                calls.append(f"open:{value}")

    sys.addaudithook(audit)
    checker = importlib.import_module("parent_fit_terminal_check_v2")
    parser_options = checker._expected_launch_args(SOURCE, pathlib.Path("/nonexistent/fold0.json"))
    sibling = importlib.import_module("teacher_s5_terminal_check_v1")
    sibling_path = pathlib.Path(sibling.__file__).resolve(strict=True)
    support_path = prep / "propensity_terminal_support.py"
    _eq_hashes = (
        _sha256(sibling_path) == S5_HELPER_SHA,
        _sha256(support_path) == SUPPORT_SHA,
    )
    if not all(_eq_hashes):
        raise ValueError("checker helper SHA differs")
    trainer = SOURCE / "scripts/train_rnn.py"
    parsed = sibling._launch_cli_args({"command": [str(trainer), "--name", checker.RUN]}, SOURCE)
    tensor_identity = sibling._canonical_tensor_identity(SOURCE)
    identity = tensor_identity(torch.tensor([1.0], dtype=torch.float32))
    if parsed.get("name") != checker.RUN or identity.get("shape") != [1]:
        raise ValueError("source-qualified helper smoke failed")
    synthetic_rng = torch.arange(16, dtype=torch.uint8)
    synthetic_signature = checker._tensor_rng_signature(synthetic_rng)
    checker._check_cuda_rng(
        [synthetic_rng],
        {"type": "builtins.list", "cardinality": 1, "components": [synthetic_signature]},
    )
    diagnostic_path = (
        CAMPAIGN / "records/parent_fit_r2_v1/rng_diagnostic_v2/capture/RNG_DIAGNOSTIC.json"
    )
    diagnostic_raw = diagnostic_path.read_bytes()
    if hashlib.sha256(diagnostic_raw).hexdigest() != RNG_DIAGNOSTIC_SHA:
        raise ValueError("actual RNG diagnostic report pin differs")
    diagnostic = json.loads(diagnostic_raw)
    entries = checker._validate_rng_diagnostic(
        diagnostic,
        diagnostic["state_file_stat_only"],
        checker.RNG_DIAGNOSTIC_TORCH_VERSION,
    )
    if entries["cuda"] != checker.EXPECTED_RNG_TENSORS["cuda"]:
        raise ValueError("actual CUDA RNG diagnostic signature differs")
    modules = {
        name: module
        for name, module in sys.modules.items()
        if isinstance(module, ModuleType) and (name == "wnn" or name.startswith("wnn."))
    }
    manifest_raw = MANIFEST_PATH.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != MANIFEST_SHA:
        raise ValueError("source manifest SHA differs")
    manifest = json.loads(manifest_raw)
    if manifest.get("source_commit") != SOURCE_COMMIT or len(manifest.get("files", [])) != 167:
        raise ValueError("source manifest identity differs")
    pins = _module_pins(modules, SOURCE, manifest)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if (
        commit != SOURCE_COMMIT
        or len(parser_options) != 191
        or len(pins) != 26
        or calls
        or torch.cuda.is_initialized()
    ):
        raise ValueError("cold import/read sentinel failed")
    result = {
        "schema": "fg19.parent-fit-r2-terminal-cold-import.v3",
        "status": "PASS",
        "checker_path": str(pathlib.Path(checker.__file__).resolve()),
        "checker_sha256": _sha256(pathlib.Path(checker.__file__).resolve()),
        "rng_diagnostic_path": str(diagnostic_path.resolve()),
        "rng_diagnostic_sha256": RNG_DIAGNOSTIC_SHA,
        "rng_runtime_torch_version_binding": checker.RNG_DIAGNOSTIC_TORCH_VERSION,
        "synthetic_cuda_rng_signature": synthetic_signature,
        "proof_script_sha256": _sha256(pathlib.Path(__file__).resolve()),
        "source_commit": commit,
        "source_file_count": 167,
        "source_manifest_path": str(MANIFEST_PATH),
        "source_manifest_sha256": MANIFEST_SHA,
        "canonical_parser_option_count": len(parser_options),
        "sibling_helper_path": str(sibling_path),
        "sibling_helper_sha256": _sha256(sibling_path),
        "support_helper_path": str(support_path.resolve()),
        "support_helper_sha256": _sha256(support_path),
        "source_helper_smoke": {
            "launch_cli_arg_count": len(parsed),
            "tensor_identity": identity,
        },
        "wnn_runtime_module_count": len(pins),
        "wnn_runtime_modules": pins,
        "foreign_origins": {},
        "reader_sentinel_calls": calls,
        "numpy_load_guard": np.load is not original_numpy_load,
        "numpy_memmap_guard": np.memmap is not original_numpy_memmap,
        "torch_load_guard": torch.load is not original_torch_load,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    fd = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

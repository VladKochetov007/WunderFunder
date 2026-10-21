"""Prove the exact seven-file AllThree checker import boundary locally."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

REPO = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
PYTHON = REPO / ".venv/bin/python"
CAMPAIGN = REPO / "research/campaign_2026-10-19/full_gru_tweaks_r1"
PACKET = CAMPAIGN / "preparation/child_allthree_terminal_check_v2"
PROOF = PACKET / "PRIVATE_IMPORT_BOUNDARY_PROOF_V2.json"
PINS = {
    "child_allthree_terminal_check_v2.py": (
        40_442,
        "6ff4ba93a4020a4b34ade46a2512ac77b871328e95c47f2ca4561b8dcbd435b5",
    ),
    "child_nll_terminal_check_v1.py": (
        40_474,
        "a3cf3b10515bb16275561a0ab9338227c00661347afbdf081888ec258dd1f7a2",
    ),
    "child_control_terminal_check_v3.py": (
        79_972,
        "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167",
    ),
    "child_ema_terminal_check_v3.py": (
        38_267,
        "676d548c1cd25c3c1379548999cca1300ef135dd41b4a8f73689ef8e59de3b5d",
    ),
    "parent_fit_terminal_check_v3.py": (
        64_550,
        "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41",
    ),
    "propensity_terminal_support.py": (
        13_107,
        "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
    ),
    "teacher_s5_terminal_check_v1.py": (
        37_448,
        "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5",
    ),
}
SOURCE_FILES = {
    "child_allthree_terminal_check_v2.py": (PACKET / "child_allthree_terminal_check_v2.py"),
    "child_nll_terminal_check_v1.py": (
        CAMPAIGN / "preparation/child_nll_terminal_check_v1/child_nll_terminal_check_v1.py"
    ),
    "child_control_terminal_check_v3.py": (
        CAMPAIGN
        / "preparation/child_control_terminal_verifier_v3/child_control_terminal_check_v3.py"
    ),
    "child_ema_terminal_check_v3.py": (
        CAMPAIGN / "preparation/child_ema_terminal_verifier_v3/child_ema_terminal_check_v3.py"
    ),
    "parent_fit_terminal_check_v3.py": CAMPAIGN / "preparation/parent_fit_terminal_check_v3.py",
    "propensity_terminal_support.py": CAMPAIGN / "preparation/propensity_terminal_support.py",
    "teacher_s5_terminal_check_v1.py": CAMPAIGN / "preparation/teacher_s5_terminal_check_v1.py",
}

CHILD_PROBE = r"""
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

private_dir = Path(sys.argv[1])
private_dir_text = os.path.realpath(os.path.abspath(os.fspath(private_dir)))
private_dir = Path(private_dir_text)
pins = json.loads(sys.stdin.read())
expected_names = set(pins)
expected_interpreter = os.environ.get("FG19_EXPECTED_LITERAL_INTERPRETER")
if not expected_interpreter or sys.executable != expected_interpreter:
    raise RuntimeError("child sys.executable differs from the literal repository venv path")
if sys.dont_write_bytecode is not True or sys.pycache_prefix is not None:
    raise RuntimeError("child bytecode controls differ from the required -E -B defaults")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty")
if {path.name for path in private_dir.iterdir()} != expected_names:
    raise RuntimeError("temporary private directory must contain exactly seven pinned code files")
if (private_dir / "__pycache__").exists() or any(
    path.suffix == ".pyc" for path in private_dir.rglob("*.pyc")
):
    raise RuntimeError("temporary private directory must have no cached bytecode before import")

def _file_pin(path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a regular nonsymlink code file: {path.name}")
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

pre_import = {name: _file_pin(private_dir / name) for name in sorted(expected_names)}
expected = {name: {"bytes": pin[0], "sha256": pin[1]} for name, pin in pins.items()}
if pre_import != expected:
    raise RuntimeError("one or more private code files differ before checker import")
sys.path.insert(0, private_dir_text)

open_attempts = []
open_categories = {
    "worker_or_B_root": 0,
    "model_or_training_state": 0,
    "dictionary": 0,
    "cache": 0,
    "data_or_split": 0,
}
private_code_open_count = 0
private_files = {os.path.abspath(os.fspath(private_dir / name)) for name in expected_names}
remote_roots = ("/home/", "/mnt/", "/workspace/")
local_b_root = (
    "/users/nb/documents/wunder/wunderfunder-maintenance/"
    "research/campaign_2026-10-19/full_gru_tweaks_r1/b/"
)
local_cache_roots = (
    "/users/nb/documents/wunder/wunderfunder-maintenance/research_tmp/",
    "/users/nb/documents/wunder/wunderfunder-strong-gru-tweaks-20261019/research_tmp/",
)
local_data_roots = (
    "/users/nb/documents/wunder/wunderfunder-maintenance/data/",
    "/users/nb/documents/wunder/wunderfunder-maintenance/splits/",
    "/users/nb/documents/wunder/wunderfunder-strong-gru-tweaks-20261019/data/",
    "/users/nb/documents/wunder/wunderfunder-strong-gru-tweaks-20261019/splits/",
)
payload_names = {
    "model.pt", "training_state.pt", "input_representation.pt", "dictionary.pt",
    "model_ep0.pt", "model_ep1.pt", "model_ep2.pt", "model_ep3.pt", "model_ep4.pt",
    "model_ep5.pt", "train.npy", "valid.npy", "train.parquet", "valid.parquet",
}
payload_suffixes = {".pt", ".pth", ".npy", ".npz", ".safetensors", ".parquet", ".arrow"}

def _audit(event, args):
    global private_code_open_count
    if event != "open" or not args:
        return
    raw_path = args[0]
    if isinstance(raw_path, bytes):
        path = os.fsdecode(raw_path)
    elif isinstance(raw_path, (str, os.PathLike)):
        path = os.fspath(raw_path)
    else:
        return
    normalized = os.path.abspath(path)
    lower = normalized.lower()
    base = os.path.basename(lower)
    categories = []
    if lower.startswith(remote_roots) or lower.startswith(local_b_root):
        categories.append("worker_or_B_root")
    if base in payload_names or Path(lower).suffix in {".pt", ".pth", ".safetensors"}:
        categories.append("model_or_training_state")
    if base in {"input_representation.pt", "dictionary.pt"}:
        categories.append("dictionary")
    if any(lower.startswith(root) for root in local_cache_roots) or "/research_tmp/" in lower:
        categories.append("cache")
    if any(lower.startswith(root) for root in local_data_roots) or Path(lower).suffix in {
        ".npy", ".npz", ".parquet", ".arrow"
    }:
        categories.append("data_or_split")
    if categories:
        for category in set(categories):
            open_categories[category] += 1
        open_attempts.append({"path": normalized, "categories": sorted(set(categories))})
        raise PermissionError("AllThree import proof denied a worker/data/model payload open")
    if normalized in private_files:
        private_code_open_count += 1

sys.addaudithook(_audit)

# Local Torch import is baseline setup, not a worker-runtime proof. The
# is_initialized observer is separate from the action-call counter.
import torch

cuda_initialized_observations = 0
original_is_initialized = torch.cuda.is_initialized

def _observe_initialized():
    global cuda_initialized_observations
    cuda_initialized_observations += 1
    return original_is_initialized()

torch.cuda.is_initialized = _observe_initialized
if torch.cuda.is_initialized():
    raise RuntimeError("CUDA was initialized before action guards were installed")

guarded_cuda_calls = []
guarded_cuda_api_names = []

def _reject_cuda_action(name):
    def reject(*args, **kwargs):
        guarded_cuda_calls.append(name)
        raise RuntimeError(f"guarded CUDA action API called: {name}")
    return reject

for name in (
    "init", "_lazy_init", "set_device", "current_device", "device_count", "is_available",
    "synchronize", "empty_cache", "memory_allocated", "max_memory_allocated",
    "memory_reserved", "max_memory_reserved", "reset_peak_memory_stats",
    "reset_accumulated_memory_stats", "get_device_properties", "get_device_name",
    "get_device_capability", "mem_get_info", "ipc_collect", "list_gpu_processes",
    "manual_seed", "manual_seed_all", "seed", "seed_all", "get_rng_state",
    "get_rng_state_all", "set_rng_state", "set_rng_state_all", "current_stream",
    "default_stream", "stream", "device", "graph", "CUDAGraph", "Stream", "Event",
    "_exchange_device", "_set_device", "_get_device", "_get_device_index",
    "_get_device_properties",
):
    value = getattr(torch.cuda, name, None)
    if callable(value):
        setattr(torch.cuda, name, _reject_cuda_action(f"torch.cuda.{name}"))
        guarded_cuda_api_names.append(f"torch.cuda.{name}")

def _is_cuda_target(value):
    if isinstance(value, str):
        return value.strip().lower().startswith("cuda")
    if isinstance(value, torch.device):
        return value.type == "cuda"
    target = getattr(value, "device", None)
    return isinstance(target, torch.device) and target.type == "cuda"

def _guard_factory(name, original):
    def guarded(*args, **kwargs):
        if any(_is_cuda_target(value) for value in (*args, *kwargs.values())):
            guarded_cuda_calls.append(name)
            raise RuntimeError(f"guarded CUDA tensor factory called: {name}")
        return original(*args, **kwargs)
    return guarded

for name in (
    "empty", "zeros", "ones", "full", "rand", "randn", "randint", "randperm",
    "arange", "linspace", "logspace", "eye", "tensor", "as_tensor", "empty_like",
    "zeros_like", "ones_like", "full_like", "rand_like", "randn_like",
):
    original = getattr(torch, name, None)
    if callable(original):
        setattr(torch, name, _guard_factory(f"torch.{name}[CUDA-device-target]", original))
        guarded_cuda_api_names.append(f"torch.{name}[CUDA-device-target]")

torch.Tensor.cuda = _reject_cuda_action("torch.Tensor.cuda")
torch.nn.Module.cuda = _reject_cuda_action("torch.nn.Module.cuda")
guarded_cuda_api_names.extend(("torch.Tensor.cuda", "torch.nn.Module.cuda"))

def _guard_transfer(name, original):
    def guarded(self, *args, **kwargs):
        if any(_is_cuda_target(value) for value in (*args, *kwargs.values())):
            guarded_cuda_calls.append(name)
            raise RuntimeError(f"guarded CUDA transfer called: {name}")
        return original(self, *args, **kwargs)
    return guarded

torch.Tensor.to = _guard_transfer("torch.Tensor.to[CUDA-device-target]", torch.Tensor.to)
torch.nn.Module.to = _guard_transfer("torch.nn.Module.to[CUDA-device-target]", torch.nn.Module.to)
guarded_cuda_api_names.extend(("torch.Tensor.to[CUDA-device-target]", "torch.nn.Module.to[CUDA-device-target]"))

if any(name == "wnn" or name.startswith("wnn.") for name in sys.modules):
    raise RuntimeError("source-68 WNN modules must not be imported in the narrow private proof")
private_module_names = {Path(name).stem for name in expected_names}
preloaded_private_module_names = sorted(
    name for name in private_module_names if name in sys.modules
)
if preloaded_private_module_names:
    raise RuntimeError(
        "private checker modules were preloaded before the import-boundary check: "
        f"{preloaded_private_module_names!r}"
    )
checker = importlib.import_module("child_allthree_terminal_check_v2")
checker_path = os.path.abspath(os.fspath(Path(checker.__file__)))
expected_checker_path = os.path.abspath(os.fspath(private_dir / "child_allthree_terminal_check_v2.py"))
if checker_path != expected_checker_path:
    raise RuntimeError("AllThree checker origin escaped the exact temporary private directory")

helper_closure = checker.assert_helper_closure()
module_by_file = {"child_allthree_terminal_check_v2.py": checker}
module_by_file.update({name: module for name, (_, _, module) in checker.HELPER_PINS.items()})
post_import = {}
module_origins = {}
for name, module in module_by_file.items():
    expected_path = os.path.abspath(os.fspath(private_dir / name))
    actual_path = os.path.abspath(os.fspath(Path(getattr(module, "__file__", ""))))
    if actual_path != expected_path:
        raise RuntimeError(
            f"private code origin mismatch after import: {name}: "
            f"actual={actual_path!r}, expected={expected_path!r}"
        )
    post_import[name] = _file_pin(private_dir / name)
    module_origins[name] = actual_path
if post_import != expected:
    raise RuntimeError("one or more private code files changed across import")
if {path.name for path in private_dir.iterdir()} != expected_names:
    raise RuntimeError("temporary private directory changed from the exact seven-code-file set")
if (private_dir / "__pycache__").exists() or any(
    path.suffix == ".pyc" for path in private_dir.rglob("*.pyc")
):
    raise RuntimeError("private import created or exposed cached bytecode")
if sys.dont_write_bytecode is not True or sys.pycache_prefix is not None:
    raise RuntimeError("private import changed the child bytecode controls")
if any(name == "wnn" or name.startswith("wnn.") for name in sys.modules):
    raise RuntimeError("unexpected source-68 WNN module import")
if open_attempts or any(open_categories.values()):
    raise RuntimeError("a protected worker/data/model payload open was attempted")
if guarded_cuda_calls:
    raise RuntimeError("a guarded CUDA initialization/allocation/device/transfer API was attempted")
if torch.cuda.is_initialized():
    raise RuntimeError("CUDA became initialized during the private-code import")

print(json.dumps({
    "status": "PASS_NARROW_ALLTHREE_PRIVATE_IMPORT_BOUNDARY_V2",
    "observed_sys_executable": sys.executable,
    "python_flags": ["-E", "-B", "-I"],
    "dont_write_bytecode": sys.dont_write_bytecode,
    "pycache_prefix": sys.pycache_prefix,
    "private_bytecode_absent_before_import": True,
    "private_bytecode_absent_after_import": True,
    "temporary_private_file_names": sorted(expected_names),
    "pre_import_file_pins": pre_import,
    "post_import_file_pins": post_import,
    "module_origins": module_origins,
    "preloaded_private_module_names": preloaded_private_module_names,
    "helper_closure_recheck": helper_closure,
    "checker_functions_called": ["assert_helper_closure"],
    "verify_terminal_called": False,
    "main_or_cli_called": False,
    "fit_or_evaluation_called": False,
    "wnn_source_modules_loaded": 0,
    "local_torch_version": torch.__version__,
    "local_torch_proves_worker_runtime": False,
    "worker_torch_runtime_remains_fresh_gate": "2.14.0+cu130",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cuda_initialized_before_and_after": False,
    "cuda_initialized_observer_calls": cuda_initialized_observations,
    "cuda_initialized_observer_is_separate_from_action_calls": True,
    "guarded_cuda_action_api_names": sorted(guarded_cuda_api_names),
    "guarded_cuda_action_api_name_count": len(set(guarded_cuda_api_names)),
    "guarded_cuda_action_api_attempts": len(guarded_cuda_calls),
    "protected_open_audit_scope": "Python open audit events for remote worker/B roots, known local B/data/research_tmp/splits roots, and model/state/dictionary/cache/data payload suffixes/names; focused boundary only, not an exhaustive OS syscall monitor.",
    "protected_open_attempts": len(open_attempts),
    "protected_open_attempts_by_category": open_categories,
    "private_code_open_count": private_code_open_count,
    "source_cli_run_identity_representation_or_model_imports_exercised": False,
}, sort_keys=True))
"""


def _pin(path: Path, expected: tuple[int, str]) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a regular nonsymlink file: {path}")
    raw = path.read_bytes()
    actual = (len(raw), hashlib.sha256(raw).hexdigest())
    if actual != expected:
        raise RuntimeError(f"pinned source changed: {path.name}: {actual!r}")
    return {"bytes": actual[0], "sha256": actual[1]}


def main() -> int:
    if sys.executable != str(PYTHON):
        raise RuntimeError("run with literal repository maintenance/.venv/bin/python")
    if PROOF.exists():
        raise RuntimeError("refusing to overwrite the private import-boundary proof receipt")
    source_pins = {name: _pin(path, PINS[name]) for name, path in SOURCE_FILES.items()}
    with tempfile.TemporaryDirectory(prefix="fg19_allthree_import_boundary_") as temp_name:
        private_dir = Path(temp_name) / "private"
        private_dir.mkdir()
        for name, source in SOURCE_FILES.items():
            shutil.copyfile(source, private_dir / name)
        copied_pins = {name: _pin(private_dir / name, PINS[name]) for name in PINS}
        if copied_pins != source_pins:
            raise RuntimeError("temporary seven-file closure differs from pre-copy source pins")

        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "FG19_EXPECTED_LITERAL_INTERPRETER": str(PYTHON),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        completed = subprocess.run(
            [str(PYTHON), "-E", "-B", "-I", "-c", CHILD_PROBE, str(private_dir)],
            capture_output=True,
            check=False,
            env=env,
            input=json.dumps(PINS),
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "focused child probe failed:\n"
                f"stdout={completed.stdout[-4_000:]}\nstderr={completed.stderr[-4_000:]}"
            )
        child = json.loads(completed.stdout)
        if child.get("status") != "PASS_NARROW_ALLTHREE_PRIVATE_IMPORT_BOUNDARY_V2":
            raise RuntimeError("child probe did not report a pass")
        proof = {
            "schema": "fg19.child-allthree-terminal-import-boundary-proof.v2",
            "status": child["status"],
            "captured_utc": datetime.now(UTC).isoformat(),
            "harness": {
                "path": os.path.abspath(os.fspath(Path(__file__))),
                "bytes": Path(__file__).stat().st_size,
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            "interpreter_binding": {
                "literal_requested_interpreter": str(PYTHON),
                "observed_sys_executable": sys.executable,
                "parent_sys_executable_matches_literal": sys.executable == str(PYTHON),
                "child_observed_sys_executable": child["observed_sys_executable"],
                "child_sys_executable_matches_literal": (
                    child["observed_sys_executable"] == str(PYTHON)
                ),
                "interpreter_path_resolution_used": False,
                "child_python_flags": child["python_flags"],
                "child_dont_write_bytecode": child["dont_write_bytecode"],
                "child_pycache_prefix": child["pycache_prefix"],
                "private_bytecode_absent_before_import": child[
                    "private_bytecode_absent_before_import"
                ],
                "private_bytecode_absent_after_import": child[
                    "private_bytecode_absent_after_import"
                ],
            },
            "private_closure": {
                "exact_file_count": 7,
                "pre_copy_source_pins": source_pins,
                "temporary_copy_pins_before_child_import": copied_pins,
                "child_pre_import_pins": child["pre_import_file_pins"],
                "child_post_import_pins": child["post_import_file_pins"],
                "child_module_origins": child["module_origins"],
                "child_preloaded_module_names": child["preloaded_private_module_names"],
                "helper_closure_recheck": child["helper_closure_recheck"],
                "exact_file_set_before_and_after_import": child["temporary_private_file_names"],
            },
            "entrypoint_scope": {
                "checker_functions_called": child["checker_functions_called"],
                "verify_terminal_called": child["verify_terminal_called"],
                "main_or_cli_called": child["main_or_cli_called"],
                "fit_or_evaluation_called": child["fit_or_evaluation_called"],
                "source_cli_run_identity_representation_or_model_imports_exercised": child[
                    "source_cli_run_identity_representation_or_model_imports_exercised"
                ],
                "wnn_source_modules_loaded": child["wnn_source_modules_loaded"],
            },
            "protected_open_guard": {
                "scope": child["protected_open_audit_scope"],
                "attempts": child["protected_open_attempts"],
                "attempts_by_category": child["protected_open_attempts_by_category"],
                "private_code_open_count": child["private_code_open_count"],
                "model_or_state_dictionary_cache_data_payload_read_or_hashed": False,
            },
            "cuda_guard": {
                "cuda_visible_devices": child["cuda_visible_devices"],
                "initialized_before_and_after": child["cuda_initialized_before_and_after"],
                "is_initialized_observer_calls": child["cuda_initialized_observer_calls"],
                "observer_is_separate_from_action_counter": child[
                    "cuda_initialized_observer_is_separate_from_action_calls"
                ],
                "guarded_api_names": child["guarded_cuda_action_api_names"],
                "guarded_api_name_count": child["guarded_cuda_action_api_name_count"],
                "guarded_action_attempts": child["guarded_cuda_action_api_attempts"],
            },
            "runtime_scope": {
                "local_torch_version": child["local_torch_version"],
                "local_torch_proves_worker_runtime": child["local_torch_proves_worker_runtime"],
                "worker_torch_runtime_remains_fresh_gate": child[
                    "worker_torch_runtime_remains_fresh_gate"
                ],
                "limits": "This proof covers import of only the exact seven-file private checker closure. It does not replay source-68 CLI/run_identity/representation imports or prove worker runtime, model/state acceptance, payload I/O behavior during terminal verification, or exhaustive OS-level I/O/CUDA controls.",
            },
        }
    raw = (json.dumps(proof, sort_keys=True, indent=2) + "\n").encode()
    with PROOF.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "proof_path": str(PROOF),
                "proof_bytes": len(raw),
                "proof_sha256": hashlib.sha256(raw).hexdigest(),
                "status": proof["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

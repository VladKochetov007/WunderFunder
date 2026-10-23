"""Cold-import proof for the dictionary checker's exact private/source closure.

This probe imports only pinned checker/helper code and selected source-68 code.
It never calls a checker entry point or opens a fit, model, dictionary, cache,
split, or dataset payload.
"""

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

REPO = Path(__file__).resolve().parents[5]
CAMPAIGN = REPO / "research/campaign_2026-10-19/full_gru_tweaks_r1"
PACKET = CAMPAIGN / "preparation/child_dictionary_terminal_check_v1"
SOURCE_ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-strong-gru-tweaks-20261019")
SOURCE_MANIFEST = CAMPAIGN / "preparation/teacher_cache_cpu_repair_v2/SOURCE_MANIFEST.json"
LAUNCH = CAMPAIGN / "records/child_dictionary_fit_v1/LAUNCH.json"
PROOF = PACKET / "COLD_IMPORT_PROOF.json"

PINS = {
    "child_dictionary_terminal_check_v1.py": (
        34_164,
        "d06818648362d571e3b06637bc026b97190476a6b59474f4a57ec84c349e4d82",
    ),
    "child_ema_terminal_check_v3.py": (
        38_267,
        "676d548c1cd25c3c1379548999cca1300ef135dd41b4a8f73689ef8e59de3b5d",
    ),
    "child_control_terminal_check_v3.py": (
        79_972,
        "618b9b0fe7c0cf85e94848b24a837a63e27df46f81e35f2f8536c049072e5167",
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

CHILD_PROBE = r"""
import copy
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

private_dir = Path(sys.argv[1]).resolve(strict=True)
source_root = Path(sys.argv[2]).resolve(strict=True)
manifest_path = Path(sys.argv[3]).resolve(strict=True)
pycache_root = Path(sys.argv[4]).resolve(strict=False)
launch = json.loads(sys.stdin.read())
source_dir = (source_root / "src").resolve(strict=True)
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CUDA must be hidden for the local cold proof")
if any(name == "wnn" or name.startswith("wnn.") for name in sys.modules):
    raise RuntimeError("WNN was already imported before the cold probe")

# Direct all source imports to fresh bytecode cache space so source paths below
# are the files Python parses in this process, not stale local pyc files.
sys.pycache_prefix = str(pycache_root)
sys.path.insert(0, str(private_dir))

blocked_attempts = []
guarded_cuda_calls = []
cuda_initialized_observations = 0
opened = {"private_code": 0, "source_code": 0, "source_manifest": 0}
allowed_private = {
    str((private_dir / name).resolve(strict=True))
    for name in (
        "child_dictionary_terminal_check_v1.py",
        "child_ema_terminal_check_v3.py",
        "child_control_terminal_check_v3.py",
        "parent_fit_terminal_check_v3.py",
        "propensity_terminal_support.py",
        "teacher_s5_terminal_check_v1.py",
    )
}
manifest_name = str(manifest_path)
source_path = str(source_dir)
remote_prefixes = ("/home/", "/mnt/", "/workspace/")
payload_names = {
    "model.pt",
    "training_state.pt",
    "input_representation.pt",
    "dictionary.pt",
    "cache.pt",
    "model_ep0.pt",
    "model_ep1.pt",
    "model_ep2.pt",
    "model_ep3.pt",
    "model_ep4.pt",
    "model_ep5.pt",
}
payload_suffixes = {".pt", ".pth", ".npy", ".npz", ".safetensors", ".parquet"}


def _audit(event, args):
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
    source_code = normalized.startswith(source_path + os.sep) and lower.endswith(".py")
    private_code = normalized in allowed_private
    manifest_read = normalized == manifest_name
    data_path = any(
        lower.startswith(prefix)
        for prefix in (
            "/users/nb/documents/wunder/wunderfunder-strong-gru-tweaks-20261019/data/",
            "/users/nb/documents/wunder/wunderfunder-maintenance/data/",
            "/users/nb/documents/wunder/wunderfunder-maintenance/research_tmp/",
            "/users/nb/documents/wunder/wunderfunder-maintenance/splits/",
        )
    ) or any(
        marker in lower
        for marker in ("/artifacts/runs/", "/research_tmp/", "/splits/")
    )
    payload = (
        base in payload_names
        or Path(lower).suffix in payload_suffixes
        or data_path
        or any(lower.startswith(prefix) for prefix in remote_prefixes)
    )
    if payload and not manifest_read and not source_code:
        blocked_attempts.append(normalized)
        raise PermissionError("cold proof denied a B/data/model/cache payload open")
    if private_code:
        opened["private_code"] += 1
    elif source_code:
        opened["source_code"] += 1
    elif manifest_read:
        opened["source_manifest"] += 1


sys.addaudithook(_audit)
manifest_raw = manifest_path.read_bytes()
manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
if manifest_sha != "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621":
    raise RuntimeError("source manifest bytes differ from the accepted 167-file authority")
manifest = json.loads(manifest_raw)
if (
    manifest.get("source_commit") != "68ff682977b822f38351aac0f621146f5f13b440"
    or len(manifest.get("files", [])) != 167
):
    raise RuntimeError("source manifest identity differs from source-68 authority")
rows = {row["path"]: row for row in manifest["files"]}

# Import the local Torch runtime before the private helper closure, then guard
# CUDA APIs. `is_initialized()` is an observer and is counted separately.
import torch

original_is_initialized = torch.cuda.is_initialized


def _observe_cuda_initialized():
    global cuda_initialized_observations
    cuda_initialized_observations += 1
    return original_is_initialized()


torch.cuda.is_initialized = _observe_cuda_initialized
if torch.cuda.is_initialized():
    raise RuntimeError("CUDA was initialized before the cold-import guards")


def _reject_cuda_call(name):
    def reject(*args, **kwargs):
        guarded_cuda_calls.append(name)
        raise RuntimeError(f"guarded CUDA API was called: {name}")

    return reject


guarded_cuda_apis = []
for name in (
    "init",
    "_lazy_init",
    "set_device",
    "current_device",
    "device_count",
    "is_available",
    "synchronize",
    "empty_cache",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "reset_peak_memory_stats",
    "reset_accumulated_memory_stats",
    "get_device_properties",
    "get_device_name",
    "get_device_capability",
    "mem_get_info",
    "ipc_collect",
    "list_gpu_processes",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "current_stream",
    "default_stream",
    "stream",
    "device",
    "graph",
    "CUDAGraph",
    "Stream",
    "Event",
    "_exchange_device",
    "_set_device",
    "_get_device",
    "_get_device_index",
    "_get_device_properties",
):
    value = getattr(torch.cuda, name, None)
    if callable(value):
        setattr(torch.cuda, name, _reject_cuda_call(f"torch.cuda.{name}"))
        guarded_cuda_apis.append(f"torch.cuda.{name}")


def _is_cuda_target(value):
    if isinstance(value, str):
        return value.strip().lower().startswith("cuda")
    if isinstance(value, torch.device):
        return value.type == "cuda"
    target_device = getattr(value, "device", None)
    return isinstance(target_device, torch.device) and target_device.type == "cuda"


def _guard_cuda_target(name, original):
    def guarded(*args, **kwargs):
        values = [kwargs.get("device"), *args]
        if any(_is_cuda_target(value) for value in values):
            guarded_cuda_calls.append(name)
            raise RuntimeError(f"guarded CUDA allocation/device path was called: {name}")
        return original(*args, **kwargs)

    return guarded


for name in (
    "empty",
    "zeros",
    "ones",
    "full",
    "rand",
    "randn",
    "randint",
    "randperm",
    "arange",
    "linspace",
    "logspace",
    "eye",
    "tensor",
    "as_tensor",
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "rand_like",
    "randn_like",
):
    original = getattr(torch, name, None)
    if callable(original):
        setattr(torch, name, _guard_cuda_target(f"torch.{name}", original))
        guarded_cuda_apis.append(f"torch.{name}[CUDA-device-target]")


def _guard_explicit_cuda_method(name):
    def reject(*args, **kwargs):
        guarded_cuda_calls.append(name)
        raise RuntimeError(f"guarded CUDA API was called: {name}")

    return reject


torch.Tensor.cuda = _guard_explicit_cuda_method("torch.Tensor.cuda")
torch.nn.Module.cuda = _guard_explicit_cuda_method("torch.nn.Module.cuda")


def _guard_to(name, original):
    def guarded(self, *args, **kwargs):
        if any(_is_cuda_target(value) for value in [kwargs.get("device"), *args]):
            guarded_cuda_calls.append(name)
            raise RuntimeError(f"guarded CUDA transfer path was called: {name}")
        return original(self, *args, **kwargs)

    return guarded


torch.Tensor.to = _guard_to("torch.Tensor.to", torch.Tensor.to)
torch.nn.Module.to = _guard_to("torch.nn.Module.to", torch.nn.Module.to)

import child_dictionary_terminal_check_v1 as checker

helper_closure = checker._check_helper_closure()
checker_path = Path(checker.__file__).resolve(strict=True)
if checker_path != private_dir / "child_dictionary_terminal_check_v1.py":
    raise RuntimeError("checker did not load from the exact private directory")

# Bind and parse the exact retained trainer argv. Only its absolute entrypoint
# is rewritten to this local source checkout so the pure CLI resolver can run.
argv_sha = checker.ema_v3._trainer_argv_sha(launch)
if argv_sha != checker.TRAINER_ARGV_SHA:
    raise RuntimeError("retained launch argv digest differs from checker pin")
local_launch = copy.deepcopy(launch)
command = local_launch.get("command")
if not isinstance(command, list):
    raise RuntimeError("retained launch command is malformed")
trainer_positions = [
    index for index, item in enumerate(command) if item.endswith("/scripts/train_rnn.py")
]
if len(trainer_positions) != 1:
    raise RuntimeError("retained launch must identify one trainer entrypoint")
trainer_index = trainer_positions[0]
command[trainer_index] = str(source_root / "scripts/train_rnn.py")
launch_args = checker._launch_cli_args(local_launch, source_root)
launch_args.update(
    {
        "valid_fold": 0,
        "n_valid_folds": 4,
        "eval_source": "vfold",
        "split_manifest_sha256": checker.SPLIT_SHA,
    }
)
split_path = Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json"
)
cache_root = Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_t3_fit_homogeneous_cache_v1"
)
checker._check_recipe_args(launch_args, split_path, cache_root)

# Exercise the exact lazy run-identity import with a new two-element CPU tensor.
tensor_identity = checker._canonical_tensor_identity(source_root)
import torch

if torch.__version__ != "2.14.1":
    raise RuntimeError(f"local proof expected torch 2.14.1; observed {torch.__version__}")
identity = tensor_identity(torch.tensor([0.125, -0.25], dtype=torch.float32, device="cpu"))

representation_modules = [
    importlib.import_module("wnn.training.input_representation"),
    importlib.import_module("wnn.training.input_representation_fit"),
    importlib.import_module("wnn.training.input_representation_dictionary"),
]
required_names = {
    "wnn",
    "wnn.run_identity",
    "wnn.training.cli",
    "wnn.training.input_representation",
    "wnn.training.input_representation_fit",
    "wnn.training.input_representation_dictionary",
}
if not required_names.issubset(sys.modules):
    raise RuntimeError("the requested canonical source modules were not all imported")

loaded_wnn = {}
for name, module in sorted(sys.modules.items()):
    if name != "wnn" and not name.startswith("wnn."):
        continue
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str):
        raise RuntimeError(f"canonical WNN module has no file origin: {name}")
    origin_path = Path(origin).resolve(strict=True)
    if not origin_path.is_relative_to(source_dir) or origin_path.suffix != ".py":
        raise RuntimeError(f"WNN module came from outside canonical source68: {name}: {origin}")
    rel = origin_path.relative_to(source_root).as_posix()
    row = rows.get(rel)
    if row is None:
        raise RuntimeError(f"loaded WNN path is absent from the accepted 167-file manifest: {rel}")
    raw = origin_path.read_bytes()
    actual = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    expected = {"bytes": row.get("bytes"), "sha256": row.get("sha256")}
    if actual != expected:
        raise RuntimeError(f"loaded WNN source differs from 167-file manifest: {rel}")
    loaded_wnn[name] = {
        "path": rel,
        "bytes": actual["bytes"],
        "sha256": actual["sha256"],
    }

private_origins = {
    "child_dictionary_terminal_check_v1": checker,
    **{name.removesuffix(".py"): module for name, (_, _, module) in checker.HELPER_PINS.items()},
}
for name, module in private_origins.items():
    expected = (private_dir / f"{name}.py").resolve(strict=True)
    actual = Path(getattr(module, "__file__", "")).resolve(strict=True)
    if actual != expected:
        raise RuntimeError(f"private helper origin escaped exact six-file closure: {name}")

if torch.cuda.is_initialized():
    raise RuntimeError("CUDA initialized during a CPU-only cold proof")
if blocked_attempts:
    raise RuntimeError("a protected payload open was attempted")
if guarded_cuda_calls:
    raise RuntimeError("a guarded CUDA initialization/allocation/device API was attempted")
if opened["source_manifest"] != 1:
    raise RuntimeError("the exact source manifest was not read once through the whitelist")
if opened["source_code"] == 0 or opened["private_code"] == 0:
    raise RuntimeError("the cold probe did not read both source and private code")

print(
    json.dumps(
        {
            "checker_main_invocations": 0,
            "trainer_main_invocations": 0,
            "fit_or_eval_invocations": 0,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_initialized": torch.cuda.is_initialized(),
            "cuda_guarded_api_names": sorted(guarded_cuda_apis),
            "cuda_guarded_call_attempts": len(guarded_cuda_calls),
            "cuda_initialized_observer_call_count": cuda_initialized_observations,
            "cuda_initialized_observer_is_not_a_guarded_cuda_operation": True,
            "local_torch_version": torch.__version__,
            "local_torch_is_worker_runtime_proof": False,
            "worker_torch_runtime_gate": "2.14.0+cu130 remains an actual-worker admission requirement",
            "private_closure_files": {
                name: {"bytes": pin[0], "sha256": pin[1]}
                for name, pin in checker.HELPER_PINS.items()
            },
            "checker_private_origin": str(checker_path),
            "helper_closure": helper_closure,
            "source_manifest": {
                "path": str(manifest_path),
                "bytes": len(manifest_raw),
                "sha256": manifest_sha,
                "source_commit": manifest["source_commit"],
                "file_count": len(manifest["files"]),
            },
            "trainer_cli": {
                "actual_retained_argv_sha256": argv_sha,
                "representation_mode": launch_args["representation_mode"],
                "run_name": launch_args["name"],
                "parsed_by": str(Path(sys.modules["wnn.training.cli"].__file__).resolve()),
                "local_entrypoint_substitution_only": True,
                "all_trainer_arguments_unchanged": True,
            },
            "run_identity": {
                "module": str(Path(sys.modules["wnn.run_identity"].__file__).resolve()),
                "cpu_synthetic_identity": identity,
            },
            "representation_modules": {
                module.__name__: str(Path(module.__file__).resolve())
                for module in representation_modules
            },
            "loaded_canonical_wnn_modules": loaded_wnn,
            "loaded_canonical_wnn_module_count": len(loaded_wnn),
            "protected_payload_open_attempts": 0,
            "protected_payload_open_categories": {
                "B_root": 0,
                "model_or_training_state": 0,
                "dictionary": 0,
                "cache": 0,
                "split_or_dataset": 0,
            },
            "allowed_open_counts": opened,
            "private_directory_file_names": sorted(
                path.name for path in private_dir.iterdir() if path.is_file()
            ),
            "source_arrays_or_payloads_read_or_hashed": False,
            "checker_entrypoint_called": False,
            "trainer_entrypoint_called": False,
            "status": "PASS_FOCUSED_COLD_PRIVATE_AND_SOURCE_IMPORT_PROOF",
        },
        sort_keys=True,
    )
)
"""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pin_file(path: Path, expected: tuple[int, str]) -> dict[str, object]:
    if path.is_symlink():
        raise RuntimeError(f"pinned source code file is a symlink: {path}")
    raw = path.read_bytes()
    actual = (len(raw), _sha256(raw))
    if actual != expected:
        raise RuntimeError(f"pinned source code changed: {path.name}: {actual!r}")
    return {"bytes": actual[0], "sha256": actual[1]}


def main() -> int:
    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"canonical source root is unavailable: {SOURCE_ROOT}")
    manifest_raw = SOURCE_MANIFEST.read_bytes()
    manifest_sha = _sha256(manifest_raw)
    manifest = json.loads(manifest_raw)
    if (
        len(manifest_raw) != 27_569
        or manifest_sha != "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
        or manifest.get("source_commit") != "68ff682977b822f38351aac0f621146f5f13b440"
        or len(manifest.get("files", [])) != 167
    ):
        raise SystemExit("the canonical 167-file source manifest differs from its frozen pin")
    launch_raw = LAUNCH.read_bytes()
    launch_sha = _sha256(launch_raw)
    if (
        len(launch_raw) != 8_072
        or launch_sha != "9cbcc8b4df3886620429cae42d678d2e281aff980402f5b19d63d9f515b6ad3c"
    ):
        raise SystemExit("the retained dictionary launch metadata differs from its frozen pin")
    launch = json.loads(launch_raw)
    if launch.get("schema") != "fg19-child-dictionary-fit-launch.v1":
        raise SystemExit("unexpected retained dictionary launch metadata schema")

    source_files: dict[str, Path] = {
        "child_dictionary_terminal_check_v1.py": PACKET / "child_dictionary_terminal_check_v1.py",
        "child_ema_terminal_check_v3.py": (
            CAMPAIGN / "preparation/child_ema_terminal_verifier_v3/child_ema_terminal_check_v3.py"
        ),
        "child_control_terminal_check_v3.py": (
            CAMPAIGN
            / "preparation/child_control_terminal_verifier_v3/child_control_terminal_check_v3.py"
        ),
        "parent_fit_terminal_check_v3.py": CAMPAIGN / "preparation/parent_fit_terminal_check_v3.py",
        "propensity_terminal_support.py": CAMPAIGN / "preparation/propensity_terminal_support.py",
        "teacher_s5_terminal_check_v1.py": CAMPAIGN / "preparation/teacher_s5_terminal_check_v1.py",
    }
    prehashes = {name: _pin_file(path, PINS[name]) for name, path in source_files.items()}

    with tempfile.TemporaryDirectory(prefix="fg19-dictionary-cold-proof-") as temp_name:
        temp = Path(temp_name)
        private_dir = temp / "private"
        private_dir.mkdir()
        for name, source in source_files.items():
            shutil.copyfile(source, private_dir / name)
        copied = {name: _pin_file(private_dir / name, PINS[name]) for name in source_files}
        if copied != prehashes:
            raise SystemExit("private six-file copies differ from their pre-import hashes")

        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        command = [
            sys.executable,
            "-E",
            "-B",
            "-c",
            CHILD_PROBE,
            str(private_dir),
            str(SOURCE_ROOT),
            str(SOURCE_MANIFEST),
            str(temp / "fresh-pycache"),
        ]
        completed = subprocess.run(
            command,
            input=launch_raw.decode("utf-8"),
            capture_output=True,
            check=False,
            env=env,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"cold subprocess failed:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)
        if result.get("status") != "PASS_FOCUSED_COLD_PRIVATE_AND_SOURCE_IMPORT_PROOF":
            raise SystemExit("cold subprocess did not report its exact PASS status")
        if result.get("private_directory_file_names") != sorted(PINS):
            raise SystemExit("cold private directory did not contain exactly six code files")
        if result.get("local_torch_version") != "2.14.1":
            raise SystemExit(f"unexpected local torch version: {result.get('local_torch_version')}")

    proof = {
        "schema": "fg19.child-dictionary-terminal-cold-import-proof.v1",
        "status": result["status"],
        "recorded_utc": datetime.now(UTC).isoformat(),
        "scope": {
            "fresh_subprocess": True,
            "temporary_private_directory": True,
            "private_code_file_count": 6,
            "all_six_files_prehashed_before_import": True,
            "source68_root": str(SOURCE_ROOT),
            "source_manifest": {
                "path": str(SOURCE_MANIFEST),
                "bytes": len(manifest_raw),
                "sha256": manifest_sha,
                "file_count": 167,
            },
            "launch_metadata": {
                "path": str(LAUNCH),
                "bytes": len(launch_raw),
                "sha256": launch_sha,
            },
            "source_arrays_or_payloads_read_or_hashed": False,
            "checker_or_trainer_entrypoint_called": False,
            "dictionary_or_model_fit_or_evaluation_run": False,
        },
        "private_code_preimport_pins": prehashes,
        "cold_subprocess_result": result,
        "runtime_limit": (
            "Local Torch 2.14.1 import proof only; worker Torch 2.14.0+cu130 remains "
            "a fresh actual-worker admission gate."
        ),
    }
    encoded = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode("utf-8")
    PROOF.write_bytes(encoded)
    print(
        json.dumps(
            {
                "path": str(PROOF),
                "bytes": len(encoded),
                "sha256": _sha256(encoded),
                "status": proof["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

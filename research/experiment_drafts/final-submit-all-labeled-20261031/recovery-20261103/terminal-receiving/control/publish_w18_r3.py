"""Publish W18 after strict cache-teacher closure and run-reference joins.

The original final-fit runtime remains read-only except for its canonical W18
checkpoint and child receipt. The recovery source is used only by the W18 child,
package builder and evidence collection; the worker never waits for TLead/LSTM.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import json
import math
import os
import runpy
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

RUNTIME_DEFAULT = Path(
    "/data/epanov_s/tabdata/runtime/final-submit-all-labeled-20261031-r1"
)
RECOVERY_DEFAULT = Path(
    "/data/epanov_s/tabdata/runtime/w18-submit-recovery-20261103-r1"
)
DEADLINE_DEFAULT = 1_793_673_140.7359152
SOURCE_VALIDATION_SHA256 = "1e8a8a794a0b8d9677a76030e570681074503d38aa2d54b80e36067566b94ce0"
PLAN_SHA256 = "29802499ab6ba62586b612cd10164d27c9737a96cf8de41c09fef3f6d0402cf4"
QUANTIZATION_SHA256 = "ec1ce8811ceb43249aca3c1c79295c4124a6a80f6cc8320aeaadb71075ff9a3f"
PACKAGE_SCRIPT_SHA256 = "10b8a990d2c1f2236ec5c7e8772553c709e0864baf1785b722598118afdf5c73"
SOLUTION_SHA256 = "68ec7a3901ef34e36ab87c3db13b1da657e96c63363d0d900c5b82095c721024"
CHECKER_SHA256 = "ec2bd7025a4103f21b8f6339ee22b26daa5ac1056d9e41175a93d678e4d9077d"
R11_PACKAGE_SHA256 = "4ff9d3975285e3c41fa8fa82ceeffa724258b6bbc069c2f540a4f65f7dfda101"
R5_CHECKER_SHA256 = "ff5ea18f968fae6f122f77750e7d80d00fec88de966eab78444e41c2a70e5018"
R5_SYNTHETIC_SHA256 = "f58e569fdaa962ed1d5a821215f6ec69d2524d8cac3fecbac6f1855636014f9e"
GITHUB_HELPER_SHA256 = "05fe6aa62a0a0d5b700f29f8f4fcacf3bf28278e3dc1fb6e89e653b991e45712"
TOKEN_HELPER_SHA256 = "d26280db335907578c2583709189913d7b3059be730f2f531ff50aa7ccdcdcb8"
TOKEN_COMMON_SHA256 = "edf9325b085361e18168c3fb2d0390a57cd3d2d28f3dfb190b4ac1f5dc28cf9d"
EXPECTED_W18_ROLES = {
    "w18final_distilled_s5",
    "fbc1k_s5",
    "ffull_s3",
    "ffull_s4",
    "frawW_s3ft",
    "frawW_s4ft",
    "orgbase",
    "fin256_s3",
    "fin256_s4",
    "fmu128_s3",
    "fmu128_s4",
    "frawW_s3",
    "frawW_s4",
}
FIT_TIMEOUT_CAP = 21_600
PACKAGE_TIMEOUT_SECONDS = 900
CHECK_TIMEOUT_SECONDS = 1_200
DELIVERY_RESERVE_SECONDS = 1_800
MAX_EVIDENCE_ZIP_BYTES = 1_900_000_000
POLL_MIN_SECONDS = 5
POLL_MAX_SECONDS = 60


class DeliveryError(RuntimeError):
    """Fail-closed operational or evidence validation error."""


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise DeliveryError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path, *, max_bytes: int = 8_000_000) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
        raise DeliveryError(f"missing, unsafe, or oversized JSON evidence: {path}")
    size = path.stat().st_size
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(max_bytes + 1)
        after = os.fstat(stream.fileno())
    if (
        len(raw) != size
        or len(raw) > max_bytes
        or (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
    ):
        raise DeliveryError(f"JSON evidence changed while being read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise DeliveryError(f"JSON evidence must be an object: {path}")
    return value


def _write_json_create(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _record(state_path: Path, phase: str, **details: object) -> None:
    if state_path.is_symlink():
        raise DeliveryError("publisher state path must not be a symbolic link")
    value = {
        "schema": "wnn.final-w18-early-publication-state.v1",
        "phase": phase,
        "updated_epoch": time.time(),
        "score_computed": False,
        "independent_score_claim": "forbidden",
        **details,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, state_path)


def _runtime_paths(runtime: Path, recovery: Path) -> dict[str, Path]:
    return {
        "source": recovery / "source",
        "original_source": runtime / "source",
        "recovery": recovery,
        "root_authority": recovery / "ROOT_AUTHORITY_R1.json",
        "source_binding": recovery / "inputs" / "SOURCE_BINDING_R1.json",
        "source_capsule": recovery / "inputs" / "w18-source-recovery-r1.tar.gz",
        "control": runtime / "control" / "final-submit-chain",
        "control_code": runtime / "control-code-r2" / "scripts",
        "serving": recovery / "serving-w18-package-r5",
        "r11": runtime / "serving-r11",
        "r5": runtime / "control-package-check-r5",
        "native_r12": runtime / "native-libs-r12",
        "native_r11": runtime / "native-libs-r11" / "v3" / "wnn_native.abi3.so",
        "membership": runtime / "inputs" / "final-fit-membership-r1.json",
        "source_audit": runtime / "inputs" / "SOURCE_VALIDATION_R1.json",
        "plan": runtime / "inputs" / "final-plan-r2.json",
        "quantization": runtime / "inputs" / "w18-final-quantization-admission-r1.json",
        "train": runtime / "data" / "datasets" / "train.parquet",
        "data": runtime / "data",
        "references": runtime / "reference-checkpoints",
        "reports": recovery / "reports-publish-r3",
        "original_reports": runtime / "reports",
        "token": recovery / "private" / "publish-r3.token",
        "python312": Path("/data/epanov_s/tabdata/.venv/bin/python"),
        "python311": runtime / "tools" / "build-env" / "bin" / "python",
    }


def _validate_recovery_authority(paths: dict[str, Path], deadline_epoch: float) -> dict[str, Any]:
    authority = _read_json(paths["root_authority"], max_bytes=64_000)
    parent_runtime = authority.get("parent_runtime")
    if not isinstance(parent_runtime, str) or Path(parent_runtime).resolve() != paths["original_source"].parent.resolve():
        raise DeliveryError("W18 recovery authority names a different parent runtime")
    if (
        authority.get("schema") != "wnn.w18-submit-recovery-root-authority.v1"
        or authority.get("hard_deadline_epoch") != DEADLINE_DEFAULT
        or authority.get("max_scientific_fits") != 1
        or authority.get("max_profile_attempts") != 2
        or authority.get("fit_epochs") != 2
        or authority.get("fixed_endpoint") != 1
        or not math.isfinite(deadline_epoch)
        or deadline_epoch <= 0
        or deadline_epoch > authority["hard_deadline_epoch"]
    ):
        raise DeliveryError("W18 recovery authority does not bind this parent runtime and deadline")
    for field in ("max_scientific_fits", "max_profile_attempts", "fit_epochs", "fixed_endpoint"):
        if type(authority.get(field)) is not int:
            raise DeliveryError(f"W18 recovery authority has invalid {field}")
    return authority


def _validate_source_binding(paths: dict[str, Path]) -> dict[str, Any]:
    binding = _read_json(paths["source_binding"], max_bytes=128_000)
    overlay_relative = Path("src/wnn/training/final_submit_bridge_cache.py")
    overlay_path = paths["source"] / overlay_relative
    base_overlay_path = paths["original_source"] / overlay_relative
    source_capsule = paths["source_capsule"]
    cache_metadata_sha256 = binding.get("cache_metadata_sha256")
    cache_metadata_path = (
        paths["data"]
        / "artifacts"
        / "runs"
        / "final-fit-w18-mixed-targets"
        / "metadata.json"
    )
    changed = binding.get("changed_files")
    if (
        binding.get("schema") != "wnn.w18-recovery-source-binding.v1"
        or not _valid_sha(binding.get("base_capsule_sha256"))
        or not _valid_sha(binding.get("source_capsule_sha256"))
        or binding.get("source_capsule_sha256") != _sha256(source_capsule)
        or not _valid_sha(cache_metadata_sha256)
        or cache_metadata_sha256 != _sha256(cache_metadata_path)
        or not isinstance(changed, list)
        or len(changed) != 1
    ):
        raise DeliveryError("W18 recovery source binding is incomplete or differs from its capsule")
    entry = changed[0]
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "old_sha256", "new_sha256"}
        or entry.get("path") != overlay_relative.as_posix()
        or entry.get("old_sha256") != _sha256(base_overlay_path)
        or entry.get("new_sha256") != _sha256(overlay_path)
    ):
        raise DeliveryError("W18 recovery source is not the bound cache-validator-only overlay")
    capsule_value = binding.get("source_capsule")
    if not isinstance(capsule_value, str):
        raise DeliveryError("W18 recovery binding omits its source capsule path")
    capsule_path = Path(capsule_value)
    if not capsule_path.is_absolute():
        capsule_path = paths["recovery"] / capsule_path
    if capsule_path.resolve() != source_capsule.resolve():
        raise DeliveryError("W18 recovery binding points to a different source capsule")
    return {
        "binding_sha256": _sha256(paths["source_binding"]),
        "capsule_sha256": _sha256(source_capsule),
        "cache_validator_sha256": entry["new_sha256"],
        "cache_metadata_sha256": cache_metadata_sha256,
    }


def _setup_import_paths(paths: dict[str, Path]) -> None:
    sys.path[:0] = [
        str(paths["serving"]),
        str(paths["r5"]),
        str(paths["control_code"]),
        str(paths["r11"]),
        str(paths["source"]),
        str(paths["source"] / "src"),
        str(paths["source"] / "scripts"),
    ]


def _subprocess_env(paths: dict[str, Path]) -> dict[str, str]:
    pythonpath = [
        paths["serving"],
        paths["r5"],
        paths["control_code"],
        paths["r11"],
        paths["source"],
        paths["source"] / "src",
        paths["source"] / "scripts",
    ]
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(str(item) for item in pythonpath),
        }
    )
    return env


def _require_hash(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise DeliveryError(f"pinned serving input changed: {path}")


def _preflight(
    paths: dict[str, Path], env: dict[str, str], cache_validator_sha256: str
) -> dict[str, str]:
    serving = paths["serving"]
    r11 = paths["r11"]
    r5 = paths["r5"]
    _require_hash(serving / "scripts" / "package_final_w18.py", PACKAGE_SCRIPT_SHA256)
    _require_hash(serving / "submission" / "solution_final_w18.py", SOLUTION_SHA256)
    _require_hash(serving / "scripts" / "check_final_w18_package.py", CHECKER_SHA256)
    _require_hash(r11 / "scripts" / "package_final_tl3.py", R11_PACKAGE_SHA256)
    _require_hash(r5 / "scripts" / "check_final_tl3_package.py", R5_CHECKER_SHA256)
    synthetic_candidates = [
        root / "scripts" / "check_final_tl3_synthetic.py"
        for root in (r5, r11, paths["source"])
        if (root / "scripts" / "check_final_tl3_synthetic.py").is_file()
    ]
    synthetic_path = next(
        (path for path in synthetic_candidates if _sha256(path) == R5_SYNTHETIC_SHA256),
        None,
    )
    if synthetic_path is None:
        raise DeliveryError("pinned W18 old-reference helper is absent from R5/R11/source roots")
    for root in (serving, r11, r5, paths["source"]):
        if (root / "scripts" / "__init__.py").exists():
            raise DeliveryError(
                "serving helper roots must remain namespace packages so R3, R11, and R5 imports resolve"
            )
    if _sha256(paths["quantization"]) != QUANTIZATION_SHA256:
        raise DeliveryError("W18 quantization admission contract differs from its frozen SHA")
    _require_hash(paths["source"] / "scripts" / "publish_checkpoint_releases.py", GITHUB_HELPER_SHA256)
    _require_hash(paths["source"] / "scripts" / "galam_l4_quality8" / "publisher.py", TOKEN_HELPER_SHA256)
    _require_hash(paths["source"] / "scripts" / "galam_l4_quality8" / "common.py", TOKEN_COMMON_SHA256)

    imports = {
        "scripts.package_final_w18": (serving / "scripts" / "package_final_w18.py", PACKAGE_SCRIPT_SHA256),
        "scripts.check_final_w18_package": (serving / "scripts" / "check_final_w18_package.py", CHECKER_SHA256),
        "scripts.package_final_tl3": (r11 / "scripts" / "package_final_tl3.py", R11_PACKAGE_SHA256),
        "scripts.check_final_tl3_package": (r5 / "scripts" / "check_final_tl3_package.py", R5_CHECKER_SHA256),
        "scripts.check_final_tl3_synthetic": (synthetic_path, R5_SYNTHETIC_SHA256),
        "wnn.training.final_submit_bridge_cache": (
            paths["source"] / "src" / "wnn" / "training" / "final_submit_bridge_cache.py",
            cache_validator_sha256,
        ),
    }
    probe = "\n".join(
        (
            "import hashlib, importlib, json",
            "from pathlib import Path",
            f"expected = { {name: (str(path.resolve()), digest) for name, (path, digest) in imports.items()}!r}",
            "actual = {}",
            "for name, (expected_path, expected_sha) in expected.items():",
            "    module = importlib.import_module(name)",
            "    path = Path(module.__file__).resolve()",
            "    digest = hashlib.sha256(path.read_bytes()).hexdigest()",
            "    if str(path) != expected_path or digest != expected_sha:",
            "        raise SystemExit(f'import source mismatch: {name}: {path}')",
            "    actual[name] = digest",
            "print(json.dumps(actual, sort_keys=True))",
        )
    )
    result = subprocess.run(
        [str(paths["python311"]), "-c", probe],
        cwd=serving,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    if result.returncode != 0:
        raise DeliveryError("Python 3.11 helper import preflight failed: " + result.stderr[-2000:])
    imported = json.loads(result.stdout)
    if imported != {name: digest for name, (_, digest) in imports.items()}:
        raise DeliveryError("Python 3.11 helper import preflight returned unexpected hashes")

    bridge_probe = "\n".join(
        (
            "import hashlib, importlib",
            "from pathlib import Path",
            "module = importlib.import_module('wnn.training.final_submit_bridge_cache')",
            "path = Path(module.__file__).resolve()",
            f"expected = {str((paths['source'] / 'src' / 'wnn' / 'training' / 'final_submit_bridge_cache.py').resolve())!r}",
            f"expected_sha = {cache_validator_sha256!r}",
            "if str(path) != expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:",
            "    raise SystemExit(f'cache-validator import source mismatch: {path}')",
        )
    )
    result = subprocess.run(
        [str(paths["python312"]), "-c", bridge_probe],
        cwd=paths["source"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    if result.returncode != 0:
        raise DeliveryError("Python 3.12 cache-validator import preflight failed: " + result.stderr[-2000:])

    for python, script in (
        (paths["python312"], serving / "scripts" / "package_final_w18.py"),
        (paths["python311"], serving / "scripts" / "check_final_w18_package.py"),
    ):
        result = subprocess.run(
            [str(python), str(script), "--help"],
            cwd=paths["source"],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=90,
        )
        if result.returncode != 0:
            raise DeliveryError(f"serving helper CLI preflight failed: {script.name}")
    return imported


def _w18_closure_nodes(nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    root_id = "w18final_distilled_s5"
    pending = [root_id]
    found: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in found:
            continue
        node = nodes.get(node_id)
        if not isinstance(node, dict):
            raise DeliveryError(f"W18 plan is missing closure node {node_id}")
        found.add(node_id)
        pending.extend(str(parent) for parent in node.get("parents", ()))
    if found != EXPECTED_W18_ROLES or any("tlead" in node_id or node_id == "source4_lstm192x1" for node_id in found):
        raise DeliveryError("W18 plan closure is not the exact 13-role teacher ancestry")
    return [nodes[node_id] | {"node_id": node_id} for node_id in sorted(found)]


def _validate_fit_receipt(
    receipt_path: Path,
    *,
    node: dict[str, Any],
    source_root: Path,
    data_root: Path,
    fit_python: Path,
) -> dict[str, Any]:
    record = _read_json(receipt_path, max_bytes=2_000_000)
    command = record.get("command")
    expected_output = (data_root / "artifacts" / "runs" / str(node["run"])).resolve()
    started = record.get("started_at_epoch")
    finished = record.get("finished_at_epoch")
    timeout = record.get("timeout_seconds")
    if (
        record.get("schema") != "wnn.multistage-owned-process.v1"
        or record.get("mode") != "final_submit_finite_chain"
        or type(record.get("stage")) is not int
        or record.get("stage") != 1
        or record.get("status") != "completed"
        or type(record.get("exit_code")) is not int
        or record.get("exit_code") != 0
        or record.get("timed_out") is not False
        or not isinstance(command, list)
        or not all(isinstance(token, str) for token in command)
        or len(command) != 6 + len(node["args"])
        or command[:3] != ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s"]
        or not command[3].endswith("s")
        or not command[3][:-1].isdecimal()
        or not 0 < int(command[3][:-1]) <= 21_590
        or Path(command[4]).resolve() != fit_python.resolve()
        or Path(command[5]).resolve() != (source_root / "scripts" / "train_rnn.py").resolve()
        or command[6:] != node["args"]
        or record.get("cwd") != str(source_root.resolve())
        or record.get("output_dir") != str(expected_output)
        or record.get("command_sha256")
        != hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
        or type(timeout) is not int
        or not 0 < timeout <= FIT_TIMEOUT_CAP
        or int(command[3][:-1]) > timeout
        or not isinstance(started, (int, float))
        or isinstance(started, bool)
        or not isinstance(finished, (int, float))
        or isinstance(finished, bool)
        or not math.isfinite(float(started))
        or not math.isfinite(float(finished))
        or not 0 < float(started) <= float(finished)
    ):
        raise DeliveryError(f"fit receipt does not prove successful fixed-plan exit: {node['id']}")
    epoch = node.get("endpoint")
    if type(epoch) is not int or epoch <= 0:
        raise DeliveryError(f"W18 closure node has an unsupported endpoint: {node['id']}")
    checkpoint = expected_output / f"model_ep{epoch}.pt"
    checkpoint_sha = _sha256(checkpoint)
    raw_sha = _sha256(receipt_path)
    return {
        "node_id": node["node_id"],
        "run": node["run"],
        "checkpoint_file": checkpoint.name,
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "fit_process_receipt_path": receipt_path,
        "fit_process_receipt_sha256": raw_sha,
        "command_sha256": record.get("command_sha256"),
        "record": record,
    }


def _collect_w18_closure(
    *,
    paths: dict[str, Path],
    nodes: dict[str, dict[str, Any]],
    deadline: float,
    poll_seconds: int,
    state_path: Path,
) -> list[dict[str, Any]]:
    closure_nodes = _w18_closure_nodes(nodes)
    while True:
        ready: list[dict[str, Any]] = []
        missing: list[str] = []
        failed: list[str] = []
        for node in closure_nodes:
            node_id = str(node["node_id"])
            receipt_path = paths["control"] / "receipts" / "fits" / f"{node_id}.json"
            if not receipt_path.exists():
                missing.append(node_id)
                continue
            record = _read_json(receipt_path, max_bytes=2_000_000)
            status = record.get("status")
            if status not in {"completed", "running", "launching"}:
                failed.append(node_id)
                continue
            if status != "completed":
                missing.append(node_id)
                continue
            try:
                ready.append(
                    _validate_fit_receipt(
                        receipt_path,
                        node=node,
                        source_root=(
                            paths["source"]
                            if node_id == "w18final_distilled_s5"
                            else paths["original_source"]
                        ),
                        data_root=paths["data"],
                        fit_python=paths["python312"],
                    )
                )
            except FileNotFoundError:
                missing.append(node_id)
        if failed:
            raise DeliveryError(f"W18 lineage fit process failed: {sorted(failed)}")
        if len(ready) == 13:
            ready.sort(key=lambda item: item["run"])
            if {item["node_id"] for item in ready} != EXPECTED_W18_ROLES:
                raise DeliveryError("completed W18 fit receipts do not cover the exact 13 roles")
            return ready
        if time.time() >= deadline - DELIVERY_RESERVE_SECONDS:
            raise TimeoutError(
                f"W18 publication deadline reserve reached; ready={len(ready)}/13, missing={sorted(set(missing))}"
            )
        _record(
            state_path,
            "waiting_for_w18_closure",
            ready_roles=sorted(item["node_id"] for item in ready),
            missing_roles=sorted(set(missing)),
            deadline_epoch=deadline,
        )
        time.sleep(min(poll_seconds, max(0.0, deadline - DELIVERY_RESERVE_SECONDS - time.time())))


def _run_bounded(
    command: list[str], *, env: dict[str, str], cwd: Path, timeout: int, stderr_path: Path
) -> subprocess.CompletedProcess[str]:
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("x", encoding="utf-8") as error_stream:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=error_stream,
            check=False,
            timeout=timeout,
        )


def _make_closure_evidence(
    *,
    destination: Path,
    closure: list[dict[str, Any]],
    inputs: list[tuple[Path, str]],
    model_receipt: Path,
    build_receipt: Path,
    process_receipt: Path,
    package_check: Path,
    package: Path,
    membership_sha256: str,
) -> dict[str, Any]:
    records = []
    manifest_items: list[tuple[Path, str, int]] = []
    for item in closure:
        checkpoint = item["checkpoint_path"]
        receipt = item["fit_process_receipt_path"]
        manifest_items.extend(
            [
                (checkpoint, f"models/{item['run']}/{item['checkpoint_file']}", zipfile.ZIP_STORED),
                (receipt, f"receipts/{item['node_id']}.json", zipfile.ZIP_DEFLATED),
            ]
        )
        records.append(
            {
                key: item[key]
                for key in (
                    "node_id",
                    "run",
                    "checkpoint_file",
                    "checkpoint_sha256",
                    "fit_process_receipt_sha256",
                    "command_sha256",
                )
            }
        )
    extras = [
        (model_receipt, "receipts/final-w18-model-receipt.json", zipfile.ZIP_DEFLATED),
        (build_receipt, "receipts/final-w18-build-receipt.json", zipfile.ZIP_DEFLATED),
        (process_receipt, "receipts/final-w18-fit-process.json", zipfile.ZIP_DEFLATED),
        (package_check, "reports/final-w18-package-check.json", zipfile.ZIP_DEFLATED),
        (package, "submission/tabdata-w18-final.zip", zipfile.ZIP_STORED),
    ]
    manifest_items.extend(extras)
    manifest_items.extend((path, name, zipfile.ZIP_DEFLATED) for path, name in inputs)
    names = [name for _, name, _ in manifest_items]
    if len(names) != len(set(names)) or any(
        PurePosixPath(name).is_absolute()
        or ".." in PurePosixPath(name).parts
        or "\\" in name
        for name in names
    ):
        raise DeliveryError("W18 evidence archive contains duplicate or unsafe member paths")
    manifest = {
        "schema": "wnn.final-w18-closure-evidence.v1",
        "scope": "single W18 final model with exact 13-role training ancestry",
        "fit_sequences": 12_480,
        "independent_evaluation_sequences": 0,
        "independent_score_claim": "forbidden",
        "score_computed": False,
        "membership_sha256": membership_sha256,
        "plan_sha256": PLAN_SHA256,
        "source_validation_sha256": SOURCE_VALIDATION_SHA256,
        "lineage_roles": records,
        "evidence_files": [
            {"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path, name, _ in manifest_items
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path, name, compression in manifest_items:
            archive.write(path, name, compress_type=compression)
        archive.writestr(
            "MANIFEST.json",
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
            compress_type=zipfile.ZIP_DEFLATED,
        )
    if destination.stat().st_size > MAX_EVIDENCE_ZIP_BYTES:
        raise DeliveryError("W18 closure evidence archive exceeds its bounded asset size")
    with zipfile.ZipFile(destination) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise DeliveryError(f"W18 closure evidence archive CRC failed: {corrupt}")
    return manifest


def _validate_receipt_joins(
    *,
    receipt_path: Path,
    process_receipt_path: Path,
    closure: list[dict[str, Any]],
    membership: Any,
    expected_cache_sha256: str,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path, max_bytes=1_000_000)
    fit = receipt.get("final_fit")
    teacher_cache = receipt.get("teacher_cache")
    if (
        receipt.get("schema") != "wnn.final-w18-model-receipt.v1"
        or not isinstance(fit, dict)
        or fit.get("manifest_sha256") != membership.manifest_sha256
        or fit.get("fit_members_sha256") != membership.fit_members_sha256
        or fit.get("source_validation_sha256") != SOURCE_VALIDATION_SHA256
        or receipt.get("fit_sequences") != 12_480
        or receipt.get("independent_evaluation_members") != []
        or receipt.get("independent_evaluation_sequences") != 0
        or receipt.get("independent_score_claim") != "forbidden"
        or not isinstance(teacher_cache, dict)
        or teacher_cache.get("metadata_sha256") != expected_cache_sha256
    ):
        raise DeliveryError("W18 receipt fit membership or no-score policy differs")
    members = receipt.get("members")
    lineage = receipt.get("lineage_closure")
    if (
        not isinstance(members, dict)
        or set(members) != {"w18"}
        or not isinstance(lineage, dict)
        or lineage.get("schema") != "wnn.final-w18-lineage-closure.v1"
    ):
        raise DeliveryError("W18 receipt contains an unsupported active-member or lineage set")
    records = lineage.get("roles")
    if not isinstance(records, list) or len(records) != len(closure):
        raise DeliveryError("W18 model receipt lacks the exact completed 13-role closure")
    by_run = {record.get("run"): record for record in records if isinstance(record, dict)}
    if len(by_run) != 13 or set(by_run) != {item["run"] for item in closure}:
        raise DeliveryError("W18 model receipt lineage roles differ from completed fit receipts")
    for item in closure:
        record = by_run[item["run"]]
        if (
            record.get("checkpoint_file") != item["checkpoint_file"]
            or record.get("checkpoint_sha256") != item["checkpoint_sha256"]
            or record.get("final_fit_manifest_sha256") != membership.manifest_sha256
            or record.get("final_fit_members_sha256") != membership.fit_members_sha256
            or record.get("source_validation_sha256") != SOURCE_VALIDATION_SHA256
        ):
            raise DeliveryError(f"W18 lineage receipt does not bind actual fit output: {item['run']}")
    process_binding = receipt.get("fit_process_receipt")
    process_record = _read_json(process_receipt_path, max_bytes=2_000_000)
    w18_member = members["w18"]
    w18_lineage = by_run.get("tl3final_w18final_distilled_s5")
    if (
        not isinstance(w18_member, dict)
        or w18_member.get("run") != "tl3final_w18final_distilled_s5"
        or w18_member.get("checkpoint_sha256") != w18_lineage.get("checkpoint_sha256")
        or lineage.get("sha256")
        != hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        or not isinstance(process_binding, dict)
        or process_binding.get("path") != "w18_fit_process_receipt.json"
        or process_binding.get("sha256") != _sha256(process_receipt_path)
        or process_binding.get("record") != process_record
    ):
        raise DeliveryError("W18 receipt does not bind its successful child process bytes")
    return receipt


def _make_api(token: str, paths: dict[str, Path]) -> tuple[Any, str]:
    publisher = runpy.run_path(str(paths["source"] / "scripts" / "publish_checkpoint_releases.py"))
    github_type = publisher["GitHub"]
    api = github_type.__new__(github_type)
    api.token = token
    api.opener = __import__("urllib.request", fromlist=["build_opener"]).build_opener(
        publisher["SafeRedirect"]()
    )
    return api, str(publisher["REPOSITORY"])


def _publish_assets(
    *,
    paths: dict[str, Path],
    token: str,
    assets: list[Path],
    release_tag: str,
    release_body: str,
) -> tuple[str, list[dict[str, Any]]]:
    api, repository_name = _make_api(token, paths)
    user = api.request("GET", "/user")
    if not isinstance(user, dict) or user.get("login") != "SemyonEpanov":
        raise DeliveryError("W18 artifact publication token belongs to an unexpected account")
    repository = api.request("GET", f"/repos/{repository_name}")
    if (
        not isinstance(repository, dict)
        or str(repository.get("full_name", "")).lower() != repository_name.lower()
        or repository.get("private") is not True
        or not isinstance(repository.get("permissions"), dict)
        or repository["permissions"].get("push") is not True
    ):
        raise DeliveryError("private repository publication permission is unavailable")

    releases = api.request("GET", f"/repos/{repository_name}/releases?per_page=100")
    if not isinstance(releases, list):
        raise DeliveryError("GitHub release inventory is malformed")
    release = next(
        (record for record in releases if isinstance(record, dict) and record.get("tag_name") == release_tag),
        None,
    )
    if release is None:
        release = api.request(
            "POST",
            f"/repos/{repository_name}/releases",
            {
                "tag_name": release_tag,
                "target_commitish": "experiment/final-submit-all-labeled-20261031",
                "name": "W18 full-corpus single-member artifact",
                "draft": False,
                "prerelease": True,
                "make_latest": "false",
                "body": release_body,
            },
        )
    if (
        not isinstance(release, dict)
        or release.get("author", {}).get("login") != "SemyonEpanov"
        or release.get("draft") is not False
        or release.get("prerelease") is not True
        or release.get("target_commitish") != "experiment/final-submit-all-labeled-20261031"
    ):
        raise DeliveryError("W18 release tag already exists with a different owner or scope")
    release_id = int(release["id"])
    release_url = str(release.get("html_url", ""))
    if not release_url.startswith(f"https://github.com/{repository_name}/releases/tag/"):
        raise DeliveryError("GitHub returned an unexpected W18 release URL")
    remote_assets = api.request("GET", f"/repos/{repository_name}/releases/{release_id}/assets?per_page=100")
    if not isinstance(remote_assets, list):
        raise DeliveryError("GitHub W18 asset inventory is malformed")
    by_name = {record.get("name"): record for record in remote_assets if isinstance(record, dict)}
    verified: list[dict[str, Any]] = []
    roundtrip = paths["reports"] / "download-verified-w18"
    roundtrip.mkdir(parents=True, exist_ok=True)
    for path in assets:
        digest = _sha256(path)
        asset = by_name.get(path.name)
        if asset is None:
            asset = api.upload(release_id, path)
            by_name[path.name] = asset
        if (
            not isinstance(asset, dict)
            or asset.get("state") != "uploaded"
            or asset.get("size") != path.stat().st_size
            or asset.get("digest") not in (None, "sha256:" + digest)
        ):
            raise DeliveryError(f"W18 release asset metadata differs: {path.name}")
        downloaded = roundtrip / path.name
        if downloaded.exists():
            if _sha256(downloaded) != digest:
                raise DeliveryError(f"cached W18 release download differs: {path.name}")
        else:
            api.download(asset, downloaded, digest)
        if path.suffix == ".zip":
            with zipfile.ZipFile(downloaded) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise DeliveryError(f"downloaded W18 ZIP CRC failed at {bad}")
        verified.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "actual_download_verified": True,
            }
        )
    return release_url, verified


def _run(args: argparse.Namespace) -> int:
    if sys.platform != "linux":
        raise DeliveryError("W18 early publisher is restricted to the Linux receiving host")
    if not POLL_MIN_SECONDS <= args.poll_seconds <= POLL_MAX_SECONDS:
        raise DeliveryError("poll interval must be between 5 and 60 seconds")

    runtime = args.runtime_root.resolve(strict=True)
    recovery = args.recovery_root.resolve(strict=True)
    paths = _runtime_paths(runtime, recovery)
    if paths["source"].resolve(strict=True) == paths["original_source"].resolve(strict=True):
        raise DeliveryError("W18 recovery source must be isolated from the frozen parent source")
    authority = _validate_recovery_authority(paths, args.deadline_epoch)
    source_binding = _validate_source_binding(paths)
    _setup_import_paths(paths)
    env = _subprocess_env(paths)
    import final_submit_chain_contract as contract

    plan = contract.read_json(paths["plan"])
    plan_nodes = contract.verify_plan(plan, paths["plan"])
    if plan.get("plan_sha256") != PLAN_SHA256:
        raise DeliveryError("W18 publication plan differs from the frozen all-corpus plan")
    membership = contract.verify_sources(
        data_root=paths["data"],
        membership_path=paths["membership"],
        audit_path=paths["source_audit"],
        audit_sha256=SOURCE_VALIDATION_SHA256,
    )
    imported = _preflight(paths, env, source_binding["cache_validator_sha256"])
    _record(
        args.state_file or paths["reports"] / "w18-recovery-publication-state-20261103-r1.json",
        "inputs_verified_waiting_for_w18_closure",
        plan_sha256=plan["plan_sha256"],
        membership_sha256=membership.manifest_sha256,
        imported_helpers=imported,
        root_authority_sha256=_sha256(paths["root_authority"]),
        source_binding_sha256=source_binding["binding_sha256"],
        cache_validator_sha256=source_binding["cache_validator_sha256"],
    )
    state_path = args.state_file or paths["reports"] / "w18-recovery-publication-state-20261103-r1.json"
    deadline = min(args.deadline_epoch, time.time() + args.max_wait_seconds)
    closure = _collect_w18_closure(
        paths=paths,
        nodes=plan_nodes,
        deadline=deadline,
        poll_seconds=args.poll_seconds,
        state_path=state_path,
    )
    w18 = next(item for item in closure if item["node_id"] == "w18final_distilled_s5")
    reports = paths["reports"]
    package_path = reports / "tabdata-w18-final-20261103-r1.zip"
    model_receipt = reports / "tabdata-w18-final-model-receipt-20261103-r1.json"
    build_receipt = reports / "tabdata-w18-final-build-receipt-20261103-r1.json"
    process_receipt = w18["fit_process_receipt_path"]
    package_check = reports / "tabdata-w18-final-package-check-20261103-r1.json"
    evidence_zip = reports / "tabdata-w18-final-13-role-evidence-20261103-r1.zip"
    for output in (package_path, model_receipt, build_receipt, package_check, evidence_zip):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"W18 publisher uses create-only outputs: {output}")

    package_script = paths["serving"] / "scripts" / "package_final_w18.py"
    package_command = [
        str(paths["python312"]),
        str(package_script),
        "--w18",
        str(w18["checkpoint_path"]),
        "--data-root",
        str(paths["data"]),
        "--membership",
        str(paths["membership"]),
        "--source-audit",
        str(paths["source_audit"]),
        "--fit-process-receipt",
        str(process_receipt),
        "--plan",
        str(paths["plan"]),
        "--native-from",
        str(paths["native_r12"]),
        "--output-zip",
        str(package_path),
        "--model-receipt",
        str(model_receipt),
        "--build-receipt",
        str(build_receipt),
    ]
    _record(state_path, "building_w18_only_package", closure_roles=13)
    package_result = _run_bounded(
        package_command,
        env=env,
        cwd=paths["source"],
        timeout=PACKAGE_TIMEOUT_SECONDS,
        stderr_path=reports / "tabdata-w18-package-build-20261103-r1.stderr.log",
    )
    if package_result.returncode != 0:
        raise DeliveryError(
            "W18-only packager failed; see bounded stderr log: "
            + str(reports / "tabdata-w18-package-build-20261103-r1.stderr.log")
        )
    _validate_receipt_joins(
        receipt_path=model_receipt,
        process_receipt_path=process_receipt,
        closure=closure,
        membership=membership,
        expected_cache_sha256=source_binding["cache_metadata_sha256"],
    )

    checker_script = paths["serving"] / "scripts" / "check_final_w18_package.py"
    checker_command = [
        str(paths["python311"]),
        str(checker_script),
        "--zip",
        str(package_path),
        "--build-receipt",
        str(build_receipt),
        "--model-receipt",
        str(model_receipt),
        "--fit-process-receipt",
        str(process_receipt),
        "--checkpoint",
        str(w18["checkpoint_path"]),
        "--train-data",
        str(paths["train"]),
        "--reference-checkpoints-root",
        str(paths["references"]),
        "--r11-native",
        str(paths["native_r11"]),
        "--quantization-contract",
        str(paths["quantization"]),
        "--quantization-contract-sha256",
        QUANTIZATION_SHA256,
        "--sequence-index",
        "0",
    ]
    _record(state_path, "checking_cpu_python311_package", package_sha256=_sha256(package_path))
    checker_result = _run_bounded(
        checker_command,
        env=env,
        cwd=paths["source"],
        timeout=CHECK_TIMEOUT_SECONDS,
        stderr_path=reports / "tabdata-w18-package-check-20261103-r1.stderr.log",
    )
    if checker_result.returncode not in (0, 2):
        raise DeliveryError(
            "W18 package checker failed; see bounded stderr log: "
            + str(reports / "tabdata-w18-package-check-20261103-r1.stderr.log")
        )
    with package_check.open("x", encoding="utf-8") as stream:
        stream.write(checker_result.stdout)
    check = _read_json(package_check)
    if (
        check.get("score_computed") is not False
        or check.get("independent_score_claim") != "forbidden"
        or check.get("artifact_sha256") != _sha256(package_path)
        or check.get("w18_checkpoint_sha256") != w18["checkpoint_sha256"]
        or check.get("lineage_runs") != 13
    ):
        raise DeliveryError("W18 package check report does not bind the expected no-score artifact")
    if checker_result.returncode != 0 or check.get("eligible_for_w18_only_delivery") is not True:
        _record(
            state_path,
            "package_checked_not_eligible_for_delivery",
            report_path=str(package_check),
            gates=check.get("gates"),
            score_computed=False,
            independent_score_claim="forbidden",
        )
        return 2

    evidence_paths = [
        (paths["plan"], "inputs/final-plan-r2.json"),
        (paths["membership"], "inputs/final-fit-membership-r1.json"),
        (paths["source_audit"], "inputs/SOURCE_VALIDATION_R1.json"),
        (paths["quantization"], "inputs/w18-final-quantization-admission-r1.json"),
        (reports / "w18-final-quant-numeric-ceilings-r1.json", "inputs/w18-final-quant-numeric-ceilings-r1.json"),
        (reports / "w18-old-reference-quantization-fidelity-contract-r1.json", "inputs/w18-old-reference-quantization-fidelity-contract-r1.json"),
        (reports / "w18-only-old-reference-numeric-calibration-r11.json", "reports/w18-only-old-reference-numeric-calibration-r11.json"),
        (reports / "w18-old-reference-f32-exact-valid-r1.json", "reports/w18-old-reference-f32-exact-valid-r1.json"),
        (reports / "w18-old-reference-q8vnni-fast-valid-r1.json", "reports/w18-old-reference-q8vnni-fast-valid-r1.json"),
        (paths["root_authority"], "ROOT_AUTHORITY_R1.json"),
        (paths["source_binding"], "inputs/SOURCE_BINDING_R1.json"),
        (paths["source_capsule"], "inputs/w18-source-recovery-r1.tar.gz"),
    ]
    for path, _ in evidence_paths:
        _sha256(path)
    archive_manifest = _make_closure_evidence(
        destination=evidence_zip,
        closure=closure,
        inputs=evidence_paths,
        model_receipt=model_receipt,
        build_receipt=build_receipt,
        process_receipt=process_receipt,
        package_check=package_check,
        package=package_path,
        membership_sha256=membership.manifest_sha256,
    )
    release_tag = "tabdata-pr54-final-w18-20261103-r1"
    release_body = (
        "Single-member W18 full-corpus submission artifact. The model was fit on all 12,480 "
        "official train and valid sequences; independent evaluation is zero and no score is "
        "claimed. The exact 13-role W18 training ancestry, child-process receipts, cache and "
        "membership provenance are included. The included Python 3.11/Linux CPU package check "
        "passed the 20 MB, reset/causality, numeric, memory, and 60-minute projection gates. "
        "Platform upload remains a human step."
    )
    manifest_path = reports / "tabdata-w18-final-delivery-manifest-20261103-r1.json"
    payload_manifest = {
        "schema": "wnn.final-w18-delivery-manifest.v1",
        "release_tag": release_tag,
        "package_sha256": _sha256(package_path),
        "package_bytes": package_path.stat().st_size,
        "model_receipt_sha256": _sha256(model_receipt),
        "build_receipt_sha256": _sha256(build_receipt),
        "fit_process_receipt_sha256": _sha256(process_receipt),
        "package_check_sha256": _sha256(package_check),
        "closure_evidence_sha256": _sha256(evidence_zip),
        "closure_evidence_bytes": evidence_zip.stat().st_size,
        "closure_roles": archive_manifest["lineage_roles"],
        "root_authority_sha256": _sha256(paths["root_authority"]),
        "source_binding_sha256": source_binding["binding_sha256"],
        "source_capsule_sha256": source_binding["capsule_sha256"],
        "cache_validator_sha256": source_binding["cache_validator_sha256"],
        "cache_metadata_sha256": source_binding["cache_metadata_sha256"],
        "hard_deadline_epoch": authority["hard_deadline_epoch"],
        "fit_sequences": 12_480,
        "independent_evaluation_sequences": 0,
        "score_computed": False,
        "independent_score_claim": "forbidden",
        "human_platform_upload_required": True,
    }
    _write_json_create(manifest_path, payload_manifest)

    from galam_l4_quality8.publisher import read_token, remove_owned_token

    _record(
        state_path,
        "publishing_checked_w18_artifacts",
        release_tag=release_tag,
        package_sha256=payload_manifest["package_sha256"],
        evidence_sha256=payload_manifest["closure_evidence_sha256"],
        score_computed=False,
        independent_score_claim="forbidden",
    )
    token, token_identity = read_token(paths["token"])
    token_removed = False

    def cleanup_token() -> None:
        nonlocal token_removed
        if not token_removed:
            token_removed = remove_owned_token(paths["token"], token_identity)

    def terminate(signum: int, frame: object) -> None:
        _record(state_path, "publisher_termination_requested", signal=signum)
        raise SystemExit(128 + signum)

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    atexit.register(cleanup_token)
    try:
        release_url, uploaded = _publish_assets(
            paths=paths,
            token=token,
            assets=[
                package_path,
                model_receipt,
                build_receipt,
                process_receipt,
                package_check,
                evidence_zip,
                manifest_path,
            ],
            release_tag=release_tag,
            release_body=release_body,
        )
    finally:
        cleanup_token()
        atexit.unregister(cleanup_token)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if not token_removed:
            raise DeliveryError("private publisher token identity changed before cleanup")
    delivery = payload_manifest | {
        "schema": "wnn.final-w18-early-publication.v1",
        "release_url": release_url,
        "uploaded_assets": uploaded,
        "serving_check": check,
        "published_epoch": time.time(),
    }
    delivery_path = reports / "tabdata-w18-final-delivery-receipt-20261103-r1.json"
    _write_json_create(delivery_path, delivery)
    _record(
        state_path,
        "w18_only_artifact_published",
        release_url=release_url,
        package_sha256=payload_manifest["package_sha256"],
        closure_roles=13,
        serving_eligible=True,
        score_computed=False,
        independent_score_claim="forbidden",
        human_platform_upload_required=True,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_DEFAULT)
    parser.add_argument("--recovery-root", type=Path, default=RECOVERY_DEFAULT)
    parser.add_argument("--deadline-epoch", type=float, default=DEADLINE_DEFAULT)
    parser.add_argument("--max-wait-seconds", type=int, default=21_600)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--state-file", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_wait_seconds <= 21_600:
        parser.error("--max-wait-seconds must be between 1 and 21600")
    try:
        result = _run(args)
    except Exception as error:
        failure_state = args.state_file or args.recovery_root / "reports" / "w18-recovery-publication-state-20261103-r1.json"
        with contextlib.suppress(OSError, DeliveryError):
            _record(
                failure_state,
                "delivery_failed",
                error_type=type(error).__name__,
                error=str(error)[:2_000],
                score_computed=False,
                independent_score_claim="forbidden",
            )
        raise
    raise SystemExit(result)


if __name__ == "__main__":
    main()

"""Private T4 worker for the preregistered six-cell E482 replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
E481_DIR = ROOT / "research/campaign_2026-10-05/recurrent_activation_curvature"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_optimizer"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_placement"))
sys.path.insert(0, str(E481_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
import fit_recurrent_curvature as e481  # noqa: E402

from wnn.optim import orthogonalize  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

SEED = 271828
SHAPES = ("baseline", "sine", "cubic")
OPTIMIZERS = ("adamw", "muon_adamw")
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
E481_SHA256 = "3d97d2d07b116b3cf04b365c33e3d3caed2c96c35108bd28f3a6ff63b54d65cc"
E470_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E474_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
RAM_LIMIT_BYTES = 6 * 1024**3
GPU_ALLOCATOR_BYTES = 2 * 1024**3
GPU_STOP_BYTES = 3_000_000_000
ADMISSION_DEADLINE = datetime(2026, 10, 5, 7, 0, tzinfo=UTC)


def file_sha256(path: Path) -> str:
    """Hash a file incrementally so source and data validation stay memory bounded."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    """Return exact frozen E481 and parent implementation identities."""
    paths = {
        "e481": E481_DIR / "fit_recurrent_curvature.py",
        "e470": e470.__file__,
        "e474": e474.__file__,
        "muon_optimizer": ROOT / "src/wnn/optim.py",
        "remote_worker": HERE / "e482_remote.py",
    }
    return {name: file_sha256(Path(path)) for name, path in paths.items()}


def verify_inputs() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Verify sealed sources/data and load fitting rows only."""
    parents = e481.check_parent_sources()
    if parents != {"e470": e481.E470_SHA256, "e474": e481.E474_SHA256}:
        raise ValueError("E481 frozen parent-source identities differ")
    actual_sources = source_hashes()
    expected_sources = {"e481": E481_SHA256, "e470": E470_SHA256, "e474": E474_SHA256}
    if any(actual_sources[name] != digest for name, digest in expected_sources.items()):
        raise ValueError("E481/E470/E474 source identity differs from E482 registration")
    if file_sha256(e470.COHORT_PATH) != COHORT_SHA256:
        raise ValueError("E482 cohort identity mismatch")
    cohort_document = json.loads(e470.COHORT_PATH.read_text())
    expected_nodes = [node for node in cohort_document["cohort"] if node["role"] == "fit"]
    if file_sha256(e470.DATA_PATH) != DATASET_SHA256:
        raise ValueError("E482 mounted valid.parquet identity mismatch")
    manifest, x, y, need, _score_mask, nodes = e470.load_cohort(include_test=False)
    if (
        x.shape != (24, 20_000, e481.FEATURES)
        or y.shape != (24, 20_000, 2)
        or need.shape != (24, 20_000)
        or len(nodes) != 24
        or {node["role"] for node in nodes} != {"fit"}
        or nodes != expected_nodes
        or len({node["component"] for node in nodes}) != 24
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("worker loader did not return the exact 24 finite fit sequences")
    return manifest, x, y, need, nodes


def configure_worker() -> dict[str, object]:
    """Apply the declared one-thread and 2 GiB allocator limits on an assigned T4."""
    torch.set_num_threads(1)
    configure_torch()
    expected_uuid = os.environ.get("E482_GPU_UUID")
    assigned = validate_worker_device(
        gpu_inventory(),
        str(expected_uuid),
        torch.cuda.device_count(),
        torch.cuda.get_device_name(0) if torch.cuda.device_count() == 1 else "",
    )
    actual_fraction = torch.cuda.get_per_process_memory_fraction()
    expected_fraction = min(
        0.70, GPU_ALLOCATOR_BYTES / torch.cuda.get_device_properties(0).total_memory
    )
    if abs(actual_fraction - expected_fraction) > 1e-8:
        raise ValueError("configured CUDA allocator fraction differs from the 2 GiB cap")
    return {
        **assigned,
        "allocator_fraction": actual_fraction,
        "muon_cuda_orthogonalization": {
            "implementation_sha256": file_sha256(Path(orthogonalize.__code__.co_filename)),
            "method": "five-step quintic Newton-Schulz",
            "input_dtype": "bfloat16 when input tensor is CUDA",
            "output_dtype": "original gradient dtype",
        },
    }


def validate_worker_device(
    physical_devices: list[dict[str, int | str]],
    expected_uuid: str,
    visible_device_count: int,
    visible_device_name: str,
) -> dict[str, int | str]:
    """Bind one CUDA-visible worker to one matching UUID in host inventory."""
    matches = [device for device in physical_devices if device["uuid"] == expected_uuid]
    if len(matches) != 1 or "T4" not in str(matches[0]["name"]):
        raise ValueError("assigned worker UUID is absent or ambiguous in physical inventory")
    if visible_device_count != 1 or "T4" not in visible_device_name:
        raise ValueError("PyTorch worker must see exactly its assigned T4")
    return {
        "uuid": str(matches[0]["uuid"]),
        "name": str(matches[0]["name"]),
        "total_memory_bytes": int(matches[0]["total_bytes"]),
    }


def require_admission_window(now: datetime) -> None:
    """Refuse new remote work at or after the preregistered 07:00 UTC cutoff."""
    if now.tzinfo is None or now >= ADMISSION_DEADLINE:
        raise ValueError("E482 admission cutoff has passed or current time lacks UTC timezone")


def close_tensors(left: torch.Tensor, right: torch.Tensor, atol: float, rtol: float) -> None:
    """Fail the parity gate when a tensor departs from the native GRU reference."""
    torch.testing.assert_close(left, right, atol=atol, rtol=rtol)


def native_parity(optimizer_name: str) -> dict:
    """Check baseline output/state/gradients and clipped optimizer-step parity on CUDA."""
    torch.manual_seed(48_103)
    reference = e474.PlacementRNN("gelu", "gelu").to("cuda").float()
    state_dict = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    candidate = e481.initialize_arm("baseline", state_dict).to("cuda").float()
    x_ref = torch.randn(2, 17, e481.FEATURES, device="cuda", requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    with e474.full_ieee_float32():
        y_ref, s_ref = reference(x_ref)
        y_new, s_new = candidate(x_new)
        close_tensors(y_ref, y_new, 2e-5, 2e-4)
        close_tensors(s_ref, s_new, 2e-5, 2e-4)
        (y_ref.square().mean() + s_ref.square().mean()).backward()
        (y_new.square().mean() + s_new.square().mean()).backward()
        close_tensors(x_ref.grad, x_new.grad, 2e-5, 2e-4)
        ref_params = dict(reference.named_parameters())
        new_params = dict(candidate.named_parameters())
        if ref_params.keys() != new_params.keys():
            raise ValueError("native GRU parameter names differ")
        for name in ref_params:
            close_tensors(ref_params[name].grad, new_params[name].grad, 2e-5, 2e-4)

        reference.load_state_dict(state_dict, strict=True)
        candidate.load_state_dict(state_dict, strict=True)
        reference_optimizer = e481.make_optimizer(reference, optimizer_name)
        candidate_optimizer = e481.make_optimizer(candidate, optimizer_name)
        torch.manual_seed(48_104)
        x_step = torch.randn(2, 13, e481.FEATURES, device="cuda")
        y_step = torch.randn(2, 13, 2, device="cuda")
        for model, optimizer in (
            (reference, reference_optimizer),
            (candidate, candidate_optimizer),
        ):
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(x_step)
            loss = (
                prediction.square().mean()
                + 0.1 * state.square().mean()
                + 0.01 * (prediction * y_step).mean()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if not torch.isfinite(loss):
                raise ValueError("native parity objective became nonfinite")
        for name, value in reference.named_parameters():
            close_tensors(value, dict(candidate.named_parameters())[name], 2e-4, 2e-3)
    return {
        "optimizer": optimizer_name,
        "output_state_input_gradient_parameter_gradient_and_clipped_step": "passed",
        "parameter_tensors": len(ref_params),
        "precision": e474.precision_metadata(),
    }


def run_probe(optimizer_name: str, output: Path) -> dict:
    """Run one actual B4×256 CUDA update per shape plus native parity checks."""
    if optimizer_name not in OPTIMIZERS or output.exists():
        raise ValueError("invalid probe optimizer or pre-existing output path")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device_receipt = configure_worker()
    torch.cuda.reset_peak_memory_stats()
    e481.check_parent_sources()
    parity = native_parity(optimizer_name)
    _manifest, x, y, need, nodes = verify_inputs()
    states = e481.seed_states()
    order = np.arange(24)
    np.random.default_rng(SEED).shuffle(order)
    ids = order[:4].tolist()
    rows = []
    with e474.full_ieee_float32():
        for shape in SHAPES:
            torch.manual_seed(SEED)
            model = e481.initialize_arm(shape, states[SEED]).to("cuda").float()
            optimizer = e481.make_optimizer(model, optimizer_name)
            before = [p.detach().clone() for p in model.parameters()]
            loss, telemetry = e481.one_update(
                model,
                optimizer,
                x,
                y,
                need,
                ids,
                torch.device("cuda"),
                f"{SEED}/{optimizer_name}/{shape}/probe",
            )
            changed = [
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters(), strict=True)
            ]
            if not any(changed) or not np.isfinite(loss):
                raise ValueError(f"probe update failed finite/change gate for {shape}")
            if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
                raise ValueError(f"probe produced nonfinite parameters for {shape}")
            rows.append(
                {
                    "seed": SEED,
                    "optimizer": optimizer_name,
                    "shape": shape,
                    "fit_row_groups": [nodes[i]["row_group"] for i in ids],
                    "batch": 4,
                    "chunk_rows": 256,
                    "updates": 1,
                    "loss": loss,
                    "changed_parameter_tensors": sum(changed),
                    "parameters": e481.N_PARAMETERS,
                    "telemetry_step0": telemetry,
                }
            )
            del model, optimizer
            torch.cuda.synchronize()
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    if max(allocated, reserved) > GPU_ALLOCATOR_BYTES:
        raise ValueError("worker CUDA allocator use exceeded the 2 GiB ceiling")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    result = {
        "status": "probe_passed",
        "phase": "probe",
        "fit_only": True,
        "test_rows_loaded": False,
        "optimizer": optimizer_name,
        "arms": rows,
        "native_parity": parity,
        "device": torch.cuda.get_device_name(0),
        "assigned_device": device_receipt,
        "device_properties": {
            "name": torch.cuda.get_device_properties(0).name,
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "uuid": os.environ.get("E482_GPU_UUID", "unavailable"),
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "precision": e474.precision_metadata(),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "dataset_sha256": DATASET_SHA256,
        "cohort_sha256": COHORT_SHA256,
        "source_sha256": source_hashes(),
    }
    (output / "probe.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def run_fits(optimizer_name: str, output: Path) -> dict:
    """Fit the three preregistered shapes for one optimizer without changing E481 globals."""
    if optimizer_name not in OPTIMIZERS or output.exists():
        raise ValueError("invalid fit optimizer or pre-existing output path")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device_receipt = configure_worker()
    parents = e481.check_parent_sources()
    manifest, x, y, need, nodes = verify_inputs()
    states = e481.seed_states()
    fit_manifest = {**manifest, "cohort": [n for n in manifest["cohort"] if n["role"] == "fit"]}
    output.mkdir(parents=True, exist_ok=False)
    arms = []
    torch.cuda.reset_peak_memory_stats()
    with e474.full_ieee_float32():
        for shape in SHAPES:
            arms.append(
                e481.run_fit(
                    SEED,
                    optimizer_name,
                    shape,
                    x,
                    y,
                    need,
                    nodes,
                    states[SEED],
                    torch.device("cuda"),
                    output,
                )
            )
            torch.cuda.synchronize()
            allocated = int(torch.cuda.max_memory_allocated())
            reserved = int(torch.cuda.max_memory_reserved())
            if max(allocated, reserved) > GPU_ALLOCATOR_BYTES:
                raise ValueError("fit CUDA allocator use exceeded the 2 GiB ceiling")
    expected = {(SEED, optimizer_name, shape) for shape in SHAPES}
    actual = {(a["seed"], a["optimizer"], a["shape"]) for a in arms}
    if actual != expected:
        raise ValueError("fit worker did not complete the exact three assigned cells")
    result = {
        "status": "three_fit_arms_complete_test_unread",
        "phase": "fit",
        "fit_only": True,
        "test_rows_loaded": False,
        "seed": SEED,
        "optimizer": optimizer_name,
        "arms": arms,
        "planned_fit_count": 3,
        "batch": 4,
        "chunk_rows": 256,
        "updates_per_arm": 474,
        "architecture": "E481 frozen; 42 features, two-layer H64 GRU, 61,058 parameters",
        "parent_source_sha256": parents,
        "source_sha256": source_hashes(),
        "dataset_sha256": DATASET_SHA256,
        "cohort_sha256": COHORT_SHA256,
        "fit_only_cohort": fit_manifest,
        "precision": e474.precision_metadata(),
        "assigned_device": device_receipt,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    (output / "fit_receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def worker_main(argv: list[str] | None = None) -> None:
    """Dispatch exactly one optimizer-specific phase for an assigned T4 process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("probe", "fit"), required=True)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="validate worker CLI arguments without loading data or touching CUDA",
    )
    args = parser.parse_args(argv)
    if args.parse_only:
        print(
            json.dumps(
                {
                    "status": "worker_cli_arguments_validated_no_work",
                    "phase": args.phase,
                    "optimizer": args.optimizer,
                    "output": str(args.output),
                }
            ),
            flush=True,
        )
        return
    if args.phase == "probe":
        result = run_probe(args.optimizer, args.output)
    else:
        result = run_fits(args.optimizer, args.output)
    print(json.dumps({"status": result["status"], "optimizer": args.optimizer}), flush=True)


def gpu_inventory() -> list[dict[str, int | str]]:
    """Read physical CUDA inventory independently through nvidia-smi."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    items = []
    for line in output.strip().splitlines():
        index, uuid, name, total_mb, used_mb = (part.strip() for part in line.split(","))
        items.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "total_bytes": int(total_mb) * 1024**2,
                "used_bytes": int(used_mb) * 1024**2,
            }
        )
    return sorted(items, key=lambda item: int(item["index"]))


def host_inventory() -> dict[str, int | list[int]]:
    """Measure the active container and affinity before admitting worker processes."""
    memory = psutil.virtual_memory()
    capacity, available = memory.total, memory.available
    maximum = Path("/sys/fs/cgroup/memory.max")
    if maximum.exists() and maximum.read_text().strip() != "max":
        capacity = min(capacity, int(maximum.read_text().strip()))
        current = Path("/sys/fs/cgroup/memory.current")
        if current.exists():
            available = min(available, capacity - int(current.read_text().strip()))
    return {
        "capacity_bytes": capacity,
        "available_bytes": max(0, min(available, capacity)),
        "affinity": sorted(os.sched_getaffinity(0)),
    }


def cgroup_memory_events() -> dict[str, int] | None:
    """Read container OOM event counters when the provider exposes cgroup v2."""
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.exists():
        return None
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        values[key] = int(value)
    return values


def host_budget(host: dict[str, int | list[int]]) -> int:
    """Apply the same 80% capacity and 20% remaining-headroom guard as reviewed runner."""
    capacity = int(host["capacity_bytes"])
    available = int(host["available_bytes"])
    return min(int(capacity * 0.8), available - int(capacity * 0.2))


def admit_workers(host: dict[str, int | list[int]], devices: list[dict[str, int | str]]) -> None:
    """Require room for both capped workers and controller support before any probe."""
    if len(host["affinity"]) < 3:
        raise ValueError("E482 requires at least three available CPUs")
    if len(devices) != 2 or any("T4" not in str(item["name"]) for item in devices):
        raise ValueError("E482 requires exactly two visible T4 GPUs")
    if (
        len({item["index"] for item in devices}) != 2
        or len({item["uuid"] for item in devices}) != 2
    ):
        raise ValueError("E482 requires two unique T4 indices and UUIDs")
    if {int(item["index"]) for item in devices} != {0, 1}:
        raise ValueError("provider T4 indices differ from the preregistered device map")
    if any(int(item["used_bytes"]) > 128 * 1024**2 for item in devices):
        raise ValueError("unexplained preexisting GPU memory use blocks E482")
    required = 2 * RAM_LIMIT_BYTES + 1 * 1024**3
    if host_budget(host) < required:
        raise ValueError("two 6 GiB workers plus 1 GiB controller support exceed host budget")
    if any(int(item["total_bytes"]) <= GPU_STOP_BYTES for item in devices):
        raise ValueError("3 GB stop guard does not fit both physical GPU capacities")


def validate_probe_receipts(
    receipts: list[dict],
    assigned_devices: dict[str, str] | None = None,
    expected_sources: dict[str, str] | None = None,
) -> None:
    """Require the exact six successful probes before any full-fit worker starts."""
    probed = {
        (int(arm["seed"]), arm["optimizer"], arm["shape"])
        for receipt in receipts
        for arm in receipt.get("arms", [])
    }
    expected = {(SEED, optimizer, shape) for optimizer in OPTIMIZERS for shape in SHAPES}
    if probed != expected or len(receipts) != len(OPTIMIZERS):
        raise ValueError("probe receipt set is incomplete; full fits remain blocked")
    if any(receipt.get("status") != "probe_passed" for receipt in receipts):
        raise ValueError("a T4 probe failed; full fits remain blocked")
    for receipt in receipts:
        optimizer = receipt.get("optimizer")
        arm_set = {
            (int(arm["seed"]), arm["optimizer"], arm["shape"]) for arm in receipt.get("arms", [])
        }
        if arm_set != {(SEED, optimizer, shape) for shape in SHAPES}:
            raise ValueError("optimizer probe receipt does not contain its exact three shapes")
        if (
            receipt.get("dataset_sha256") != DATASET_SHA256
            or receipt.get("cohort_sha256") != COHORT_SHA256
            or (expected_sources is not None and receipt.get("source_sha256") != expected_sources)
            or receipt.get("fit_only") is not True
            or receipt.get("test_rows_loaded") is not False
            or receipt.get("native_parity", {}).get("optimizer") != optimizer
            or receipt.get("native_parity", {}).get(
                "output_state_input_gradient_parameter_gradient_and_clipped_step"
            )
            != "passed"
            or int(receipt.get("peak_allocated_bytes", GPU_ALLOCATOR_BYTES + 1))
            > GPU_ALLOCATOR_BYTES
            or int(receipt.get("peak_reserved_bytes", GPU_ALLOCATOR_BYTES + 1))
            > GPU_ALLOCATOR_BYTES
        ):
            raise ValueError("probe source or input hashes differ from E482 registration")
        if any(
            arm.get("batch") != 4
            or arm.get("chunk_rows") != 256
            or arm.get("updates") != 1
            or int(arm.get("changed_parameter_tensors", 0)) <= 0
            for arm in receipt.get("arms", [])
        ):
            raise ValueError("probe cell differs from the exact finite changed B4x256 update")
    if assigned_devices is not None:
        for receipt in receipts:
            optimizer = receipt.get("optimizer")
            identity = receipt.get("assigned_device", {})
            if (
                optimizer not in assigned_devices
                or identity.get("uuid") != assigned_devices[optimizer]
                or "T4" not in str(identity.get("name", ""))
                or receipt.get("device_properties", {}).get("uuid") != assigned_devices[optimizer]
            ):
                raise ValueError("probe device identity does not match the sealed assignment")


def validate_fit_receipts(receipts: list[dict], assigned_devices: dict[str, str]) -> None:
    """Require each completed optimizer triplet to remain on its assigned distinct T4."""
    expected = {(SEED, optimizer, shape) for optimizer in OPTIMIZERS for shape in SHAPES}
    actual = {
        (int(arm["seed"]), arm["optimizer"], arm["shape"])
        for receipt in receipts
        for arm in receipt.get("arms", [])
    }
    if actual != expected or len(receipts) != len(OPTIMIZERS):
        raise ValueError("fit receipt set is incomplete")
    cohort_document = json.loads(e470.COHORT_PATH.read_text())
    expected_nodes = [node for node in cohort_document["cohort"] if node["role"] == "fit"]
    row_order = np.arange(24)
    np.random.default_rng(SEED).shuffle(row_order)
    expected_row_groups = [expected_nodes[index]["row_group"] for index in row_order]
    if len(expected_row_groups) != 24 or len(set(expected_row_groups)) != 24:
        raise ValueError("registered E482 fit cohort is incomplete or duplicated")
    for receipt in receipts:
        optimizer = receipt.get("optimizer")
        identity = receipt.get("assigned_device", {})
        arm_set = {
            (int(arm["seed"]), arm["optimizer"], arm["shape"]) for arm in receipt.get("arms", [])
        }
        if (
            receipt.get("status") != "three_fit_arms_complete_test_unread"
            or optimizer not in assigned_devices
            or identity.get("uuid") != assigned_devices[optimizer]
            or "T4" not in str(identity.get("name", ""))
            or arm_set != {(SEED, optimizer, shape) for shape in SHAPES}
            or receipt.get("dataset_sha256") != DATASET_SHA256
            or receipt.get("cohort_sha256") != COHORT_SHA256
            or receipt.get("fit_only") is not True
            or receipt.get("test_rows_loaded") is not False
        ):
            raise ValueError("fit device identity or status differs from the assigned worker")
        for arm in receipt["arms"]:
            if (
                arm.get("updates") != 474
                or arm.get("parameters") != e481.N_PARAMETERS
                or arm.get("fit_row_groups") != expected_row_groups
                or [item.get("update_index_zero_based") for item in arm.get("telemetry", [])]
                != list(e481.TELEMETRY_STEPS)
                or not re.fullmatch(r"[0-9a-f]{64}", str(arm.get("checkpoint_sha256", "")))
            ):
                raise ValueError("fit arm cadence/cohort/checkpoint identity is incomplete")


def process_tree_rss(pid: int) -> int:
    """Conservatively sum RSS for a process and its descendants."""
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def stop_worker(process: subprocess.Popen[str]) -> None:
    """Stop only the worker process group on a resource threshold breach."""
    try:
        os.killpg(process.pid, 15)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, 9)
        process.wait(timeout=5)


def supervise_processes(jobs: list[dict], devices: list[dict[str, int | str]]) -> dict:
    """Run two one-thread workers while enforcing per-worker and whole-host/GPU limits."""
    start_host = host_inventory()
    start_devices = gpu_inventory()
    admit_workers(start_host, start_devices)
    expected_uuids = {str(item["uuid"]) for item in devices}
    if {str(item["uuid"]) for item in start_devices} != expected_uuids:
        raise ValueError("T4 device inventory changed between admission and worker launch")
    cpu_ids = list(os.sched_getaffinity(0))
    running = []
    resource_start = {
        "host": start_host,
        "devices": start_devices,
        "memory_events": cgroup_memory_events(),
    }
    peak_rss = {str(job["optimizer"]): 0 for job in jobs}
    for index, job in enumerate(jobs):
        device = next(item for item in devices if item["index"] == job["device_index"])
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(device["uuid"]),
                "E482_GPU_UUID": str(device["uuid"]),
                "WNN_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMBA_NUM_THREADS": "1",
                "POLARS_MAX_THREADS": "1",
                "ARROW_NUM_THREADS": "1",
                "WNN_GPU_MEMORY_BYTES": str(GPU_ALLOCATOR_BYTES),
                "WNN_GPU_FRACTION": "0.70",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(ROOT / "src"),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        )
        log_path = job["output"].parent / f"{job['phase']}_{job['optimizer']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w")
        command = [
            sys.executable,
            str(HERE / "e482_remote.py"),
            "--worker",
            "--phase",
            str(job["phase"]),
            "--optimizer",
            str(job["optimizer"]),
            "--output",
            str(job["output"]),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=lambda cpu=cpu_ids[index]: os.sched_setaffinity(0, {cpu}),
                text=True,
            )
        except OSError:
            log_file.close()
            for active in running:
                stop_worker(active["process"])
                active["log_file"].close()
            raise
        running.append({**job, "process": process, "log_file": log_file, "device": device})

    failure = None
    while any(item["process"].poll() is None for item in running):
        failed = [
            (item["optimizer"], item["process"].returncode)
            for item in running
            if item["process"].poll() is not None and item["process"].returncode != 0
        ]
        if failed:
            failure = f"worker exited unsuccessfully before sibling completion: {failed}"
            break
        try:
            host = host_inventory()
            live = [item for item in running if item["process"].poll() is None]
            rss = {item["optimizer"]: process_tree_rss(item["process"].pid) for item in live}
            physical = gpu_inventory()
            by_uuid = {str(item["uuid"]): item for item in physical}
            device_usage = {
                str(item["device"]["uuid"]): int(by_uuid[str(item["device"]["uuid"])]["used_bytes"])
                for item in live
            }
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, psutil.Error) as exc:
            failure = f"resource monitoring failed closed: {type(exc).__name__}: {exc}"
            break
        if any(value >= RAM_LIMIT_BYTES for value in rss.values()):
            failure = f"worker RSS reached its 6 GiB cap: {rss}"
            break
        if int(host["capacity_bytes"]) - int(host["available_bytes"]) >= int(
            int(host["capacity_bytes"]) * 0.9
        ):
            failure = "whole-host memory crossed the predeclared 90% stop"
            break
        if any(value >= GPU_STOP_BYTES for value in device_usage.values()):
            failure = f"physical GPU memory reached 3 GB stop: {device_usage}"
            break
        for optimizer, value in rss.items():
            peak_rss[optimizer] = max(peak_rss[optimizer], value)
        time.sleep(1.0)
    if failure is not None:
        for item in running:
            if item["process"].poll() is None:
                stop_worker(item["process"])
    exit_codes = {}
    for item in running:
        exit_codes[str(item["optimizer"])] = item["process"].wait()
        item["log_file"].close()
    if failure is not None:
        raise RuntimeError(failure)
    if any(code != 0 for code in exit_codes.values()):
        raise RuntimeError(f"one or more phase workers failed: {exit_codes}")
    resource_end = {
        "host": host_inventory(),
        "devices": gpu_inventory(),
        "memory_events": cgroup_memory_events(),
    }
    return {
        "resource_start": resource_start,
        "resource_end": resource_end,
        "peak_process_tree_rss_bytes_by_optimizer": peak_rss,
        "exit_codes": exit_codes,
        "thresholds": {
            "ram_worker_bytes": RAM_LIMIT_BYTES,
            "gpu_physical_stop_bytes": GPU_STOP_BYTES,
            "host_stop_fraction": 0.9,
        },
    }


def run_remote(output: Path) -> dict:
    """Probe both T4s fully before admitting the two optimizer-specific fit jobs."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite E482 output: {output}")
    require_admission_window(datetime.now(UTC))
    output.mkdir(parents=True)
    host = host_inventory()
    devices = gpu_inventory()
    admit_workers(host, devices)
    input_receipt = {
        "dataset_sha256": file_sha256(e470.DATA_PATH),
        "cohort_sha256": file_sha256(e470.COHORT_PATH),
    }
    if input_receipt != {"dataset_sha256": DATASET_SHA256, "cohort_sha256": COHORT_SHA256}:
        raise ValueError("E482 external input bytes differ from preregistered hashes")
    after_hash_host = host_inventory()
    after_hash_devices = gpu_inventory()
    admit_workers(after_hash_host, after_hash_devices)
    if {str(item["uuid"]) for item in after_hash_devices} != {
        str(item["uuid"]) for item in devices
    }:
        raise ValueError("T4 inventory changed while validating mounted inputs")
    (output / "admission.json").write_text(
        json.dumps(
            {
                "host": host,
                "devices": devices,
                "host_budget_bytes": host_budget(host),
                "after_input_hash_host": after_hash_host,
                "after_input_hash_devices": after_hash_devices,
                "after_input_hash_host_budget_bytes": host_budget(after_hash_host),
                "inputs": input_receipt,
            },
            indent=2,
        )
        + "\n"
    )
    device_by_optimizer = {
        "adamw": next(device for device in devices if device["index"] == 0),
        "muon_adamw": next(device for device in devices if device["index"] == 1),
    }
    probe_jobs = [
        {
            "phase": "probe",
            "optimizer": optimizer,
            "device_index": device_by_optimizer[optimizer]["index"],
            "output": output / "probes" / optimizer,
        }
        for optimizer in OPTIMIZERS
    ]
    probe_resources = supervise_processes(probe_jobs, devices)
    probe_receipts = [json.loads((job["output"] / "probe.json").read_text()) for job in probe_jobs]
    device_uuids = {
        optimizer: str(device_by_optimizer[optimizer]["uuid"]) for optimizer in OPTIMIZERS
    }
    expected_source_hashes = source_hashes()
    validate_probe_receipts(probe_receipts, device_uuids, expected_source_hashes)
    expected = {(SEED, optimizer, shape) for optimizer in OPTIMIZERS for shape in SHAPES}
    before_fit_host = host_inventory()
    before_fit_devices = gpu_inventory()
    admit_workers(before_fit_host, before_fit_devices)
    if {str(item["uuid"]) for item in before_fit_devices} != set(device_uuids.values()):
        raise ValueError("T4 inventory changed after probes; full fits remain blocked")
    fit_jobs = [
        {
            "phase": "fit",
            "optimizer": optimizer,
            "device_index": device_by_optimizer[optimizer]["index"],
            "output": output / "fits" / optimizer,
        }
        for optimizer in OPTIMIZERS
    ]
    fit_resources = supervise_processes(fit_jobs, devices)
    fit_receipts = [
        json.loads((job["output"] / "fit_receipt.json").read_text()) for job in fit_jobs
    ]
    validate_fit_receipts(fit_receipts, device_uuids)
    for fit_receipt in fit_receipts:
        if fit_receipt.get("source_sha256") != expected_source_hashes:
            raise ValueError("fit source hashes differ from the successful probe identities")
        probe = next(
            receipt
            for receipt in probe_receipts
            if receipt["optimizer"] == fit_receipt["optimizer"]
        )
        if fit_receipt["assigned_device"]["uuid"] != probe["assigned_device"]["uuid"]:
            raise ValueError("fit moved to a different T4 than its successful probe")
        for arm in fit_receipt["arms"]:
            checkpoint_name = str(arm.get("checkpoint", ""))
            if Path(checkpoint_name).name != checkpoint_name:
                raise ValueError("fit checkpoint path is not a safe basename")
            checkpoint = output / "fits" / fit_receipt["optimizer"] / checkpoint_name
            if not checkpoint.is_file() or file_sha256(checkpoint) != arm["checkpoint_sha256"]:
                raise ValueError("fit checkpoint bytes do not match its arm receipt")
    result = {
        "kind": "E482 six-recipe private T4 replication",
        "status": "six_fits_complete_test_unread",
        "fit_only": True,
        "test_rows_loaded": False,
        "seed": SEED,
        "shapes": list(SHAPES),
        "optimizers": list(OPTIMIZERS),
        "fit_count": len(expected),
        "probes": probe_receipts,
        "probe_resources": probe_resources,
        "before_fit_host": before_fit_host,
        "before_fit_devices": before_fit_devices,
        "fits": fit_receipts,
        "fit_resources": fit_resources,
        "host_initial": host,
        "devices_initial": devices,
        "source_sha256": source_hashes(),
        "dataset_sha256": DATASET_SHA256,
        "cohort_sha256": COHORT_SHA256,
    }
    (output / "e482_receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def orchestrator_main(argv: list[str] | None = None) -> None:
    """Run the sealed remote job under the kernel's outer whole-job watchdog."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in raw_args:
        raw_args.remove("--worker")
        worker_main(raw_args)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(raw_args)
    try:
        result = run_remote(args.output)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        if args.output.exists():
            (args.output / "E482_FAILURE.json").write_text(
                json.dumps(
                    {
                        "status": "failed_before_or_during_full_fit",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                        "test_rows_loaded": False,
                        "host": host_inventory(),
                        "devices": gpu_inventory(),
                        "memory_events": cgroup_memory_events(),
                    },
                    indent=2,
                )
                + "\n"
            )
        raise
    print(json.dumps({"status": result["status"], "fit_count": result["fit_count"]}), flush=True)


if __name__ == "__main__":
    orchestrator_main()

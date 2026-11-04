from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]

from wnn.multistage_search import atomic_write_json  # noqa: E402
from wnn.multistage_search_owned_command import run_owned_command  # noqa: E402
from wnn.multistage_search_runtime import (  # noqa: E402
    base_environment,
    current_process_owner,
    hold_gpu_slot,
)
from wnn.splits import file_sha256  # noqa: E402

SCHEMA = "wnn.l4-teacher-capacity-gru.v1"


def read_panel(path: Path) -> dict[str, object]:
    """Read and validate the six-cell GRU handoff without changing it."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4_000_000:
        raise ValueError("panel must be a regular JSON file no larger than 4 MB")
    panel = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if (
        panel.get("schema") != SCHEMA
        or panel.get("status") != "gru_recipes_frozen_waiting_for_source_admission"
    ):
        raise ValueError("panel schema/status is not the frozen GRU handoff")
    cells = panel.get("cells")
    if not isinstance(cells, list) or len(cells) != 6:
        raise ValueError("the GRU stage must contain exactly six cells")
    ids = [str(cast(dict[str, object], cell)["cell_id"]) for cell in cells]
    uuids = [str(cast(dict[str, object], cell)["gpu_uuid"]) for cell in cells]
    if len(set(ids)) != 6 or len(set(uuids)) != 6:
        raise ValueError("cell IDs and physical GPU UUIDs must be unique")
    return panel


def panel_path(panel: dict[str, object], cell: dict[str, object], phase: str) -> dict[str, Path]:
    """Return unique one-time request, log, receipt, and model-output paths."""
    paths = cast(dict[str, object], panel["paths"])
    runtime = Path(str(paths["runtime_root"]))
    project = Path(str(paths["project_root"]))
    cell_id, name = str(cell["cell_id"]), str(cell["run_name"])
    label = "profiles" if phase == "profile" else "fits"
    suffix = "_profile_r1" if phase == "profile" else ""
    return {
        "request": runtime / "requests" / label / f"{cell_id}.json",
        "log": runtime / "logs" / label / f"{cell_id}.log",
        "receipt": runtime / "receipts" / label / f"{cell_id}.json",
        "output": project / "artifacts" / "runs" / f"{name}{suffix}",
    }


def phase_state_path(panel: dict[str, object]) -> Path:
    """Return the GRU-stage deadline and phase state file."""
    paths = cast(dict[str, object], panel["paths"])
    return Path(str(paths["runtime_root"])) / "GRU_STATE_R1.json"


def start_or_check_state(panel: dict[str, object], phase: str) -> dict[str, object]:
    """Start the shared 12-hour deadline at first profile and enforce profile-before-fit."""
    path = phase_state_path(panel)
    budget = cast(dict[str, object], panel["budget"])
    if path.is_file():
        state = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        if state.get("panel_sha256") != file_sha256(Path(str(panel["_panel_path"]))):
            raise ValueError("state belongs to a different frozen GRU panel")
        expected = {"fit": "profile_complete"}.get(phase)
        if phase == "profile" or state.get("phase") != expected:
            raise ValueError(f"phase {phase} cannot follow {state.get('phase')}")
        return state
    if phase != "profile":
        raise ValueError("the first profile must establish the original deadline")
    started = time.time()
    state: dict[str, object] = {
        "schema": "wnn.l4-teacher-capacity-gru-state.v1",
        "panel_sha256": file_sha256(Path(str(panel["_panel_path"]))),
        "first_profile_started_epoch": started,
        "hard_deadline_epoch": started + int(budget["max_wave_seconds"]),
        "collection_reserve_seconds": int(budget["collection_reserve_seconds"]),
        "profile_attempts": 6,
        "fit_attempts": 0,
        "phase": "profile_running",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, state, replace=False)
    return state


def environment(panel: dict[str, object], cell: dict[str, object]) -> dict[str, str]:
    """Build the bounded strict-FP32 environment for exactly one physical L4."""
    paths = cast(dict[str, object], panel["paths"])
    threads = int(cell["cpu_threads"])
    env = base_environment(str(cell["gpu_uuid"]), Path(str(paths["source_root"])))
    env.update(
        {
            "WNN_PROJECT_ROOT": str(paths["project_root"]),
            "WNN_THREADS": str(threads),
            "WNN_LOADER_WORKERS": "2",
            "WNN_LOADER_RAM_GB": "8",
            "PYTHONUNBUFFERED": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        }
    )
    return env


def verify_external_receipt(path: Path, digest: str) -> dict[str, object]:
    """Require an unchanged admission or independent-review receipt by its SHA256."""
    if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
        raise ValueError(f"receipt missing or changed: {path}")
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def check_budget(panel: dict[str, object]) -> dict[str, int]:
    """Enforce the new-output cap and physical free-space floor for this GRU wave."""
    paths = cast(dict[str, object], panel["paths"])
    budget = cast(dict[str, object], panel["budget"])
    runtime = Path(str(paths["runtime_root"]))
    project = Path(str(paths["project_root"]))
    roots = [
        runtime / folder / phase
        for folder in ("logs", "requests", "receipts")
        for phase in ("profiles", "fits")
    ]
    for cell in cast(list[dict[str, object]], panel["cells"]):
        name = str(cell["run_name"])
        roots.extend(
            (
                project / "artifacts" / "runs" / f"{name}_profile_r1",
                project / "artifacts" / "runs" / name,
            )
        )
    written = 0
    for root in roots:
        if not root.exists() or root.is_symlink():
            continue
        candidates = root.rglob("*") if root.is_dir() else (root,)
        for candidate in candidates:
            if candidate.is_file() and not candidate.is_symlink():
                written += candidate.stat().st_size
    free = shutil.disk_usage(Path(str(paths["disk_mount"]))).free
    if written > int(budget["max_output_bytes"]):
        raise RuntimeError("GRU outputs crossed the frozen byte cap")
    if free < int(budget["minimum_free_bytes"]):
        raise RuntimeError("data2 free space crossed the frozen reserve")
    return {"output_bytes": written, "disk_free_bytes": free}


def run_cell(
    panel: dict[str, object],
    cell: dict[str, object],
    phase: str,
    state: dict[str, object],
    index: int,
) -> dict[str, object]:
    """Dispatch one immutable argv vector under a PID/birth/PGID ownership receipt."""
    paths = cast(dict[str, object], panel["paths"])
    budget = cast(dict[str, object], panel["budget"])
    policy = cast(dict[str, object], panel["profile_policy"])
    files = panel_path(panel, cell, phase)
    if any(item.exists() for item in files.values()):
        raise FileExistsError(f"{cell['cell_id']}: one-time output already exists")
    check_budget(panel)
    split = Path(str(paths["split_manifest"]))
    expected_split = str(cast(dict[str, object], panel["identity"])["split_sha256"])
    if file_sha256(split) != expected_split:
        raise ValueError("source-bound chronological split hash changed")
    command = cast(list[str], cell[f"{phase}_argv"])
    env = environment(panel, cell)
    request = {
        "schema": f"wnn.l4-teacher-capacity-gru-{phase}-request.v1",
        "panel_sha256": file_sha256(Path(str(panel["_panel_path"]))),
        "cell_id": cell["cell_id"],
        "configuration_id": cell["configuration_id"],
        "seed": cell["seed"],
        "gpu_uuid": cell["gpu_uuid"],
        "argv": command,
        "argv_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
        "split_sha256": expected_split,
        "admission_receipt_sha256": state["admission_receipt_sha256"],
        "source_binding_sha256": state["source_binding_sha256"],
        "data_binding_sha256": state["data_binding_sha256"],
        "fit_members_sha256": state["fit_members_sha256"],
        "created_utc": datetime.now(UTC).isoformat(),
        "profile_review_sha256": state.get("profile_review_sha256") if phase == "fit" else None,
    }
    files["request"].parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(files["request"], request, replace=False)
    owner = current_process_owner()
    owner.update(
        {
            "panel_sha256": request["panel_sha256"],
            "cell_id": cell["cell_id"],
            "run_name": cell["run_name"],
            "split_sha256": expected_split,
            "recipe_sha256": request["argv_sha256"],
        }
    )
    timeout = (
        int(policy["timeout_seconds"])
        if phase == "profile"
        else int(cast(dict[str, object], panel["execution"])["fit_timeout_seconds"])
    )
    remaining = int(
        float(state["hard_deadline_epoch"])
        - time.time()
        - int(budget["collection_reserve_seconds"])
    )
    timeout = min(timeout, remaining)
    if timeout <= 0:
        raise TimeoutError("only the collection reserve remains")
    with hold_gpu_slot(Path(str(paths["control_root"])), str(cell["gpu_uuid"])):
        code, timed_out, _ = run_owned_command(
            command,
            cwd=Path(str(paths["source_root"])),
            env=env,
            log_path=files["log"],
            receipt_path=files["receipt"],
            owner=owner,
            stage=0 if phase == "profile" else 1,
            trial_number=index,
            mode=f"l4_teacher_capacity_gru_{phase}",
            gpu_key=str(cell["gpu_uuid"]),
            output_dir=files["output"],
            timeout_seconds=timeout,
        )
    if code != 0 or timed_out:
        raise RuntimeError(f"{cell['cell_id']}: {phase} ended with {code}, timeout={timed_out}")
    return {"cell_id": cell["cell_id"], "receipt": str(files["receipt"]), "exit_code": code}


def run_phase(
    panel: dict[str, object],
    phase: str,
    admission_receipt: Path,
    admission_sha256: str,
    profile_review_receipt: Path | None,
    profile_review_sha256: str | None,
) -> None:
    """Run all six GRUs concurrently, preserving receipts and the global deadline."""
    admission = verify_external_receipt(admission_receipt, admission_sha256)
    admission_status = admission.get("status")
    if not isinstance(admission_status, str) or not admission_status.startswith("approved"):
        raise ValueError("source/data admission receipt is not admitted")
    identity = cast(dict[str, object], panel["identity"])
    panel_sha256 = file_sha256(Path(str(panel["_panel_path"])))
    if admission.get("panel_sha256") != panel_sha256:
        raise ValueError("source/data admission receipt belongs to a different frozen panel")
    bindings = (
        "split_sha256",
        "source_binding_sha256",
        "data_binding_sha256",
        "fit_members_sha256",
    )
    for binding in bindings:
        expected = identity.get(binding)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"panel identity must bind {binding} before dispatch")
        if admission.get(binding) != expected:
            raise ValueError(f"source/data admission receipt disagrees on {binding}")
    state = start_or_check_state(panel, phase)
    state.update(
        {
            "admission_receipt_sha256": admission_sha256,
            "source_binding_sha256": identity["source_binding_sha256"],
            "data_binding_sha256": identity["data_binding_sha256"],
            "fit_members_sha256": identity["fit_members_sha256"],
        }
    )
    atomic_write_json(phase_state_path(panel), state)
    if phase == "fit":
        if profile_review_receipt is None or profile_review_sha256 is None:
            raise ValueError("fit requires an independent profile-review receipt and SHA256")
        review = verify_external_receipt(profile_review_receipt, profile_review_sha256)
        review_status = review.get("status")
        if not isinstance(review_status, str) or not review_status.startswith("approved"):
            raise ValueError("independent profile review is not approved")
        review_bindings = {
            "panel_sha256": panel_sha256,
            "admission_receipt_sha256": admission_sha256,
            **{key: identity[key] for key in bindings},
        }
        if any(review.get(key) != value for key, value in review_bindings.items()):
            raise ValueError("profile review does not bind this panel, admission, split, and source")
        cells = cast(list[dict[str, object]], panel["cells"])
        for cell in cells:
            receipt = panel_path(panel, cell, "profile")["receipt"]
            if not receipt.is_file() or json.loads(receipt.read_text()).get("exit_code") != 0:
                raise ValueError(f"{cell['cell_id']}: zero-exit profile receipt is required")
        state["profile_review_sha256"] = profile_review_sha256
        state["fit_attempts"] = 6
        state["phase"] = "fit_running"
        atomic_write_json(phase_state_path(panel), state)
    results: list[dict[str, object]] = []
    errors: list[str] = []
    cells = cast(list[dict[str, object]], panel["cells"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(run_cell, panel, cell, phase, state, i) for i, cell in enumerate(cells, 1)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                errors.append(f"{type(error).__name__}: {error}")
    if phase == "profile" and not errors:
        state["phase"] = "profile_complete"
        atomic_write_json(phase_state_path(panel), state)
    if phase == "fit" and not errors:
        state["phase"] = "fit_complete"
        atomic_write_json(phase_state_path(panel), state)
    paths = cast(dict[str, object], panel["paths"])
    receipt_path = Path(str(paths["runtime_root"])) / "receipts" / f"GRU_{phase.upper()}_R1.json"
    atomic_write_json(
        receipt_path,
        {
            "schema": f"wnn.l4-teacher-capacity-gru-{phase}.v1",
            "panel_sha256": file_sha256(Path(str(panel["_panel_path"]))),
            "admission_sha256": admission_sha256,
            "profile_review_sha256": profile_review_sha256,
            "finished_utc": datetime.now(UTC).isoformat(),
            "results": sorted(results, key=lambda item: str(item["cell_id"])),
            "errors": errors,
        },
        replace=False,
    )
    if errors:
        raise RuntimeError("; ".join(errors))


def main() -> int:
    """Run the profile or fit stage from the frozen six-GRU panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("profile", "fit"))
    parser.add_argument("--admission-receipt", required=True, type=Path)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--profile-review-receipt", type=Path)
    parser.add_argument("--profile-review-sha256")
    args = parser.parse_args()
    panel = read_panel(args.panel)
    panel["_panel_path"] = str(args.panel.resolve())
    run_phase(
        panel,
        args.phase,
        args.admission_receipt,
        args.admission_sha256,
        args.profile_review_receipt,
        args.profile_review_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

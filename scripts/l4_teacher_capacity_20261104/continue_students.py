"""One-shot continuation for the two frozen L4 teacher-family student queues."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]

from scripts.l4_teacher_capacity_20261104.stage2_stream_common import (  # noqa: E402
    EXPECTED_RUNS,
    PR60_CACHE_METADATA_SHA256,
    PR60_MAIN_TRAIN_SHA256,
    TARGET_BYTES,
    partial_root,
    state_path,
)
from scripts.l4_teacher_capacity_20261104.stage2_stream_teachers import (  # noqa: E402
    verify_sequence_teacher,
)
from scripts.w18_l4_quality_grade_20261104 import (  # noqa: E402
    collect_member,
    expected_row_order,
    reference_checkpoint_filename,
)
from wnn.distillation_cache import row_groups_sha256  # noqa: E402
from wnn.multistage_search import atomic_write_json  # noqa: E402
from wnn.multistage_search_owned_command import run_owned_command  # noqa: E402
from wnn.multistage_search_runtime import (  # noqa: E402
    base_environment,
    current_process_owner,
    hold_gpu_slot,
)
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

PANEL_SCHEMA = "wnn.l4-teacher-capacity-continuation.v1"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
SEEDS = (1001, 1002)
PROFILE_UPDATES = 680
PROFILE_WARMUP = 65
PROFILE_TIMED = 3
MAX_WAVE_SECONDS = 43_200
COLLECTION_RESERVE_SECONDS = 2_700
MINIMUM_FREE_BYTES = 3_000_000_000
STOP_ON_EXISTING = "existing output or receipt found; preserve it and do not retry"


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 << 20:
        raise ValueError(f"expected a regular JSON object no larger than 16 MiB: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return file_sha256(path)


def _option(args: list[str], flag: str) -> str:
    positions = [index for index, item in enumerate(args) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(args):
        raise ValueError(f"student argv must contain exactly one {flag} VALUE")
    return args[positions[0] + 1]


def validate_panel(path: Path) -> tuple[dict[str, Any], str]:
    """Validate exact student cells against the frozen PR60 control argv."""
    panel = _read(path)
    if panel.get("schema") != PANEL_SCHEMA or panel.get("status") != "frozen":
        raise ValueError("continuation panel must be frozen with the packet-specific schema")
    paths = panel.get("paths")
    identity = panel.get("identity")
    families = panel.get("families")
    students = panel.get("students")
    if not all(isinstance(item, dict) for item in (paths, identity, families, students)):
        raise ValueError("continuation panel is missing paths, identities, families or students")
    for key in (
        "source_root",
        "project_root",
        "data_root",
        "python",
        "split_manifest",
        "runtime_root",
        "control_root",
        "disk_mount",
        "shared_state",
        "pr60_panel",
        "old_cache",
    ):
        value = paths.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"paths.{key} must be an absolute receiving-host path")
    for key in (
        "source_binding_sha256",
        "data_binding_sha256",
        "fit_members_sha256",
        "baseline_ancestry_sha256",
        "pr60_panel_sha256",
    ):
        if not isinstance(identity.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", identity[key]
        ):
            raise ValueError(f"identity.{key} must be a bound SHA256")
    if identity.get("split_sha256") != SPLIT_SHA256:
        raise ValueError("continuation split differs from the frozen chronological split")
    if file_sha256(Path(paths["split_manifest"])) != SPLIT_SHA256:
        raise ValueError("official split bytes differ from the frozen chronological split")
    if not isinstance(families.get("wide_gru"), dict) or not isinstance(
        families.get("causal_transformer"), dict
    ):
        raise ValueError("both frozen target-family bindings are required")
    cells = students.get("cells")
    if not isinstance(cells, list) or len(cells) != 4:
        raise ValueError("exactly four student fits are authorized in this continuation")
    refs = {(str(cell.get("family")), cell.get("seed")) for cell in cells if isinstance(cell, dict)}
    expected = {(family, seed) for family in families for seed in SEEDS}
    if refs != expected:
        raise ValueError("student cells must be exactly two families x seeds 1001/1002")
    ids: set[str] = set()
    names: set[str] = set()
    uuids_by_family: dict[str, set[str]] = {"wide_gru": set(), "causal_transformer": set()}
    affinity_by_family: dict[str, list[set[int]]] = {"wide_gru": [], "causal_transformer": []}
    baseline = _read(Path(paths["pr60_panel"]))
    if file_sha256(Path(paths["pr60_panel"])) != identity["pr60_panel_sha256"]:
        raise ValueError("frozen PR60 control panel bytes differ from the continuation identity")
    if baseline.get("schema") != "wnn.w18-l4-cohort.v1":
        raise ValueError("frozen PR60 W18 control panel is unavailable")
    baseline_cells = baseline.get("cells")
    if not isinstance(baseline_cells, list):
        raise ValueError("frozen PR60 W18 panel has no cells")
    controls = {
        cell.get("seed"): cell
        for cell in baseline_cells
        if isinstance(cell, dict) and cell.get("configuration_id") == "control"
    }
    if set(controls) != set(SEEDS):
        raise ValueError("frozen PR60 panel must provide both exact control recipes")
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("student cell must be an object")
        cell_id, run_name = cell.get("cell_id"), cell.get("run_name")
        seed, family = cell.get("seed"), cell.get("family")
        if not isinstance(cell_id, str) or not cell_id or cell_id in ids:
            raise ValueError("student cell IDs must be unique strings")
        if (
            not isinstance(run_name, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name)
            or run_name in names
        ):
            raise ValueError("student run names must be unique safe names")
        if type(seed) is not int or seed not in SEEDS or family not in families:
            raise ValueError("student cell family/seed differs from the frozen allocation")
        if not isinstance(cell.get("gpu_uuid"), str) or not re.fullmatch(
            r"GPU-[0-9A-Fa-f-]+", cell["gpu_uuid"]
        ):
            raise ValueError("student cell requires one physical GPU UUID")
        if cell["gpu_uuid"] in uuids_by_family[family]:
            raise ValueError(f"{family}: matched student cells may not share a physical GPU")
        if cell["gpu_uuid"] != controls[seed].get("gpu_uuid"):
            raise ValueError(
                f"{cell_id}: student GPU must remain the same physical PR60 control GPU"
            )
        args = cell.get("train_args")
        if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
            raise ValueError("student train_args must be a frozen string vector")
        source_args = list(controls[seed]["train_args"])
        expected_args = source_args.copy()
        for flag, replacement in (
            ("--name", run_name),
            ("--distill-cache-root", str(families[family]["target_cache"])),
        ):
            index = expected_args.index(flag) + 1
            expected_args[index] = replacement
        if args != expected_args:
            raise ValueError(
                f"{cell_id}: argv differs from exact PR60 control except run/cache identity"
            )
        required = {
            "--seed": str(seed),
            "--train-source": "train",
            "--train-seqs": "10325",
            "--eval-source": "dev",
            "--split-manifest": paths["split_manifest"],
            "--precision": "fp32",
            "--epochs": "2",
            "--checkpoint-policy": "fixed",
            "--batch": "16",
            "--chunk": "2000",
            "--init": "tl3c_fbc1k_s5@ep3",
            "--distill-alpha": ".7",
            "--distill-family-weight": ".1",
        }
        for flag, wanted in required.items():
            if _option(args, flag) != wanted:
                raise ValueError(f"{cell_id}: {flag} differs from the fixed student recipe")
        checkpoint_epoch = [i for i, token in enumerate(args[:-1]) if token == "--checkpoint-epoch"]
        if checkpoint_epoch and _option(args, "--checkpoint-epoch") != "1":
            raise ValueError(f"{cell_id}: fixed epoch-1 endpoint cannot change")
        if (
            "--full-valid" not in args
            or "--save-training-state" not in args
            or "--inherit-init-preprocessing" not in args
        ):
            raise ValueError(
                f"{cell_id}: full-valid, provenance and fit-only initializer reuse are required"
            )
        if type(cell.get("cpu_threads")) is not int or not 1 <= cell["cpu_threads"] <= 8:
            raise ValueError(f"{cell_id}: CPU allocation must be in 1..8")
        affinity = cell.get("cpu_affinity")
        if (
            not isinstance(affinity, list)
            or len(affinity) != cell["cpu_threads"]
            or len(set(affinity)) != len(affinity)
            or any(type(cpu) is not int or not 0 <= cpu <= 47 for cpu in affinity)
        ):
            raise ValueError(f"{cell_id}: an exact disjoint CPU affinity is required")
        ids.add(cell_id)
        names.add(run_name)
        uuids_by_family[family].add(cell["gpu_uuid"])
        affinity_by_family[family].append(set(affinity))
    for family, groups in affinity_by_family.items():
        if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
            raise ValueError(f"{family}: concurrently fitting students need disjoint CPU affinity")
    admission_path = Path(str(panel.get("admission_receipt", "")))
    if not admission_path.is_file() or admission_path.is_symlink():
        raise FileNotFoundError("independent source/data/cache admission receipt is required")
    admission = _read(admission_path)
    if _sha(admission_path) != panel.get("admission_receipt_sha256"):
        raise ValueError("admission receipt SHA differs from the frozen continuation panel")
    if not str(admission.get("status", "")).startswith("approved"):
        raise ValueError("source/data/cache admission has not been approved")
    for key in (
        "source_binding_sha256",
        "data_binding_sha256",
        "split_sha256",
        "fit_members_sha256",
        "baseline_ancestry_sha256",
        "pr60_panel_sha256",
    ):
        if admission.get(key) != identity.get(key):
            raise ValueError(f"admission receipt differs on {key}")
    if admission.get("continuation_panel_sha256") not in (None, file_sha256(path)):
        raise ValueError("admission receipt belongs to another continuation panel")
    return panel, file_sha256(path)


def _deadline(panel: dict[str, Any]) -> int:
    state_path = Path(panel["paths"]["shared_state"])
    state = _read(state_path)
    deadline = state.get("hard_deadline_epoch")
    reserve = state.get("collection_reserve_seconds")
    if not isinstance(deadline, (float, int)) or reserve != COLLECTION_RESERVE_SECONDS:
        raise ValueError("shared teacher deadline/reserve state is invalid")
    left = int(deadline - time.time() - reserve)
    if left <= 0:
        raise TimeoutError("only the reserved collection window remains")
    if deadline - state["first_profile_started_epoch"] != MAX_WAVE_SECONDS:
        raise ValueError("shared deadline differs from the 12-hour frozen wave")
    return int(deadline)


def _remaining(deadline: int) -> int:
    left = int(deadline - time.time() - COLLECTION_RESERVE_SECONDS)
    if left <= 0:
        raise TimeoutError("only the reserved collection window remains")
    return left


def _check_free_floor(panel: dict[str, Any]) -> None:
    free = shutil.disk_usage(Path(panel["paths"]["disk_mount"])).free
    if free < MINIMUM_FREE_BYTES:
        raise OSError(f"data2 free space crossed the {MINIMUM_FREE_BYTES}-byte reserve")


def _teacher_cells(panel: dict[str, Any], family: str) -> list[dict[str, Any]]:
    binding = panel["families"][family]
    panel_path = Path(binding["teacher_panel"])
    if (
        not panel_path.is_file()
        or panel_path.is_symlink()
        or _sha(panel_path) != binding["teacher_panel_sha256"]
    ):
        raise ValueError(f"{family}: teacher panel is absent or changed")
    teacher = _read(panel_path)
    teacher_identity = teacher.get("identity")
    if not isinstance(teacher_identity, dict):
        raise ValueError(f"{family}: teacher panel lacks independent source/data identities")
    runs = list(EXPECTED_RUNS[family])
    actual = teacher.get("cells")
    if not isinstance(actual, list):
        raise ValueError(f"{family}: actual teacher panel has no cells")
    by_run = {cell.get("run_name"): cell for cell in actual if isinstance(cell, dict)}
    if not set(runs).issubset(by_run):
        raise ValueError(f"{family}: teacher panel omits a fixed run")
    rows = []
    for run in runs:
        cell = by_run[run]
        cell_id = str(cell["cell_id"])
        receipt = Path(binding["fit_receipts"][run])
        rows.append(
            {
                "run": run,
                "cell_id": cell_id,
                "receipt": receipt,
                "teacher_panel": panel_path,
                "teacher_panel_sha256": _sha(panel_path),
                "teacher_cell": cell,
                "source_binding_sha256": teacher_identity.get("source_binding_sha256"),
                "data_binding_sha256": teacher_identity.get("data_binding_sha256"),
            }
        )
    return rows


def _wait_teachers(panel: dict[str, Any], family: str, deadline: int) -> list[dict[str, Any]]:
    rows = _teacher_cells(panel, family)
    while True:
        pending = []
        for row in rows:
            receipt_path = row["receipt"]
            if not receipt_path.is_file():
                pending.append(row["run"])
                continue
            receipt = _read(receipt_path)
            if receipt.get("status") in (
                "failed",
                "timed_out",
                "progress_callback_failed",
            ) or receipt.get("exit_code") not in (None, 0):
                raise RuntimeError(
                    f"teacher failed; preserve its receipt and do not retry: {row['run']}"
                )
            if (
                receipt.get("status") == "completed"
                and receipt.get("exit_code") == 0
                and receipt.get("timed_out") is False
            ):
                owner = receipt.get("owner", {})
                cell = row["teacher_cell"]
                owner_panel = owner.get("panel_sha256", owner.get("cohort_panel_sha256"))
                expected_output = str(Path(str(cell.get("fit_output", ""))).resolve())
                if (
                    owner_panel != row["teacher_panel_sha256"]
                    or owner.get("cell_id") != row["cell_id"]
                    or (owner.get("run_name") is not None and owner.get("run_name") != row["run"])
                    or receipt.get("command") != cell.get("fit_argv")
                    or receipt.get("output_dir") != expected_output
                    or (
                        owner.get("source_binding_sha256") is not None
                        and owner.get("source_binding_sha256") != row["source_binding_sha256"]
                    )
                    or (
                        owner.get("data_binding_sha256") is not None
                        and owner.get("data_binding_sha256") != row["data_binding_sha256"]
                    )
                ):
                    raise ValueError(f"teacher receipt ownership differs: {row['run']}")
                continue
            pending.append(row["run"])
        if not pending:
            return rows
        if _remaining(deadline) < 30:
            raise TimeoutError(f"teachers still pending as collection reserve begins: {pending}")
        time.sleep(min(15, max(1, _remaining(deadline) - 15)))


def _runtime(panel: dict[str, Any]) -> Path:
    return Path(panel["paths"]["runtime_root"])


def _output_dir(cell: dict[str, Any], phase: str, project: Path) -> Path:
    run = cell["run_name"]
    if phase == "profile":
        return project / "artifacts" / "runs" / f"{run}_profile_r1"
    if phase == "fit":
        return project / "artifacts" / "runs" / run
    return project / "cache"


def _receipt(panel: dict[str, Any], cell: dict[str, Any], phase: str) -> tuple[Path, Path]:
    runtime = _runtime(panel)
    folder = {
        "profile": "profiles",
        "fit": "fits",
        "student_export": "exports",
        "teacher_export": "exports",
        "stage2": "stage2",
    }[phase]
    return (
        runtime / "logs" / folder / f"{cell['cell_id']}.log",
        runtime / "receipts" / folder / f"{cell['cell_id']}.json",
    )


def _run_owned(
    panel: dict[str, Any],
    panel_hash: str,
    cell: dict[str, Any],
    phase: str,
    command: list[str],
    deadline: int,
    *,
    output_dir: Path,
    gpu_uuid: str | None,
    stage: int,
    source_root: Path | None = None,
) -> dict[str, Any]:
    log_path, receipt_path = _receipt(panel, cell, phase)
    if receipt_path.exists() or log_path.exists():
        raise FileExistsError(f"{cell['cell_id']} {phase}: {STOP_ON_EXISTING}")
    command_source = source_root or Path(panel["paths"]["source_root"])
    env = base_environment(gpu_uuid or "", command_source)
    env.update(
        {
            "WNN_PROJECT_ROOT": str(panel["paths"]["project_root"]),
            "WNN_THREADS": str(cell.get("cpu_threads", 4)),
            "WNN_LOADER_WORKERS": str(min(2, max(1, int(cell.get("cpu_threads", 4)) // 2))),
            "WNN_LOADER_RAM_GB": "8",
            "PYTHONUNBUFFERED": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        }
    )
    affinity = cell.get("cpu_affinity")
    if not isinstance(affinity, list) or any(type(cpu) is not int for cpu in affinity):
        raise ValueError(f"{cell['cell_id']}: invalid CPU affinity")
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required to enforce the frozen 64-core ceiling")
    command = [taskset, "-c", ",".join(map(str, affinity)), *command]
    owner = current_process_owner()
    owner.update(
        {
            "continuation_panel_sha256": panel_hash,
            "panel_sha256": cell.get("teacher_panel_sha256", panel_hash),
            "cell_id": cell["cell_id"],
            "run_name": cell["run_name"],
            "family": cell.get("family"),
            "source_binding_sha256": cell.get(
                "source_binding_sha256", panel["identity"]["source_binding_sha256"]
            ),
            "data_binding_sha256": cell.get(
                "data_binding_sha256", panel["identity"]["data_binding_sha256"]
            ),
            "split_sha256": panel["identity"]["split_sha256"],
            "command_sha256": hashlib.sha256(
                json.dumps(command, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    )
    defaults = {
        "profile": 1_800,
        "fit": 10_800,
        "student_export": 3_600,
        "teacher_export": 36_000,
        "stage2": 36_000,
    }
    timeout = min(
        int(cell.get(f"{phase}_timeout_seconds", defaults.get(phase, 10_800))), _remaining(deadline)
    )
    _check_free_floor(panel)
    if gpu_uuid is None:
        code, timed_out, _ = run_owned_command(
            command,
            cwd=command_source,
            env=env,
            log_path=log_path,
            receipt_path=receipt_path,
            owner=owner,
            stage=stage,
            trial_number=int(cell.get("seed", 0)),
            mode=f"l4tc_{phase}",
            gpu_key="cpu-stage2",
            output_dir=output_dir,
            timeout_seconds=timeout,
        )
    else:
        with hold_gpu_slot(Path(panel["paths"]["control_root"]), gpu_uuid):
            code, timed_out, _ = run_owned_command(
                command,
                cwd=command_source,
                env=env,
                log_path=log_path,
                receipt_path=receipt_path,
                owner=owner,
                stage=stage,
                trial_number=int(cell.get("seed", 0)),
                mode=f"l4tc_{phase}",
                gpu_key=gpu_uuid,
                output_dir=output_dir,
                timeout_seconds=timeout,
            )
    if code != 0 or timed_out:
        raise RuntimeError(
            f"{cell['cell_id']} {phase} failed ({code}, timeout={timed_out}); preserve receipt"
        )
    return _read(receipt_path)


def _run_stage2(
    panel: dict[str, Any], panel_hash: str, family: str, args: list[str], deadline: int, index: int
) -> None:
    paths = panel["paths"]
    run_dir = _runtime(panel) / "logs" / "continuation" / family
    run_dir.mkdir(parents=True, exist_ok=True)
    pseudo = {
        "cell_id": f"{family}_stage2_{index:02d}",
        "run_name": f"{family}_stage2_{index:02d}",
        "seed": index,
        "cpu_threads": 8,
        "cpu_affinity": list(range(32, 40)),
    }
    command = [
        str(paths["python"]),
        str(
            Path(paths["source_root"])
            / "scripts/l4_teacher_capacity_20261104/stream_stage2_targets.py"
        ),
        *args,
    ]
    _run_owned(
        panel,
        panel_hash,
        pseudo,
        "stage2",
        command,
        deadline,
        output_dir=Path(args[args.index("--output-cache") + 1]),
        gpu_uuid=None,
        stage=2,
    )


def _record_progress(panel: dict[str, Any], panel_hash: str, payload: dict[str, Any]) -> None:
    path = _runtime(panel) / "receipts" / "continuation" / "STUDENT_CONTINUATION_START_R1.json"
    if path.exists():
        raise FileExistsError("one-shot continuation state exists; inspect it and do not retry")
    payload.update(
        {"schema": "wnn.l4-teacher-capacity-continuation-state.v1", "panel_sha256": panel_hash}
    )
    atomic_write_json(path, payload, replace=False)


def _teacher_export(
    panel: dict[str, Any],
    panel_hash: str,
    family: str,
    row: dict[str, Any],
    deadline: int,
    index: int,
) -> Path:
    paths = panel["paths"]
    root = Path(paths["project_root"]) / "cache" / "l4_teacher_sequence_exports" / row["run"]
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"existing TF export must be preserved; no automatic retry: {root}")
    export_gpu = panel["families"][family].get("export_gpu_uuid")
    if not isinstance(export_gpu, str) or not export_gpu.startswith("GPU-"):
        raise ValueError("GPU 2 full-training exporter must have a root-verified L4 UUID")
    cell = {
        "cell_id": f"{family}_export_{index:02d}_{row['run']}",
        "run_name": row["run"],
        "seed": index,
        "gpu_uuid": export_gpu,
        "cpu_threads": int(panel["families"][family].get("export_cpu_threads", 8)),
        "cpu_affinity": list(panel["families"][family].get("export_cpu_affinity", range(16, 24))),
        "teacher_export_timeout_seconds": 36_000,
        "source_binding_sha256": row["source_binding_sha256"],
        "data_binding_sha256": row["data_binding_sha256"],
        "teacher_panel_sha256": row["teacher_panel_sha256"],
    }
    teacher_panel = _read(row["teacher_panel"])
    teacher_source = Path(teacher_panel["paths"]["source_root"])
    command = [
        str(teacher_panel["paths"]["python"]),
        str(teacher_source / "scripts/predict_sequence.py"),
        "--checkpoint",
        str(Path(paths["project_root"]) / "artifacts/runs" / row["run"] / "model.pt"),
        "--data-root",
        str(teacher_panel["paths"]["data_root"]),
        "--split-manifest",
        str(paths["split_manifest"]),
        "--output",
        str(root),
        "--membership",
        "fit",
        "--batch-size",
        "32",
    ]
    receipt = _run_owned(
        panel,
        panel_hash,
        cell,
        "teacher_export",
        command,
        deadline,
        output_dir=root,
        gpu_uuid=cell["gpu_uuid"],
        stage=3,
        source_root=teacher_source,
    )
    if receipt.get("exit_code") != 0:
        raise RuntimeError("Transformer fit export did not complete successfully")
    # Validate the fit-only export before the streaming aggregator is allowed to unlink it.
    split = load_campaign_split(Path(paths["split_manifest"]), Path(paths["project_root"]))
    verify_sequence_teacher(
        row["run"], split, Path(paths["project_root"]), root, row["teacher_panel"]
    )
    return root


def _native_gru_export(
    panel: dict[str, Any],
    panel_hash: str,
    family: str,
    row: dict[str, Any],
    deadline: int,
    index: int,
) -> dict[str, Any]:
    """Export and audit one full-index native GRU bank immediately before its add."""
    if family != "wide_gru":
        raise ValueError("native full-index exports are reserved for the wide-GRU family")
    paths = panel["paths"]
    teacher_panel = _read(row["teacher_panel"])
    teacher_source = Path(teacher_panel["paths"]["source_root"])
    teacher_python = str(teacher_panel["paths"]["python"])
    run = row["run"]
    project = Path(paths["project_root"])
    prediction = project / "cache" / f"pred_{run}_train.npy"
    sidecar = prediction.with_suffix(".json")
    if prediction.exists() or prediction.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(
            f"native GRU training prediction exists; preserve it and do not overwrite: {prediction}"
        )
    cell = {
        "cell_id": f"wide_gru_native_export_{index:02d}_{run}",
        "run_name": run,
        "family": family,
        "seed": row["teacher_cell"]["seed"],
        "gpu_uuid": panel["families"][family]["export_gpu_uuid"],
        "cpu_threads": int(panel["families"][family].get("export_cpu_threads", 8)),
        "cpu_affinity": list(panel["families"][family].get("export_cpu_affinity", range(16, 24))),
        "teacher_export_timeout_seconds": 36_000,
        "source_binding_sha256": row["source_binding_sha256"],
        "data_binding_sha256": row["data_binding_sha256"],
        "teacher_panel_sha256": row["teacher_panel_sha256"],
    }
    command = [
        teacher_python,
        str(teacher_source / "scripts/predict_run.py"),
        run,
        "train",
        "--epoch",
        "3",
        "--campaign-split-manifest",
        str(paths["split_manifest"]),
        "--campaign-full-train-index-feature-only",
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]
    _, receipt_path = _receipt(panel, cell, "teacher_export")
    receipt = _run_owned(
        panel,
        panel_hash,
        cell,
        "teacher_export",
        command,
        deadline,
        output_dir=project / "cache",
        gpu_uuid=cell["gpu_uuid"],
        stage=3,
        source_root=teacher_source,
    )
    expected_command = [
        shutil.which("taskset"),
        "-c",
        ",".join(map(str, cell["cpu_affinity"])),
        *command,
    ]
    if (
        receipt.get("status") != "completed"
        or receipt.get("exit_code") != 0
        or receipt.get("timed_out") is not False
        or receipt.get("command") != expected_command
        or receipt.get("gpu_key") != cell["gpu_uuid"]
        or receipt.get("output_dir") != str((project / "cache").resolve())
    ):
        raise ValueError(f"{run}: owned native-export process receipt differs")
    owner = receipt.get("owner", {})
    if (
        owner.get("continuation_panel_sha256") != panel_hash
        or owner.get("panel_sha256") != row["teacher_panel_sha256"]
        or owner.get("cell_id") != cell["cell_id"]
        or owner.get("run_name") != run
        or owner.get("source_binding_sha256") != row["source_binding_sha256"]
        or owner.get("data_binding_sha256") != row["data_binding_sha256"]
    ):
        raise ValueError(f"{run}: native-export owner receipt identity differs")

    from scripts.build_distill_cache import _target_order
    from scripts.l4_teacher_capacity_20261104.stage2_stream_native import verify_native_gru

    split = load_campaign_split(Path(paths["split_manifest"]), project)
    source_order = _target_order(project, "train")
    fit_order = list(map(int, split.train_groups))
    record, values, positions = verify_native_gru(run, split, project, source_order, fit_order)
    del values
    state = _read(state_path(Path(panel["families"][family]["target_cache"])))
    position_sha = hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes()).hexdigest()
    if (
        state.get("schema") != "wnn.l4-teacher-stage2-stream.v2"
        or source_order != state.get("source_train_index_row_groups")
        or record.get("source_train_index_sha256") != state.get("source_train_index_sha256")
        or fit_order != state.get("fit_row_groups")
        or record.get("fit_source_positions_sha256") != position_sha
        or record.get("fit_source_positions_sha256") != state.get("fit_source_positions_sha256")
        or record.get("prediction_row_groups_sha256") != state.get("source_train_index_sha256")
    ):
        raise ValueError(f"{run}: native export does not match the initialized R2 fit mapping")
    return {
        "run_name": run,
        "cell_id": cell["cell_id"],
        "process_receipt_path": str(receipt_path.resolve()),
        "process_receipt_sha256": _sha(receipt_path),
        "prediction_path": record["prediction_path"],
        "prediction_sha256": record["prediction_sha256"],
        "prediction_sidecar_path": str(sidecar.resolve()),
        "prediction_sidecar_sha256": record["prediction_sidecar_sha256"],
        "checkpoint_path": record["checkpoint_path"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "provenance_sha256": record["provenance_sha256"],
        "source_train_index_rows": record["source_train_index_rows"],
        "source_train_index_sha256": record["source_train_index_sha256"],
        "fit_rows": record["fit_rows"],
        "fit_row_groups_sha256": record["fit_row_groups_sha256"],
        "fit_source_positions_sha256": record["fit_source_positions_sha256"],
        "prediction_row_groups_sha256": record["prediction_row_groups_sha256"],
        "campaign_export_scope": "chronological-full-train-index-feature-only",
        "feature_only": True,
        "endpoint_epoch": 3,
        "inference_precision": "fp32",
        "bank_lifecycle": "one export immediately before matching stage2 add; actor removes bank after hashing and aggregation",
    }


def _family_target(
    panel: dict[str, Any], panel_hash: str, family: str, rows: list[dict[str, Any]], deadline: int
) -> Path:
    binding = panel["families"][family]
    teacher_admission_path = Path(binding["teacher_admission_receipt"])
    if (
        not teacher_admission_path.is_file()
        or teacher_admission_path.is_symlink()
        or _sha(teacher_admission_path) != binding["teacher_admission_receipt_sha256"]
    ):
        raise FileNotFoundError(f"{family}: frozen independent teacher/cache admission is required")
    teacher_admission = _read(teacher_admission_path)
    if (
        not str(teacher_admission.get("status", "")).startswith("approved")
        or teacher_admission.get("family") != family
        or teacher_admission.get("teacher_panel_sha256") != _sha(Path(binding["teacher_panel"]))
        or teacher_admission.get("source_binding_sha256") != rows[0]["source_binding_sha256"]
        or teacher_admission.get("data_binding_sha256") != rows[0]["data_binding_sha256"]
        or teacher_admission.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError(f"{family}: independent teacher/cache approval identity is invalid")
    output = Path(binding["target_cache"])
    base_args = [
        "init",
        "--family",
        family,
        "--output-cache",
        str(output),
        "--project-root",
        panel["paths"]["project_root"],
        "--split-manifest",
        panel["paths"]["split_manifest"],
        "--old-cache",
        panel["paths"]["old_cache"],
    ]
    init_receipt_raw = binding.get("stage2_init_receipt")
    if init_receipt_raw is None:
        if partial_root(output).exists():
            raise FileExistsError(
                "stage-2 partial exists without its frozen initialization receipt; preserve it"
            )
        _run_stage2(panel, panel_hash, family, base_args, deadline, 1)
    else:
        init_receipt_path = Path(init_receipt_raw)
        while not init_receipt_path.is_file():
            if _remaining(deadline) < 30:
                raise TimeoutError(f"{family}: root-owned stage-2 initialization is still pending")
            time.sleep(5)
        if _sha(init_receipt_path) != binding.get("stage2_init_receipt_sha256"):
            raise ValueError(f"{family}: preinitialized target receipt SHA changed")
        init_receipt = _read(init_receipt_path)
        state_file = state_path(output)
        if not state_file.is_file() or state_file.is_symlink():
            raise FileNotFoundError(f"{family}: stage-2 initialized stream state is missing")
        state = _read(state_file)
        accumulator = Path(state.get("accumulator", ""))
        fit_groups = state.get("fit_row_groups")
        split = load_campaign_split(
            Path(panel["paths"]["split_manifest"]), Path(panel["paths"]["project_root"])
        )
        if (
            not isinstance(fit_groups, list)
            or len(fit_groups) != 10_325
            or len(set(fit_groups)) != 10_325
            or set(fit_groups) != set(split.train_groups)
            or row_groups_sha256(list(map(int, fit_groups))) != state.get("fit_row_groups_sha256")
        ):
            raise ValueError(
                f"{family}: preinitialized stream must preserve exact full train membership and its actual row order"
            )
        stream_schema = state.get("schema")
        if stream_schema == "wnn.l4-teacher-stage2-stream.v2":
            source_groups = state.get("source_train_index_row_groups")
            source_positions = state.get("fit_source_positions")
            if (
                not isinstance(source_groups, list)
                or len(source_groups) != 10_607
                or len(set(source_groups)) != 10_607
                or row_groups_sha256(list(map(int, source_groups)))
                != state.get("source_train_index_sha256")
                or not isinstance(source_positions, list)
                or len(source_positions) != 10_325
                or any(
                    type(position) is not int or not 0 <= position < len(source_groups)
                    for position in source_positions
                )
            ):
                raise ValueError(f"{family}: R2 source-index to fit-member mapping is incomplete")
            source_position_by_group = {
                group: position for position, group in enumerate(source_groups)
            }
            expected_positions = [source_position_by_group[group] for group in fit_groups]
            positions_sha = hashlib.sha256(
                np.asarray(source_positions, dtype="<i8").tobytes()
            ).hexdigest()
            if (
                source_positions != expected_positions
                or state.get("fit_source_positions_sha256") != positions_sha
            ):
                raise ValueError(
                    f"{family}: R2 fit source-position mapping differs from actual row-group IDs"
                )
        if (
            init_receipt.get("status") not in ("completed", "approved", "complete")
            or init_receipt.get("exit_code") != 0
            or init_receipt.get("family") != family
            or Path(str(init_receipt.get("output_cache", output))).resolve() != output.resolve()
            or stream_schema
            not in (
                "wnn.l4-teacher-stage2-stream.v1",
                "wnn.l4-teacher-stage2-stream.v2",
            )
            or state.get("family") != family
            or state.get("expected_runs") != list(EXPECTED_RUNS[family])
            or state.get("processed_runs") != []
            or state.get("pending_run") is not None
            or Path(state.get("output_cache", "")).resolve() != output.resolve()
            or Path(state.get("project_root", "")).resolve()
            != Path(panel["paths"]["project_root"]).resolve()
            or Path(state.get("split_manifest", "")).resolve()
            != Path(panel["paths"]["split_manifest"]).resolve()
            or state.get("split_manifest_sha256") != SPLIT_SHA256
            or state.get("old_cache_metadata_sha256") != PR60_CACHE_METADATA_SHA256
            or state.get("old_main_sha256") != PR60_MAIN_TRAIN_SHA256
            or Path(state.get("old_cache", "")).resolve()
            != Path(panel["paths"]["old_cache"]).resolve()
            or state.get("target_array_bytes") != TARGET_BYTES
            or not accumulator.is_file()
            or _sha(accumulator) != init_receipt.get("accumulator_sha256")
            or init_receipt.get("zero_filled") is not True
        ):
            raise ValueError(
                f"{family}: preinitialized one-array stream fails its exact empty-state receipt"
            )
        array = np.load(accumulator, mmap_mode="r", allow_pickle=False)
        if array.shape != (10_325, 20_000, 2) or array.dtype != np.float32:
            raise ValueError(f"{family}: preinitialized accumulator schema differs")
        del array
    for index, row in enumerate(rows, start=2):
        if family == "causal_transformer":
            export = _teacher_export(panel, panel_hash, family, row, deadline, index)
            add = [
                "add",
                "--family",
                family,
                "--output-cache",
                str(output),
                "--run",
                row["run"],
                "--sequence-export",
                str(export),
                "--teacher-panel",
                str(row["teacher_panel"]),
            ]
        else:
            artifact = _native_gru_export(panel, panel_hash, family, row, deadline, index)
            binding.setdefault("teacher_export_artifacts", []).append(artifact)
            add = ["add", "--family", family, "--output-cache", str(output), "--run", row["run"]]
        _run_stage2(panel, panel_hash, family, add, deadline, index)
    _run_stage2(
        panel,
        panel_hash,
        family,
        ["finalize", "--family", family, "--output-cache", str(output)],
        deadline,
        len(rows) + 2,
    )
    metadata = output / "metadata.json"
    if not metadata.is_file():
        raise FileNotFoundError("finalized student target cache metadata is missing")
    cache_sha = _sha(metadata)
    cache_review_path = Path(binding["cache_admission_receipt"])
    while not cache_review_path.is_file():
        if _remaining(deadline) < 30:
            raise TimeoutError(
                f"{family}: independent target-cache review did not arrive before reserve"
            )
        time.sleep(10)
    cache_review = _read(cache_review_path)
    if (
        not str(cache_review.get("status", "")).startswith("approved")
        or cache_review.get("family") != family
        or cache_review.get("continuation_panel_sha256") != panel_hash
        or cache_review.get("target_cache_path") != str(output.resolve())
        or cache_review.get("target_metadata_sha256") != cache_sha
        or cache_review.get("source_binding_sha256") != panel["identity"]["source_binding_sha256"]
        or cache_review.get("data_binding_sha256") != panel["identity"]["data_binding_sha256"]
        or cache_review.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError(
            f"{family}: independent finalized target-cache review is not bound to its bytes"
        )
    binding["target_metadata_sha256"] = cache_sha
    return output


def _profile_evidence(path: Path) -> dict[str, Any]:
    lines = [
        line.partition(" ")[2]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("PROFILE_UPDATE_EVIDENCE ")
    ]
    if len(lines) != 1:
        raise ValueError("profile log must contain exactly one update evidence record")
    evidence = json.loads(lines[0])
    expected = {
        "optimizer_attempts": PROFILE_UPDATES,
        "optimizer_completed_updates": PROFILE_UPDATES,
        "amp_skipped_updates": 0,
        "loss_checks": PROFILE_UPDATES,
        "gradient_stats_steps": PROFILE_UPDATES,
        "gradient_stats_nonfinite_steps": 0,
        "loss_finite": True,
        "gradient_finite": True,
        "valid_for_throughput": True,
    }
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise ValueError("profile evidence failed the frozen 680-update finite-gradient gate")
    return evidence


def _compact_valid_bank(
    panel: dict[str, Any], run: str
) -> tuple[Path, Path, dict[int, np.ndarray], dict[int, tuple[int, int]]]:
    """Keep only full-inference predictions at the frozen scored valid positions."""
    import numpy as np
    import pyarrow.parquet as pq

    project = Path(panel["paths"]["project_root"])
    full = project / "cache" / f"pred_{run}_valid.npy"
    full_meta = full.with_suffix(".json")
    if not full.is_file() or not full_meta.is_file():
        raise FileNotFoundError(f"full-valid inference is incomplete for {run}")
    metadata = _read(full_meta)
    groups = expected_row_order(project)
    split = load_campaign_split(Path(panel["paths"]["split_manifest"]), project)
    if (
        metadata.get("run") != run
        or metadata.get("split") != "valid"
        or metadata.get("epoch") != 1
        or metadata.get("split_manifest_sha256") != split.sha256
        or metadata.get("inference_precision") != "fp32"
        or metadata.get("row_groups") != groups
        or metadata.get("prediction_sha256") != _sha(full)
    ):
        raise ValueError(
            f"full-valid student predictions lack exact endpoint, precision or row coverage: {run}"
        )
    compact = full.with_name(full.stem + "_scored.npy")
    temporary = compact.with_name(compact.name + ".partial")
    if compact.exists() or compact.with_suffix(".json").exists() or temporary.exists():
        raise FileExistsError(
            f"compact prediction output exists; preserve it and do not retry: {compact}"
        )
    array = np.load(full, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (1873, 20_000, 2):
        raise ValueError("full-valid transient bank must cover 1873 complete FP32 sequences")
    parquet = pq.ParquetFile(project / "datasets" / "valid.parquet")
    selected: dict[int, np.ndarray] = {}
    offsets: dict[int, tuple[int, int]] = {}
    count, digest = 0, hashlib.sha256()
    mask_bank = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(3_528_273, 2)
    )
    for index, group in enumerate(groups):
        table = parquet.read_row_group(group, columns=["is_scored"], use_threads=False)
        if table.num_rows != 20_000:
            raise ValueError("official validation row group is not one full sequence")
        mask = table["is_scored"].to_numpy().astype(bool, copy=True)
        mask[:99] = False
        positions = np.flatnonzero(mask).astype("<i4")
        selected[group] = positions
        offsets[group] = (count, count + len(positions))
        digest.update(np.asarray(("valid", group), dtype="U16").tobytes())
        digest.update(positions.tobytes())
        mask_bank[count : count + len(positions)] = np.asarray(
            array[index, positions], dtype=np.float32
        )
        count += len(positions)
    mask_bank.flush()
    del mask_bank, array
    if count != 3_528_273:
        raise ValueError("compact full-valid export has an unexpected scored-row count")
    temporary.replace(compact)
    run_dir = project / "artifacts" / "runs" / run
    compact_meta = {
        "schema": "wnn.l4tc-scored-row-prediction.v1",
        "run": run,
        "split": "valid",
        "epoch": 1,
        "checkpoint_file": "model_ep1.pt",
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "run_provenance_sha256": _sha(run_dir / "provenance.json"),
        "prediction_sha256": _sha(compact),
        "source_full_prediction_sha256": metadata["prediction_sha256"],
        "split_manifest_sha256": split.sha256,
        "inference_precision": "fp32",
        "row_groups": groups,
        "row_order_sha256": digest.hexdigest(),
        "scored_rows": count,
        "warmup_excluded_before_row99": True,
        "stored_values": "raw FP32 outputs at exactly is_scored and row>=99 positions",
    }
    compact.with_suffix(".json").write_text(
        json.dumps(compact_meta, indent=2) + "\n", encoding="utf-8"
    )
    full.unlink()
    full_meta.unlink()
    return compact, compact.with_suffix(".json"), selected, offsets


class _CompactScoredSource:
    """Expand one retained scored-row bank in memory for the canonical scorer mask."""

    def __init__(
        self, bank: np.ndarray, selected: dict[int, np.ndarray], offsets: dict[int, tuple[int, int]]
    ) -> None:
        self.bank, self.selected, self.offsets = bank, selected, offsets

    def sequence(self, group: int) -> np.ndarray:
        start, stop = self.offsets[group]
        values = np.asarray(self.bank[start:stop], dtype=np.float32)
        positions = self.selected[group]
        if values.shape != (len(positions), 2) or not np.isfinite(values).all():
            raise ValueError("compact scored-row member has invalid shape or values")
        expanded = np.zeros((20_000, 2), dtype=np.float32)
        expanded[positions] = values
        return expanded


def _train_cell(
    panel: dict[str, Any], panel_hash: str, cell: dict[str, Any], phase: str, deadline: int
) -> dict[str, Any]:
    paths = panel["paths"]
    args = list(cell["train_args"])
    if phase == "profile":
        if "--save-training-state" in args:
            args.remove("--save-training-state")
        args[args.index("--name") + 1] += "_profile_r1"
        args.extend(
            ["--profile-warmup", str(PROFILE_WARMUP), "--profile-steps", str(PROFILE_TIMED)]
        )
    command = [
        str(paths["python"]),
        str(Path(paths["source_root"]) / "scripts/train_rnn.py"),
        *args,
    ]
    output_dir = _output_dir(cell, phase, Path(paths["project_root"]))
    return _run_owned(
        panel,
        panel_hash,
        cell,
        phase,
        command,
        deadline,
        output_dir=output_dir,
        gpu_uuid=cell["gpu_uuid"],
        stage=4 if phase == "profile" else 5,
    )


def _student_family(
    panel: dict[str, Any], panel_hash: str, family: str, deadline: int
) -> dict[str, Any]:
    family_cells = sorted(
        [cell for cell in panel["students"]["cells"] if cell["family"] == family],
        key=lambda cell: cell["seed"],
    )
    binding = panel["families"][family]
    cache = Path(binding["target_cache"])
    if not cache.is_dir() or not (cache / "metadata.json").is_file():
        raise FileNotFoundError(
            f"{family}: completed independently reviewed target cache is required"
        )
    if (
        binding.get("target_metadata_sha256") is not None
        and _sha(cache / "metadata.json") != binding["target_metadata_sha256"]
    ):
        raise ValueError(f"{family}: target cache metadata changed after the admission review")
    if any(
        _option(cell["train_args"], "--distill-cache-root") != str(cache) for cell in family_cells
    ):
        raise ValueError(f"{family}: student argv does not point at its finalized target cache")
    profile_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_train_cell, panel, panel_hash, cell, "profile", deadline)
            for cell in family_cells
        ]
        for future in concurrent.futures.as_completed(futures):
            profile_results.append(future.result())
    for cell in family_cells:
        log, receipt = _receipt(panel, cell, "profile")
        if not receipt.is_file() or _read(receipt).get("exit_code") != 0:
            raise ValueError(f"{cell['cell_id']}: zero-exit profile process receipt is required")
        _profile_evidence(log)
    fit_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_train_cell, panel, panel_hash, cell, "fit", deadline)
            for cell in family_cells
        ]
        for future in concurrent.futures.as_completed(futures):
            fit_results.append(future.result())
    fit_artifacts = []
    for cell in family_cells:
        _, fit_receipt_path = _receipt(panel, cell, "fit")
        fit_receipt = _read(fit_receipt_path)
        checkpoint = (
            Path(panel["paths"]["project_root"])
            / "artifacts"
            / "runs"
            / cell["run_name"]
            / "model_ep1.pt"
        )
        if (
            fit_receipt.get("status") != "completed"
            or fit_receipt.get("exit_code") != 0
            or fit_receipt.get("timed_out") is not False
            or not checkpoint.is_file()
        ):
            raise ValueError(f"{cell['cell_id']}: fixed epoch-1 fit receipt/checkpoint is missing")
        fit_artifacts.append(
            {
                "cell_id": cell["cell_id"],
                "run_name": cell["run_name"],
                "seed": cell["seed"],
                "endpoint_epoch": 1,
                "fit_receipt_path": str(fit_receipt_path.resolve()),
                "fit_receipt_sha256": _sha(fit_receipt_path),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha(checkpoint),
            }
        )
    export_results = []
    for cell in family_cells:
        project = Path(panel["paths"]["project_root"])
        run = cell["run_name"]
        prediction = project / "cache" / f"pred_{run}_valid.npy"
        metadata = prediction.with_suffix(".json")
        if prediction.exists() or metadata.exists():
            raise FileExistsError(f"{cell['cell_id']}: {STOP_ON_EXISTING}")
        command = [
            str(panel["paths"]["python"]),
            str(Path(panel["paths"]["source_root"]) / "scripts/predict_run.py"),
            run,
            "valid",
            "--epoch",
            "1",
            "--campaign-split-manifest",
            panel["paths"]["split_manifest"],
            "--campaign-full-valid-feature-only",
            "--device",
            "cuda",
            "--inference-precision",
            "fp32",
        ]
        export_cell = dict(cell)
        export_cell["cell_id"] = f"{cell['cell_id']}_valid_export"
        _run_owned(
            panel,
            panel_hash,
            export_cell,
            "student_export",
            command,
            deadline,
            output_dir=project / "cache",
            gpu_uuid=binding["export_gpu_uuid"],
            stage=6,
        )
        compact, _, _, _ = _compact_valid_bank(panel, run)
        export_results.append(str(compact))
    grade_path = _grade_family(panel, panel_hash, family, family_cells, deadline)
    grade_receipt_path = grade_path / "grade_receipt.json"
    grade_receipt = _read(grade_receipt_path)
    return {
        "family": family,
        "profiles": profile_results,
        "fits": fit_results,
        "fit_artifacts": fit_artifacts,
        "valid_exports": export_results,
        "grade": str(grade_path),
        "grade_receipt_path": str(grade_receipt_path.resolve()),
        "grade_receipt_sha256": _sha(grade_receipt_path),
        "grade_status": grade_receipt["status"],
        "grade_candidate_rows": grade_receipt["candidate_rows"],
        "target_cache_metadata_sha256": _sha(cache / "metadata.json"),
    }


def _grade_family(
    panel: dict[str, Any], panel_hash: str, family: str, cells: list[dict[str, Any]], deadline: int
) -> Path:
    import numpy as np
    import pyarrow.parquet as pq

    from scripts.heterogeneous_ensemble_contract import SPLIT_SHA256, TL3_WEIGHTS, load_bank
    from scripts.heterogeneous_ensemble_score import (
        blend_predictions,
        ensemble_requirements,
        score_groups,
    )
    from scripts.heterogeneous_galam_probe import scored_row_layout
    from wnn.chronological_source_index import OFFICIAL_SOURCE_PINS
    from wnn.training.chronological_teacher import validate_chronological_teacher_membership

    paths = panel["paths"]
    project = Path(paths["project_root"])
    split_path = Path(paths["split_manifest"])
    split = load_campaign_split(split_path, project)
    if split.sha256 != SPLIT_SHA256 or len(split.evaluation_groups) != 1_873:
        raise ValueError(
            "canonical grade requires the frozen full chronological validation partition"
        )
    old_panel_path = Path(paths["pr60_panel"])
    old_panel = _read(old_panel_path)
    row_order = expected_row_order(project)
    member_records: dict[str, Any] = {}
    for reference in old_panel["reference_predictions"]:
        run = str(reference["run"])
        alias = {
            "tl3c_w18final_distilled_s5": "w18",
            "tl3c_fttall_tlead_s5": "tiedlead_s5",
            "tl3c_fttall_tlead_s6": "tiedlead_s6",
        }.get(run)
        if alias is None:
            raise ValueError(f"unexpected PR60 reference member {run}")
        if float(reference.get("weight", -1)) != TL3_WEIGHTS[alias]:
            raise ValueError(f"PR60 reference weight differs from canonical TL3: {alias}")
        epoch = 1 if alias == "w18" else 3
        prediction = Path(reference["prediction_path"])
        metadata = Path(reference["sidecar_path"])
        root = prediction.parent.parent
        checkpoint = root / "artifacts/runs" / run / reference_checkpoint_filename(metadata, epoch)
        provenance = root / "artifacts/runs" / run / "provenance.json"
        member_records[alias] = collect_member(
            run=run,
            epoch=epoch,
            prediction=prediction,
            metadata_path=metadata,
            checkpoint=checkpoint,
            provenance=provenance,
            row_order=row_order,
            expected_checkpoint_sha256=reference["checkpoint_sha256"],
            expected_prediction_sha256=reference["prediction_sha256"],
        )
    sources: dict[str, Any] = {
        alias: load_bank(record, _runtime(panel), split, row_order)
        for alias, record in member_records.items()
    }
    old_project = Path(old_panel["paths"]["project_root"])
    old_control_cells = {
        int(cell["seed"]): cell
        for cell in old_panel["cells"]
        if isinstance(cell, dict) and cell.get("configuration_id") == "control"
    }
    if set(old_control_cells) != set(SEEDS):
        raise ValueError("frozen PR60 R3 panel must bind both existing matched control seeds")
    for seed in SEEDS:
        control = old_control_cells[seed]
        run = str(control["run_name"])
        prediction = old_project / "cache" / f"pred_{run}_valid.npy"
        control_record = collect_member(
            run=run,
            epoch=1,
            prediction=prediction,
            metadata_path=prediction.with_suffix(".json"),
            checkpoint=old_project / "artifacts/runs" / run / "model_ep1.pt",
            provenance=old_project / "artifacts/runs" / run / "provenance.json",
            row_order=row_order,
        )
        control_name = f"control_s{seed}"
        member_records[control_name] = control_record
        sources[control_name] = load_bank(control_record, _runtime(panel), split, row_order)
    ensembles: dict[str, dict[str, float]] = {"TL3": TL3_WEIGHTS}
    selected, offsets, row_hash, selected_rows = scored_row_layout(
        project / "datasets/valid.parquet", list(split.evaluation_groups)
    )
    for cell in cells:
        run, seed = str(cell["run_name"]), int(cell["seed"])
        name = f"{family}_s{seed}"
        prediction = project / "cache" / f"pred_{run}_valid_scored.npy"
        metadata_path = prediction.with_suffix(".json")
        checkpoint = project / "artifacts/runs" / run / "model_ep1.pt"
        provenance_path = project / "artifacts/runs" / run / "provenance.json"
        metadata = _read(metadata_path)
        checkpoint_sha = _sha(checkpoint)
        provenance = _read(provenance_path)
        validate_chronological_teacher_membership(provenance, split, run)
        bank = np.load(prediction, mmap_mode="r", allow_pickle=False)
        if (
            metadata.get("schema") != "wnn.l4tc-scored-row-prediction.v1"
            or metadata.get("run") != run
            or metadata.get("epoch") != 1
            or metadata.get("checkpoint_sha256") != checkpoint_sha
            or metadata.get("run_provenance_sha256") != _sha(provenance_path)
            or metadata.get("prediction_sha256") != _sha(prediction)
            or metadata.get("split_manifest_sha256") != split.sha256
            or metadata.get("inference_precision") != "fp32"
            or metadata.get("row_groups") != row_order
            or metadata.get("row_order_sha256") != row_hash
            or metadata.get("scored_rows") != selected_rows
            or bank.dtype != np.float32
            or bank.shape != (selected_rows, 2)
            or not np.isfinite(bank).all()
        ):
            raise ValueError(f"compact student endpoint identity, mask or values differ: {run}")
        sources[name] = _CompactScoredSource(bank, selected, offsets)
        member_records[name] = {
            "kind": "compact_scored_rows",
            "run": run,
            "epoch": 1,
            "predictions": {"path": str(prediction.resolve()), "sha256": _sha(prediction)},
            "metadata": {"path": str(metadata_path.resolve()), "sha256": _sha(metadata_path)},
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha},
            "provenance": {"path": str(provenance_path.resolve()), "sha256": _sha(provenance_path)},
            "row_order_sha256": row_hash,
            "scored_rows": selected_rows,
        }
        ensembles[f"TL3_plus_{name}"] = {"TL3": 0.9, name: 0.1}
    valid_path = project / "datasets/valid.parquet"
    manifest = {
        "schema": "wnn.l4tc-fixed-student-blend-inputs.v1",
        "baseline": "TL3",
        "data_root": str(project.resolve()),
        "split": {"path": str(split_path.resolve()), "sha256": SPLIT_SHA256},
        "validation": {
            "path": str(valid_path.resolve()),
            "sha256": OFFICIAL_SOURCE_PINS["valid"].sha256,
        },
        "members": member_records,
        "ensembles": ensembles,
        "assembly": {
            "schema": "wnn.l4tc-fixed-student-blend.v1",
            "family": family,
            "continuation_panel_sha256": panel_hash,
            "fixed_candidate_weight": 0.1,
            "selection": "two predeclared seeds; no validation-fitted coefficients",
        },
    }
    grade_root = _runtime(panel) / "grades" / family
    if grade_root.exists() or grade_root.is_symlink():
        raise FileExistsError(f"grade output exists; no automatic retry: {grade_root}")
    grade_root.mkdir(parents=True)
    manifest_path = grade_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    requirements = ensemble_requirements(ensembles, set(sources))
    if any(requirements[name] - sources.keys() for name in ensembles):
        raise ValueError("fixed blend is missing a canonical TL3 or compact student source")
    scores, member_scores, actual_rows = score_groups(
        pq.ParquetFile(valid_path),
        list(split.evaluation_groups),
        sources,
        ensembles,
        list(ensembles),
    )
    if actual_rows != 3_528_273:
        raise ValueError("canonical scorer observed the wrong official-valid mask size")
    if abs(scores["TL3"]["wp"] - 0.6923774245652725) > 1e-7:
        raise ValueError("canonical TL3 baseline WP differs from its known full-valid result")
    # Compute the canonical training WMSE diagnostic beside WP without fitting weights.
    numerator = {name: np.zeros(2, dtype=np.float64) for name in [*sources, *ensembles]}
    denominator = np.zeros(2, dtype=np.float64)
    parquet = pq.ParquetFile(valid_path)
    for group in split.evaluation_groups:
        table = parquet.read_row_group(group, columns=["t0", "t1", "is_scored"], use_threads=False)
        targets = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")]).astype(
            np.float32
        )
        mask = table["is_scored"].to_numpy().astype(bool, copy=True)
        mask[:99] = False
        clipped = np.clip(targets.astype(np.float64), -2.0, 2.0)
        weights = np.abs(clipped) * mask[:, None]
        denominator += weights.sum(axis=0, dtype=np.float64)
        values = {name: source.sequence(group) for name, source in sources.items()}
        for name in [*sources, *ensembles]:
            prediction = (
                values[name] if name in sources else blend_predictions(name, values, ensembles)
            )
            error = prediction.astype(np.float64) - clipped
            numerator[name] += (weights * error**2).sum(axis=0, dtype=np.float64)
    wmse_by_target = {
        name: {
            "wmse_t0": float(value[0] / denominator[0]),
            "wmse_t1": float(value[1] / denominator[1]),
            "wmse": float(value.sum() / denominator.sum()),
        }
        for name, value in numerator.items()
    }
    baseline = scores["TL3"]
    rows = [
        {
            "name": key,
            "weights": weights,
            "score": scores[key],
            "delta_to_tl3": scores[key]["wp"] - baseline["wp"],
            "wmse": wmse_by_target[key],
            "status": "complete",
            "missing_members": [],
        }
        for key, weights in ensembles.items()
    ]
    report = {
        "schema": "wnn.heterogeneous-ensemble-score.v1",
        "status": "complete",
        "baseline": "TL3",
        "baseline_available": True,
        "input_manifest_sha256": _sha(manifest_path),
        "input_manifest": manifest,
        "split_sha256": split.sha256,
        "valid_sequences": len(split.evaluation_groups),
        "selected_rows": actual_rows,
        "rows": rows,
        "member_scores": member_scores,
        "pending_members": {},
        "member_wmse": {name: wmse_by_target[name] for name in sources},
        "matched_student_deltas": {
            str(seed): {
                "student_member": f"{family}_s{seed}",
                "control_member": f"control_s{seed}",
                "student_wp": member_scores[f"{family}_s{seed}"]["wp"],
                "control_wp": member_scores[f"control_s{seed}"]["wp"],
                "delta_student_minus_control_wp": (
                    member_scores[f"{family}_s{seed}"]["wp"]
                    - member_scores[f"control_s{seed}"]["wp"]
                ),
            }
            for seed in SEEDS
        },
        "exposure": "full official valid is development-exposed; fixed coefficients, no new public/private result",
        "uncertainty": "two matched fitting seeds; no validation-fitted coefficients or new independent holdout",
        "scoring_kernel": "canonical heterogeneous_ensemble_score.score_groups; compact scored-only FP32 student members",
    }
    atomic_write_json(grade_root / "report.json", report, replace=False)
    binding = panel["families"][family]
    target_metadata = _read(Path(binding["target_cache"]) / "metadata.json")
    cache_review = Path(binding["cache_admission_receipt"])
    teacher_rows = _teacher_cells(panel, family)
    teacher_checkpoints = []
    for row in teacher_rows:
        checkpoint_name = "model_ep3.pt" if family == "wide_gru" else "model.pt"
        checkpoint = project / "artifacts" / "runs" / row["run"] / checkpoint_name
        teacher_checkpoints.append(
            {
                "run_name": row["run"],
                "fit_receipt_path": str(row["receipt"].resolve()),
                "fit_receipt_sha256": _sha(row["receipt"]),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha(checkpoint),
                "endpoint_epoch": 3,
            }
        )
    fit_artifacts = []
    for cell in cells:
        _, receipt_path = _receipt(panel, cell, "fit")
        checkpoint = project / "artifacts" / "runs" / cell["run_name"] / "model_ep1.pt"
        fit_artifacts.append(
            {
                "cell_id": cell["cell_id"],
                "run_name": cell["run_name"],
                "seed": cell["seed"],
                "fit_receipt_path": str(receipt_path.resolve()),
                "fit_receipt_sha256": _sha(receipt_path),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha(checkpoint),
                "endpoint_epoch": 1,
            }
        )
    report_path = grade_root / "report.json"
    atomic_write_json(
        grade_root / "grade_receipt.json",
        {
            "schema": "wnn.l4tc-fixed-student-grade.v1",
            "family": family,
            "status": report["status"],
            "panel_sha256": panel_hash,
            "admission_receipt_sha256": panel["admission_receipt_sha256"],
            "teacher_admission_receipt_sha256": binding["teacher_admission_receipt_sha256"],
            "cache_admission_receipt_path": str(cache_review.resolve()),
            "cache_admission_receipt_sha256": _sha(cache_review),
            "target_cache_path": str(Path(binding["target_cache"]).resolve()),
            "target_cache_metadata_sha256": _sha(Path(binding["target_cache"]) / "metadata.json"),
            "target_contract": target_metadata.get("target_contract", {}),
            "target_teacher_records": target_metadata.get("target_contract", {}).get(
                "teachers", []
            ),
            "teacher_fit_artifacts": teacher_checkpoints,
            "teacher_export_artifacts": binding.get("teacher_export_artifacts", []),
            "student_fit_artifacts": fit_artifacts,
            "endpoint_epoch": 1,
            "fixed_weights": {"TL3": 0.9, "candidate": 0.1},
            "canonical_score_kernel": "heterogeneous_ensemble_score.score_groups",
            "selected_rows": report["selected_rows"],
            "valid_sequences": report["valid_sequences"],
            "candidate_rows": [
                row for row in report["rows"] if row["name"].startswith("TL3_plus_")
            ],
            "student_member_rows": {
                str(seed): {
                    "score": member_scores[f"{family}_s{seed}"],
                    "wmse": wmse_by_target[f"{family}_s{seed}"],
                }
                for seed in SEEDS
            },
            "matched_student_deltas": report["matched_student_deltas"],
            "baseline_row": next(row for row in report["rows"] if row["name"] == "TL3"),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha(manifest_path),
            "report_path": str(report_path.resolve()),
            "report_sha256": _sha(report_path),
            "compact_prediction_artifacts": [
                {
                    "path": str(
                        (project / "cache" / f"pred_{cell['run_name']}_valid_scored.npy").resolve()
                    ),
                    "sha256": _sha(project / "cache" / f"pred_{cell['run_name']}_valid_scored.npy"),
                }
                for cell in cells
            ],
        },
        replace=False,
    )
    return grade_root


def run(panel_path: Path) -> dict[str, Any]:
    """Run wide-GRU then causal-Transformer queues; fail closed and never retry."""
    panel, panel_hash = validate_panel(panel_path)
    deadline = _deadline(panel)
    progress_path = (
        _runtime(panel) / "receipts" / "continuation" / "STUDENT_CONTINUATION_START_R1.json"
    )
    if progress_path.exists():
        raise FileExistsError(
            "one-shot continuation already started; inspect receipts, do not restart"
        )
    _record_progress(
        panel,
        panel_hash,
        {
            "status": "running",
            "first_started_epoch": time.time(),
            "hard_deadline_epoch": deadline,
            "source_binding_sha256": panel["identity"]["source_binding_sha256"],
            "data_binding_sha256": panel["identity"]["data_binding_sha256"],
            "split_sha256": panel["identity"]["split_sha256"],
            "families": [],
        },
    )
    results = []
    current_family = None
    current_phase = "teacher_wait"
    try:
        for family in ("wide_gru", "causal_transformer"):
            current_family = family
            current_phase = "teacher_wait"
            teacher_rows = _wait_teachers(panel, family, deadline)
            current_phase = "stage2_target"
            output = _family_target(panel, panel_hash, family, teacher_rows, deadline)
            current_phase = "student_profile_fit_export_grade"
            result = _student_family(panel, panel_hash, family, deadline)
            results.append({"target_cache": str(output), **result})
            family_receipt = (
                _runtime(panel) / "receipts" / "continuation" / f"{family}_COMPLETE_R1.json"
            )
            atomic_write_json(
                family_receipt,
                {
                    "schema": "wnn.l4tc-family-continuation.v1",
                    "panel_sha256": panel_hash,
                    **results[-1],
                },
                replace=False,
            )
    except Exception as error:
        failure_path = (
            _runtime(panel) / "receipts" / "continuation" / "STUDENT_CONTINUATION_FAILED_R1.json"
        )
        if not failure_path.exists():
            atomic_write_json(
                failure_path,
                {
                    "schema": "wnn.l4-teacher-capacity-continuation-failure.v1",
                    "status": "failed",
                    "panel_sha256": panel_hash,
                    "family": current_family,
                    "phase": current_phase,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_epoch": time.time(),
                    "source_binding_sha256": panel["identity"]["source_binding_sha256"],
                    "data_binding_sha256": panel["identity"]["data_binding_sha256"],
                    "completed_families": results,
                },
                replace=False,
            )
        raise
    final = {
        "status": "complete",
        "finished_epoch": time.time(),
        "hard_deadline_epoch": deadline,
        "source_binding_sha256": panel["identity"]["source_binding_sha256"],
        "data_binding_sha256": panel["identity"]["data_binding_sha256"],
        "split_sha256": panel["identity"]["split_sha256"],
        "baseline_ancestry_sha256": panel["identity"]["baseline_ancestry_sha256"],
        "families": results,
        "admission_receipt_path": panel["admission_receipt"],
        "admission_receipt_sha256": panel["admission_receipt_sha256"],
        "checkpoint_endpoint_epoch": 1,
        "canonical_fixed_weights": {"TL3": 0.9, "candidate": 0.1},
        "all_family_grades_complete": all(
            item.get("grade_status") == "complete" for item in results
        ),
    }
    atomic_write_json(
        _runtime(panel) / "receipts" / "continuation" / "STUDENT_CONTINUATION_COMPLETE_R1.json",
        {
            "schema": "wnn.l4-teacher-capacity-continuation-state.v1",
            "panel_sha256": panel_hash,
            **final,
        },
        replace=False,
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument(
        "--execute", action="store_true", help="run the frozen one-shot continuation"
    )
    args = parser.parse_args()
    panel, digest = validate_panel(args.panel)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "validated_no_launch",
                    "panel_sha256": digest,
                    "teacher_families": list(EXPECTED_RUNS),
                }
            )
        )
        return
    print(json.dumps(run(args.panel), allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        sys.stderr.write(f"L4 student continuation stopped: {type(error).__name__}: {error}\n")
        raise SystemExit(2) from error

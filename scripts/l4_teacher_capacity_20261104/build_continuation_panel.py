"""Build and locally validate the frozen four-student continuation panel.

This is a plan builder only. It never invokes a trainer, exporter or stage-two
command. The actual PR60 controls supply the two student argv vectors and GPU
UUIDs; actual teacher panels supply teacher run, cell and output identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from scripts.l4_teacher_capacity_20261104.continue_students import (  # noqa: E402
    PANEL_SCHEMA,
    SPLIT_SHA256,
    validate_panel,
)
from scripts.l4_teacher_capacity_20261104.stage2_stream_common import (  # noqa: E402
    EXPECTED_RUNS,
)
from wnn.splits import file_sha256  # noqa: E402

RUN_SUFFIX = "20261104"
FAMILIES = ("wide_gru", "causal_transformer")
HASH_KEYS = (
    "source_binding_sha256",
    "data_binding_sha256",
    "split_sha256",
    "fit_members_sha256",
    "baseline_ancestry_sha256",
    "pr60_panel_sha256",
)


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 << 20:
        raise ValueError(f"expected a regular JSON file no larger than 32 MiB: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def absolute(path: Path, label: str, *, must_exist: bool = True) -> Path:
    value = path.expanduser().absolute()
    if must_exist and not value.exists():
        raise FileNotFoundError(f"{label} does not exist: {value}")
    if value.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {value}")
    return value


def python_launcher(path: Path, qualified_path: Path) -> tuple[Path, dict[str, str]]:
    """Allow only the exact qualified virtualenv launcher to be a symlink."""
    value = path.expanduser().absolute()
    qualified = qualified_path.expanduser().absolute()
    if value != qualified:
        raise ValueError(
            f"Python launcher must equal the PR60-qualified interpreter path: {qualified}"
        )
    if not value.exists():
        raise FileNotFoundError(f"qualified Python launcher is missing: {value}")
    resolved = value.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"qualified Python target must be a regular executable file: {resolved}")
    if value.is_symlink() and not value.is_file():
        raise ValueError(f"qualified Python symlink does not resolve to a regular file: {value}")
    return value, {
        "path": str(value),
        "resolved_target": str(resolved),
        "resolved_target_sha256": file_sha256(resolved),
    }


def digest(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def replace_arg(args: list[str], flag: str, value: str) -> None:
    indices = [i for i, arg in enumerate(args) if arg == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(args):
        raise ValueError(f"PR60 control argv must contain exactly one {flag} VALUE")
    args[indices[0] + 1] = value


def teacher_binding(
    *,
    family: str,
    panel_path: Path,
    admission_path: Path,
    target_cache: Path,
    cache_review_path: Path,
    runtime_root: Path,
    export_gpu_uuid: str,
    receipt_overrides: dict[str, str],
    stage2_init_receipt: Path | None,
) -> dict[str, Any]:
    panel_path = absolute(panel_path, f"{family} teacher panel")
    panel = read_json(panel_path)
    identity = panel.get("identity")
    cells = panel.get("cells")
    if not isinstance(identity, dict) or not isinstance(cells, list):
        raise ValueError(f"{family}: actual teacher panel needs identity and cells")
    for key in ("source_binding_sha256", "data_binding_sha256"):
        digest(identity.get(key), f"{family} teacher identity.{key}")
    digest(identity.get("split_sha256"), f"{family} teacher identity.split_sha256")
    if identity["split_sha256"] != SPLIT_SHA256:
        raise ValueError(f"{family}: teacher panel uses another chronological split")
    by_run: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get("run_name"), str):
            continue
        run = cell["run_name"]
        if run in by_run:
            raise ValueError(f"{family}: duplicate teacher run name {run}")
        by_run[run] = cell
    expected = list(EXPECTED_RUNS[family])
    if not set(expected).issubset(by_run):
        raise ValueError(f"{family}: teacher panel does not bind exact fixed runs {expected}")
    fit_receipts: dict[str, str] = {}
    for run in expected:
        cell = by_run[run]
        if (
            not isinstance(cell.get("cell_id"), str)
            or not isinstance(cell.get("fit_output"), str)
            or not isinstance(cell.get("fit_argv"), list)
            or not cell["fit_argv"]
        ):
            raise ValueError(f"{run}: actual teacher cell lacks frozen id/argv/output")
        seed_match = re.search(r"_s(1001|1002)_", run)
        if seed_match is None or int(cell.get("seed", -1)) != int(seed_match.group(1)):
            raise ValueError(f"{run}: teacher run/cell seed identity differs")
        if run in receipt_overrides:
            receipt = absolute(Path(receipt_overrides[run]), f"{run} fit receipt", must_exist=False)
        elif family == "causal_transformer":
            receipt = runtime_root / "receipts" / "fits" / f"tf_s{seed_match.group(1)}.json"
        else:
            receipt = runtime_root / "receipts" / "fits" / f"{cell['cell_id']}.json"
        fit_receipts[run] = str(receipt)
    admission_path = absolute(admission_path, f"{family} teacher admission receipt")
    admission = read_json(admission_path)
    if not str(admission.get("status", "")).startswith("approved"):
        raise ValueError(f"{family}: teacher admission is not approved")
    if (
        admission.get("family") != family
        or admission.get("teacher_panel_sha256") != file_sha256(panel_path)
        or admission.get("source_binding_sha256") != identity["source_binding_sha256"]
        or admission.get("data_binding_sha256") != identity["data_binding_sha256"]
        or admission.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError(f"{family}: teacher admission receipt does not bind this panel")
    if not re.fullmatch(r"GPU-[0-9A-Fa-f-]+", export_gpu_uuid):
        raise ValueError("a root-verified physical export GPU UUID is required")
    result: dict[str, Any] = {
        "teacher_panel": str(panel_path),
        "teacher_panel_sha256": file_sha256(panel_path),
        "fit_receipts": fit_receipts,
        "teacher_admission_receipt": str(admission_path),
        "teacher_admission_receipt_sha256": file_sha256(admission_path),
        "target_cache": str(absolute(target_cache, f"{family} target cache", must_exist=False)),
        "cache_admission_receipt": str(
            absolute(cache_review_path, f"{family} future cache review", must_exist=False)
        ),
        "export_gpu_uuid": export_gpu_uuid,
        "export_cpu_threads": 8,
        "export_cpu_affinity": list(range(16, 24)),
    }
    if stage2_init_receipt is not None:
        receipt_path = absolute(stage2_init_receipt, f"{family} stage2 init receipt")
        receipt = read_json(receipt_path)
        if (
            receipt.get("status") not in ("completed", "approved", "complete")
            or receipt.get("exit_code") != 0
            or Path(str(receipt.get("output_cache", ""))).resolve()
            != Path(result["target_cache"]).resolve()
            or receipt.get("family") != family
        ):
            raise ValueError(f"{family}: stage2 init receipt is not successful/bound")
        result["stage2_init_receipt"] = str(receipt_path)
        result["stage2_init_receipt_sha256"] = file_sha256(receipt_path)
    return result


def build(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    pr60_path = absolute(args.pr60_panel, "PR60 control panel")
    pr60 = read_json(pr60_path)
    if pr60.get("schema") != "wnn.w18-l4-cohort.v1":
        raise ValueError("the source panel must be the frozen PR60 W18 cohort")
    control_cells = {
        int(cell["seed"]): cell
        for cell in pr60.get("cells", [])
        if isinstance(cell, dict) and cell.get("configuration_id") == "control"
    }
    if set(control_cells) != {1001, 1002}:
        raise ValueError("PR60 panel must contain the exact two matched controls")
    old_identity = pr60.get("identity")
    if not isinstance(old_identity, dict):
        raise ValueError("PR60 panel is missing frozen ancestry identity")

    runtime = absolute(args.runtime_root, "continuation runtime")
    project = absolute(args.project_root or (runtime / "project"), "project root")
    data_root = absolute(args.data_root or project, "data root")
    source_root = absolute(args.source_root, "delivery source root")
    python, python_binding = python_launcher(
        args.python or Path(pr60["paths"]["python"]), Path(pr60["paths"]["python"])
    )
    split = absolute(args.split_manifest or Path(pr60["paths"]["split_manifest"]), "split manifest")
    control_root = absolute(args.control_root or (runtime / "control"), "control root")
    shared_state = absolute(args.shared_state or (runtime / "GRU_STATE_R1.json"), "shared state")
    disk_mount = absolute(args.disk_mount, "disk mount")
    old_cache = absolute(args.old_cache, "qualified PR60 target cache")
    if file_sha256(split) != SPLIT_SHA256:
        raise ValueError("receiving official chronological split bytes differ from the frozen SHA")

    admission_path = absolute(args.admission_receipt, "continuation admission receipt")
    admission = read_json(admission_path)
    if not str(admission.get("status", "")).startswith("approved"):
        raise ValueError("continuation admission receipt is not approved")
    identity = {
        key: digest(admission.get(key), f"continuation admission.{key}")
        for key in HASH_KEYS
        if key != "pr60_panel_sha256"
    }
    identity["pr60_panel_sha256"] = file_sha256(pr60_path)
    identity["split_sha256"] = SPLIT_SHA256
    for key in ("fit_members_sha256", "baseline_ancestry_sha256"):
        if old_identity.get(key) != identity[key]:
            raise ValueError(f"continuation admission and PR60 panel differ on {key}")
    for key in HASH_KEYS:
        if admission.get(key) != identity[key]:
            raise ValueError(f"continuation admission receipt does not bind {key}")

    receipt_overrides: dict[str, str] = {}
    if args.fit_receipt_map is not None:
        value = read_json(absolute(args.fit_receipt_map, "teacher fit receipt map"))
        receipt_overrides = value.get("fit_receipts", value)
        if not isinstance(receipt_overrides, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in receipt_overrides.items()
        ):
            raise ValueError("fit receipt map must map run names to absolute paths")

    family_paths = {
        "wide_gru": (
            args.gru_panel,
            args.gru_teacher_admission,
            args.gru_target_cache,
            args.gru_cache_review,
            args.stage2_init_wide,
        ),
        "causal_transformer": (
            args.transformer_panel,
            args.transformer_teacher_admission,
            args.transformer_target_cache,
            args.transformer_cache_review,
            args.stage2_init_transformer,
        ),
    }
    families = {
        family: teacher_binding(
            family=family,
            panel_path=panel_path,
            admission_path=teacher_admission,
            target_cache=target_cache,
            cache_review_path=cache_review,
            runtime_root=runtime,
            export_gpu_uuid=args.export_gpu_uuid,
            receipt_overrides=receipt_overrides,
            stage2_init_receipt=init_receipt,
        )
        for family, (
            panel_path,
            teacher_admission,
            target_cache,
            cache_review,
            init_receipt,
        ) in family_paths.items()
    }

    students = []
    for family in FAMILIES:
        target_cache = families[family]["target_cache"]
        for index, seed in enumerate((1001, 1002)):
            control = control_cells[seed]
            run = f"l4tc_student_{family}_s{seed}_{RUN_SUFFIX}"
            train_args = list(control["train_args"])
            replace_arg(train_args, "--name", run)
            replace_arg(train_args, "--distill-cache-root", target_cache)
            students.append(
                {
                    "family": family,
                    "seed": seed,
                    "cell_id": f"student_{family}_s{seed}",
                    "run_name": run,
                    "gpu_uuid": control["gpu_uuid"],
                    "cpu_threads": 8,
                    "cpu_affinity": list(range(index * 8, (index + 1) * 8)),
                    "profile_timeout_seconds": 1_800,
                    "fit_timeout_seconds": 10_800,
                    "student_export_timeout_seconds": 3_600,
                    "train_args": train_args,
                }
            )

    panel: dict[str, Any] = {
        "schema": PANEL_SCHEMA,
        "status": "frozen",
        "paths": {
            "source_root": str(source_root),
            "project_root": str(project),
            "data_root": str(data_root),
            "python": str(python),
            "split_manifest": str(split),
            "runtime_root": str(runtime),
            "control_root": str(control_root),
            "disk_mount": str(disk_mount),
            "shared_state": str(shared_state),
            "pr60_panel": str(pr60_path),
            "old_cache": str(old_cache),
        },
        "identity": identity,
        "python_launcher": python_binding,
        "families": families,
        "students": {"cells": students},
        "admission_receipt": str(admission_path),
        "admission_receipt_sha256": file_sha256(admission_path),
    }
    return panel, hashlib.sha256(
        (json.dumps(panel, indent=2, allow_nan=False) + "\n").encode()
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pr60-panel", required=True, type=Path)
    parser.add_argument("--gru-panel", required=True, type=Path)
    parser.add_argument("--transformer-panel", required=True, type=Path)
    parser.add_argument("--admission-receipt", required=True, type=Path)
    parser.add_argument("--gru-teacher-admission", required=True, type=Path)
    parser.add_argument("--transformer-teacher-admission", required=True, type=Path)
    parser.add_argument("--gru-target-cache", required=True, type=Path)
    parser.add_argument("--transformer-target-cache", required=True, type=Path)
    parser.add_argument("--gru-cache-review", required=True, type=Path)
    parser.add_argument("--transformer-cache-review", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--export-gpu-uuid", required=True)
    parser.add_argument("--old-cache", required=True, type=Path)
    parser.add_argument("--disk-mount", type=Path, default=Path("/data2"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--shared-state", type=Path)
    parser.add_argument("--fit-receipt-map", type=Path)
    parser.add_argument("--stage2-init-wide", type=Path)
    parser.add_argument("--stage2-init-transformer", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"preserve existing continuation panel: {output}")
    panel, panel_sha = build(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"preserve existing temporary file: {temporary}")
    temporary.write_text(json.dumps(panel, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        parsed, validated_sha = validate_panel(temporary)
        if validated_sha != panel_sha or parsed != panel:
            raise ValueError("continuation manifest bytes changed during validation")
        temporary.replace(output)
    except BaseException:
        # Preserve any partial/failed evidence rather than silently deleting it.
        raise
    print(
        json.dumps({"status": "frozen_no_launch", "panel": str(output), "panel_sha256": panel_sha})
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"Continuation panel builder stopped: {type(error).__name__}: {error}\n")
        raise SystemExit(2) from error

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]

from scripts.l4_teacher_capacity_20261104 import runner  # noqa: E402
from wnn.multistage_search import atomic_write_json  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402

IDENTITY_KEYS = [
    "source_binding_sha256",
    "data_binding_sha256",
    "split_sha256",
    "fit_members_sha256",
]
CELL_COUNT = 4
EXTERNAL_RESERVE_BYTES = 500_000_000


def _read(path: Path) -> dict[str, object]:
    """Read a bounded regular JSON object."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 << 20:
        raise ValueError(f"expected regular JSON no larger than 16 MiB: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, object], value)


def validate_phase(
    panel_path: Path,
    authority_path: Path,
    admission_path: Path,
    phase: str,
    review_path: Path | None,
) -> tuple[dict[str, object], dict[str, object], str, str, str, str | None]:
    """Verify root authority and source identity before dispatching any cell."""
    panel, authority, admission = map(_read, (panel_path, authority_path, admission_path))
    panel_sha, authority_sha, admission_sha = map(
        file_sha256, (panel_path, authority_path, admission_path)
    )
    if (
        panel.get("schema") != "wnn.l4-objective-panel.v1"
        or authority.get("schema") != "wnn.root-authority.v1"
    ):
        raise ValueError("objective panel or root-authority schema differs")
    bounds = (
        ("max_new_scientific_fits", 6),
        ("collection_reserve_seconds", 2700),
        ("max_new_output_bytes", 2_500_000_000),
        ("minimum_free_data2_bytes", 3_000_000_000),
        ("reserved_external_output_bytes", EXTERNAL_RESERVE_BYTES),
        ("max_aggregate_cpu_cores", 64),
        ("new_fit_gpu_indices", [2, 3, 4, 5]),
    )
    if any(authority.get(key) != value for key, value in bounds):
        raise ValueError("root-authority resource bounds differ")
    deadline = authority.get("hard_deadline_epoch")
    if type(deadline) not in (int, float) or time.time() + 2700 >= deadline:
        raise ValueError("root-authority deadline leaves no collection reserve")
    identity, cells, paths = panel.get("identity"), panel.get("cells"), panel.get("paths")
    if not isinstance(identity, dict) or not isinstance(cells, list) or len(cells) != CELL_COUNT:
        raise ValueError("panel must bind identity and exactly four cells")
    if not isinstance(paths, dict) or not all(isinstance(cell, dict) for cell in cells):
        raise ValueError("panel paths/cells are malformed")
    identity = cast(dict[str, object], identity)
    if any(
        not isinstance(identity.get(key), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get(key)))
        for key in IDENTITY_KEYS
    ):
        raise ValueError("panel identity must bind four lowercase SHA256 digests")
    if (
        admission.get("schema") != "wnn.l4-objective-admission.v1"
        or admission.get("status") != "approved"
        or admission.get("authority_sha256") != authority_sha
        or admission.get("panel_sha256") != panel_sha
        or any(admission.get(key) != identity[key] for key in IDENTITY_KEYS)
    ):
        raise ValueError("admission does not bind authority, panel and fitting identities")
    cells = cast(list[dict[str, object]], cells)
    ids, slots = [c.get("cell_id") for c in cells], [c.get("gpu_slot") for c in cells]
    uuids, threads = [c.get("gpu_uuid") for c in cells], [c.get("cpu_threads") for c in cells]
    if (
        len(set(ids)) != CELL_COUNT
        or set(slots) != {2, 3, 4, 5}
        or any(not isinstance(uuid, str) for uuid in uuids)
        or len(set(uuids)) != CELL_COUNT
    ):
        raise ValueError("four unique cells must use distinct GPUs in admitted slots 2-5")
    if any(type(value) is not int or value <= 0 for value in threads) or sum(threads) > 64:
        raise ValueError("objective CPU threads exceed the aggregate 64-core cap")
    if any(
        any(
            not isinstance(c.get(k), list) or not c[k] or not all(isinstance(a, str) for a in c[k])
            for k in ("profile_argv", "fit_argv")
        )
        for c in cells
    ):
        raise ValueError("each cell needs nonempty string argv for profile and fit")
    budget = panel.get("budget")
    if (
        not isinstance(budget, dict)
        or type(budget.get("max_output_bytes")) is not int
        or not 0 < budget["max_output_bytes"] <= 2_500_000_000
    ):
        raise ValueError("panel output budget is malformed or exceeds root authority")
    panel["_panel_path"] = str(panel_path.resolve())
    budget.update(minimum_free_bytes=3_500_000_000, collection_reserve_seconds=2700)
    budget["max_output_bytes"] = min(budget["max_output_bytes"], 2_500_000_000)
    if (
        shutil.disk_usage(Path(str(paths.get("disk_mount", "")))).free
        < budget["minimum_free_bytes"]
    ):
        raise RuntimeError("physical free space is below 3 GB plus external 0.5 GB reserve")
    review_sha = None
    if phase == "fit":
        if review_path is None:
            raise ValueError("fit phase requires an independent --profile-review receipt")
        review, review_sha = _read(review_path), file_sha256(review_path)
        if (
            review.get("schema") != "wnn.l4-objective-profile-review.v1"
            or review.get("status") != "approved"
            or review.get("panel_sha256") != panel_sha
            or review.get("admission_receipt_sha256") != admission_sha
            or any(review.get(key) != identity[key] for key in IDENTITY_KEYS)
        ):
            raise ValueError("profile review does not bind this panel, admission and source")
        receipts = review.get("profile_receipts")
        if not isinstance(receipts, dict) or set(receipts) != set(ids):
            raise ValueError("profile review must bind all four cell receipts")
        for cell in cells:
            receipt_path = runner.panel_path(panel, cell, "profile")["receipt"]
            receipt = _read(receipt_path)
            owner = receipt.get("owner")
            if (
                file_sha256(receipt_path) != receipts[cell["cell_id"]]
                or receipt.get("status") != "completed"
                or receipt.get("exit_code") != 0
                or receipt.get("timed_out") is not False
                or not isinstance(owner, dict)
                or owner.get("cell_id") != cell["cell_id"]
                or owner.get("panel_sha256") != panel_sha
            ):
                raise ValueError(f"{cell['cell_id']}: changed or unsuccessful profile receipt")
    elif phase != "profile" or review_path is not None:
        raise ValueError("profile phase accepts no review; choose profile or fit")
    return panel, authority, panel_sha, authority_sha, admission_sha, review_sha


def run_phase(
    panel: dict[str, object],
    authority: dict[str, object],
    phase: str,
    panel_sha: str,
    authority_sha: str,
    admission_sha: str,
    review_sha: str | None,
) -> dict[str, object]:
    """Dispatch four cells concurrently and preserve accurate phase counts."""
    paths, identity = (
        cast(dict[str, object], panel["paths"]),
        cast(dict[str, object], panel["identity"]),
    )
    cells = cast(list[dict[str, object]], panel["cells"])
    summary_path = (
        Path(str(paths["runtime_root"]))
        / "receipts/objective"
        / f"OBJECTIVE_{phase.upper()}_R1.json"
    )
    if summary_path.exists() or summary_path.is_symlink():
        raise FileExistsError(f"preserve existing objective phase summary: {summary_path}")
    state = {
        "hard_deadline_epoch": authority["hard_deadline_epoch"],
        "collection_reserve_seconds": 2700,
        "admission_receipt_sha256": admission_sha,
        "source_binding_sha256": identity["source_binding_sha256"],
        "data_binding_sha256": identity["data_binding_sha256"],
        "fit_members_sha256": identity["fit_members_sha256"],
        "profile_review_sha256": review_sha,
    }
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CELL_COUNT) as pool:
        futures = [
            pool.submit(runner.run_cell, panel, cell, phase, state, i)
            for i, cell in enumerate(cells, 1)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
    summary = {
        "schema": "wnn.l4-objective-phase.v1",
        "status": "failed" if errors else "complete",
        "phase": phase,
        "panel_sha256": panel_sha,
        "authority_sha256": authority_sha,
        "admission_receipt_sha256": admission_sha,
        "profile_review_sha256": review_sha,
        "identity": identity,
        "expected_cells": CELL_COUNT,
        "attempted_cells": len(futures),
        "completed_cells": len(results),
        "failed_cells": len(errors),
        "results": sorted(results, key=lambda row: str(row["cell_id"])),
        "errors": errors,
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(summary_path, summary, replace=False)
    if errors:
        raise RuntimeError("objective phase failed; per-cell receipts and summary were preserved")
    return summary


def main() -> int:
    """Run the admitted four-cell profile or fit phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("panel", "authority", "admission"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("profile", "fit"))
    parser.add_argument("--profile-review", type=Path)
    args = parser.parse_args()
    values = validate_phase(
        args.panel, args.authority, args.admission, args.phase, args.profile_review
    )
    run_phase(*values[:2], args.phase, *values[2:5], values[5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

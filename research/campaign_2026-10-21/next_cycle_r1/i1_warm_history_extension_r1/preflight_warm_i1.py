from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import cast

from warm_i1_order_amendment import verify_f1_order_expectation
from warm_i1_preflight_amendment import verify_f1_source_amendment


def _sha256(path: Path) -> str:
    """Hash source and metadata files with a fixed-size buffer."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_plan(path: Path) -> dict[str, object]:
    """Read the frozen plan under a small explicit size limit."""
    size = path.stat().st_size
    if size <= 0 or size > 200_000:
        raise ValueError(f"frozen warm-history plan has unsupported size: {size}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError("frozen plan changed while it was being read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("frozen plan must be a JSON object")
    return cast(dict[str, object], value)


def _verify_source(plan: dict[str, object], source_root: Path) -> dict[str, str]:
    """Require every canonical source file to match the plan before imports."""
    source = cast(dict[str, object], plan["source"])
    pinned = cast(dict[str, str], source["current_source_sha256"])
    actual = {name: _sha256(source_root / name) for name in pinned}
    if actual != pinned:
        changed = [name for name in pinned if actual.get(name) != pinned[name]]
        raise ValueError(f"canonical source pins differ from the plan: {changed}")
    return actual


def main() -> None:
    """Run create-only matched setup and initialization-parity gates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--source-amendment", type=Path)
    parser.add_argument("--order-amendment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    plan = _bounded_plan(plan_path)
    source = cast(dict[str, object], plan["source"])
    env = cast(dict[str, str], source["environment"])
    expected_project = Path(env["WNN_PROJECT_ROOT"]).resolve()
    project_root = (args.project_root or expected_project).resolve()
    source_root = (args.source_root or project_root).resolve()
    if project_root != expected_project:
        raise ValueError("project root differs from the source-qualified plan")
    source_hashes = _verify_source(plan, source_root)
    amendment_record: dict[str, object] | None = None
    order_record: dict[str, object] | None = None
    if args.fold == 1:
        if args.source_amendment is None or args.source_sha is None:
            raise ValueError("F1 preflight requires the exact R4 source amendment and commit")
        if args.order_amendment is None:
            raise ValueError("F1 preflight requires the exact seed-4 order expectation amendment")
        amendment_record = verify_f1_source_amendment(
            args.source_amendment,
            args.source_sha,
            source_root,
            project_root,
            plan_path,
            plan,
        )
        order_record = verify_f1_order_expectation(
            args.order_amendment, source_root, project_root, plan_path, plan
        )
    elif (
        args.source_amendment is not None
        or args.source_sha is not None
        or args.order_amendment is not None
    ):
        raise ValueError("F0 preflight uses the frozen R1 source path without amendments")

    helper_dir = Path(__file__).resolve().parent
    helper_paths = {
        "entrypoint": Path(__file__).resolve(),
        "identity": helper_dir / "warm_i1_preflight_identity.py",
        "setup": helper_dir / "warm_i1_preflight_setup.py",
        "parity": helper_dir / "warm_i1_preflight_parity.py",
        "pair": helper_dir / "warm_i1_preflight_pair.py",
        "source_amendment": helper_dir / "warm_i1_preflight_amendment.py",
        "order_amendment": helper_dir / "warm_i1_order_amendment.py",
    }
    helper_hashes = {name: _sha256(path) for name, path in helper_paths.items()}

    scratch = Path("/home/vlad/development/wunder/research_tmp").resolve()
    output = args.output.resolve()
    if scratch not in output.parents or output.exists():
        raise ValueError("output must be a new file under physical research_tmp")
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.update(env)
    os.environ["WNN_PROJECT_ROOT"] = str(project_root)
    os.chdir(project_root)
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(helper_dir))
    from warm_i1_preflight_pair import run_fold_preflight

    result = {
        "schema_version": 1,
        "status": "PASS_SETUP_AND_INITIALIZATION_PARITY",
        "scientific_fit_performed": False,
        "optimizer_steps": 0,
        "fold": args.fold,
        "plan_sha256": _sha256(plan_path),
        "project_root": str(project_root),
        "source_root": str(source_root),
        "canonical_source_sha256": source_hashes,
        "source_amendment": amendment_record,
        "order_expectation_amendment": order_record,
        "preflight_helper_sha256": helper_hashes,
        "checks": run_fold_preflight(plan, args.fold, project_root, source_root, order_record),
        "execution_limits": {"device": "cpu", "torch_threads": 1},
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

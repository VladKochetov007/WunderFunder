"""Drive the genuine upstream recursion API with immutable research boundaries."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from harness2.core import recursion, selection, tree
from harness2.core.library import Library
from harness2.core.proposer import Proposer
from harness2.core.spec import RunView

from . import UPSTREAM_REVISION
from .adapter import ResearchAdapter
from .proposer import fixture_executor
from .runtime import Budget, write_json
from .task import Task
from .upstream import verify_installation


def fingerprint(value: object) -> str:
    """Address a JSON-compatible identity using stable canonical bytes."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hashes(root: Path) -> dict[str, str]:
    """Bind every retained harness byte and reject symlinks."""
    if root.is_symlink():
        raise ValueError(f"Symlink is not allowed as an evidence root: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink is not allowed in a harness: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


def evidence_hashes(root: Path) -> dict[str, str]:
    """Bind attempt evidence excluding the receipt that contains this manifest."""
    return {name: digest for name, digest in file_hashes(root).items() if name != "COMPLETE.json"}


def refine(
    task: Task,
    output: Path,
    base: Path,
    library_root: Path,
    budget: Budget,
    proposer: Proposer,
    executor: Callable[[Path, int], None],
    *,
    mode: str = "parallel",
    k: int = 1,
    executor_identity: object = None,
) -> dict[str, object]:
    """Refine process guidance; never launch or select a scientific experiment."""
    if mode not in ("parallel", "sequential") or not 1 <= k <= 2:
        raise ValueError("Use parallel/sequential mode and k=1 or k=2")
    upstream = verify_installation()
    is_fixture = executor is fixture_executor and executor_identity == "fixture"
    task.assert_unchanged()
    base = base.resolve(strict=True)
    output = output.resolve()
    base_hashes = file_hashes(base)
    integration = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.iterdir())
        if path.suffix in (".py", ".json") and path.is_file()
    }
    identity = {
        "integration_files": integration,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_files": upstream,
        "task": task.fingerprint,
        "base": base_hashes,
        "mode": mode,
        "k": k,
        "channel": proposer.spec.kind,
        "model": proposer.spec.model,
        "effort": proposer.spec.effort,
        "steps": proposer.spec.steps,
        "call_timeout": proposer.spec.timeout,
        "worker": executor_identity,
        "wall_seconds": budget.seconds,
        "max_rollouts": budget.max_rollouts,
        "max_calls": budget.max_calls,
        "library_root": str(library_root.resolve()),
    }
    complete = output / "COMPLETE.json"
    if complete.is_file():
        saved = json.loads(complete.read_text(encoding="utf-8"))
        if (
            saved["identity"] != identity
            or file_hashes(output / "adopted") != saved["adopted_files"]
            or evidence_hashes(output) != saved["evidence_files"]
        ):
            raise ValueError("Completed run identity/artifacts changed; use a new output directory")
        if (
            saved["library_entry"]
            and file_hashes(Path(saved["library_entry"])) != saved["library_entry_files"]
        ):
            raise ValueError("Retained library entry changed; completed result cannot be resumed")
        return saved
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "IDENTITY.json", identity)

    rollout_timings: dict[str, float] = {}

    def execute(workspace: Path, timeout: int) -> None:
        budget.admit_rollout()
        started = time.monotonic()
        try:
            executor(workspace, min(timeout, budget.remaining()))
        finally:
            rollout_timings[workspace.name] = round(time.monotonic() - started, 6)
            write_json(output / "ROLLOUT_TIMINGS.json", rollout_timings)
        budget.remaining()

    try:
        adapter = ResearchAdapter(task, output, base, execute, budget.remaining)
        adapter.spec = replace(
            adapter.spec, channels={"proposer": proposer.spec, "selector": proposer.spec}
        )
        base_tree = tree.read_tree(adapter.substrate.surface(), base)
        adapter.validate_tree(base_tree)
        if set(base_tree) != set(base_hashes):
            raise ValueError("Base contains files outside the editable guidance surface")
        # Source, recipe, harness and model changes isolate prior process lessons.
        namespace = fingerprint(
            {
                **{key: value for key, value in identity.items() if key != "task"},
                "domain": task.domain,
                "prompt": task.prompt,
                "protected": task.protected,
                "sources": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in task.sources.items()
                },
            }
        )
        library = Library(
            library_root.resolve() / namespace, task.domain, adapter.substrate.surface()
        )
        if task.task_id in library.entries():
            raise ValueError("Task is already retained in this library; resume its original output")
        library_before = file_hashes(library.dir) if library.dir.exists() else {}
        references = []
        for index in range(1, k + 1):
            label = f"base{index}"
            run_dir = adapter.execute(task.task_id, None, label)
            references.append(RunView(label, label, run_dir, True, adapter.delivered(run_dir)))
        token = "agg" if mode == "parallel" else "seq"
        result = recursion.run_mode(
            token,
            task.task_id,
            k,
            references,
            adapter=adapter,
            prop=proposer,
            work_dir=output,
            run_id_of=lambda label: label,
            base_tree=base_tree,
            library_dir=library.context_dir(),
            propose_workers=1,
            rollout_workers=1,
        )
        write_json(output / "RECURSION.json", result.to_dict())
        if not result.ok:
            raise RuntimeError(f"Upstream recursion did not complete: {result.errors}")
        candidates = selection.selection_pool(token, k, result.rollouts, references)
        chosen = selection.run_selection(
            task.task_id,
            k,
            token,
            candidates,
            adapter=adapter,
            prop=proposer,
            work_dir=output,
            pool_kind="finals",
        )
        if not chosen.ok or chosen.fallback:
            raise RuntimeError(f"Selection failed or used fallback: {chosen.error}")
        experience = selection.reflect_adoption(
            task.task_id,
            chosen,
            candidates,
            adapter=adapter,
            prop=proposer,
            work_dir=output,
        )
        adopted = result.trees[chosen.chosen_label]
        adapter.validate_tree(adopted)
        task.assert_unchanged()
        verify_installation()
        budget.remaining()
        if file_hashes(base) != base_hashes:
            raise ValueError("Base harness changed during refinement")
        current_integration = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.iterdir())
            if path.suffix in (".py", ".json") and path.is_file()
        }
        if current_integration != integration:
            raise ValueError("Integration source changed during refinement")
        tree.write_tree(adapter.substrate.surface(), adopted, output / "adopted", stubs=False)
        retained = library.add(
            task.task_id,
            adopted,
            why=chosen.why,
            label=chosen.chosen_label,
            run_id=chosen.chosen_run_id,
            experience=experience,
        )
        summary = {
            "schema": "wnn-harness2-result.v1",
            "identity": identity,
            "task_id": task.task_id,
            "status": "workflow_selected",
            "selection": chosen.to_dict(),
            "recursion": result.to_dict(),
            "library_namespace": namespace,
            "library_entry": str(retained) if retained else None,
            "library_before": library_before,
            "adopted_files": file_hashes(output / "adopted"),
            "rollouts_attempted": budget.rollouts,
            "editor_calls_attempted": budget.calls,
            "rollout_wall_seconds": rollout_timings,
            "elapsed_wall_seconds": round(time.monotonic() - budget.started, 6),
            "cost_usd": 0 if is_fixture else None,
            "editor_reported_cost_usd": sum(reply.cost_usd for reply in proposer._calls),
            "scope": "Process guidance only; selection is not a scientific quality verdict",
            "scientific_fits_started": 0 if is_fixture else None,
            "scientific_dispatches_by_driver": 0,
            "model_quality_tested": False,
            "fixture": is_fixture,
            "library_entry_files": file_hashes(retained) if retained else {},
            "evidence_files": evidence_hashes(output),
        }
        write_json(complete, summary)
        return summary
    except BaseException as error:
        write_json(
            output / "FAILURE.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "rollouts_attempted": budget.rollouts,
                "editor_calls_attempted": budget.calls,
                "complete": False,
            },
        )
        raise

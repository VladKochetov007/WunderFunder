"""Exercise upstream recursion, isolation, persistence and campaign failure paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("harness2", reason="Optional Python 3.12 Harness² research environment")

from wnn.harness2_workflow.engine import refine
from wnn.harness2_workflow.proposer import FixtureProposer, fixture_executor
from wnn.harness2_workflow.runtime import Budget, run_worker
from wnn.harness2_workflow.task import Task

ROOT = Path(__file__).resolve().parents[1]


def make_task(tmp_path: Path) -> Task:
    """Build input evidence independently of the demo writer and production adapter."""
    (tmp_path / "events.json").write_text(
        json.dumps(
            [
                {"id": "alpha", "stage": "exported"},
                {"id": "beta", "stage": "checkpointed"},
                {"id": "gamma", "stage": "evaluated"},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "fixed.py").write_text("SCIENTIFIC_METRIC = 'unchanged'\n", encoding="utf-8")
    path = tmp_path / "task.json"
    path.write_text(
        json.dumps(
            {
                "schema": "wnn-harness2-task.v1",
                "id": "test-handoff",
                "domain": "handoffs",
                "prompt": "Prepare a factual handoff from the recorded events.",
                "sources": {"events.json": "events.json"},
                "protected_files": ["fixed.py"],
            }
        ),
        encoding="utf-8",
    )
    return Task.load(path, tmp_path)


@pytest.mark.parametrize(("mode", "k"), [("parallel", 1), ("sequential", 1), ("parallel", 2)])
def test_real_upstream_loop_and_exact_resume(tmp_path: Path, mode: str, k: int) -> None:
    task = make_task(tmp_path)
    output = tmp_path / "attempt"
    budget = Budget(120, 4 * k, 6 * k + 2)
    result = refine(
        task,
        output,
        ROOT / "tools/harness2/base",
        tmp_path / "library",
        budget,
        FixtureProposer(budget),
        fixture_executor,
        mode=mode,
        k=k,
        executor_identity="fixture",
    )
    assert result["rollouts_attempted"] == 4 * k
    assert result["scientific_fits_started"] == 0
    assert result["model_quality_tested"] is False
    assert result["library_entry"]
    chosen = result["selection"]["chosen_label"]
    handoff = (output / "rollouts" / chosen / "deliverable.md").read_text(encoding="utf-8")
    assert "- checkpointed: 1" in handoff
    assert "- exported: 1" in handoff
    assert "- evaluated: 1" in handoff
    assert "completed: 3" in (output / "rollouts/base1/deliverable.md").read_text(encoding="utf-8")
    repeated = Budget(120, 4 * k, 6 * k + 2)
    resumed = refine(
        task,
        output,
        ROOT / "tools/harness2/base",
        tmp_path / "library",
        repeated,
        FixtureProposer(repeated),
        fixture_executor,
        mode=mode,
        k=k,
        executor_identity="fixture",
    )
    assert resumed == result
    assert repeated.rollouts == repeated.calls == 0
    (output / "adopted/AGENTS.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="identity/artifacts changed"):
        refine(
            task,
            output,
            ROOT / "tools/harness2/base",
            tmp_path / "library",
            repeated,
            FixtureProposer(repeated),
            fixture_executor,
            mode=mode,
            k=k,
            executor_identity="fixture",
        )


def test_changed_source_never_starts_worker(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    (tmp_path / "events.json").write_text("[]", encoding="utf-8")
    budget = Budget(120, 4, 8)
    with pytest.raises(RuntimeError):
        refine(
            task,
            tmp_path / "attempt",
            ROOT / "tools/harness2/base",
            tmp_path / "library",
            budget,
            FixtureProposer(budget),
            fixture_executor,
        )
    assert budget.rollouts == 0


def test_worker_mutation_preserved_as_failure_without_learning(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    budget = Budget(120, 4, 8)

    def bad_worker(workspace: Path, timeout: int) -> None:
        fixture_executor(workspace, timeout)
        (tmp_path / "fixed.py").write_text("changed", encoding="utf-8")

    output = tmp_path / "attempt"
    with pytest.raises(RuntimeError):
        refine(
            task,
            output,
            ROOT / "tools/harness2/base",
            tmp_path / "library",
            budget,
            FixtureProposer(budget),
            bad_worker,
        )
    assert (output / "FAILURE.json").is_file()
    assert not (output / "COMPLETE.json").exists()
    assert not list((tmp_path / "library").rglob("EXPERIENCE.md"))


def test_budget_bounds_and_no_silent_retry() -> None:
    with pytest.raises(ValueError):
        Budget(14401, 4, 8)
    budget = Budget(120, 1, 1)
    budget.admit_rollout()
    with pytest.raises(RuntimeError, match="rollout"):
        budget.admit_rollout()
    budget.admit_call()
    with pytest.raises(RuntimeError, match="call"):
        budget.admit_call()


def test_worker_timeout_preserves_logs(tmp_path: Path) -> None:
    import subprocess

    with pytest.raises(subprocess.TimeoutExpired):
        run_worker([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, 1)
    assert (tmp_path / "worker.stdout.log").is_file()
    assert (tmp_path / "worker.stderr.log").is_file()


@pytest.mark.parametrize("artifact", ["deliverable", "trajectory", "recursion", "lesson"])
def test_resume_rejects_modified_retained_evidence(tmp_path: Path, artifact: str) -> None:
    task = make_task(tmp_path)
    output = tmp_path / "attempt"
    library = tmp_path / "library"
    budget = Budget(120, 4, 8)
    result = refine(
        task,
        output,
        ROOT / "tools/harness2/base",
        library,
        budget,
        FixtureProposer(budget),
        fixture_executor,
        executor_identity="fixture",
    )
    chosen = result["selection"]["chosen_label"]
    target = {
        "deliverable": output / "rollouts" / chosen / "deliverable.md",
        "trajectory": output / "rollouts" / chosen / "trajectory.md",
        "recursion": output / "RECURSION.json",
        "lesson": Path(result["library_entry"]) / "EXPERIENCE.md",
    }[artifact]
    target.write_text("tampered", encoding="utf-8")
    fresh = Budget(120, 4, 8)
    with pytest.raises(ValueError, match="changed"):
        refine(
            task,
            output,
            ROOT / "tools/harness2/base",
            library,
            fresh,
            FixtureProposer(fresh),
            fixture_executor,
            executor_identity="fixture",
        )
    assert fresh.rollouts == fresh.calls == 0


def test_unspecified_executor_identity_does_not_claim_zero_cost(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    budget = Budget(120, 4, 8)

    def custom_worker(workspace: Path, timeout: int) -> None:
        fixture_executor(workspace, timeout)

    result = refine(
        task,
        tmp_path / "attempt",
        ROOT / "tools/harness2/base",
        tmp_path / "library",
        budget,
        FixtureProposer(budget),
        custom_worker,
    )
    assert result["fixture"] is False
    assert result["cost_usd"] is None
    assert result["scientific_fits_started"] is None


def test_missing_reflection_does_not_write_a_lesson(tmp_path: Path) -> None:
    from harness2.core.proposer import Reply

    class NoReflection(FixtureProposer):
        def call(
            self, mission: str, context_dir: Path | str | None, label: str, **kwargs: object
        ) -> Reply:
            if label.startswith("experience."):
                self.budget.admit_call()
                return Reply(label=label, text="NO_ENTRY", ok=True)
            return super().call(mission, context_dir, label, **kwargs)

    task = make_task(tmp_path)
    budget = Budget(120, 4, 8)
    result = refine(
        task,
        tmp_path / "attempt",
        ROOT / "tools/harness2/base",
        tmp_path / "library",
        budget,
        NoReflection(budget),
        fixture_executor,
    )
    assert result["library_entry"] is None
    assert not list((tmp_path / "library").rglob("EXPERIENCE.md"))


def test_real_editor_gets_one_attempt_and_shared_remaining_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harness2.core.proposer import Proposer, ProposerError, Reply
    from harness2.core.spec import ChannelSpec

    from wnn.harness2_workflow.proposer import BoundedProposer

    calls = []

    def fake_call(
        self: Proposer, mission: str, context_dir: Path | str | None, label: str, **kwargs: object
    ) -> Reply:
        calls.append(kwargs)
        return Reply(label=label, text="reply", ok=True)

    monkeypatch.setattr(Proposer, "call", fake_call)
    budget = Budget(120, 4, 1)
    editor = BoundedProposer(ChannelSpec("opencode", "local/fake", timeout=3600), budget, tmp_path)
    editor.call("work", tmp_path, "inv1", max_attempts=7)
    assert calls[0]["max_attempts"] == 1
    assert 1 <= calls[0]["timeout"] <= 120
    with pytest.raises(ProposerError, match="call budget"):
        editor.call("work", tmp_path, "inv2")
    assert len(calls) == 1

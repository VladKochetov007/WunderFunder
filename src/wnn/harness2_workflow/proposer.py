"""Use upstream model editors, with a deterministic fixture editor for verification."""

from __future__ import annotations

from pathlib import Path

from harness2.core.proposer import Proposer, ProposerError, Reply
from harness2.core.spec import ChannelSpec

from .runtime import Budget

LESSON = "Report each lifecycle stage separately."


class BoundedProposer(Proposer):
    """Bound real upstream model calls; transport retries are charged separately."""

    def __init__(self, spec: ChannelSpec, budget: Budget, audit_dir: Path) -> None:
        super().__init__(spec, audit_dir=audit_dir, attempts=1)
        self.budget = budget

    def call(
        self,
        mission: str,
        context_dir: Path | str | None,
        label: str,
        *,
        system: str = "",
        timeout: int | None = None,
        max_attempts: int | None = None,
        harvest: str | None = None,
    ) -> Reply:
        """Use at most one transport attempt inside the shared remaining time."""
        del max_attempts
        try:
            self.budget.admit_call()
            remaining = self.budget.remaining()
        except (TimeoutError, RuntimeError) as error:
            raise ProposerError(str(error), attempts=0) from error
        return super().call(
            mission,
            context_dir,
            label,
            system=system,
            timeout=min(timeout or self.spec.timeout, remaining),
            max_attempts=1,
            harvest=harvest,
        )


class FixtureProposer(Proposer):
    """Exercise the real upstream loop without an LLM or a scientific quality claim."""

    def __init__(self, budget: Budget) -> None:
        super().__init__(ChannelSpec("opencode", "local/deterministic"))
        self.budget = budget

    def call(
        self,
        mission: str,
        context_dir: Path | str | None,
        label: str,
        *,
        system: str = "",
        timeout: int | None = None,
        max_attempts: int | None = None,
        harvest: str | None = None,
    ) -> Reply:
        """Edit, select and reflect using only the fixture's visible process evidence."""
        del mission, system, timeout, max_attempts, harvest
        self.budget.admit_call()
        if context_dir is None:
            raise ValueError("Fixture editor requires a context directory")
        context = Path(context_dir)
        workspace = None
        if label.startswith("select."):
            candidates = sorted((context / "candidates").glob("*.md"))
            chosen = next(
                path.stem
                for path in candidates
                if "checkpointed: 1" in path.read_text(encoding="utf-8")
            )
            text = f"<why>The handoff separates the three observed lifecycle stages.</why>\n```choice\n{chosen}\n```"
        elif label.startswith("experience."):
            text = (
                "<experience>\ncue: Handoff includes mixed lifecycle stages\n"
                f"lesson: {LESSON}\n"
                "evidence: The base called three items completed; the probe separated checkpointed, exported and evaluated items.\n"
                "confidence: deterministic fixture only; no ML-quality claim\n</experience>"
            )
        else:
            workspace = context / "harness"
            agents = workspace / "AGENTS.md"
            content = agents.read_text(encoding="utf-8")
            if LESSON not in content:
                agents.write_text(content.rstrip() + "\n\n" + LESSON + "\n", encoding="utf-8")
            text = "<notes>Separate lifecycle stages using the task's recorded events.</notes>"
        reply = Reply(label=label, text=text, ok=True, attempts=1, workspace=workspace)
        self._calls.append(reply)
        return reply


def fixture_executor(workspace: Path, timeout: int) -> None:
    """Replay actual fixture metadata under the proposed reporting instruction."""
    import json
    from collections import Counter

    if timeout < 1:
        raise TimeoutError("Fixture budget expired")
    rows = json.loads((workspace / "source/events.json").read_text(encoding="utf-8"))
    guidance = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    if LESSON in guidance:
        counts = Counter(row["stage"] for row in rows)
        text = (
            "# Lifecycle handoff\n\n"
            + "\n".join(f"- {stage}: {counts[stage]}" for stage in sorted(counts))
            + "\n"
        )
    else:
        text = f"# Lifecycle handoff\n\n- completed: {len(rows)}\n"
    (workspace / "deliverable.md").write_text(text, encoding="utf-8")
    (workspace / "trajectory.md").write_text(
        "[ACTION] Read selected event metadata and prepare a lifecycle handoff.\n[OBSERVATION]\n"
        + text,
        encoding="utf-8",
    )

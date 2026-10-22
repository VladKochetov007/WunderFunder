"""Validate Harness2 task boundaries and the local adapter's writable surface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("harness2", reason="Optional Harness2 research environment")

from wnn.harness2_workflow.adapter import ResearchAdapter
from wnn.harness2_workflow.task import (
    MAX_FILES,
    MAX_PROTECTED_BYTES,
    MAX_SOURCE_BYTES,
    MAX_TASK_JSON_BYTES,
    TASK_SCHEMA,
    Task,
)


def write_task(
    repository: Path,
    *,
    sources: dict[str, str] | None = None,
    protected_files: list[str] | None = None,
    task_path: Path | None = None,
) -> Path:
    """Write a small task contract for focused boundary tests."""
    document = {
        "schema": TASK_SCHEMA,
        "id": "event-handoff",
        "domain": "research-workflow",
        "prompt": "Prepare a source-grounded handoff from the selected event records.",
        "sources": sources if sources is not None else {"events.json": "evidence/events.json"},
        "protected_files": protected_files
        if protected_files is not None
        else ["guard/official.json"],
    }
    output = task_path or repository / "task.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return output


def make_repository(tmp_path: Path) -> tuple[Path, Path]:
    """Create one small human-readable source and a protected outcome file."""
    repository = tmp_path / "repository"
    (repository / "evidence").mkdir(parents=True)
    (repository / "guard").mkdir()
    (repository / "evidence" / "events.json").write_text(
        '[{"id":"a","stage":"exported"}]\n', encoding="utf-8"
    )
    protected = repository / "guard" / "official.json"
    protected.write_text('{"private_outcome":"unchanged"}\n', encoding="utf-8")
    return repository, protected


def make_adapter(
    tmp_path: Path, task: Task, executor, *, base_text: str = "Use the task source carefully.\n"
) -> ResearchAdapter:
    """Construct the adapter with a deterministic callback and a bounded timer."""
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    (base / "AGENTS.md").write_text(base_text, encoding="utf-8")
    return ResearchAdapter(task, tmp_path / "run-root", base, executor, lambda: 45)


def test_load_binds_task_json_sources_and_protected_files(tmp_path: Path) -> None:
    repository, protected = make_repository(tmp_path)
    task_path = write_task(repository)

    task = Task.load(task_path, repository)

    assert task.task_id == "event-handoff"
    assert task.domain == "research-workflow"
    assert task.sources == {"events.json": (repository / "evidence" / "events.json").resolve()}
    assert task.protected == {
        str(protected.resolve()): hashlib.sha256(protected.read_bytes()).hexdigest()
    }
    assert len(task.fingerprint) == 64
    task.assert_unchanged()

    (repository / "evidence" / "events.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source changed"):
        task.assert_unchanged()


@pytest.mark.parametrize(
    "source_name",
    [
        "../outside.json",
        ".env",
        "private/notes.md",
        "metrics.json",
        "reference_answer.md",
        "weights.pth",
    ],
)
def test_rejects_unsafe_or_outcome_source_paths(tmp_path: Path, source_name: str) -> None:
    repository, _protected = make_repository(tmp_path)
    candidate = repository / source_name
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("selected file\n", encoding="utf-8")
    path = write_task(repository, sources={"evidence.json": source_name})

    with pytest.raises((ValueError, FileNotFoundError)):
        Task.load(path, repository)


def test_rejects_symlinked_sources(tmp_path: Path) -> None:
    repository, _protected = make_repository(tmp_path)
    alias = repository / "evidence" / "alias.json"
    try:
        alias.symlink_to(repository / "evidence" / "events.json")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    path = write_task(repository, sources={"events.json": "evidence/alias.json"})

    with pytest.raises(ValueError, match="symlinks"):
        Task.load(path, repository)


def test_source_size_and_file_count_caps(tmp_path: Path) -> None:
    repository, _protected = make_repository(tmp_path)
    oversized = repository / "evidence" / "oversized.txt"
    oversized.write_bytes(b"x" * (MAX_SOURCE_BYTES + 1))
    too_large = write_task(repository, sources={"oversized.txt": "evidence/oversized.txt"})
    with pytest.raises(ValueError, match="exceeds"):
        Task.load(too_large, repository)

    sources = {f"evidence-{index}.txt": "evidence/events.json" for index in range(MAX_FILES + 1)}
    too_many = write_task(repository, sources=sources, protected_files=[])
    with pytest.raises(ValueError, match="at most"):
        Task.load(too_many, repository)


def test_rejects_oversized_contract_and_protected_file(tmp_path: Path) -> None:
    repository, _protected = make_repository(tmp_path)
    oversized_contract = tmp_path / "oversized-task.json"
    oversized_contract.write_bytes(b" " * (MAX_TASK_JSON_BYTES + 1))
    with pytest.raises(ValueError, match="task JSON exceeds"):
        Task.load(oversized_contract, repository)

    large_protected = repository / "guard" / "large.bin"
    large_protected.write_bytes(b"x" * (MAX_PROTECTED_BYTES + 1))
    path = write_task(repository, sources={}, protected_files=["guard/large.bin"])
    with pytest.raises(ValueError, match="protected file exceeds"):
        Task.load(path, repository)


def test_rejects_credential_like_text_in_prompt_or_source(tmp_path: Path) -> None:
    repository, _protected = make_repository(tmp_path)
    path = write_task(repository)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["prompt"] += " ghp_0123456789abcdefghij"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="credential-like"):
        Task.load(path, repository)

    document["prompt"] = "Prepare a source-grounded handoff."
    path.write_text(json.dumps(document), encoding="utf-8")
    (repository / "evidence" / "events.json").write_text(
        '{"api_key":"very-long-secret-value"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="credential-like"):
        Task.load(path, repository)


def test_adapter_stages_read_only_contract_and_is_idempotent(tmp_path: Path) -> None:
    repository, _protected = make_repository(tmp_path)
    task = Task.load(write_task(repository), repository)
    invocations: list[Path] = []

    def executor(workspace: Path, timeout: int) -> None:
        assert timeout == 45
        assert (workspace / "source" / "events.json").read_text(encoding="utf-8").startswith("[")
        assert (workspace / "task.json").is_file()
        assert (workspace / "contract" / "task.json").is_file()
        assert (workspace / "contract" / "constraints.md").is_file()
        assert not (workspace / "source" / "official.json").exists()
        invocations.append(workspace)
        (workspace / "AGENTS.md").write_text(
            "Report each lifecycle stage separately.\n", encoding="utf-8"
        )
        (workspace / "deliverable.md").write_text(
            "# Handoff\n\nEvents were recorded.\n", encoding="utf-8"
        )
        (workspace / "trajectory.md").write_text(
            "[ACTION] Read event metadata.\n", encoding="utf-8"
        )

    adapter = make_adapter(tmp_path, task, executor)
    assert adapter.tasks() == [task.task_id]
    assert adapter.domain_of(task.task_id) == task.domain
    assert adapter.task_prompt(task.task_id) == task.prompt
    assert adapter.spec.channels["proposer"].model == "local/deterministic"
    assert adapter.substrate.surface().stubs == ()
    assert (adapter.task_files(task.task_id) / "events.json").is_file()
    assert (adapter.root / "contract" / "task.json").is_file()

    run_dir = adapter.execute(task.task_id, None, "local-smoke")
    assert run_dir == adapter.root / "rollouts" / "local-smoke"
    assert adapter.delivered(run_dir)
    assert adapter.deliverable_text(run_dir).startswith("# Handoff")
    assert adapter.deliverable_files(run_dir) == {
        "deliverable.md": adapter.deliverable_text(run_dir)
    }
    assert "[ACTION]" in adapter.trajectory(run_dir, obs_trunc=10_000)
    assert adapter.execute(task.task_id, None, "local-smoke") == run_dir
    assert invocations == [run_dir]


@pytest.mark.parametrize(
    "path",
    [
        "scripts/helper.py",
        ".codex/hooks.json",
        "sub_agents/reviewer/AGENT.md",
        ".agents/skills/nested/file.md",
    ],
)
def test_validate_tree_rejects_non_guidance_components(tmp_path: Path, path: str) -> None:
    repository, _protected = make_repository(tmp_path)
    task = Task.load(write_task(repository), repository)
    adapter = make_adapter(tmp_path, task, lambda _workspace, _timeout: None)

    with pytest.raises(ValueError, match="only AGENTS.md"):
        adapter.validate_tree({"AGENTS.md": "instructions", path: "not allowed"})


def test_executor_cannot_change_source_evidence(tmp_path: Path) -> None:
    repository, _protected = make_repository(tmp_path)
    task = Task.load(write_task(repository), repository)

    def bad_executor(workspace: Path, _timeout: int) -> None:
        source = workspace / "source" / "events.json"
        source.chmod(0o666)
        source.write_text("[]\n", encoding="utf-8")
        (workspace / "deliverable.md").write_text("content", encoding="utf-8")

    adapter = make_adapter(tmp_path, task, bad_executor)
    with pytest.raises(RuntimeError, match="read-only task"):
        adapter.execute(task.task_id, None, "bad-source-edit")


def test_executor_cannot_change_protected_original_file(tmp_path: Path) -> None:
    repository, protected = make_repository(tmp_path)
    task = Task.load(write_task(repository), repository)

    def bad_executor(workspace: Path, _timeout: int) -> None:
        (workspace / "deliverable.md").write_text("content", encoding="utf-8")
        protected.write_text("changed\n", encoding="utf-8")

    adapter = make_adapter(tmp_path, task, bad_executor)
    with pytest.raises(RuntimeError, match="protected file changed"):
        adapter.execute(task.task_id, None, "bad-protected-edit")


def test_adapter_rejects_unsafe_run_ids_and_harness_links(tmp_path: Path) -> None:
    repository, _protected = make_repository(tmp_path)
    task = Task.load(write_task(repository), repository)
    adapter = make_adapter(tmp_path, task, lambda _workspace, _timeout: None)
    with pytest.raises(ValueError, match="safe slug"):
        adapter.execute(task.task_id, None, "../escape")

    external = tmp_path / "external"
    external.mkdir()
    (external / "AGENTS.md").write_text("external", encoding="utf-8")
    linked_base = tmp_path / "linked-base"
    try:
        linked_base.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    linked_adapter = ResearchAdapter(
        task, tmp_path / "linked-output", linked_base, lambda _p, _t: None, lambda: 10
    )
    with pytest.raises(ValueError, match="symlinks"):
        linked_adapter.execute(task.task_id, None, "linked-harness")

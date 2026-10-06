from __future__ import annotations

import json
import os
import signal
import sys
from contextlib import suppress
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

NIGHT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NIGHT / "kaggle"))
sys.path.insert(0, str(NIGHT / "width"))
import bounded_loader  # noqa: E402
import host_guard  # noqa: E402
from derive_source import derive_trainer  # noqa: E402

from wnn.data import FEATURE_COLUMNS, SEQUENCE_LENGTH  # noqa: E402


def test_full_dataset_tensors_match_after_teacher_and_propensity_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the actual loader boundary with reordered teacher rows and masks."""
    root = NIGHT.parents[2]
    original_path = root / "research/campaign_2026-10-05/w384_v7/bundle_source/scripts/train_rnn.py"
    original_source = original_path.read_text()
    monkeypatch.setitem(sys.modules, "wnn.bounded_loader", bounded_loader)
    original, derived = ModuleType("old_width"), ModuleType("new_width")
    for module, source in ((original, original_source), (derived, derive_trainer(original_source))):
        module.__file__ = str(original_path)
        exec(compile(source, str(original_path), "exec"), module.__dict__)
        module.ROOT = tmp_path
        module.HEURISTICS = False
    (tmp_path / "datasets").mkdir()
    (tmp_path / "cache").mkdir()
    rows = np.arange(SEQUENCE_LENGTH, dtype=np.float32)
    values = {"seq_ix": np.repeat([71, 99], SEQUENCE_LENGTH)}
    for number, name in enumerate(FEATURE_COLUMNS):
        column = np.concatenate([rows + number / 7, -rows - number / 5]).astype(np.float32)
        column[[0, SEQUENCE_LENGTH]] = -0.0
        values[name] = column
    values.update(
        t0=np.tile(rows / 31, 2), t1=np.tile(-rows / 41, 2), is_scored=np.tile(rows % 7 == 0, 2)
    )
    pq.write_table(
        pa.table(values), tmp_path / "datasets/train.parquet", row_group_size=SEQUENCE_LENGTH
    )
    np.savez(tmp_path / "cache/train_targets.npz", row_group=np.array([1, 0]))
    teacher = np.arange(2 * SEQUENCE_LENGTH * 2, dtype=np.float32).reshape(2, SEQUENCE_LENGTH, 2)
    propensity = np.arange(2 * SEQUENCE_LENGTH, dtype=np.float32).reshape(2, SEQUENCE_LENGTH)
    np.save(tmp_path / "cache/pred_a_train.npy", teacher)
    np.save(tmp_path / "cache/pred_b_train.npy", -teacher / 3)
    np.save(tmp_path / "cache/prop.npy", propensity)
    owners = [
        module.Sequences(
            "train",
            [1, 0],
            teachers=[("a", 0.7), ("b", 1.3)],
            prop_cache=tmp_path / "cache/prop.npy",
        )
        for module in (original, derived)
    ]
    held = None
    for index in (0, 1, 0):
        expected, actual = (dataset[index] for dataset in owners)
        assert len(actual) == len(expected) == 4
        for left, right in zip(expected, actual, strict=True):
            assert left.dtype == right.dtype
            assert left.shape == right.shape
            assert left.numpy().tobytes() == right.numpy().tobytes()
            assert right.numpy().flags.writeable
        if held is None:
            held = [(tensor, tensor.numpy().tobytes()) for tensor in actual]
        for tensor, identity in held:
            assert tensor.numpy().tobytes() == identity
    assert not owners[1]._prop.flags.writeable
    assert all(not array.flags.writeable for array, _ in owners[1]._teach)


def test_intermediate_cgroup_limit_is_part_of_effective_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "/proc/meminfo": "MemTotal: 64000000 kB\nMemAvailable: 60000000 kB\n",
        "/proc/self/cgroup": "0::/parent/worker\n",
        "/sys/fs/cgroup/memory.max": "max",
        "/sys/fs/cgroup/memory.current": "1000000",
        "/sys/fs/cgroup/parent/memory.max": "12000000000",
        "/sys/fs/cgroup/parent/memory.current": "4000000000",
        "/sys/fs/cgroup/parent/worker/memory.max": "max",
        "/sys/fs/cgroup/parent/worker/memory.current": "1000000",
    }
    monkeypatch.setattr(Path, "read_text", lambda path, **_: files[str(path)])
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) in files)
    monkeypatch.setattr(Path, "exists", lambda path: str(path) in files)
    sampled = host_guard.host_sample()
    assert sampled["capacity_bytes"] == 12_000_000_000
    assert sampled["available_bytes"] == 8_000_000_000


def test_v1_intermediate_cgroup_limit_is_part_of_effective_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "/proc/meminfo": "MemTotal: 64000000 kB\nMemAvailable: 60000000 kB\n",
        "/proc/self/cgroup": "1:memory:/parent/worker\n",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": "1000000",
        "/sys/fs/cgroup/memory/parent/memory.limit_in_bytes": "18000000000",
        "/sys/fs/cgroup/memory/parent/memory.usage_in_bytes": "7000000000",
        "/sys/fs/cgroup/memory/parent/worker/memory.limit_in_bytes": "9223372036854771712",
        "/sys/fs/cgroup/memory/parent/worker/memory.usage_in_bytes": "1000000",
    }
    monkeypatch.setattr(Path, "read_text", lambda path, **_: files[str(path)])
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) in files)
    monkeypatch.setattr(Path, "exists", lambda path: str(path) in files)
    sampled = host_guard.host_sample()
    assert sampled["capacity_bytes"] == 18_000_000_000
    assert sampled["available_bytes"] == 11_000_000_000


def test_v1_oom_kill_counter_is_available_to_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "/proc/meminfo": "MemTotal: 64000000 kB\nMemAvailable: 60000000 kB\n",
        "/proc/self/cgroup": "1:memory:/parent/worker\n",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes": "1000000",
        "/sys/fs/cgroup/memory/parent/memory.limit_in_bytes": "18000000000",
        "/sys/fs/cgroup/memory/parent/memory.usage_in_bytes": "7000000000",
        "/sys/fs/cgroup/memory/parent/memory.oom_control": "oom_kill 2\nunder_oom 0\n",
        "/sys/fs/cgroup/memory/parent/worker/memory.limit_in_bytes": "9223372036854771712",
        "/sys/fs/cgroup/memory/parent/worker/memory.usage_in_bytes": "1000000",
        "/sys/fs/cgroup/memory/parent/worker/memory.oom_control": "oom_kill 2\nunder_oom 0\n",
    }
    monkeypatch.setattr(Path, "read_text", lambda path, **_: files[str(path)])
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) in files)
    monkeypatch.setattr(Path, "exists", lambda path: str(path) in files)
    sampled = host_guard.host_sample()
    assert sampled["cgroups"]["/sys/fs/cgroup/memory/parent"]["memory.oom_control"]["oom_kill"] == 2
    assert (
        host_guard.oom_counts(sampled)[
            "/sys/fs/cgroup/memory/parent/memory.oom_control/oom_kill"
        ]
        == 2
    )


@pytest.mark.parametrize("malformed", [False, True])
def test_v1_missing_or_malformed_ancestor_fails_closed(
    tmp_path: Path, malformed: bool
) -> None:
    root, parent, leaf = (
        tmp_path / name for name in ("memory", "memory/parent", "memory/parent/worker")
    )
    for path in (root, parent, leaf):
        path.mkdir(parents=True, exist_ok=True)
    for path, limit in ((root, "16000000000"), (leaf, "8000000000")):
        (path / "memory.limit_in_bytes").write_text(limit)
        (path / "memory.usage_in_bytes").write_text("1000000000")
    if malformed:
        (parent / "memory.limit_in_bytes").write_text("unknown")
        (parent / "memory.usage_in_bytes").write_text("1000000000")
    with pytest.raises(RuntimeError, match="not fully observable"):
        host_guard.sample_v1_hierarchy(
            [leaf, parent, root], 32_000_000_000, 30_000_000_000
        )


def test_fast_parent_does_not_leave_orphan_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generous = {"capacity_bytes": 32 * 1024**3, "available_bytes": 32 * 1024**3, "cgroups": {}}
    monkeypatch.setattr(host_guard, "host_sample", lambda: generous)
    source = (
        "import os,subprocess,sys,time; time.sleep(.2); "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(15)']); "
        "print(os.getpid(),p.pid,flush=True); time.sleep(.2)"
    )
    output = tmp_path / "attempt"
    try:
        with pytest.raises(RuntimeError, match="surviving descendants"):
            host_guard.supervised(
                [sys.executable, "-c", source],
                tmp_path,
                output,
                os.environ.copy(),
                timeout=10,
                rss_limit=128 * 1024**2,
            )
        parent, child = map(int, (output / "worker.log").read_text().split())
        observed = host_guard.tree_sample(child)
        assert not observed or observed[0]["rss_bytes"] == 0
        receipt = json.loads((output / "result.json").read_text())
        assert receipt["reason"] == "descendants_after_root_exit"
        assert receipt["teardown"]["status"] == "complete"
        assert receipt["teardown"]["remaining"] == []
    finally:
        log = output / "worker.log"
        if log.exists() and log.read_text().strip():
            parent, _ = map(int, log.read_text().split())
            with suppress(ProcessLookupError):
                os.killpg(parent, signal.SIGKILL)

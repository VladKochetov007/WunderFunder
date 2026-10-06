from __future__ import annotations

import fcntl
import shutil
from pathlib import Path

import pytest

from wnn import campaign, campaign_local, campaign_spec


def _source(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    for filename in campaign.RUNTIME_FILES:
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("identity fixture\n")
    contract = root / "contract.md"
    contract.write_text("fixed allowance\n")
    source = tmp_path / "source"
    campaign.initialize(source, root, contract, "local001")
    campaign.atomic_json(
        source / "attempts/charged.json",
        {
            "id": "charged",
            "status": "finished",
            "kind": "gpu",
            "block_id": "local001",
            "charged_seconds": 48511.247415846,
        },
    )
    policy = root / "policy.json"
    campaign.atomic_json(
        policy,
        {
            "version": 1,
            "local_start_cutoff_utc": "2026-10-06T07:00:00Z",
            "parent_ram_max_bytes": 40_000_000_000,
            "parent_ram_high_bytes": 36_000_000_000,
            "mem_available_reserve_bytes": 8_000_000_000,
        },
    )
    return root, source, policy


def test_one_source_allowance_cannot_fund_two_sibling_controllers(tmp_path: Path) -> None:
    root, source, policy = _source(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    campaign.initialize(first, root, root / "contract.md", "local001", policy, source)
    assert campaign.budget(first, "local001")["remaining_seconds"] == 37888.752584154
    with pytest.raises((ValueError, FileExistsError, RuntimeError)):
        campaign.initialize(second, root, root / "contract.md", "local001", policy, source)


def test_carried_gpu_lock_blocks_independent_open_file_handle(tmp_path: Path) -> None:
    root, source, policy = _source(tmp_path)
    destination = tmp_path / "continued"
    campaign.initialize(destination, root, root / "contract.md", "local001", policy, source)
    value = campaign.ready(destination, root)
    lock_path = Path(value["gpu_lock_path"])
    assert lock_path == source / "gpu.lock"
    with lock_path.open("a") as old, lock_path.open("a") as new:
        fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            fcntl.flock(new, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_transfer_closes_source_and_copied_consumer_is_not_authority(tmp_path: Path) -> None:
    root, source, policy = _source(tmp_path)
    destination, copied = tmp_path / "continued", tmp_path / "copied"
    campaign.initialize(destination, root, root / "contract.md", "local001", policy, source)
    with pytest.raises(campaign.BudgetExhausted):
        campaign.ready(source, root)
    shutil.copytree(destination, copied)
    with pytest.raises(ValueError, match="controller state"):
        campaign.ready(copied, root)
    with pytest.raises(ValueError, match="own the source"):
        campaign.budget(copied, "local001")


def test_crash_after_claim_cannot_reopen_or_duplicate_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source, policy = _source(tmp_path)
    destination = tmp_path / "continued"
    original = campaign.atomic_json

    def crash_on_readiness(path: Path, value: object) -> None:
        if path == destination / "READY.json":
            raise OSError("simulated publication failure")
        original(path, value)

    monkeypatch.setattr(campaign, "atomic_json", crash_on_readiness)
    with pytest.raises(OSError, match="publication"):
        campaign.initialize(destination, root, root / "contract.md", "local001", policy, source)
    assert (source / "CONTINUATION.json").exists()
    with pytest.raises(campaign.BudgetExhausted):
        campaign.ready(source, root)
    with pytest.raises(ValueError, match="continuation consumer"):
        campaign.initialize(
            tmp_path / "retry", root, root / "contract.md", "local001", policy, source
        )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "MemAvailable: nan kB\n",
        "MemAvailable: -1 kB\n",
        "MemAvailable: 9 MB\n",
        "MemAvailable: 9 kB\nMemAvailable: 9 kB\n",
    ],
)
def test_untrusted_host_memory_cannot_admit(text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: text)
    with pytest.raises(ValueError):
        campaign.mem_available()


def test_exact_cutoff_and_parent_soft_limit_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _source_state, path = _source(tmp_path)
    policy = campaign_spec.validate_admission_policy(campaign.read_json(path))
    monkeypatch.setattr(campaign.time, "time", lambda: campaign_spec.admission_cutoff(policy))
    with pytest.raises(campaign.AdmissionClosed):
        campaign.check_admission(policy, [])
    parent = {
        "CPUQuotaPerSecUSec": "12s",
        "MemoryMax": "40000000000",
        "MemoryHigh": "36000000000",
        "MemorySwapMax": "0",
        "ControlGroup": "/fixture/wnn-campaign.slice",
    }
    campaign_local.check_parent(parent, policy)
    with pytest.raises((RuntimeError, ValueError)):
        campaign_local.check_parent({**parent, "MemoryHigh": "max"}, policy)

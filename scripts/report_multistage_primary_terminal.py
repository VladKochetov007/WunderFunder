from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

TERMINAL_STATES = {"completed", "failed", "censored_timeout", "complete"}
LEGACY_STATES = {"terminal_unverified", "artifacts_complete_exit_unknown", "running"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4_000_000:
        raise ValueError("terminal qualification requires a bounded regular JSON file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("terminal qualification input must be a JSON object")
    return value


def _reference(path: object, expected_sha: object) -> tuple[Path, dict[str, object]]:
    source = Path(str(path))
    value = _read_json(source)
    if _sha256(source) != expected_sha:
        raise ValueError("legacy owner/observation reference bytes changed")
    return source, value


def _process_snapshot() -> list[dict[str, object]]:
    if os.name != "posix" or not Path("/proc/self/stat").is_file():
        raise RuntimeError("terminal process qualification requires the actual POSIX receiver")
    processes = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        if len(processes) >= 65_536:
            raise RuntimeError("process inventory exceeds the terminal qualification bound")
        try:
            raw = (path / "stat").read_text(encoding="ascii")
            with (path / "cmdline").open("rb") as stream:
                command_raw = stream.read(65_537)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if len(command_raw) > 65_536:
            raise RuntimeError("process argv exceeds the terminal qualification bound")
        close = raw.rfind(")")
        fields = raw[close + 2 :].split() if close >= 0 else []
        if len(fields) <= 19:
            raise ValueError("receiver process identity is malformed")
        processes.append(
            {
                "pid": int(path.name),
                "pgid": int(fields[2]),
                "birth": fields[19],
                "state": fields[0],
                "command": [os.fsdecode(token) for token in command_raw.split(b"\0") if token],
            }
        )
    return processes


def _qualify_legacy(
    row: Mapping[str, object], processes: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    owner_path, owner = _reference(row.get("owner_receipt_path"), row.get("owner_receipt_sha256"))
    observation_path, observation = _reference(
        row.get("observation_path"), row.get("observation_sha256")
    )
    identity = {key: owner.get(key) for key in ("pid", "birth", "pgid")}
    command = owner.get("command")
    command_sha = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
    if (
        type(identity["pid"]) is not int
        or identity["pid"] <= 0
        or type(identity["pgid"]) is not int
        or identity["pgid"] != identity["pid"]
        or not isinstance(identity["birth"], str)
        or not identity["birth"].isdecimal()
        or not isinstance(command, list)
        or not command
        or owner.get("command_sha256") != command_sha
        or owner.get("stage") != row.get("stage")
        or owner.get("output_dir") != row.get("output_dir")
        or observation.get("schema") != "wnn.multistage-external-fit-observation.v1"
        or observation.get("attempt_id") != row.get("attempt_id")
        or observation.get("stage") != row.get("stage")
        or observation.get("status") != row.get("status")
        or observation.get("owner_process") != identity
        or observation.get("owner_receipt_path") != str(owner_path)
        or observation.get("owner_receipt_sha256") != row.get("owner_receipt_sha256")
        or observation.get("config_sha256") != row.get("config_sha256")
        or observation.get("output_dir") != row.get("output_dir")
    ):
        raise ValueError("legacy fit observation does not bind its original owned worker")
    active = [process for process in processes if process.get("state") not in {"Z", "X"}]
    if any(
        all(process.get(key) == value for key, value in identity.items())
        or process.get("pgid") == identity["pgid"]
        for process in active
    ):
        raise ValueError(
            f"scientific fit still has a live owned process/group: {row['attempt_id']}"
        )
    for process in active:
        argv = process.get("command", [])
        if not isinstance(argv, list):
            raise ValueError("receiver process argv inventory is malformed")
        if any(
            token == f"--output={row['output_dir']}"
            or (
                token == "--output"
                and index + 1 < len(argv)
                and argv[index + 1] == row["output_dir"]
            )
            for index, token in enumerate(argv)
        ):
            raise ValueError(
                f"scientific fit still has a process using its output: {row['attempt_id']}"
            )
    return {
        "attempt_id": row["attempt_id"],
        "ledger_status": row["status"],
        "qualification": "owned_process_and_group_gone_exit_unknown",
        "matching_output_process_absent": True,
        "process_exit_code_observed": False,
        "exit_code": None,
        "censoring": "outer_process_exit_unobserved",
        "scientific_completion": "not_inferred_from_process_absence; candidate artifacts validated separately",
        "owner_process": identity,
        "owner_receipt_path": str(owner_path),
        "owner_receipt_sha256": row["owner_receipt_sha256"],
        "observation_path": str(observation_path),
        "observation_sha256": row["observation_sha256"],
        "historical_observation_status": observation["status"],
    }


def qualify_primary_fit_terminal(run_dir: Path) -> dict[str, object]:
    """Observe fit termination without changing ledger states or inventing process exits."""
    budget_path = run_dir / "budget.json"
    budget_sha = _sha256(budget_path)
    budget = _read_json(budget_path)
    attempts = budget.get("fit_attempts")
    if (
        budget.get("schema") != "wnn.multistage-budget.v1"
        or not isinstance(attempts, list)
        or not 1 <= len(attempts) <= 48
        or any(not isinstance(row, dict) for row in attempts)
        or len({row.get("attempt_id") for row in attempts}) != len(attempts)
    ):
        raise ValueError("primary report requires the bounded unique scientific-fit ledger")
    processes = (
        _process_snapshot() if any(row.get("status") in LEGACY_STATES for row in attempts) else []
    )
    qualifications = []
    for row in attempts:
        status = row.get("status")
        if status in LEGACY_STATES:
            qualifications.append(_qualify_legacy(row, processes))
        elif status in TERMINAL_STATES:
            qualifications.append(
                {
                    "attempt_id": row["attempt_id"],
                    "ledger_status": status,
                    "qualification": "ledger_terminal",
                }
            )
        else:
            raise ValueError(f"scientific fit is not qualified terminal: {row.get('attempt_id')}")
    if _sha256(budget_path) != budget_sha:
        raise ValueError("fit ledger changed during read-only terminal qualification")
    return {
        "schema": "wnn.multistage-primary-terminal-qualification.v1",
        "status": "all_fit_processes_qualified_terminal",
        "budget_path": str(budget_path.resolve()),
        "budget_sha256": budget_sha,
        "ledger_modified": False,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "fit_attempts": qualifications,
    }

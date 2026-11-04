from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from wnn.multistage_search_study import StudyHandle


class AskTrial(Protocol):
    """Optuna trial values used by exact-profile workers."""

    number: int
    user_attrs: Mapping[str, object]

    def set_user_attr(self, key: str, value: object) -> None: ...


@dataclass(frozen=True)
class ProfileJob:
    """Immutable context for one exact-identity profile attempt."""

    run_dir: Path
    run_manifest: Mapping[str, object]
    worker_id: str
    gpu_key: str
    device: Mapping[str, object]
    study: StudyHandle
    trial: AskTrial
    trial_number: int
    family: str
    config_path: Path
    config_sha256: str
    repo_root: Path
    source_sha256: str
    auxiliary_path: Path
    profile_updates: int
    output_dir: Path
    profile_dir: Path
    native_profile_path: Path
    worker_receipt_path: Path
    plan_path: Path
    proposal_key: str
    request: dict[str, object]
    worker_env: Mapping[str, str]
    stage: int = 1

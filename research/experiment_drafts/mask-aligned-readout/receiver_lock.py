"""Cross-process ownership for independent per-cell receiver workers."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from receiver_assets import output_root


@contextmanager
def claim_receiver_stage(root: Path, stage: str, fold: str) -> Iterator[None]:
    """Claim one profile or paired-fit cell until its receiver stage completes.

    The exclusive file creation prevents duplicate workers for one fold from
    racing on shared receipts or run directories. A lock left by a killed
    process is preserved as a fail-closed signal for operator inspection.
    """
    if stage not in ("profile", "fit") or fold not in ("F0", "F2"):
        raise ValueError(f"unsupported receiver stage/fold: {stage}/{fold}")
    path = output_root(root) / "locks" / f"{stage}_{fold}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        try:
            with path.open("x", encoding="utf-8") as stream:
                created = True
                json.dump(
                    {
                        "schema_version": 1,
                        "stage": stage,
                        "fold": fold,
                        "pid": os.getpid(),
                    },
                    stream,
                    sort_keys=True,
                )
                stream.write("\n")
        except FileExistsError as exc:
            raise FileExistsError(
                f"receiver {stage} lock exists for {fold}; inspect the owner before retrying: {path}"
            ) from exc
        yield
    finally:
        if created:
            path.unlink(missing_ok=True)

"""Feature-only teacher exports padded explicitly outside the clean fitting pool."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

from audit import require_fit, teacher_targets
from commands import prediction_command, run_name
from common import command_environment, digest, output_root, read_json, write_new

# Canonical metadata is heterogeneous; no rows/labels enter receipts.


def export_prediction(
    assets: dict[str, Any], cell: str, role: str, source: str, scope: str
) -> None:
    """Run the canonical FP32 feature-only predictor, preserving every existing attempt."""
    require_fit(assets, cell, role)
    root = Path(assets["project_root"])
    name = f"{run_name(cell, role)}_{scope}"
    final = root / "cache" / f"pred_{name}_{source}.npy"
    if final.exists() or final.with_suffix(".json").exists():
        raise FileExistsError(f"prediction already exists: {final}")
    log = output_root(assets) / "logs" / f"predict_{name}_{source}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        subprocess.run(
            prediction_command(assets, cell, role, source, scope),
            env=command_environment(assets),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def scatter_fit_predictions(assets: dict[str, Any], cell: str, role: str, source: str) -> None:
    """Adapt a fit-only export to the legacy cache index without predicting excluded rows."""
    import numpy as np

    from wnn.splits import load_campaign_split

    root = Path(assets["project_root"])
    split = load_campaign_split(Path(assets["splits"][cell]["path"]), root)
    run = run_name(cell, role)
    approved = require_fit(assets, cell, role)
    incoming = root / "cache" / f"pred_{run}_fit_{source}.npy"
    meta = read_json(incoming.with_suffix(".json"))
    groups = list(split.train_groups if source == "train" else split.valid_groups)
    if (
        meta["row_groups"] != groups
        or meta["campaign_export_scope"] != "fit"
        or meta["feature_only"] is not True
        or meta["inference_precision"] != "fp32"
        or meta["split_manifest_sha256"] != split.sha256
        or meta["checkpoint_sha256"] != approved["checkpoint_sha256"]
        or meta["epoch"] != 3
        or digest(incoming) != meta["prediction_sha256"]
    ):
        raise ValueError("fit-only export is not the expected teacher/checkpoint/membership")
    with np.load(root / "cache" / f"{source}_targets.npz", allow_pickle=False) as targets:
        order = list(map(int, targets["row_group"]))
    if len(set(order)) != len(order) or not set(groups).issubset(order):
        raise ValueError("target cache row groups are duplicated or omit fitting groups")
    values = np.load(incoming, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(groups), 20000, 2) or values.dtype != np.float32:
        raise ValueError("fit-only export shape/dtype changed")
    final = root / "cache" / f"pred_{run}_{source}.npy"
    partial = final.with_suffix(".partial.npy")
    if any(path.exists() for path in (final, partial, final.with_suffix(".json"))):
        raise FileExistsError(f"cache output already exists: {final}")
    if shutil.disk_usage(final.parent).free < len(order) * 20000 * 8 + 5 * (1 << 30):
        raise OSError("insufficient physical scratch for padded cache plus 5 GiB reserve")
    out = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(len(order), 20000, 2)
    )
    positions = {group: index for index, group in enumerate(groups)}
    for index, group in enumerate(order):
        if group in positions:
            value = values[positions[group]]
            if not np.isfinite(value).all():
                raise ValueError("nonfinite fitting teacher prediction")
            out[index] = value
        else:
            out[index] = 0
    out.flush()
    del out
    partial.rename(final)
    meta.update(
        row_groups=order,
        evaluation_membership_sha256=hashlib.sha256(
            np.asarray(order, dtype="<i8").tobytes()
        ).hexdigest(),
        prediction_sha256=digest(final),
        fitting_only_groups=groups,
        zeros_outside_fit=True,
        campaign_export_scope="fit-only-scattered-to-full-index",
        unpadded_prediction_sha256=digest(incoming),
        unpadded_sidecar_sha256=digest(incoming.with_suffix(".json")),
    )
    write_new(final.with_suffix(".json"), meta)


def build_targets(assets: dict[str, Any], cell: str) -> None:
    """Materialize both newly trained teacher caches; never mix folds or production weights."""
    for role in ("teacher_a", "teacher_b"):
        for source in ("train", "valid"):
            root = Path(assets["project_root"])
            incoming = root / "cache" / f"pred_{run_name(cell, role)}_fit_{source}.npy"
            final = root / "cache" / f"pred_{run_name(cell, role)}_{source}.npy"
            if not incoming.exists():
                export_prediction(assets, cell, role, source, "fit")
            if not final.exists():
                scatter_fit_predictions(assets, cell, role, source)
    targets = teacher_targets(assets, cell)
    receipt = output_root(assets) / "targets" / f"{cell}.json"
    record = {"cell": cell, "teachers": targets}
    if receipt.exists():
        if read_json(receipt) != record:
            raise ValueError("existing teacher cache receipt changed")
    else:
        write_new(receipt, record)

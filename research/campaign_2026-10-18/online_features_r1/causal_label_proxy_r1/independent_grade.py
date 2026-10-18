from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import cast

import numpy as np

from wnn.data import WARMUP, SequenceFile
from wnn.gmetric import METRIC_ID, GlobalAccumulator
from wnn.splits import load_campaign_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(
    recipe_path: Path,
    predictions_path: Path,
    fit_grade_path: Path,
    root: Path,
) -> dict[str, object]:
    recipe_bytes = recipe_path.read_bytes()
    recipe = cast(dict[str, object], json.loads(recipe_bytes))
    if recipe["metric_id"] != METRIC_ID:
        raise ValueError(f"unexpected metric: {recipe['metric_id']}")

    selection = cast(dict[str, object], recipe["selection"])
    split_path = root / str(selection["split_manifest"])
    split = load_campaign_split(split_path, root)
    if split.sha256 != selection["split_manifest_sha256"]:
        raise ValueError("split manifest hash differs from the frozen recipe")
    group_components = {(side, group): component for side, group, component in split.components}
    groups = cast(list[dict[str, int]], selection["evaluation_groups"])
    evaluation_ids = {group for group in split.evaluation_groups}
    for item in groups:
        group_id = int(item["row_group"])
        component = int(item["component"])
        if group_id not in evaluation_ids:
            raise ValueError(f"row group {group_id} is outside frozen evaluation membership")
        if group_components.get(("valid", group_id)) != component:
            raise ValueError(f"component mismatch for row group {group_id}")

    predictions = np.load(predictions_path, allow_pickle=False)
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError(f"expected a two-target prediction matrix, got {predictions.shape}")
    if not np.isfinite(predictions).all():
        raise ValueError("prediction matrix contains nonfinite values")

    scorer = GlobalAccumulator()
    source = SequenceFile(root / "datasets/valid.parquet")
    offset = 0
    components: list[int] = []
    for item in groups:
        group_id = int(item["row_group"])
        component = int(item["component"])
        sequence = source.read(group_id, targets=True, mask=True)
        if sequence.seq_ix != int(item["seq_ix"]):
            raise ValueError(f"sequence identity mismatch at row group {group_id}")
        if sequence.targets is None or sequence.scored is None:
            raise ValueError(f"target or mask missing at row group {group_id}")

        target_indices = np.arange(WARMUP, sequence.features.shape[0] - 1)
        count = target_indices.size
        prediction_slice = predictions[offset : offset + count]
        if prediction_slice.shape[0] != count:
            raise ValueError("prediction rows ended before the frozen sequence list")
        scorer.add(
            sequence.targets[target_indices],
            prediction_slice,
            sequence.scored[target_indices],
            group=component,
        )
        components.append(component)
        offset += count
    if offset != predictions.shape[0]:
        raise ValueError(f"unused prediction rows remain: {predictions.shape[0] - offset}")

    observed = scorer.result()
    fit_grade = cast(dict[str, object], json.loads(fit_grade_path.read_bytes()))
    expected_wp = float(fit_grade["lagged_proxy_wp"])
    delta = abs(float(observed["wp"]) - expected_wp)
    if delta > 1e-12:
        raise ValueError(f"independent WP differs from the runner receipt by {delta}")
    return {
        "id": "OPS-ONLINE-LAGGED-PROXY-INDEPENDENT-GRADE-20261018-R1",
        "status": "independent_metric_regrade_passed",
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "split_manifest_sha256": split.sha256,
        "prediction_sha256": _sha256(predictions_path),
        "fit_grade_receipt_sha256": _sha256(fit_grade_path),
        "metric_id": observed["metric_id"],
        "lagged_proxy_wp": observed["wp"],
        "target_wp": {"t0": observed["t0"], "t1": observed["t1"]},
        "runner_receipt_wp": expected_wp,
        "absolute_wp_difference": delta,
        "scored_rows": scorer.selected_rows,
        "evaluation_row_groups": len(groups),
        "evaluation_components": components,
        "prediction_rows": int(predictions.shape[0]),
        "alignment": "independently paired each prediction at step t with target/mask row t-1",
        "exposure": selection["exposure"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently regrade a frozen proxy cache.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--fit-grade", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _score(
        args.recipe.resolve(strict=True),
        args.predictions.resolve(strict=True),
        args.fit_grade.resolve(strict=True),
        args.root.resolve(strict=True),
    )
    output = args.output.resolve()
    if not output.is_relative_to(args.root.resolve(strict=True)):
        raise ValueError("independent grade receipt must stay in the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)


if __name__ == "__main__":
    main()

"""Exact admission checks for the 2026-10-06 C1-C4 campaign recipes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

CAMPAIGN_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
SOURCE_COEFFICIENT_SUM = 0.8125
FAMILY_ORDER = ("strong", "full_finetuned", "raw_wmse", "samefold_baseline")
RECIPE_VARIANTS = {
    "C1": {
        "id": "E-504",
        "variant": "homogeneous",
        "family_weight": 0.0,
        "normalized": False,
        "cache_root": "cache/bd26_four_family_f0_v1",
        "recipe_path": "research/campaign_2026-10-06/bandit_continuation/local/E504_C1_PAIRED.json",
        "recipe_sha256": "cc0709e6127d8bb38f119606e64d99c65683467098faeb3e0ddb24983515e515",
    },
    "C2": {
        "id": "E-505",
        "variant": "source",
        "family_weight": 0.0,
        "normalized": False,
        "cache_root": "cache/bd26_four_family_f0_v1",
        "recipe_path": "research/campaign_2026-10-06/bandit_continuation/local/E505_C2_PAIRED.json",
        "recipe_sha256": "3b432a086cee575f0215f11a2d8c45783db3bd313e1171d7e0bf32bb20d3196a",
    },
    "C3": {
        "id": "E-506",
        "variant": "source",
        "family_weight": 0.1,
        "normalized": False,
        "cache_root": "cache/bd26_four_family_f0_v1",
        "recipe_path": "research/campaign_2026-10-06/bandit_continuation/local/E506_C3_PAIRED.json",
        "recipe_sha256": "6acd7901f05f85eab0702acd82f3375f055db0f60bcb3098bdb8bc8bf722101a",
    },
    "C4": {
        "id": "E-507",
        "variant": "normalized",
        "family_weight": 0.1,
        "normalized": True,
        "cache_root": "cache/bd26_four_family_f0_v1",
        "recipe_path": "research/campaign_2026-10-06/bandit_continuation/local/E507_C4_PAIRED.json",
        "recipe_sha256": "7835167ac07dc7b475542cf94632584af8e222fc8d895e7f0749513644abdacf",
    },
}
_EXPECTED_OPTIONS = {
    "--device",
    "--arch",
    "--hidden",
    "--layers",
    "--seed",
    "--split-manifest",
    "--valid-fold",
    "--train-source",
    "--train-seqs",
    "--init",
    "--epochs",
    "--checkpoint-policy",
    "--checkpoint-epoch",
    "--no-eval-during-fit",
    "--batch",
    "--chunk",
    "--loss",
    "--lr",
    "--wd",
    "--optim",
    "--sched",
    "--propensity",
    "--prop-power",
    "--prop-eps",
    "--valid-true-mask",
    "--aux-struct",
    "--aux-next",
    "--distill-boot-h",
    "--fast",
    "--save-training-state",
    "--distill-cache-root",
    "--distill-cache-variant",
    "--distill-alpha",
    "--distill-family-weight",
    "--name",
}


def sha256_bytes(data: bytes) -> str:
    """Return a SHA-256 digest for an exact serialized artifact."""
    return hashlib.sha256(data).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _digest(value: object, label: str) -> str:
    result = _string(value, label)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _records_by_run(value: object, label: str) -> dict[str, Mapping[str, object]]:
    records = _sequence(value, label)
    result: dict[str, Mapping[str, object]] = {}
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"{label}[{index}]")
        run = _string(record.get("run"), f"{label}[{index}].run")
        if run in result:
            raise ValueError(f"{label} repeats source run {run}")
        result[run] = record
    return result


def _source_snapshot(
    sealed_families: Mapping[str, object],
    realized_snapshot: Mapping[str, object],
    *,
    base_manifest_sha256: str,
) -> dict[str, Mapping[str, object]]:
    if realized_snapshot.get("format") != "wnn.bandit.source-families-realized":
        raise ValueError("realized source snapshot has an unsupported format")
    if realized_snapshot.get("version") != 1:
        raise ValueError("realized source snapshot has an unsupported version")
    if realized_snapshot.get("base_source_families_sha256") != base_manifest_sha256:
        raise ValueError("realized source snapshot is not tied to the sealed source manifest")
    _digest(
        realized_snapshot.get("e508_profile_review_receipt_sha256"),
        "realized snapshot E508 profile review hash",
    )

    original_rows = _records_by_run(sealed_families.get("families"), "SOURCE_FAMILIES.families")
    realized_rows = _records_by_run(realized_snapshot.get("families"), "realized.families")
    if set(original_rows) != set(realized_rows) or len(original_rows) != 8:
        raise ValueError("realized source snapshot must preserve all eight registered teachers")
    for run, original in original_rows.items():
        realized = realized_rows[run]
        original_status = original.get("status")
        if original_status == "existing_audited":
            if dict(realized) != dict(original):
                raise ValueError(f"realized snapshot changed audited source {run}")
            _digest(realized.get("checkpoint_sha256"), f"{run}.checkpoint_sha256")
            _digest(realized.get("provenance_sha256"), f"{run}.provenance_sha256")
            continue
        if original_status != "planned_fit" or original.get("family") != "samefold_baseline":
            raise ValueError(f"source {run} has an unsupported registration status")
        for key in (
            "family",
            "run",
            "seed",
            "epoch_zero_based",
            "main_coefficient",
            "hidden",
            "layers",
        ):
            if realized.get(key) != original.get(key):
                raise ValueError(f"realized E510 teacher changed registered {key}: {run}")
        if realized.get("status") != "produced_by_e510":
            raise ValueError(f"E510 teacher {run} is not marked as produced")
        _digest(realized.get("checkpoint_sha256"), f"{run}.checkpoint_sha256")
        _digest(realized.get("provenance_sha256"), f"{run}.provenance_sha256")
    return dict(realized_rows)


def _parse_command(command: object, label: str) -> tuple[int, dict[str, str | bool]]:
    values = [
        _string(item, f"{label}[{index}]") for index, item in enumerate(_sequence(command, label))
    ]
    if values[:2] != [".venv/bin/python", "scripts/train_rnn.py"]:
        raise ValueError(f"{label} does not invoke the frozen production trainer")
    options: dict[str, str | bool] = {}
    index = 2
    while index < len(values):
        option = values[index]
        if not option.startswith("--") or option in options:
            raise ValueError(f"{label} has an invalid or repeated option: {option}")
        if index + 1 < len(values) and not values[index + 1].startswith("--"):
            options[option] = values[index + 1]
            index += 2
        else:
            options[option] = True
            index += 1
    if set(options) != _EXPECTED_OPTIONS:
        missing = sorted(_EXPECTED_OPTIONS - set(options))
        unexpected = sorted(set(options) - _EXPECTED_OPTIONS)
        raise ValueError(f"{label} option set changed; missing={missing}, unexpected={unexpected}")
    seed_text = options["--seed"]
    if not isinstance(seed_text, str) or not seed_text.isdecimal():
        raise ValueError(f"{label} seed must be a decimal string")
    return int(seed_text), options


def _validate_commands(recipe: Mapping[str, object], expected: Mapping[str, object]) -> None:
    commands = _sequence(recipe.get("commands"), "recipe.commands")
    digests = _sequence(recipe.get("command_sha256"), "recipe.command_sha256")
    if len(commands) != 2 or len(digests) != 2:
        raise ValueError("preregistered recipe must contain exactly two paired commands")

    cache_root = _string(expected.get("cache_root"), "registered cache root")
    if recipe.get("cache_root") != cache_root:
        raise ValueError("preregistered recipe cache root differs from the sealed campaign root")
    expected_seeds = (3, 4)
    for index, (raw_command, expected_digest) in enumerate(zip(commands, digests, strict=True)):
        command = [
            _string(item, f"recipe.commands[{index}][{item_index}]")
            for item_index, item in enumerate(_sequence(raw_command, f"recipe.commands[{index}]"))
        ]
        actual_digest = sha256_bytes(json.dumps(command, separators=(",", ":")).encode())
        if actual_digest != _digest(expected_digest, f"recipe.command_sha256[{index}]"):
            raise ValueError(f"recipe command {index} does not match its registered SHA-256")
        seed, options = _parse_command(command, f"recipe.commands[{index}]")
        if seed != expected_seeds[index]:
            raise ValueError("paired commands must preserve seed order 3, 4")
        run = f"c20261003_combo_s{seed}_f0"
        expected_values: dict[str, str] = {
            "--name": f"bd26_{str(recipe.get('arm')).lower()}_s{seed}_f0",
            "--device": "cuda",
            "--arch": "gru",
            "--hidden": "256",
            "--layers": "2",
            "--split-manifest": "research/campaign_2026-09-29/artifacts/splits/fold0.json",
            "--valid-fold": "0",
            "--train-source": "train+valid",
            "--train-seqs": "10607",
            "--init": f"{run}@ep3",
            "--epochs": "2",
            "--checkpoint-policy": "fixed",
            "--checkpoint-epoch": "1",
            "--batch": "16",
            "--chunk": "2000",
            "--loss": "wmse",
            "--lr": "0.0001",
            "--wd": "0.01",
            "--optim": "adamw",
            "--sched": "onecycle",
            "--propensity": "c20260929_prop_f0",
            "--prop-power": "2",
            "--prop-eps": "0.03",
            "--valid-true-mask": "0.03",
            "--aux-struct": "0.3",
            "--aux-next": "0.3",
            "--distill-boot-h": "0",
            "--distill-cache-root": cache_root,
            "--distill-cache-variant": str(expected["variant"]),
            "--distill-alpha": "0.7",
            "--distill-family-weight": str(expected["family_weight"]),
        }
        for option, value in expected_values.items():
            if options.get(option) != value:
                raise ValueError(
                    f"recipe command {index} changed {option}: {options.get(option)!r}"
                )
        for flag in ("--no-eval-during-fit", "--fast", "--save-training-state"):
            if options.get(flag) is not True:
                raise ValueError(f"recipe command {index} must retain {flag}")


def verify_derived_distillation_targets(
    cache_root: Path,
    teacher_cache_root: Path,
    cache_metadata: Mapping[str, object],
    realized_snapshot: Mapping[str, object],
    target_orders: Mapping[str, Sequence[int]],
    expected_fit_groups: Mapping[str, Sequence[int]],
    *,
    variant: str,
    require_families: bool,
    chunk_rows: int = 8,
) -> dict[str, object]:
    """Rebuild selected cache targets from bound source predictions in bounded chunks.

    The generic cache validator verifies each requested array's hash, shape, dtype,
    finiteness, row order, and fold membership. This second pass proves the target
    bytes equal the exact source-scale/normalized/homogeneous/family arithmetic
    over the eight registered teacher prediction files.
    """
    import numpy as np

    from wnn.distillation import TeacherSource, combine_teacher_chunk

    if variant not in ("source", "normalized", "homogeneous"):
        raise ValueError(f"unknown derived main target variant: {variant}")
    if chunk_rows < 1 or chunk_rows > 32:
        raise ValueError("derived target verification chunk must be between 1 and 32 sequences")
    cache_root = Path(cache_root).resolve(strict=True)
    teacher_cache_root = Path(teacher_cache_root).resolve(strict=True)
    source_rows = _records_by_run(realized_snapshot.get("families"), "realized.families")
    coefficients = _records_by_run(
        cache_metadata.get("source_coefficients"), "cache source_coefficients"
    )
    metadata_sources = _records_by_run(cache_metadata.get("sources"), "cache sources")
    if set(source_rows) != set(coefficients) or set(source_rows) != set(metadata_sources):
        raise ValueError("derived-target inputs do not contain exactly the registered teachers")

    files = _mapping(cache_metadata.get("files"), "cache files")
    variants = {
        "source": "main_source",
        "normalized": "main_normalized",
        "homogeneous": "main_homogeneous",
    }
    selected = [f"{variants[variant]}_train", f"{variants[variant]}_valid"]
    if require_families:
        selected.extend(("families_train", "families_valid"))
    target_paths: dict[str, object] = {}
    for key in selected:
        entry = _mapping(files.get(key), f"cache files.{key}")
        relative = _string(entry.get("path"), f"cache files.{key}.path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or len(relative_path.parts) != 1:
            raise ValueError(f"cache target path must be one relative filename: {key}")
        target_path = (cache_root / relative_path).resolve(strict=True)
        try:
            target_path.relative_to(cache_root)
        except ValueError as error:
            raise ValueError(f"cache target escapes its registered root: {key}") from error
        array = np.load(target_path, mmap_mode="r", allow_pickle=False)
        split_name = key.rsplit("_", 1)[1]
        width = 2 if key.startswith("main_") else len(FAMILY_ORDER) * 2
        expected_shape = (len(expected_fit_groups[split_name]), 20_000, width)
        if array.shape != expected_shape or array.dtype != np.float32:
            raise ValueError(f"derived cache target shape or dtype is invalid for {key}")
        target_paths[key] = array

    source_arrays: dict[str, dict[str, object]] = {"train": {}, "valid": {}}
    source_identities: dict[str, dict[str, object]] = {}
    for run, expected in source_rows.items():
        source_record = metadata_sources[run]
        prediction_records = _mapping(source_record.get("predictions"), f"{run}.predictions")
        family = _string(expected.get("family"), f"{run}.family")
        epoch = _integer(expected.get("epoch_zero_based"), f"{run}.epoch")
        coefficient_record = coefficients[run]
        if (
            coefficient_record.get("family") != family
            or coefficient_record.get("epoch") != epoch
            or not math.isclose(
                _number(coefficient_record.get("coefficient"), f"{run}.coefficient"),
                _number(expected.get("main_coefficient"), f"{run}.registered coefficient"),
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            or source_record.get("checkpoint_sha256") != expected.get("checkpoint_sha256")
            or source_record.get("epoch") != epoch
            or source_record.get("family") != family
        ):
            raise ValueError(f"source prediction manifest differs from registered teacher {run}")
        for split_name in ("train", "valid"):
            prediction_record = _mapping(
                prediction_records.get(split_name), f"{run}.predictions.{split_name}"
            )
            path = (teacher_cache_root / f"pred_{run}_{split_name}.npy").resolve(strict=True)
            try:
                path.relative_to(teacher_cache_root)
            except ValueError as error:
                raise ValueError(
                    f"source prediction escapes its registered cache root: {run}/{split_name}"
                ) from error
            sidecar_path = path.with_suffix(".json")
            sidecar_path = sidecar_path.resolve(strict=True)
            try:
                sidecar_path.relative_to(teacher_cache_root)
            except ValueError as error:
                raise ValueError(
                    f"source prediction sidecar escapes its registered cache root: {run}/{split_name}"
                ) from error
            sidecar_bytes = sidecar_path.read_bytes()
            sidecar = json.loads(sidecar_bytes)
            expected_order = list(map(int, target_orders[split_name]))
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                array.shape != (len(expected_order), 20_000, 2)
                or array.dtype != np.float32
                or sidecar.get("run") != run
                or sidecar.get("split") != split_name
                or sidecar.get("epoch") != epoch
                or sidecar.get("checkpoint_sha256") != expected.get("checkpoint_sha256")
                or sidecar.get("split_manifest_sha256") != CAMPAIGN_SPLIT_SHA256
                or sidecar.get("row_groups") != expected_order
                or sidecar.get("feature_only_export") is not True
                or sidecar.get("label_columns_read") is not False
                or sidecar.get("mask_columns_read") is not False
            ):
                raise ValueError(
                    f"source prediction schema/order is invalid for {run}/{split_name}"
                )
            prediction_sha256 = _sha256_file(path)
            sidecar_sha256 = sha256_bytes(sidecar_bytes)
            if (
                sidecar.get("prediction_sha256") != prediction_sha256
                or prediction_record.get("prediction_sha256") != prediction_sha256
                or prediction_record.get("sidecar_sha256") != sidecar_sha256
                or prediction_record.get("checkpoint_sha256") != expected.get("checkpoint_sha256")
                or prediction_record.get("epoch") != epoch
                or prediction_record.get("row_groups_sha256")
                != hashlib.sha256(np.asarray(expected_order, dtype="<i8").tobytes()).hexdigest()
            ):
                raise ValueError(
                    f"source prediction bytes differ from cache builder record: {run}/{split_name}"
                )
            if Path(str(prediction_record.get("prediction_path", ""))).name != path.name:
                raise ValueError(f"source prediction path identity differs for {run}/{split_name}")
            source_arrays[split_name][run] = array
            source_identities[f"{run}/{split_name}"] = {
                "prediction_sha256": prediction_sha256,
                "sidecar_sha256": sidecar_sha256,
            }

    source_specs = [
        TeacherSource(
            family=_string(coefficients[run].get("family"), f"{run}.family"),
            run=run,
            coefficient=_number(coefficients[run].get("coefficient"), f"{run}.coefficient"),
            epoch=_integer(coefficients[run].get("epoch"), f"{run}.epoch"),
        )
        for run in coefficients
    ]
    verified_chunks = 0
    for split_name in ("train", "valid"):
        target_order = list(map(int, target_orders[split_name]))
        fitting = set(map(int, expected_fit_groups[split_name]))
        selected_order = [group for group in target_order if group in fitting]
        positions = {group: index for index, group in enumerate(target_order)}
        source_rows_by_index = np.asarray(
            [positions[group] for group in selected_order], dtype=np.int64
        )
        source_cache = source_arrays[split_name]
        destination_keys = [f"{variants[variant]}_{split_name}"]
        if require_families:
            destination_keys.append(f"families_{split_name}")
        for start in range(0, len(selected_order), chunk_rows):
            stop = min(start + chunk_rows, len(selected_order))
            rows = source_rows_by_index[start:stop]
            source_chunk = {
                run: np.asarray(array[rows], dtype=np.float32)
                for run, array in source_cache.items()
            }
            if any(not np.isfinite(array).all() for array in source_chunk.values()):
                raise ValueError(f"source teacher predictions are nonfinite for {split_name}")
            main, normalized, homogeneous, families = combine_teacher_chunk(
                source_chunk, source_specs, FAMILY_ORDER, "strong"
            )
            values_by_name = {
                "main_source": main,
                "main_normalized": normalized,
                "main_homogeneous": homogeneous,
                "families": families,
            }
            for key in destination_keys:
                name, _ = key.rsplit("_", 1)
                actual = target_paths[key][start:stop]
                if not np.isfinite(actual).all():
                    raise ValueError(f"derived cache target is nonfinite for {key}")
                if not np.array_equal(actual, values_by_name[name]):
                    raise ValueError(f"derived cache target values differ from teachers for {key}")
            verified_chunks += 1
    return {
        "selected_target_keys": selected,
        "source_prediction_identities": source_identities,
        "derived_chunks_compared_bitwise": verified_chunks,
        "chunk_rows": chunk_rows,
    }


def _sha256_file(path: Path) -> str:
    """Hash a large source or cache artifact in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def validate_bandit_recipe(
    recipe: Mapping[str, object],
    sealed_families: Mapping[str, object],
    realized_snapshot: Mapping[str, object],
    cache_metadata: Mapping[str, object],
    *,
    split_manifest: Mapping[str, object],
    recipe_path: str,
    recipe_sha256: str,
    sealed_families_sha256: str,
    realized_snapshot_sha256: str,
    expected_realized_snapshot_sha256: str,
    actual_artifact_hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    """Bind one E504-E507 cache to every sealed teacher and its exact fit recipe.

    Args:
        recipe: One preregistered paired recipe JSON object.
        sealed_families: The immutable ``SOURCE_FAMILIES.json`` JSON object.
        realized_snapshot: E510's source-family snapshot after both baseline fits.
        cache_metadata: The exact ``metadata.json`` emitted by the cache builder.
        recipe_path: Repository-relative path of the preregistered arm JSON.
        recipe_sha256: SHA-256 of the exact preregistration bytes.
        sealed_families_sha256: Digest of the original source-family manifest bytes.
        realized_snapshot_sha256: Digest of E510's realized source snapshot bytes.
        expected_realized_snapshot_sha256: Controller-pinned snapshot digest.
        actual_artifact_hashes: Disk hashes keyed by run and artifact kind.

    Returns:
        A compact report that can be attached to the controller's admission record.

    Raises:
        ValueError: If any recipe, source, cache, or artifact identity differs.
    """
    sealed_families_sha256 = _digest(sealed_families_sha256, "sealed source-family hash")
    realized_snapshot_sha256 = _digest(realized_snapshot_sha256, "realized snapshot hash")
    expected_realized_snapshot_sha256 = _digest(
        expected_realized_snapshot_sha256, "controller-pinned realized snapshot hash"
    )
    if sealed_families_sha256 != recipe.get("registered_manifest_sha256"):
        raise ValueError("recipe source-family hash differs from the sealed manifest")
    if recipe.get("source_families_manifest") != "SOURCE_FAMILIES.json":
        raise ValueError("recipe points to a different source-family manifest")
    if realized_snapshot_sha256 != expected_realized_snapshot_sha256:
        raise ValueError("realized source snapshot differs from the controller-pinned hash")
    expected_source_rows = _source_snapshot(
        sealed_families,
        realized_snapshot,
        base_manifest_sha256=sealed_families_sha256,
    )

    arm = _string(recipe.get("arm"), "recipe.arm")
    expected = RECIPE_VARIANTS.get(arm)
    if expected is None:
        raise ValueError(f"unregistered campaign arm: {arm}")
    if (
        recipe_path != expected["recipe_path"]
        or _digest(recipe_sha256, "preregistered recipe source hash") != expected["recipe_sha256"]
    ):
        raise ValueError(f"{arm} recipe path or bytes differ from its immutable registration")
    for key, value in expected.items():
        if key in {"variant", "recipe_path", "recipe_sha256", "cache_root"}:
            continue
        recipe_key = {
            "family_weight": "family_auxiliary_weight",
            "normalized": "normalizes_main_teacher",
        }.get(key, key)
        if recipe.get(recipe_key) != value:
            raise ValueError(f"{arm} preregistration changed {key}")
    if recipe.get("status") != "preregistered_not_admitted":
        raise ValueError("recipe must remain preregistered and unadmitted at this gate")
    if recipe.get("fold") != 0 or recipe.get("seeds") != [3, 4]:
        raise ValueError("C1-C4 are fixed to discovery fold 0 and paired seeds 3/4")
    if recipe.get("split_sha256") != CAMPAIGN_SPLIT_SHA256:
        raise ValueError("preregistered recipe uses a different fold-0 split")
    if (
        recipe.get("fit_train_groups") != 10_535
        or recipe.get("fit_valid_groups") != 1_405
        or recipe.get("evaluation_valid_groups") != 468
    ):
        raise ValueError("preregistered fit/evaluation group counts changed")
    if not math.isclose(
        _number(recipe.get("source_coefficient_sum"), "recipe source coefficient sum"),
        SOURCE_COEFFICIENT_SUM,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("recipe source coefficient sum changed")
    if not math.isclose(
        _number(recipe.get("distillation_alpha"), "recipe distillation alpha"),
        0.7,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("recipe distillation alpha changed")
    if not math.isclose(
        _number(recipe.get("family_auxiliary_weight"), "recipe family weight"),
        float(expected["family_weight"]),
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("recipe family-head weight differs from its registered arm")
    if recipe.get("normalizes_main_teacher") is not expected["normalized"]:
        raise ValueError("recipe main-target normalization differs from its registered arm")
    _validate_commands(recipe, expected)

    if cache_metadata.get("format") != "wnn.heterogeneous-distillation-cache":
        raise ValueError("cache metadata has an unexpected format")
    if cache_metadata.get("version") != 1 or cache_metadata.get("fold") != 0:
        raise ValueError("cache metadata has an unexpected version or fold")
    if cache_metadata.get("split_manifest_sha256") != CAMPAIGN_SPLIT_SHA256:
        raise ValueError("cache metadata uses a different fold-0 split")
    if split_manifest.get("fold") != 0:
        raise ValueError("sealed fold-0 split manifest has the wrong fold")
    split_fit = _mapping(split_manifest.get("fit"), "fold-0 split fit groups")
    split_evaluation = _mapping(split_manifest.get("evaluation"), "fold-0 split evaluation groups")
    expected_fit_groups = {
        "train": [
            _integer(value, f"fold-0 fit train group {index}")
            for index, value in enumerate(_sequence(split_fit.get("train"), "fit.train"))
        ],
        "valid": [
            _integer(value, f"fold-0 fit valid group {index}")
            for index, value in enumerate(_sequence(split_fit.get("valid"), "fit.valid"))
        ],
    }
    expected_evaluation_groups = [
        _integer(value, f"fold-0 evaluation group {index}")
        for index, value in enumerate(_sequence(split_evaluation.get("valid"), "evaluation.valid"))
    ]
    if (
        expected_fit_groups["train"] != sorted(set(expected_fit_groups["train"]))
        or expected_fit_groups["valid"] != sorted(set(expected_fit_groups["valid"]))
        or expected_evaluation_groups != sorted(set(expected_evaluation_groups))
        or len(expected_fit_groups["train"]) != 10_535
        or len(expected_fit_groups["valid"]) != 1_405
        or len(expected_evaluation_groups) != 468
    ):
        raise ValueError("sealed fold-0 group membership is not the registered exact split")
    if (
        cache_metadata.get("fit_groups") != expected_fit_groups
        or cache_metadata.get("evaluation_valid_groups") != expected_evaluation_groups
    ):
        raise ValueError(
            "cache fit/evaluation group identities differ from the sealed fold-0 split"
        )
    if tuple(cache_metadata.get("family_order", ())) != FAMILY_ORDER:
        raise ValueError("cache metadata family-head order changed")
    if cache_metadata.get("homogeneous_family") != "strong":
        raise ValueError("shared cache must define the C1 homogeneous target from strong")
    if not math.isclose(
        _number(cache_metadata.get("coefficient_sum"), "cache coefficient sum"),
        SOURCE_COEFFICIENT_SUM,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("cache source coefficients do not sum to 0.8125")

    coefficients_by_run = _records_by_run(
        cache_metadata.get("source_coefficients"), "cache source_coefficients"
    )
    cache_sources_by_run = _records_by_run(cache_metadata.get("sources"), "cache sources")
    if set(coefficients_by_run) != set(expected_source_rows):
        raise ValueError("cache source runs differ from the realized eight-teacher snapshot")
    if set(cache_sources_by_run) != set(expected_source_rows):
        raise ValueError("cache checkpoint records differ from the realized source snapshot")

    expected_coefficient_sum = 0.0
    for run, expected_source in expected_source_rows.items():
        coefficient = _number(expected_source.get("main_coefficient"), f"{run}.coefficient")
        expected_coefficient_sum += coefficient
        cache_coefficient = coefficients_by_run[run]
        source_record = cache_sources_by_run[run]
        if (
            cache_coefficient.get("family") != expected_source.get("family")
            or cache_coefficient.get("epoch") != expected_source.get("epoch_zero_based")
            or source_record.get("family") != expected_source.get("family")
            or source_record.get("epoch") != expected_source.get("epoch_zero_based")
        ):
            raise ValueError(f"cache family or epoch differs from registered source {run}")
        for actual, label in (
            (cache_coefficient, "source coefficient"),
            (source_record, "source record"),
        ):
            value = _number(actual.get("coefficient"), f"{run} {label} coefficient")
            if not math.isclose(value, coefficient, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(f"cache coefficient differs from registered source {run}")
        checkpoint_sha256 = _digest(
            expected_source.get("checkpoint_sha256"), f"{run} snapshot checkpoint hash"
        )
        if (
            _digest(source_record.get("checkpoint_sha256"), f"{run} cache checkpoint hash")
            != checkpoint_sha256
        ):
            raise ValueError(f"cache checkpoint hash differs from registered source {run}")
        hashes = actual_artifact_hashes.get(run)
        if hashes is None:
            raise ValueError(f"actual checkpoint/provenance hashes are absent for {run}")
        if (
            _digest(hashes.get("checkpoint_sha256"), f"{run} actual checkpoint hash")
            != checkpoint_sha256
        ):
            raise ValueError(f"checkpoint bytes differ from the realized source snapshot: {run}")
        provenance_sha256 = _digest(
            expected_source.get("provenance_sha256"), f"{run} snapshot provenance hash"
        )
        if (
            _digest(hashes.get("provenance_sha256"), f"{run} actual provenance hash")
            != provenance_sha256
        ):
            raise ValueError(f"provenance bytes differ from the realized source snapshot: {run}")
    if not math.isclose(
        expected_coefficient_sum, SOURCE_COEFFICIENT_SUM, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("realized source snapshot coefficients do not sum to 0.8125")

    files = _mapping(cache_metadata.get("files"), "cache files")
    selected_variant = str(expected["variant"])
    required_keys = {f"main_{selected_variant}_{split_name}" for split_name in ("train", "valid")}
    if float(expected["family_weight"]) > 0:
        required_keys.update(f"families_{split_name}" for split_name in ("train", "valid"))
    if not required_keys.issubset(files):
        raise ValueError("cache is missing a target required by the preregistered treatment")
    epoch_convention = _mapping(cache_metadata.get("epoch_convention"), "cache epoch convention")
    if (
        epoch_convention.get("friend_at_ep1") != "second additional epoch"
        or epoch_convention.get("student_epochs") != 2
        or epoch_convention.get("student_checkpoint_policy") != "fixed"
        or epoch_convention.get("student_checkpoint_epoch") != 1
    ):
        raise ValueError("cache student checkpoint convention differs from preregistration")

    return {
        "recipe_id": recipe.get("id"),
        "arm": arm,
        "variant": selected_variant,
        "family_weight": expected["family_weight"],
        "source_families_sha256": sealed_families_sha256,
        "realized_source_snapshot_sha256": realized_snapshot_sha256,
        "source_count": len(expected_source_rows),
        "source_coefficient_sum": expected_coefficient_sum,
        "verified_runs": sorted(expected_source_rows),
        "preregistered_commands": recipe["commands"],
    }

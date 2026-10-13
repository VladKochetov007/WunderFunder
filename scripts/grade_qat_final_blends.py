"""Grade three predeclared four-member native prediction blends on fixed F0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha256_file(path: Path, *, evict_cache: bool = False) -> str:
    """Hash a file in bounded memory and optionally drop its page cache."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        if evict_cache:
            advise_dontneed(stream.fileno())
    return digest.hexdigest()


def advise_dontneed(fd: int, offset: int = 0, length: int = 0) -> None:
    """Ask Linux to discard clean cached pages without changing the source."""
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)


def command_options(command: list[str]) -> tuple[dict[str, str], set[str]]:
    """Extract the export CLI's value options and boolean switches."""
    values: dict[str, str] = {}
    flags: set[str] = set()
    for index, token in enumerate(command):
        if token == "--packed":
            flags.add(token)
        elif token.startswith("--"):
            if index + 1 >= len(command):
                raise ValueError(f"missing value for command option {token}")
            values[token] = command[index + 1]
    return values, flags


def read_json(path: Path) -> dict[str, object]:
    """Read a small frozen JSON record."""
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def verify_admissions(root: Path, recipe: dict[str, object]) -> None:
    """Bind each member result to its frozen, hash-pinned export command."""
    admissions = recipe["admissions"]
    members = recipe["members"]
    assert isinstance(admissions, dict) and isinstance(members, dict)
    parsed: dict[str, dict[str, object]] = {}
    for key, record in admissions.items():
        assert isinstance(record, dict)
        path = root / str(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"export admission changed: {path}")
        parsed[key] = read_json(path)

    native = recipe["native"]
    panel = recipe["panel"]
    assert isinstance(native, dict) and isinstance(panel, dict)
    for name, member in members.items():
        assert isinstance(member, dict)
        admission_key = str(member["admission"])
        admission = parsed[admission_key]
        if admission.get("source_sha256") != recipe["exporter_sha256"]:
            raise ValueError(f"admission source hash mismatch: {admission_key}")
        if admission.get("native_sha256") not in (None, native["sha256"]):
            raise ValueError(f"admission native hash mismatch: {admission_key}")
        if admission.get("split_sha256") != panel["split_sha256"]:
            raise ValueError(f"admission split hash mismatch: {admission_key}")
        if admission.get("ordered_panel_count") != panel["group_count"]:
            raise ValueError(f"admission group count mismatch: {admission_key}")
        jobs = admission.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError(f"invalid jobs in admission: {admission_key}")
        selected = [job for job in jobs if job.get("attempt") == member["attempt"]]
        if len(selected) != 1:
            raise ValueError(f"expected one frozen job for {name}")
        job = selected[0]
        command = job.get("command")
        if not isinstance(command, list):
            raise ValueError(f"invalid command for {name}")
        values, flags = command_options([str(token) for token in command])
        expected_values = {
            "--weights": str(member["weights_path"]),
            "--weights-sha256": str(member["weights_sha256"]),
            "--native": str(native["path"]),
            "--native-sha256": str(native["sha256"]),
            "--split": str(panel["split_path"]),
            "--split-sha256": str(panel["split_sha256"]),
            "--data": str(panel["data_path"]),
            "--data-sha256": str(panel["data_sha256"]),
            "--output": str(member["prediction_dir"]),
        }
        for option, expected in expected_values.items():
            if values.get(option) != expected:
                raise ValueError(f"frozen command {name} differs at {option}")
        if ("--packed" in flags) != bool(member["packed"]):
            raise ValueError(f"frozen packed-code mode mismatch: {name}")
        if str(job.get("output")) != str(member["prediction_dir"]):
            raise ValueError(f"frozen output path mismatch: {name}")
        exporter = str(root / "scripts/export_native_qat_member_panel.py")
        if exporter not in command:
            raise ValueError(f"frozen command does not invoke the reviewed exporter: {name}")


def validate_member(
    root: Path, recipe: dict[str, object], name: str
) -> tuple[dict[str, object], np.memmap]:
    """Verify one complete producer result and return its read-only memmap."""
    members = recipe["members"]
    panel = recipe["panel"]
    native = recipe["native"]
    assert isinstance(members, dict) and isinstance(panel, dict) and isinstance(native, dict)
    member = members[name]
    assert isinstance(member, dict)
    prediction_dir = root / str(member["prediction_dir"])
    result_path = prediction_dir / "result.json"
    predictions_path = prediction_dir / "predictions.npy"
    result = read_json(result_path)

    expected_fields = {
        "weights": str(member["weights_path"]),
        "weights_sha256": str(member["weights_sha256"]),
        "native_sha256": str(native["sha256"]),
        "split_sha256": str(panel["split_sha256"]),
        "data_sha256": str(panel["data_sha256"]),
        "source_sha256": str(recipe["exporter_sha256"]),
        "scope": str(panel["scope"]),
    }
    for field, expected in expected_fields.items():
        if result.get(field) != expected:
            raise ValueError(f"member result {name} has wrong {field}")
    if result.get("groups") != panel["evaluation_groups"]:
        raise ValueError(f"member result {name} has wrong ordered F0 groups")
    if result.get("reset_replay_bitwise") is not True:
        raise ValueError(f"member result {name} did not pass reset replay")
    if sha256_file(root / str(member["weights_path"])) != member["weights_sha256"]:
        raise ValueError(f"member weight archive changed: {name}")
    if sha256_file(predictions_path, evict_cache=True) != result.get("prediction_sha256"):
        raise ValueError(f"member prediction bytes do not match result receipt: {name}")

    predictions = np.load(predictions_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (int(panel["group_count"]), 20_000, 2)
    if not isinstance(predictions, np.memmap):
        raise ValueError(f"member predictions are not mmap-readable NPY: {name}")
    if predictions.shape != expected_shape or predictions.dtype != np.dtype("float32"):
        raise ValueError(f"member prediction shape/dtype mismatch: {name}")
    expected_bytes = int(np.prod(expected_shape)) * np.dtype("float32").itemsize
    if predictions_path.stat().st_size != int(predictions.offset) + expected_bytes:
        raise ValueError(f"member NPY header/payload length mismatch: {name}")
    score = result.get("score")
    if not isinstance(score, dict) or score.get("metric_id") != "global_wp_clip2_v1":
        raise ValueError(f"member score does not identify official metric: {name}")
    return result, predictions


def read_group_labels(parquet: pq.ParquetFile, group: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one fixed 20k source group and return labels plus official mask."""
    columns = ["seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"]
    table = parquet.read_row_group(group, columns=columns, use_threads=False)
    if table.num_rows != 20_000:
        raise ValueError(f"row group {group} is not one complete sequence")
    seq_ix = table["seq_ix"].to_numpy()
    steps = table["step_in_seq"].to_numpy()
    need = table["need_prediction"].to_numpy().astype(bool)
    if not np.all(seq_ix == seq_ix[0]) or not np.array_equal(steps, np.arange(20_000)):
        raise ValueError(f"row group {group} has invalid sequence/step identity")
    if not np.array_equal(need, np.arange(20_000) >= 99):
        raise ValueError(f"row group {group} violates the prediction warm-up mask")
    mask = need & table["is_scored"].to_numpy().astype(bool)
    targets = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")]).astype(
        np.float32, copy=False
    )
    return targets, mask


def blend_group(
    member_arrays: dict[str, np.memmap],
    names: list[str],
    index: int,
) -> np.ndarray:
    """Sum four float32 member vectors in native index order, before clipping."""
    total = np.zeros((20_000, 2), dtype=np.float32)
    term = np.empty_like(total)
    coefficient = np.float32(0.25)
    for name in names:
        values = np.asarray(member_arrays[name][index])
        if not np.isfinite(values).all():
            raise ValueError(f"member {name} has nonfinite predictions at group {index}")
        np.multiply(values, coefficient, out=term)
        np.add(total, term, out=total)
    total[~np.isfinite(total)] = np.float32(0.0)
    return total


def close_mmaps(arrays: dict[str, np.memmap]) -> None:
    """Close open memmaps before hashing their backing files."""
    for array in arrays.values():
        array._mmap.close()
    arrays.clear()


def main() -> None:
    """Validate the six frozen member panels and grade the three fixed blends."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    recipe_path = args.recipe.resolve()
    if sha256_file(recipe_path) != args.recipe_sha256:
        raise ValueError("frozen blend recipe SHA256 mismatch")
    recipe = read_json(recipe_path)
    if str(repo_root) != str(recipe["repo_root"]):
        raise ValueError("repository root differs from frozen blend recipe")
    if sha256_file(Path(__file__).resolve()) != recipe["grader_source_sha256"]:
        raise ValueError("grader source differs from frozen recipe")
    if (
        sha256_file(repo_root / "scripts/export_native_qat_member_panel.py")
        != recipe["exporter_sha256"]
    ):
        raise ValueError("native member exporter source changed")
    if sha256_file(repo_root / "src/wnn/gmetric.py") != recipe["metric_source_sha256"]:
        raise ValueError("official metric source changed")
    if sha256_file(repo_root / "native/src/python.rs") != recipe["native_blend_source_sha256"]:
        raise ValueError("native callback blend source changed")
    native = recipe["native"]
    panel = recipe["panel"]
    assert isinstance(native, dict) and isinstance(panel, dict)
    if sha256_file(repo_root / str(native["path"])) != native["sha256"]:
        raise ValueError("native extension bytes differ from frozen release")
    if sha256_file(repo_root / str(panel["split_path"])) != panel["split_sha256"]:
        raise ValueError("F0 split bytes differ from frozen panel")

    sys_path = str(repo_root / "src")
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from wnn.gmetric import METRIC_ID, GlobalAccumulator
    from wnn.splits import load_campaign_split

    split = load_campaign_split(repo_root / str(panel["split_path"]), repo_root)
    groups = list(split.evaluation_groups)
    if split.fold != 0 or len(groups) != int(panel["group_count"]):
        raise ValueError("split is not the frozen 468-group F0 evaluation panel")
    if groups != panel["evaluation_groups"]:
        raise ValueError("loaded split order differs from the frozen F0 recipe")
    verify_admissions(repo_root, recipe)

    members = recipe["members"]
    blends = recipe["blends"]
    assert isinstance(members, dict) and isinstance(blends, dict)
    member_results: dict[str, dict[str, object]] = {}
    member_arrays: dict[str, np.memmap] = {}
    for name in members:
        result, values = validate_member(repo_root, recipe, name)
        member_results[name] = result
        member_arrays[name] = values

    output_root = repo_root / str(recipe["output_dir"])
    output_root.mkdir(parents=True, exist_ok=False)
    destinations: dict[str, np.memmap] = {}
    output_paths: dict[str, Path] = {}
    for name, blend in blends.items():
        assert isinstance(blend, dict)
        output_path = output_root / str(blend["prediction_file"])
        output_paths[name] = output_path
        destinations[name] = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(groups), 20_000, 2),
        )

    member_accumulators = {name: GlobalAccumulator() for name in members}
    blend_accumulators = {name: GlobalAccumulator() for name in blends}
    data_path = repo_root / str(panel["data_path"])
    parquet = pq.ParquetFile(data_path)
    if max(groups) >= parquet.num_row_groups:
        raise ValueError("F0 split refers to a missing validation row group")
    input_handles = {
        name: (repo_root / str(member["prediction_dir"]) / "predictions.npy").open("rb")
        for name, member in members.items()
        if isinstance(member, dict)
    }
    groups_file = (output_root / "progress.jsonl").open("x", encoding="utf-8")
    selected_rows = 0
    group_bytes = 20_000 * 2 * np.dtype("float32").itemsize
    try:
        for index, group in enumerate(groups):
            targets, mask = read_group_labels(parquet, group)
            selected_rows += int(mask.sum())
            for name, values in member_arrays.items():
                prediction = np.asarray(values[index])
                member_accumulators[name].add(targets, prediction, mask)
            for blend_name, blend in blends.items():
                assert isinstance(blend, dict)
                names = [str(item) for item in blend["members"]]
                prediction = blend_group(member_arrays, names, index)
                destinations[blend_name][index] = prediction
                blend_accumulators[blend_name].add(targets, prediction, mask)
            for name, values in member_arrays.items():
                byte_offset = int(values.offset) + index * group_bytes
                advise_dontneed(input_handles[name].fileno(), byte_offset, group_bytes)
            groups_file.write(json.dumps({"index": index, "group": group}) + "\n")
            if (index + 1) % 8 == 0:
                groups_file.flush()
                for name, destination in destinations.items():
                    destination.flush()
                    destination._mmap.madvise(mmap.MADV_DONTNEED)
                    with output_paths[name].open("rb") as stream:
                        advise_dontneed(stream.fileno())
    finally:
        groups_file.close()
        for stream in input_handles.values():
            stream.close()
    if selected_rows != int(panel["selected_rows"]):
        raise ValueError("actual official mask count differs from frozen F0 panel")
    for destination in destinations.values():
        destination.flush()
    close_mmaps(destinations)
    close_mmaps(member_arrays)

    blend_scores: dict[str, dict[str, object]] = {}
    for name, accumulator in blend_accumulators.items():
        blend_scores[name] = accumulator.result()
    member_scores: dict[str, dict[str, object]] = {}
    for name, accumulator in member_accumulators.items():
        computed = accumulator.result()
        reported = member_results[name]["score"]
        assert isinstance(reported, dict)
        for field in ("t0", "t1", "wp"):
            if not math.isclose(
                float(computed[field]), float(reported[field]), rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"member producer score mismatch for {name}/{field}")
        if int(computed["selected_rows"]) != selected_rows:
            raise ValueError(f"member selected-row total mismatch: {name}")
        member_scores[name] = computed

    baseline = blend_scores["baseline"]
    deltas: dict[str, dict[str, float]] = {}
    for name in ("qat_s3", "qat_s4"):
        deltas[name] = {
            field: float(blend_scores[name][field]) - float(baseline[field])
            for field in ("t0", "t1", "wp")
        }
    deltas["qat_s4_minus_qat_s3"] = {
        field: float(blend_scores["qat_s4"][field]) - float(blend_scores["qat_s3"][field])
        for field in ("t0", "t1", "wp")
    }

    prediction_hashes = {
        name: sha256_file(path, evict_cache=True) for name, path in output_paths.items()
    }
    result = {
        "schema": "wnn.qat-final-blend-grade.v1",
        "recipe_sha256": args.recipe_sha256,
        "grader_source_sha256": recipe["grader_source_sha256"],
        "metric_id": METRIC_ID,
        "scope": panel["scope"],
        "split_sha256": panel["split_sha256"],
        "data_sha256": panel["data_sha256"],
        "ordered_groups": groups,
        "selected_rows": selected_rows,
        "native_sha256": native["sha256"],
        "members": {
            name: {
                "prediction_sha256": member_results[name]["prediction_sha256"],
                "weights_sha256": members[name]["weights_sha256"],
                "score_recomputed": member_scores[name],
            }
            for name in members
        },
        "blends": {
            name: {
                "members_in_native_sum_order": blend["members"],
                "coefficients_f32": blend["coefficients_f32"],
                "score": blend_scores[name],
                "prediction_file": str(output_paths[name]),
                "prediction_sha256": prediction_hashes[name],
                "shape": [len(groups), 20_000, 2],
                "dtype": "float32",
            }
            for name, blend in blends.items()
        },
        "deltas_candidate_minus_baseline": deltas,
        "limits": [
            "All endpoints use exposed, fitted-data F0; this is not clean CV or a public forecast.",
            "Point scores compare only the three predeclared fixed mixtures; they do not rank every possible endpoint or establish a universally strongest model.",
            "Prediction member hashes are independently verified, but the full validation parquet was not rescanned for a whole-file SHA; its producer hash receipt and split source identity are pinned, and grading reads only the 468 selected label/mask groups.",
            "Root owns submission choice and must separately verify callback parity against the final ZIP on the planned eight full-F0 sequences.",
        ],
    }
    output_path = output_root / "result.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

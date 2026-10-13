"""Grade five predeclared four-member native blends on fixed F0.

R4 preserves the original xteam baseline and R1 S3 QAT blend, then compares
three stationary-teacher R2 fourth-member replacements: FP32/native-Q8,
same-checkpoint W4 PTQ, and learned-W4 QAT. All use fixed 0.25 weights and
native index-order float32 arithmetic on the SHA-pinned F0 panel. It accepts
both the earlier `jobs` admission record and the actual R2 `vectors`/`commands`
producer schema without changing the grade contract.
"""

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
        if isinstance(admission.get("jobs"), list):
            if admission.get("source_sha256") != recipe["exporter_sha256"]:
                raise ValueError(f"admission source hash mismatch: {admission_key}")
            if admission.get("native_sha256") not in (None, native["sha256"]):
                raise ValueError(f"admission native hash mismatch: {admission_key}")
            if admission.get("split_sha256") != panel["split_sha256"]:
                raise ValueError(f"admission split hash mismatch: {admission_key}")
            if admission.get("ordered_panel_count") != panel["group_count"]:
                raise ValueError(f"admission group count mismatch: {admission_key}")
            jobs = admission["jobs"]
            selected = [job for job in jobs if job.get("attempt") == member["attempt"]]
            if len(selected) != 1:
                raise ValueError(f"expected one frozen job for {name}")
            job = selected[0]
            command = job.get("command")
            if not isinstance(command, list):
                raise ValueError(f"invalid command for {name}")
            command_tokens = [str(token) for token in command]
            expected_output = str(job.get("output"))
        elif (
            admission.get("format") == "wnn.qat-native-export-admission"
            and admission.get("version") == 2
        ):
            # This branch mirrors the actual R2 producer receipt. Keep the
            # normalization local to this verifier so no synthetic admission
            # record can weaken the root-pinned source receipt.
            frozen = admission.get("frozen_inputs")
            evaluation = admission.get("evaluation")
            vectors = admission.get("vectors")
            commands = admission.get("commands")
            if not all(isinstance(value, dict) for value in (frozen, evaluation)):
                raise ValueError(f"invalid R2 frozen inputs: {admission_key}")
            if not isinstance(vectors, list) or not isinstance(commands, list):
                raise ValueError(f"invalid R2 vectors/commands: {admission_key}")
            if frozen.get("exporter_sha256") != recipe["exporter_sha256"]:
                raise ValueError(f"R2 exporter source hash mismatch: {admission_key}")
            if frozen.get("native_library_sha256") != native["sha256"]:
                raise ValueError(f"R2 native hash mismatch: {admission_key}")
            if frozen.get("fold0_split_sha256") != panel["split_sha256"]:
                raise ValueError(f"R2 split hash mismatch: {admission_key}")
            if frozen.get("valid_parquet_sha256") != panel["data_sha256"]:
                raise ValueError(f"R2 data hash mismatch: {admission_key}")
            if (
                evaluation.get("fold") != 0
                or evaluation.get("evaluation_groups") != panel["group_count"]
                or evaluation.get("sequence_rows") != 20_000
                or evaluation.get("output_shape") != [panel["group_count"], 20_000, 2]
            ):
                raise ValueError(f"R2 F0 panel dimensions mismatch: {admission_key}")
            vector_matches = [
                item for item in vectors if item.get("attempt") == member["attempt"]
            ]
            command_matches = [
                item for item in commands if item.get("attempt") == member["attempt"]
            ]
            if len(vector_matches) != 1 or len(command_matches) != 1:
                raise ValueError(f"expected one frozen R2 vector and command for {name}")
            vector = vector_matches[0]
            job = command_matches[0]
            if (
                vector.get("weights") != member["weights_path"]
                or vector.get("weights_sha256") != member["weights_sha256"]
                or vector.get("output") != member["prediction_dir"]
                or bool(vector.get("packed")) != bool(member["packed"])
            ):
                raise ValueError(f"R2 vector binding differs for {name}")
            for vector_field, member_field in (
                ("checkpoint", "checkpoint_path"),
                ("checkpoint_sha256", "checkpoint_sha256"),
                ("manifest", "manifest_path"),
                ("manifest_sha256", "manifest_sha256"),
            ):
                if vector.get(vector_field) != member.get(member_field):
                    raise ValueError(f"R2 {vector_field} binding differs for {name}")
            command = job.get("argv")
            if not isinstance(command, list):
                raise ValueError(f"invalid R2 argv for {name}")
            command_tokens = [str(token) for token in command]
            expected_output = str(vector["output"])
            env = job.get("env")
            if not isinstance(env, dict) or env.get("PYTHONPATH") != str(root / "src"):
                raise ValueError(f"R2 source import path differs for {name}")
        elif (
            admission.get("format") == "wnn.qat-native-export-admission-correction"
            and admission.get("version") == 1
        ):
            # The third R2 export was initially refused before worker start
            # because its 1-core launcher reservation exceeded the aggregate
            # CPU cap. This correction admits one new unique attempt while
            # preserving the failed not-started receipt and exact export argv.
            correction = admission.get("correction")
            policy = admission.get("policy")
            artifacts = admission.get("artifacts")
            attempt = admission.get("attempt")
            prior_binding = admission.get("binds_original_admission")
            failed = admission.get("preserves_prelaunch_failure")
            if not all(
                isinstance(value, dict)
                for value in (policy, artifacts, attempt, prior_binding, failed)
            ) or not isinstance(correction, str) or not correction:
                raise ValueError(f"invalid CPU-repair admission fields: {admission_key}")
            original_record = admissions.get("stationary_r2")
            if not isinstance(original_record, dict):
                raise ValueError("CPU-repair recipe must pin the original R2 admission")
            original_path = (root / str(original_record["path"])).resolve()
            if (
                prior_binding.get("path") != str(original_path)
                or prior_binding.get("sha256") != original_record["sha256"]
            ):
                raise ValueError("CPU-repair receipt does not bind the original R2 admission")
            if not (
                failed.get("status") == "not_started"
                and failed.get("worker_created") is False
                and failed.get("output_absent") is True
            ):
                raise ValueError("CPU-repair receipt does not preserve the prelaunch failure")
            if not (
                policy.get("one_re_admission_only_for_prior_not_started_cpu_refusal") is True
                and policy.get("retry_after_worker_start_or_failure") is False
                and policy.get("automatic_retries") is False
            ):
                raise ValueError("CPU-repair retry policy is not the frozen one-shot policy")
            if (
                artifacts.get("native_sha256") != native["sha256"]
                or artifacts.get("split_sha256") != panel["split_sha256"]
                or artifacts.get("data_expected_sha256") != panel["data_sha256"]
            ):
                raise ValueError("CPU-repair native/split/data identity differs")
            if (
                attempt.get("id") != member["attempt"]
                or attempt.get("weights") != member["weights_path"]
                or attempt.get("weights_sha256") != member["weights_sha256"]
                or attempt.get("output") != member["prediction_dir"]
                or bool(attempt.get("packed")) != bool(member["packed"])
            ):
                raise ValueError(f"CPU-repair vector binding differs for {name}")
            original_admission = parsed["stationary_r2"]
            original_vectors = original_admission.get("vectors")
            original_commands = original_admission.get("commands")
            if not isinstance(original_vectors, list) or not isinstance(original_commands, list):
                raise ValueError("original R2 admission lacks vectors or commands")
            previous_attempt = str(member.get("supersedes_attempt", ""))
            previous_vectors = [
                item for item in original_vectors if item.get("attempt") == previous_attempt
            ]
            previous_commands = [
                item for item in original_commands if item.get("attempt") == previous_attempt
            ]
            if len(previous_vectors) != 1 or len(previous_commands) != 1:
                raise ValueError("CPU-repair receipt does not identify one prior attempt")
            prior_vector = previous_vectors[0]
            if not all(
                prior_vector.get(field) == attempt.get(field)
                for field in ("role", "weights", "weights_sha256", "packed", "output")
            ):
                raise ValueError("CPU-repair changed the frozen model/vector identity")
            for vector_field, member_field in (
                ("checkpoint", "checkpoint_path"),
                ("checkpoint_sha256", "checkpoint_sha256"),
                ("manifest", "manifest_path"),
                ("manifest_sha256", "manifest_sha256"),
            ):
                if prior_vector.get(vector_field) != member.get(member_field):
                    raise ValueError(f"CPU-repair {vector_field} binding differs for {name}")
            prior_argv = previous_commands[0].get("argv")
            repaired_argv = attempt.get("argv")
            if not isinstance(prior_argv, list) or not isinstance(repaired_argv, list):
                raise ValueError("CPU-repair command vectors are malformed")
            if [str(token) for token in prior_argv] != [str(token) for token in repaired_argv]:
                raise ValueError("CPU-repair changed the native export command")
            command = repaired_argv
            command_tokens = [str(token) for token in command]
            expected_output = str(attempt["output"])
            env = attempt.get("env")
            if not isinstance(env, dict) or env.get("PYTHONPATH") != str(root / "src"):
                raise ValueError(f"CPU-repair import path differs for {name}")
        else:
            raise ValueError(f"unsupported admission schema: {admission_key}")

        values, flags = command_options(command_tokens)
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
        if expected_output != str(member["prediction_dir"]):
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
    """Validate frozen member panels and grade the five fixed blends."""
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
    expected_blends = {
        "baseline": ["common0", "common1", "common2", "common3"],
        "r1_s3_qat": ["common0", "common1", "common2", "r1_lsq_s3"],
        "r2_fp32": ["common0", "common1", "common2", "r2_fp32_q8"],
        "r2_w4_ptq": ["common0", "common1", "common2", "r2_w4_ptq"],
        "r2_qat_w4": ["common0", "common1", "common2", "r2_lsq_w4"],
    }
    if set(blends) != set(expected_blends):
        raise ValueError("R3 recipe must contain exactly the five predeclared blends")
    expected_member_names = {
        member for names in expected_blends.values() for member in names
    }
    if set(members) != expected_member_names:
        raise ValueError("R3 recipe member roster differs from the predeclared fixed blends")
    for name, expected_members in expected_blends.items():
        blend = blends[name]
        if not isinstance(blend, dict) or blend.get("members") != expected_members:
            raise ValueError(f"R3 fixed member/order contract changed: {name}")
        if blend.get("coefficients_f32") != [0.25, 0.25, 0.25, 0.25]:
            raise ValueError(f"R3 requires four fixed 0.25 coefficients: {name}")
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
    r1_reference = blend_scores["r1_s3_qat"]

    def score_delta(candidate: str, reference: dict[str, object]) -> dict[str, float]:
        return {
            field: float(blend_scores[candidate][field]) - float(reference[field])
            for field in ("t0", "t1", "wp")
        }

    deltas_vs_baseline = {
        name: score_delta(name, baseline)
        for name in ("r1_s3_qat", "r2_fp32", "r2_w4_ptq", "r2_qat_w4")
    }
    deltas_vs_r1_s3_qat = {
        name: score_delta(name, r1_reference)
        for name in ("r2_fp32", "r2_w4_ptq", "r2_qat_w4")
    }
    deltas_within_r2 = {
        "r2_qat_w4_minus_r2_w4_ptq": score_delta("r2_qat_w4", blend_scores["r2_w4_ptq"]),
        "r2_qat_w4_minus_r2_fp32": score_delta("r2_qat_w4", blend_scores["r2_fp32"]),
        "r2_w4_ptq_minus_r2_fp32": score_delta("r2_w4_ptq", blend_scores["r2_fp32"]),
    }

    prediction_hashes = {
        name: sha256_file(path, evict_cache=True) for name, path in output_paths.items()
    }
    result = {
        "schema": "wnn.qat-final-blend-grade-r4.v1",
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
        "deltas_vs_baseline": deltas_vs_baseline,
        "deltas_vs_r1_s3_qat": deltas_vs_r1_s3_qat,
        "matched_r2_deltas": deltas_within_r2,
        "limits": [
            "All endpoints use exposed, fitted-data F0; this is not clean CV or a public forecast.",
            "Point scores compare only the five predeclared fixed mixtures; they do not rank every possible endpoint or establish a universally strongest model.",
            "Prediction member hashes are independently verified, but the full validation parquet was not rescanned for a whole-file SHA; its producer hash receipt and split source identity are pinned, and grading reads only the 468 selected label/mask groups.",
            "Root owns submission choice and must separately verify callback parity against the final ZIP on the planned eight full-F0 sequences.",
        ],
    }
    output_path = output_root / "result.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

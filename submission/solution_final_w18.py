"""Offline callback for a receipt-bound, single-member full-corpus W18 package."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent
_SHA256 = set("0123456789abcdef")
_SOURCE_VALIDATION_SHA256 = "1e8a8a794a0b8d9677a76030e570681074503d38aa2d54b80e36067566b94ce0"
_W18_RUN = "tl3final_w18final_distilled_s5"
_W18_PLAN_SHA256 = "29802499ab6ba62586b612cd10164d27c9737a96cf8de41c09fef3f6d0402cf4"
_W18_ENDPOINT = {"configured_epochs": 2, "checkpoint_epoch": 1}
_FIT_GROUP_COUNTS = {"train": 10_607, "valid": 1_873}
_LINEAGE_CHECKPOINTS = {
    "tl3final_w18final_distilled_s5": "model_ep1.pt",
    "tl3final_fbc1k_s5": "model_ep3.pt",
    "tl3final_ffull_s3": "model_ep1.pt",
    "tl3final_ffull_s4": "model_ep1.pt",
    "tl3final_frawW_s3ft": "model_ep1.pt",
    "tl3final_frawW_s4ft": "model_ep1.pt",
    "tl3final_orgbase": "model_ep3.pt",
    "tl3final_fin256_s3": "model_ep3.pt",
    "tl3final_fin256_s4": "model_ep3.pt",
    "tl3final_fmu128_s3": "model_ep3.pt",
    "tl3final_fmu128_s4": "model_ep3.pt",
    "tl3final_frawW_s3": "model_ep3.pt",
    "tl3final_frawW_s4": "model_ep3.pt",
}
_W18_TEACHERS = (
    ("tl3final_fbc1k_s5", "strong", 0.3, 3),
    ("tl3final_ffull_s3", "full_finetuned", 0.1, 1),
    ("tl3final_ffull_s4", "full_finetuned", 0.1, 1),
    ("tl3final_frawW_s3ft", "raw_wmse", 0.125, 1),
    ("tl3final_frawW_s4ft", "raw_wmse", 0.125, 1),
    ("tl3final_orgbase", "samefold_baseline", 0.0625, 3),
)
_CACHE_FILES = {
    "families_train",
    "families_valid",
    "main_source_train",
    "main_source_valid",
}
_MEMBER_FIELDS = {
    "role",
    "run_reference",
    "checkpoint_sha256",
    "configuration_sha256",
    "provenance_sha256",
    "weights",
    "weights_sha256",
    "input_width",
}

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, max_bytes: int) -> tuple[object, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular package file: {path.name}")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"package metadata exceeds its bounded size: {path.name}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"package metadata changed while reading: {path.name}")
    return json.loads(raw), raw


def _fit_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("single-W18 receipt lacks final-fit identity")
    fields = {
        "schema",
        "policy",
        "manifest_sha256",
        "fit_members_sha256",
        "source_sha256",
        "source_validation_sha256",
        "fit_row_groups",
        "independent_evaluation_members",
        "independent_score_claim",
        "post_fit_primary_valid_scoring",
    }
    fit = dict(value)
    sources = fit.get("source_sha256")
    groups = fit.get("fit_row_groups")
    if (
        set(fit) != fields
        or fit.get("schema") != "wnn.final-fit-membership.v1"
        or fit.get("policy") != "all_official_train_and_valid_labels_fit_no_independent_score_v1"
        or not all(
            _is_sha256(fit.get(name))
            for name in ("manifest_sha256", "fit_members_sha256", "source_validation_sha256")
        )
        or fit.get("source_validation_sha256") != _SOURCE_VALIDATION_SHA256
        or not isinstance(sources, Mapping)
        or set(sources) != {"train", "valid"}
        or not all(_is_sha256(sources.get(name)) for name in ("train", "valid"))
        or not isinstance(groups, Mapping)
        or set(groups) != {"train", "valid"}
        or any(
            not isinstance(groups.get(source), list)
            or any(type(group) is not int for group in groups[source])
            or groups[source] != list(range(count))
            for source, count in _FIT_GROUP_COUNTS.items()
        )
        or fit.get("independent_evaluation_members") != []
        or fit.get("independent_score_claim") != "forbidden"
        or fit.get("post_fit_primary_valid_scoring") != "forbidden"
    ):
        raise ValueError("single-W18 receipt fit membership or no-score policy is invalid")
    members = [[source, group] for source in ("train", "valid") for group in groups[source]]
    encoded = json.dumps(members, separators=(",", ":")).encode("ascii")
    if hashlib.sha256(encoded).hexdigest() != fit["fit_members_sha256"]:
        raise ValueError("single-W18 fit membership digest differs from all 12,480 sequences")
    return fit


def _validate_receipt(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or value.get("schema") != "wnn.final-w18-model-receipt.v1":
        raise ValueError("unsupported single-W18 model receipt")
    receipt = dict(value)
    fields = {
        "schema",
        "final_fit",
        "fit_sequences",
        "independent_evaluation_members",
        "independent_evaluation_sequences",
        "independent_score_claim",
        "fixed_endpoint",
        "members",
        "plan_sha256",
        "fit_process_receipt",
        "teacher_recipe",
        "teacher_cache",
        "lineage_closure",
    }
    if set(receipt) != fields:
        raise ValueError("single-W18 receipt contains unsupported or missing fields")
    fit = _fit_identity(receipt["final_fit"])
    if (
        type(receipt.get("fit_sequences")) is not int
        or receipt["fit_sequences"] != sum(_FIT_GROUP_COUNTS.values())
        or receipt.get("independent_evaluation_members") != []
        or type(receipt.get("independent_evaluation_sequences")) is not int
        or receipt["independent_evaluation_sequences"] != 0
        or receipt.get("independent_score_claim") != "forbidden"
        or receipt.get("fixed_endpoint") != _W18_ENDPOINT
        or receipt.get("plan_sha256") != _W18_PLAN_SHA256
    ):
        raise ValueError("single-W18 receipt endpoint or evaluation policy is invalid")

    fit_process = receipt.get("fit_process_receipt")
    if (
        not isinstance(fit_process, Mapping)
        or set(fit_process) != {"path", "sha256", "record"}
        or fit_process.get("path") != "w18_fit_process_receipt.json"
        or not _is_sha256(fit_process.get("sha256"))
        or not isinstance(fit_process.get("record"), Mapping)
    ):
        raise ValueError("single-W18 receipt lacks its exact fit-process receipt binding")
    process = fit_process["record"]
    command = process.get("command")
    command_args = command[6:] if isinstance(command, list) else []
    options: dict[str, list[str]] = {}
    for index, token in enumerate(command_args):
        if not isinstance(token, str):
            continue
        if token.startswith("--") and "=" in token:
            option, _, option_value = token.partition("=")
            options.setdefault(option, []).append(option_value)
        elif token.startswith("--") and index + 1 < len(command_args):
            next_value = command_args[index + 1]
            if isinstance(next_value, str) and not next_value.startswith("--"):
                options.setdefault(token, []).append(next_value)
    if (
        process.get("schema") != "wnn.multistage-owned-process.v1"
        or type(process.get("stage")) is not int
        or process.get("stage") != 1
        or process.get("mode") != "final_submit_finite_chain"
        or process.get("status") != "completed"
        or type(process.get("exit_code")) is not int
        or process.get("exit_code") != 0
        or process.get("timed_out") is not False
        or not isinstance(command, list)
        or process.get("command_sha256")
        != hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
        or options.get("--name") != [_W18_RUN]
        or options.get("--epochs") != [str(_W18_ENDPOINT["configured_epochs"])]
        or options.get("--checkpoint-epoch") != [str(_W18_ENDPOINT["checkpoint_epoch"])]
        or options.get("--train-source") != ["train+valid"]
        or options.get("--train-seqs") != [str(_FIT_GROUP_COUNTS["train"])]
        or options.get("--final-fit-source-validation-sha256") != [_SOURCE_VALIDATION_SHA256]
        or command_args.count("--include-heldout-valid") != 1
        or command_args.count("--no-eval-during-fit") != 1
    ):
        raise ValueError(
            "W18 fit-process receipt does not prove successful final all-labeled completion"
        )
    members = receipt.get("members")
    if not isinstance(members, Mapping) or set(members) != {"w18"}:
        raise ValueError("single-W18 receipt must contain exactly W18, with no inactive members")
    member = members["w18"]
    member_fields = {"run", "checkpoint_sha256", "configuration_sha256", "provenance_sha256"}
    if (
        not isinstance(member, Mapping)
        or set(member) != member_fields
        or member.get("run") != _W18_RUN
        or any(
            not _is_sha256(member.get(name))
            for name in ("checkpoint_sha256", "configuration_sha256", "provenance_sha256")
        )
    ):
        raise ValueError("single-W18 receipt member identity is malformed")

    closure = receipt.get("lineage_closure")
    if (
        not isinstance(closure, Mapping)
        or set(closure) != {"schema", "roles", "sha256"}
        or closure.get("schema") != "wnn.final-w18-lineage-closure.v1"
        or not isinstance(closure.get("roles"), list)
        or closure.get("sha256") != _canonical_sha256(closure.get("roles"))
    ):
        raise ValueError("single-W18 receipt lineage closure is malformed")
    roles = closure["roles"]
    role_fields = {
        "run",
        "checkpoint_file",
        "checkpoint_sha256",
        "final_fit_manifest_sha256",
        "final_fit_members_sha256",
        "source_validation_sha256",
    }
    by_run: dict[str, Mapping[str, object]] = {}
    for record in roles:
        if not isinstance(record, Mapping) or set(record) != role_fields:
            raise ValueError("single-W18 lineage entry has unsupported fields")
        run = record.get("run")
        if (
            not isinstance(run, str)
            or run in by_run
            or run not in _LINEAGE_CHECKPOINTS
            or record.get("checkpoint_file") != _LINEAGE_CHECKPOINTS[run]
            or not _is_sha256(record.get("checkpoint_sha256"))
            or record.get("final_fit_manifest_sha256") != fit["manifest_sha256"]
            or record.get("final_fit_members_sha256") != fit["fit_members_sha256"]
            or record.get("source_validation_sha256") != fit["source_validation_sha256"]
        ):
            raise ValueError("single-W18 lineage entry is duplicated or escapes the final fit")
        by_run[run] = record
    if tuple(sorted(by_run)) != tuple(sorted(_LINEAGE_CHECKPOINTS)):
        raise ValueError("single-W18 receipt must contain exactly the W18 model and 12 ancestors")
    if by_run[_W18_RUN]["checkpoint_sha256"] != member["checkpoint_sha256"]:
        raise ValueError("W18 member hash differs from its lineage closure")

    recipe = receipt.get("teacher_recipe")
    if not isinstance(recipe, list) or len(recipe) != len(_W18_TEACHERS):
        raise ValueError("single-W18 receipt must bind the six fixed training teachers")
    recipe_fields = {
        "run",
        "family",
        "coefficient",
        "epoch",
        "checkpoint_file",
        "checkpoint_sha256",
    }
    for item, (run, family, coefficient, epoch) in zip(recipe, _W18_TEACHERS, strict=True):
        record = by_run[run]
        if (
            not isinstance(item, Mapping)
            or set(item) != recipe_fields
            or item.get("run") != run
            or item.get("family") != family
            or item.get("coefficient") != coefficient
            or item.get("epoch") != epoch
            or item.get("checkpoint_file") != record["checkpoint_file"]
            or item.get("checkpoint_sha256") != record["checkpoint_sha256"]
        ):
            raise ValueError("single-W18 teacher recipe differs from its verified closure")
    cache = receipt.get("teacher_cache")
    if (
        not isinstance(cache, Mapping)
        or set(cache) != {"schema", "metadata_sha256", "verified_files"}
        or cache.get("schema") != "wnn.final-w18-teacher-cache-binding.v1"
        or not _is_sha256(cache.get("metadata_sha256"))
        or not isinstance(cache.get("verified_files"), Mapping)
        or set(cache["verified_files"]) != _CACHE_FILES
        or not all(_is_sha256(digest) for digest in cache["verified_files"].values())
    ):
        raise ValueError("single-W18 receipt lacks verified six-teacher target-cache identities")
    return receipt


def _native_level() -> str:
    forced = os.environ.get("WNN_NATIVE_LEVEL")
    if forced in ("v2", "avx2", "v3"):
        return forced
    flags: set[str] = set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    if {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"} <= flags:
        return "v3"
    if {"avx", "avx2", "fma"} <= flags and (HERE / "avx2").is_dir():
        return "avx2"
    return "v2"


def _load_weights(record: Mapping[str, object]) -> dict[str, np.ndarray]:
    name = record.get("weights")
    if not isinstance(name, str):
        raise ValueError("W18 package weights path is malformed")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or len(relative.parts) != 1:
        raise ValueError("W18 weights path must be a safe package-root filename")
    path = HERE / name
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != record.get("weights_sha256"):
        raise ValueError("W18 weight archive hash differs from the package manifest")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.ascontiguousarray(archive[key], dtype=np.float32) for key in archive.files}


def _member_matches_receipt(
    package_member: Mapping[str, object], receipt_member: Mapping[str, object]
) -> bool:
    """Join package naming to the receipt schema while matching all identity hashes."""
    return all(
        package_member.get(package_field) == receipt_member.get(receipt_field)
        for package_field, receipt_field in (
            ("run_reference", "run"),
            ("checkpoint_sha256", "checkpoint_sha256"),
            ("configuration_sha256", "configuration_sha256"),
            ("provenance_sha256", "provenance_sha256"),
        )
    )


manifest_value, _manifest_bytes = _read_json(HERE / "members.json", 1_000_000)
if not isinstance(manifest_value, Mapping):
    raise ValueError("W18 package manifest must be an object")
_manifest = dict(manifest_value)
_expected_manifest_fields = {
    "schema",
    "candidate_scope",
    "training_and_selection_scope",
    "independent_evaluation_sequences",
    "independent_score_claim",
    "serving_policy",
    "precision",
    "fast_activations",
    "active_member_indices",
    "callback_output",
    "final_model_receipt",
    "fit_process_receipt",
    "members",
    "native_libraries",
}
if set(_manifest) != _expected_manifest_fields:
    raise ValueError("W18 package manifest has unsupported or missing fields")
if (
    _manifest.get("schema") != "wnn.final-w18-package-manifest.v1"
    or _manifest.get("candidate_scope") != "approved_full_corpus_final_w18"
    or _manifest.get("training_and_selection_scope")
    != "all official train and valid labels fit; no independent evaluation"
    or _manifest.get("independent_evaluation_sequences") != 0
    or _manifest.get("independent_score_claim") != "forbidden"
    or _manifest.get("serving_policy") != "w18_only"
    or _manifest.get("precision") != "q8vnni"
    or _manifest.get("fast_activations") is not True
    or _manifest.get("active_member_indices") != [0]
    or _manifest.get("callback_output")
    != {
        "schema": "wnn.single-member-output.v1",
        "role": "w18",
        "weight": 1.0,
        "clip": "official-global-clip-after-callback",
    }
):
    raise ValueError("W18 package manifest does not describe the approved single-member path")

_receipt_ref = _manifest.get("final_model_receipt")
if (
    not isinstance(_receipt_ref, Mapping)
    or set(_receipt_ref) != {"path", "schema", "sha256"}
    or _receipt_ref.get("path") != "final_w18_receipt.json"
    or _receipt_ref.get("schema") != "wnn.final-w18-model-receipt.v1"
    or not _is_sha256(_receipt_ref.get("sha256"))
):
    raise ValueError("W18 package manifest lacks its pinned single-member receipt")
_receipt_value, _receipt_bytes = _read_json(HERE / "final_w18_receipt.json", 1_000_000)
if hashlib.sha256(_receipt_bytes).hexdigest() != _receipt_ref["sha256"]:
    raise ValueError("W18 receipt file differs from the package-bound receipt SHA256")
_receipt = _validate_receipt(_receipt_value)

_fit_process_ref = _manifest.get("fit_process_receipt")
if (
    not isinstance(_fit_process_ref, Mapping)
    or set(_fit_process_ref) != {"path", "sha256"}
    or _fit_process_ref.get("path") != "w18_fit_process_receipt.json"
    or not _is_sha256(_fit_process_ref.get("sha256"))
):
    raise ValueError("W18 package manifest lacks its pinned fit-process receipt")
_fit_process_value, _fit_process_bytes = _read_json(HERE / _fit_process_ref["path"], 2_000_000)
if hashlib.sha256(_fit_process_bytes).hexdigest() != _fit_process_ref["sha256"]:
    raise ValueError("W18 fit-process receipt bytes differ from the package manifest")
_receipt_fit_process = _receipt["fit_process_receipt"]
if (
    _receipt_fit_process["path"] != _fit_process_ref["path"]
    or _receipt_fit_process["sha256"] != _fit_process_ref["sha256"]
    or dict(_receipt_fit_process["record"]) != _fit_process_value
):
    raise ValueError("W18 fit-process receipt differs from its final model receipt")

_members = _manifest.get("members")
if not isinstance(_members, list) or len(_members) != 1:
    raise ValueError("W18-only package must contain exactly one member record")
_member = _members[0]
if (
    not isinstance(_member, Mapping)
    or set(_member) != _MEMBER_FIELDS
    or _member.get("role") != "w18"
    or _member.get("run_reference") != _W18_RUN
    or _member.get("input_width") != 211
    or not _is_sha256(_member.get("checkpoint_sha256"))
    or not _is_sha256(_member.get("configuration_sha256"))
    or not _is_sha256(_member.get("provenance_sha256"))
    or not _is_sha256(_member.get("weights_sha256"))
):
    raise ValueError("W18 package member record is malformed")
_receipt_member = _receipt["members"]["w18"]
if not _member_matches_receipt(_member, _receipt_member):
    raise ValueError("W18 package member differs from its full-fit checkpoint receipt")

_native_libraries = _manifest.get("native_libraries")
_expected_native_paths = {f"{level}/wnn_native.abi3.so" for level in ("v2", "avx2", "v3")}
if not isinstance(_native_libraries, Mapping) or set(_native_libraries) != _expected_native_paths:
    raise ValueError("W18 package must include exactly the three supported native ABI libraries")
for _relative, _digest in _native_libraries.items():
    if not _is_sha256(_digest):
        raise ValueError("W18 native library digest is malformed")
    _path = HERE / _relative
    if (
        _path.is_symlink()
        or not _path.is_file()
        or hashlib.sha256(_path.read_bytes()).hexdigest() != _digest
    ):
        raise ValueError(f"W18 native library digest differs: {_relative}")

_level = _native_level()
sys.path.insert(0, str(HERE / _level))
import wnn_native  # noqa: E402


class PredictionModel:
    """Provide the scorer callback backed only by the verified W18 checkpoint."""

    def __init__(self) -> None:
        weights = _load_weights(_member)
        self.model = wnn_native.FinalW18Predictor(
            weights, precision="q8vnni", fast_activations=True
        )
        if tuple(self.model.member_widths) != (211,):
            raise ValueError("native W18 predictor loaded an unexpected input width")
        if tuple(self.model.active_member_indices) != (0,):
            raise ValueError("native W18 predictor activated unexpected member roles")
        if not str(self.model.precision).startswith("q8vnni"):
            raise ValueError("native W18 predictor did not honor q8vnni precision")
        if self.model.fast_activations is not True:
            raise ValueError("native W18 predictor did not honor fast activations")
        self.predict: Callable[[object], np.ndarray | None] = self.model.predict

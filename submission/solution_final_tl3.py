"""Fixed TL3 plus Source4 LSTM callback for the final chronological package."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent
ROLES = ("w18", "tiedlead_s5", "tiedlead_s6", "source4_lstm")
WIDTHS = (211, 263, 263, 112)
BLEND = {
    "schema": "wnn.nested-fp64-fp32.v1",
    "tl3_weights": [0.34, 0.33, 0.33],
    "outer_weights": [0.9, 0.1],
    "tl3_cast": "float32",
    "outer_cast": "float32",
    "clip": "official-global-clip-after-callback",
}
MANIFEST_SCHEMA = "wnn.final-tl3-lstm-members.v1"
ENGINEERING_SCOPE = "engineering_train_only_reference"
FINAL_SCOPE = "approved_full_corpus_final_model_set"
FINAL_FIT_SCHEMA = "wnn.final-fit-membership.v1"
FINAL_FIT_POLICY = "all_official_train_and_valid_labels_fit_no_independent_score_v1"
SOURCE_VALIDATION_SHA256 = "1e8a8a794a0b8d9677a76030e570681074503d38aa2d54b80e36067566b94ce0"
_SHA256 = set("0123456789abcdef")
_V3_FLAGS = {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "abm", "movbe", "xsave"}
_AVX2_FLAGS = {"avx", "avx2", "fma"}

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def _cpu_flags() -> set[str]:
    """Read CPU flags for selecting a compatible bundled native library."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _native_level() -> str:
    """Select the highest bundled ISA supported by the current CPU."""
    forced = os.environ.get("WNN_NATIVE_LEVEL")
    if forced in ("v2", "avx2", "v3"):
        return forced
    flags = _cpu_flags()
    if flags >= _V3_FLAGS:
        return "v3"
    if flags >= _AVX2_FLAGS and (HERE / "avx2").is_dir():
        return "avx2"
    return "v2"


def _load_member(path: str, expected_sha256: str) -> dict[str, np.ndarray]:
    """Load float32 tensors from a package-local NPZ file."""
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in path or len(relative.parts) != 1:
        raise ValueError("member weights path must be a safe package-root filename")
    weights_path = HERE / path
    digest = hashlib.sha256()
    with weights_path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"member weight archive hash mismatch: {path}")
    with np.load(weights_path, allow_pickle=False) as archive:
        return {key: np.ascontiguousarray(archive[key], dtype=np.float32) for key in archive.files}


_manifest = json.loads((HERE / "members.json").read_text(encoding="utf-8"))
if _manifest.get("schema") != MANIFEST_SCHEMA:
    raise ValueError("unsupported final predictor manifest")
if tuple(item.get("role") for item in _manifest.get("members", ())) != ROLES:
    raise ValueError("final predictor members must be ordered W18, tiedlead s5, tiedlead s6, LSTM")
if tuple(item.get("input_width") for item in _manifest["members"]) != WIDTHS:
    raise ValueError("final predictor member input schemas differ from the fixed contract")
if _manifest.get("blend") != BLEND:
    raise ValueError("final predictor blend tree differs from its fixed nested contract")
if _manifest.get("precision") != "f32":
    raise ValueError("final predictor package must preserve fp32 member weights")
if (
    _manifest.get("independent_evaluation_sequences") != 0
    or _manifest.get("independent_score_claim") != "forbidden"
):
    raise ValueError("final predictor manifest cannot claim independent evaluation or scores")
if _manifest.get("candidate_scope") == FINAL_SCOPE:
    final_fit = _manifest.get("final_fit")
    receipt = _manifest.get("final_model_receipt")
    fit_row_groups = final_fit.get("fit_row_groups") if isinstance(final_fit, dict) else None
    expected_fit_members_sha256 = hashlib.sha256(
        json.dumps(
            [
                [source, group]
                for source, count in (("train", 10_607), ("valid", 1_873))
                for group in range(count)
            ],
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    source_sha256 = final_fit.get("source_sha256") if isinstance(final_fit, dict) else None
    if (
        not isinstance(final_fit, dict)
        or final_fit.get("schema") != FINAL_FIT_SCHEMA
        or final_fit.get("policy") != FINAL_FIT_POLICY
        or final_fit.get("independent_evaluation_members") != []
        or final_fit.get("independent_score_claim") != "forbidden"
        or final_fit.get("post_fit_primary_valid_scoring") != "forbidden"
        or not isinstance(fit_row_groups, dict)
        or fit_row_groups.get("train") != list(range(10_607))
        or fit_row_groups.get("valid") != list(range(1_873))
        or any(
            type(group) is not int
            for source in ("train", "valid")
            for group in fit_row_groups.get(source, ())
        )
        or not isinstance(final_fit.get("fit_members_sha256"), str)
        or len(final_fit["fit_members_sha256"]) != 64
        or not set(final_fit["fit_members_sha256"]) <= _SHA256
        or final_fit["fit_members_sha256"] != expected_fit_members_sha256
        or not isinstance(final_fit.get("manifest_sha256"), str)
        or len(final_fit["manifest_sha256"]) != 64
        or not set(final_fit["manifest_sha256"]) <= _SHA256
        or not isinstance(source_sha256, dict)
        or set(source_sha256) != {"train", "valid"}
        or any(
            not isinstance(source_sha256.get(source), str)
            or len(source_sha256[source]) != 64
            or not set(source_sha256[source]) <= _SHA256
            for source in ("train", "valid")
        )
        or not isinstance(final_fit.get("source_validation_sha256"), str)
        or len(final_fit["source_validation_sha256"]) != 64
        or not set(final_fit["source_validation_sha256"]) <= _SHA256
        or final_fit["source_validation_sha256"] != SOURCE_VALIDATION_SHA256
        or not isinstance(receipt, dict)
        or receipt.get("schema") != "wnn.final-model-set-receipt.v1"
        or not isinstance(receipt.get("sha256"), str)
        or len(receipt["sha256"]) != 64
        or not set(receipt["sha256"]) <= _SHA256
    ):
        raise ValueError("full-corpus final manifest lacks its complete receipt-bound fit identity")
elif _manifest.get("candidate_scope") == ENGINEERING_SCOPE:
    if _manifest.get("final_fit") is not None or _manifest.get("final_model_receipt") is not None:
        raise ValueError("engineering reference manifest cannot claim final full-corpus provenance")
else:
    raise ValueError("unsupported final predictor candidate scope")
expected_runs = (
    (
        "tl3final_w18final_distilled_s5",
        "tl3final_fttall_tlead_s5",
        "tl3final_fttall_tlead_s6",
        "final-lstm192x1",
    )
    if _manifest["candidate_scope"] == FINAL_SCOPE
    else (
        "tl3c_w18final_distilled_s5",
        "tl3c_fttall_tlead_s5",
        "tl3c_fttall_tlead_s6",
        "primary-refit-03-s1t000000-lstm",
    )
)
if tuple(item.get("run_reference") for item in _manifest["members"]) != expected_runs:
    raise ValueError("final predictor run labels differ from its declared candidate scope")

_LEVEL = _native_level()
sys.path.insert(0, str(HERE / _LEVEL))
import wnn_native  # noqa: E402


class PredictionModel:
    """Expose the native stateful predictor using the competition callback contract."""

    def __init__(self) -> None:
        weights = [
            _load_member(item["weights"], item["weights_sha256"]) for item in _manifest["members"]
        ]
        self.model = wnn_native.FinalTl3Predictor(weights)
        if tuple(self.model.member_widths) != WIDTHS:
            raise ValueError("native final predictor loaded unexpected member widths")
        self.predict: Callable[[object], np.ndarray | None] = self.model.predict

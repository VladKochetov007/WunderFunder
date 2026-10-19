"""Verify and restore only the immutable sparse R15 representation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from idea_screens_r15.representation import DictionaryAutoencoder, fingerprint
from idea_screens_r15.runtime_support import sha256_file

R15_RESULT_SHA256 = "ab149a3fcf9d58c1221300ddca3eba356a16181fe5239ea1b3d7646bfe9099c3"
R15_RESULT_BYTES = 244_853
R15_ARCHIVE_SHA256 = "9f3404f921bb27ffa92912a88b51142a5d1fa76ab29df417c12c808f30d4269f"
R15_ARCHIVE_BYTES = 178_584
SPARSE_DICTIONARY_SHA256 = "7a62eac9aa42658c38b92ebe90b84489f7dcb9c8750e9b33e872db8a7b124b06"
SPARSE_WMSE_WP = 0.4824412038543008
DENSE_WMSE_WP = 0.4798730901246624
PREDICTOR_INITIAL_SHA256 = "bce617ce88cb42d0c2cce82f3ec479c90909adc2b9e8adcd4cb6dc1d50963405"
REFERENCE_ARCHIVE_PATH = "nf19-r15-v1/model_parameters.npz"

RESTORED_ARRAYS = (
    "raw_input_mean",
    "raw_input_scale",
    "sparse_encoder_weight",
    "sparse_decoder_weight",
    "sparse_downstream_mean",
    "sparse_downstream_scale",
)


def read_reference_result(path: Path) -> dict[str, object]:
    """Load the source-pinned R15 compact result and enforce fixed identities."""
    if path.stat().st_size != R15_RESULT_BYTES or sha256_file(path) != R15_RESULT_SHA256:
        raise ValueError("captured R15 reference result bytes differ")
    result = json.loads(path.read_bytes())
    if result.get("schema") != "nf19.r15.result.v1":
        raise ValueError("unexpected R15 reference result schema")
    if result.get("predictor_initialization_sha256") != PREDICTOR_INITIAL_SHA256:
        raise ValueError("R15 predictor initialization fingerprint differs")
    if result.get("dictionary_frozen_sha256", {}).get("sparse") != SPARSE_DICTIONARY_SHA256:
        raise ValueError("R15 frozen sparse dictionary fingerprint differs")
    if result.get("dictionary_final_sha256", {}).get("sparse") != SPARSE_DICTIONARY_SHA256:
        raise ValueError("R15 final sparse dictionary fingerprint differs")
    archive = result.get("model_archive", {})
    if archive.get("sha256") != R15_ARCHIVE_SHA256 or archive.get("bytes") != R15_ARCHIVE_BYTES:
        raise ValueError("R15 producer archive receipt differs")
    arms = result.get("evaluation", {}).get("arms", {})
    for arm, expected in (("sparse", SPARSE_WMSE_WP), ("dense", DENSE_WMSE_WP)):
        actual = arms.get(arm, {}).get("official_metric", {}).get("wp")
        if actual != expected:
            raise ValueError(f"stored R15 {arm} endpoint differs")
    if (
        result.get("sample_sha256")
        != "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
    ):
        raise ValueError("R15 reference sample differs")
    return result


def restore_sparse_reference(
    archive_path: Path, result: dict[str, object]
) -> tuple[DictionaryAutoencoder, dict[str, np.ndarray], dict[str, object]]:
    """Hash, validate and restore R15 sparse dictionary/normalizer arrays."""
    if archive_path.name != "model_parameters.npz":
        raise ValueError("R15 reference archive has an unexpected name")
    if archive_path.stat().st_size != R15_ARCHIVE_BYTES:
        raise ValueError("R15 reference archive size differs")
    archive_sha256 = sha256_file(archive_path)
    if archive_sha256 != R15_ARCHIVE_SHA256:
        raise ValueError("R15 reference archive SHA256 differs")
    metadata = result["model_archive"]["arrays"]
    expected_names = set(metadata)
    if not expected_names or any(
        not isinstance(name, str)
        or not isinstance(spec, dict)
        or not isinstance(spec.get("shape"), list)
        or not isinstance(spec.get("dtype"), str)
        for name, spec in metadata.items()
    ):
        raise ValueError("R15 result has an invalid numeric archive key manifest")
    restored: dict[str, np.ndarray] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_names:
            raise ValueError("R15 reference archive names differ from captured metadata")
        for name in RESTORED_ARRAYS:
            array = np.asarray(archive[name])
            spec = metadata[name]
            if list(array.shape) != spec.get("shape") or str(array.dtype) != spec.get("dtype"):
                raise ValueError(f"R15 reference array schema differs: {name}")
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError(f"R15 reference array is nonnumeric or nonfinite: {name}")
            if name.endswith("_scale") and (array <= 0).any():
                raise ValueError(f"R15 reference scale must be positive: {name}")
            if name in RESTORED_ARRAYS:
                restored[name] = array.copy()
    if set(restored) != set(RESTORED_ARRAYS):
        raise ValueError("R15 required sparse representation arrays are incomplete")
    dictionary = DictionaryAutoencoder(sparse=True)
    dictionary.load_state_dict(
        {
            "encoder.weight": torch.from_numpy(restored["sparse_encoder_weight"]),
            "decoder.weight": torch.from_numpy(restored["sparse_decoder_weight"]),
        }
    )
    actual_fingerprint = fingerprint(dictionary)
    if actual_fingerprint != SPARSE_DICTIONARY_SHA256:
        raise ValueError("restored sparse dictionary fingerprint differs from R15 post-AE state")
    for parameter in dictionary.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    dictionary.eval()
    if any(
        parameter.grad is not None or parameter.requires_grad
        for parameter in dictionary.parameters()
    ):
        raise RuntimeError("restored sparse dictionary is not frozen")
    archive_record = {
        "path": str(archive_path),
        "sha256": archive_sha256,
        "bytes": R15_ARCHIVE_BYTES,
        "archive_key_count": len(expected_names),
        "restored_arrays_validated": len(restored),
        "arrays_retained": len(restored),
        "sparse_dictionary_fingerprint": actual_fingerprint,
        "archive_key_manifest_checked": True,
        "old_predictor_arrays_loaded": False,
        "old_predictors_reinferred_or_regraded": False,
    }
    return dictionary, restored, archive_record

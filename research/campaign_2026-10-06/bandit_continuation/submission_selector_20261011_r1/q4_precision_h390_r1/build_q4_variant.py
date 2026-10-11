"""Build a rowwise Q4-rounded H390 pair for evaluation in the Q8 native backend."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

EXPECTED_SOURCE_ZIP_SHA256 = "3420ea5e6318e15b6340582125959ed7489983b17338f399c748965cb42c4a4a"
EXPECTED_WEIGHT_SHA256 = {
    "weights_0.npz": "62b11349b4372beda7abbd35f7e006d2c967237070cbd5318c7a2f15bc0d1ec6",
    "weights_1.npz": "9f019d54a147296037b1bb2c6ad8b15b87070bde0402ff5426ce9c4804f52eaf",
}
TARGET_TENSORS = {"head.0.weight"}
TARGET_TENSORS.update(
    {f"gru.weight_{kind}_l{layer}" for kind in ("ih", "hh") for layer in range(2)}
)


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of a byte string."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large sources do not require a copy."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(array: np.ndarray) -> str:
    """Hash tensor values in C order, excluding container-level NPZ metadata."""
    return sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def rowwise_q4_round(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round each matrix row to symmetric signed levels -7..7.

    Each nonzero row uses ``max(abs(row)) / 7`` as its float32 scale. Ties use
    NumPy's round-to-nearest, ties-to-even rule. Zero rows retain zero scale,
    zero codes, and exact zero reconstruction.
    """
    values = np.asarray(matrix)
    if values.ndim != 2 or values.dtype != np.float32:
        raise ValueError("rowwise Q4 rounding expects a 2D float32 matrix")
    if not np.isfinite(values).all():
        raise ValueError("Q4 source matrix contains nonfinite values")

    maxima = np.max(np.abs(values), axis=1)
    scales = (maxima / np.float32(7.0)).astype(np.float32, copy=False)
    codes = np.zeros(values.shape, dtype=np.int8)
    nonzero_rows = scales > 0
    scaled = values[nonzero_rows] / scales[nonzero_rows, None]
    codes[nonzero_rows] = np.clip(np.rint(scaled), -7, 7).astype(np.int8)
    reconstructed = codes.astype(np.float32) * scales[:, None]
    return reconstructed, codes, scales


def _is_target_tensor(name: str, array: np.ndarray) -> bool:
    """Select dense recurrent matrices and the first dense head matrix only."""
    return name in TARGET_TENSORS and array.ndim == 2


def transform_npz(
    source: bytes, member_index: int
) -> tuple[bytes, dict[str, np.ndarray], dict[str, object]]:
    """Return a rebuilt float32 weights NPZ, packed Q4 audit arrays, and receipt."""
    output_arrays: dict[str, np.ndarray] = {}
    code_arrays: dict[str, np.ndarray] = {}
    tensor_records: dict[str, dict[str, object]] = {}
    selected: set[str] = set()

    with np.load(io.BytesIO(source), allow_pickle=False) as weights:
        for name in weights.files:
            array = np.asarray(weights[name])
            original_digest = tensor_sha256(array)
            if _is_target_tensor(name, array):
                if array.dtype != np.float32:
                    raise ValueError(f"{name} must be float32, got {array.dtype}")
                restored, codes, scales = rowwise_q4_round(array)
                selected.add(name)
                output_arrays[name] = restored
                code_prefix = f"member_{member_index}__{name.replace('.', '_')}"
                code_arrays[f"{code_prefix}__codes"] = codes
                code_arrays[f"{code_prefix}__scales"] = scales
                error = restored.astype(np.float64) - array.astype(np.float64)
                tensor_records[name] = {
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "source_tensor_sha256": original_digest,
                    "variant_tensor_sha256": tensor_sha256(restored),
                    "codes_sha256": tensor_sha256(codes),
                    "scales_sha256": tensor_sha256(scales),
                    "code_values": sorted(int(value) for value in np.unique(codes)),
                    "scale_min": float(scales.min()),
                    "scale_max": float(scales.max()),
                    "max_abs_error": float(np.max(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "changed_elements": int(np.count_nonzero(restored != array)),
                    "elements": int(array.size),
                }
            else:
                output_arrays[name] = np.array(array, copy=True)
                if tensor_sha256(output_arrays[name]) != original_digest:
                    raise AssertionError(f"unchanged tensor drifted: {name}")

    if selected != TARGET_TENSORS:
        missing = sorted(TARGET_TENSORS - selected)
        extra = sorted(selected - TARGET_TENSORS)
        raise ValueError(f"unexpected selected tensors; missing={missing}, extra={extra}")

    encoded = io.BytesIO()
    np.savez_compressed(encoded, **output_arrays)
    return (
        encoded.getvalue(),
        code_arrays,
        {
            "member_index": member_index,
            "tensor_records": tensor_records,
            "unchanged_tensor_names": sorted(set(output_arrays) - selected),
        },
    )


def validate_zip_names(names: list[str]) -> None:
    """Reject duplicate or unsafe ZIP paths before writing derived package files."""
    if len(names) != len(set(names)):
        raise ValueError("source ZIP has duplicate entry names")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"unsafe ZIP entry path: {name!r}")


def build_variant(source_zip: Path, output_dir: Path) -> dict[str, object]:
    """Create an immutable Q4-rounded ZIP and tensor-code audit archive."""
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    source_digest = sha256_file(source_zip)
    if source_digest != EXPECTED_SOURCE_ZIP_SHA256:
        raise ValueError(
            "Q8 source ZIP digest mismatch: "
            f"expected {EXPECTED_SOURCE_ZIP_SHA256}, got {source_digest}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_zip = output_dir / "submission_q4rounded_q8backend.zip"
    codes_path = output_dir / "q4_codes_and_scales.npz"
    receipt_path = output_dir / "Q4_BUILD_RECEIPT_R1.json"
    changed_entries: set[str] = set()
    entry_records: list[dict[str, object]] = []
    member_records: dict[int, dict[str, object]] = {}
    audit_arrays: dict[str, np.ndarray] = {}

    with zipfile.ZipFile(source_zip, "r") as source_archive:
        infos = source_archive.infolist()
        names = [info.filename for info in infos]
        validate_zip_names(names)
        if source_archive.read("precision.txt").decode("utf-8").strip() != "q8vnni":
            raise ValueError("source package does not declare the Q8 VNNI backend")
        members = json.loads(source_archive.read("members.json"))
        expected_members = [
            {
                "weights": "weights_0.npz",
                "coef": 0.5,
                "source": "vt_all_h390_ds_homogeneous_s3_r3@ep5",
            },
            {
                "weights": "weights_1.npz",
                "coef": 0.5,
                "source": "vt_all_h390_ds_homogeneous_s4_r3@ep5",
            },
        ]
        if members != expected_members:
            raise ValueError("source package member identity/order/coefficient mismatch")
        if sha256_bytes(source_archive.read("v3/wnn_native.abi3.so")) != (
            "ce179a68b7cd321368cc9ab73cdf1b79973afcdb8266ef8f6d7e613bb6f9874e"
        ):
            raise ValueError("source package native v3 backend digest mismatch")
        with zipfile.ZipFile(variant_zip, "w") as variant_archive:
            variant_archive.comment = source_archive.comment
            for info in infos:
                original = source_archive.read(info.filename)
                derived = original
                if info.filename in {"weights_0.npz", "weights_1.npz"}:
                    member_index = int(info.filename.removeprefix("weights_").removesuffix(".npz"))
                    if sha256_bytes(original) != EXPECTED_WEIGHT_SHA256[info.filename]:
                        raise ValueError(f"source member digest mismatch: {info.filename}")
                    derived, codes, record = transform_npz(original, member_index)
                    audit_arrays.update(codes)
                    member_records[member_index] = record
                    changed_entries.add(info.filename)
                variant_archive.writestr(info, derived, compresslevel=9)
                entry_records.append(
                    {
                        "path": info.filename,
                        "source_payload_sha256": sha256_bytes(original),
                        "variant_payload_sha256": sha256_bytes(derived),
                        "payload_identical": original == derived,
                    }
                )

    if changed_entries != {"weights_0.npz", "weights_1.npz"}:
        raise AssertionError(f"unexpected package entry changes: {sorted(changed_entries)}")
    np.savez_compressed(codes_path, **audit_arrays)

    with (
        zipfile.ZipFile(source_zip, "r") as source_archive,
        zipfile.ZipFile(variant_zip, "r") as variant_archive,
    ):
        source_names = source_archive.namelist()
        variant_names = variant_archive.namelist()
        if source_names != variant_names:
            raise AssertionError("package member ordering or names changed")
        for name in source_names:
            if name not in changed_entries and source_archive.read(name) != variant_archive.read(
                name
            ):
                raise AssertionError(f"non-weight package entry changed: {name}")

    receipt: dict[str, object] = {
        "schema": "wnn.q4-rowwise-rounded-q8-backend-build.v1",
        "interpretation": (
            "Four-bit-grid rowwise weight rounding reconstructed as float32, then passed to the "
            "existing Q8 VNNI backend. This is not an INT4 runtime or speed claim."
        ),
        "quantization": {
            "selected_tensors": sorted(TARGET_TENSORS),
            "levels": list(range(-7, 8)),
            "scale": "per output row max(abs(weight))/7, float32",
            "rounding": "numpy.rint round-to-nearest, ties-to-even",
            "zero_row": "zero codes and zero scale",
            "reconstruction": "float32(code) * float32(row_scale)",
            "next_backend": "same Q8 VNNI model loader and native inference path as source ZIP",
        },
        "source": {
            "zip_path": str(source_zip.resolve()),
            "zip_sha256": source_digest,
            "source_zip_bytes": source_zip.stat().st_size,
            "member_manifest": "pkg/members.json",
            "members": [
                {
                    "weights": "weights_0.npz",
                    "source": "vt_all_h390_ds_homogeneous_s3_r3@ep5",
                    "checkpoint_sha256": "04675f8220ba949c1dfb6c5796a3f25a9b8f655bb48f83c188bf8f9bacd361e0",
                    "weight_npz_sha256": "62b11349b4372beda7abbd35f7e006d2c967237070cbd5318c7a2f15bc0d1ec6",
                },
                {
                    "weights": "weights_1.npz",
                    "source": "vt_all_h390_ds_homogeneous_s4_r3@ep5",
                    "checkpoint_sha256": "cefee5b51cae7b72247a26558444f526e915a9fb7ad3927abd6cc02e63018419",
                    "weight_npz_sha256": "9f019d54a147296037b1bb2c6ad8b15b87070bde0402ff5426ce9c4804f52eaf",
                },
            ],
        },
        "variant": {
            "zip_path": variant_zip.name,
            "zip_sha256": sha256_file(variant_zip),
            "zip_bytes": variant_zip.stat().st_size,
            "codes_path": codes_path.name,
            "codes_sha256": sha256_file(codes_path),
            "codes_bytes": codes_path.stat().st_size,
            "only_changed_package_entries": sorted(changed_entries),
        },
        "member_tensor_audits": [member_records[index] for index in sorted(member_records)],
        "package_entries": entry_records,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    """Build the pinned variant from an exact source ZIP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = build_variant(arguments.source_zip, arguments.output_dir)
    print(json.dumps(receipt["variant"], sort_keys=True))


if __name__ == "__main__":
    main()

"""Bind a SwiGLU/GELU teacher cache to its actual export and parity receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from wnn.splits import load_campaign_split


def _sha256(path: Path) -> str:
    """Hash a file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    """Hash normalized JSON using the runtime's source identity convention."""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_identity(source_root: Path, entrypoint: Path) -> dict[str, Any]:
    """Recompute the trainer's source hash without importing training modules."""
    root = source_root.resolve()
    paths = sorted((root / "src").rglob("*.py"))
    resolved_entrypoint = entrypoint.resolve()
    if resolved_entrypoint not in paths:
        paths.append(resolved_entrypoint)
    files = {path.relative_to(root).as_posix(): _sha256(path) for path in paths}
    return {"sha256": _canonical_sha256(files), "files": files}


def _matches_config(actual: object, expected: object) -> bool:
    """Require every explicitly frozen config value in the producer record."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _matches_config(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _matches_config(left, right) for left, right in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def _read(path: Path) -> dict[str, Any]:
    """Read bounded JSON metadata and require an object."""
    if path.stat().st_size > 32_000_000:
        raise ValueError(f"metadata file is oversized: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be an object: {path}")
    return value


def validate_teacher_cache(
    data_root: Path,
    split_manifest: Path,
    one: Path,
    eight: Path,
    parity_path: Path,
    checkpoint: Path,
    expected_activation: str,
    expected_width: int,
    expected_fold: int,
    expected_seed: int,
    expected_config_path: Path,
    source_root: Path,
    entrypoint: Path,
    binding_output: Path,
) -> None:
    """Verify the selected full-fit cache, its parity proof and frozen teacher arm."""
    parity = _read(parity_path)
    left, right = _read(one / "manifest.json"), _read(eight / "manifest.json")
    split = load_campaign_split(split_manifest, data_root)
    manifests = {"one": _sha256(one / "manifest.json"), "eight": _sha256(eight / "manifest.json")}
    if parity.get("schema") != "wnn.sequence-cache-parity.v1" or parity.get("passed") is not True:
        raise ValueError("teacher cache needs a passing versioned parity receipt")
    if parity.get("rtol") != 1e-5 or parity.get("atol") != 1e-6:
        raise ValueError("teacher cache parity tolerances differ from the declared contract")
    if parity.get("manifest_sha256") != manifests:
        raise ValueError("parity receipt is not bound to these exact cache manifests")
    for manifest, world in ((left, 1), (right, 8)):
        if (
            manifest.get("schema") != "sequence-teacher-cache-v1"
            or manifest.get("complete") is not True
            or manifest.get("membership") != "fit"
            or manifest.get("world_size") != world
            or manifest.get("hint_dim") != 0
        ):
            raise ValueError("teacher export does not match the declared output-cache contract")
    if left.get("split_sha256") != right.get("split_sha256") or (
        parity.get("split_sha256") != split.sha256 or right.get("split_sha256") != split.sha256
    ):
        raise ValueError("teacher cache and parity receipt have different split identities")
    producer = right.get("producer")
    if not isinstance(producer, dict) or producer != left.get("producer"):
        raise ValueError("teacher exports do not share one complete producer identity")
    checkpoint_sha256 = _sha256(checkpoint)
    if producer.get("checkpoint_sha256") != checkpoint_sha256 or (
        parity.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("teacher caches do not bind to the expected checkpoint")
    config = producer.get("config")
    model = config.get("model") if isinstance(config, dict) else None
    if not isinstance(model, dict):
        raise ValueError("teacher producer is missing its complete model config")
    expected_config = _read(expected_config_path)
    if not _matches_config(config, expected_config):
        raise ValueError("teacher producer differs from the complete frozen recipe")
    if (
        model.get("ff_activation", "gelu") != expected_activation
        or model.get("ff_hidden") != expected_width
        or model.get("hidden") != 768
        or model.get("layers") != 24
        or model.get("heads") != 12
        or model.get("context_rows") != 1024
        or model.get("hint_dim") != 0
        or config.get("precision_mode") != "bf16_bf16x9"
        or config.get("expected_fold") != expected_fold
        or config.get("seed") != expected_seed
    ):
        raise ValueError("teacher producer config differs from the frozen arm/fold/seed")
    provenance = producer.get("provenance")
    identity = provenance.get("identity") if isinstance(provenance, dict) else None
    expected_source = _source_identity(source_root, entrypoint)
    if not isinstance(identity, dict) or (
        identity.get("source_sha256") != expected_source["sha256"]
        or identity.get("source_files") != expected_source["files"]
    ):
        raise ValueError("teacher cache does not bind to the pinned runtime source tree")
    max_error = parity.get("max_abs_error")
    if isinstance(max_error, bool) or not isinstance(max_error, (int, float)):
        raise ValueError("cache parity receipt has no finite numeric error bound")
    if not math.isfinite(max_error) or max_error < 0:
        raise ValueError("cache parity error bound is not finite and nonnegative")
    binding = {
        "schema": "wnn.transformer-swiglu-teacher-cache-binding.v1",
        "status": "bound_for_student_fit",
        "activation": expected_activation,
        "ff_hidden": expected_width,
        "fold": expected_fold,
        "teacher_seed": expected_seed,
        "teacher_config_sha256": _canonical_sha256(expected_config),
        "source_sha256": expected_source["sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "split_sha256": right["split_sha256"],
        "parity_receipt_sha256": _sha256(parity_path),
        "parity_receipt": parity,
        "manifest_sha256": manifests,
        "max_abs_error": float(max_error),
        "rtol": 1e-5,
        "atol": 1e-6,
    }
    if binding_output.exists():
        if _read(binding_output) != binding:
            raise ValueError(
                "existing cache binding receipt differs from current teacher artifacts"
            )
        return
    with binding_output.open("x", encoding="utf-8") as stream:
        json.dump(binding, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    """Validate both existing cache roots before any student stage starts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--one", type=Path, required=True)
    parser.add_argument("--eight", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--activation", choices=("gelu", "swiglu"), required=True)
    parser.add_argument("--width", type=int, choices=(2048, 3072), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 2), required=True)
    parser.add_argument("--seed", type=int, choices=(101001, 101002), required=True)
    parser.add_argument("--expected-config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--binding-output", type=Path, required=True)
    args = parser.parse_args()
    validate_teacher_cache(
        args.data_root,
        args.split_manifest,
        args.one,
        args.eight,
        args.parity,
        args.checkpoint,
        args.activation,
        args.width,
        args.fold,
        args.seed,
        args.expected_config,
        args.source_root,
        args.entrypoint,
        args.binding_output,
    )


if __name__ == "__main__":
    main()

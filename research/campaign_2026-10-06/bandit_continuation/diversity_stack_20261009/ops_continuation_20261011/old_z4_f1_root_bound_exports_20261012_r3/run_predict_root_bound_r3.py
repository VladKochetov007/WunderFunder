from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--trainer-sha256", required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--predictor-sha256", required=True)
    parser.add_argument("--source-bindings", type=Path, required=True)
    parser.add_argument("--source-bindings-sha256", required=True)
    parser.add_argument("--input-bindings", type=Path, required=True)
    parser.add_argument("--input-bindings-sha256", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=("valid",), required=True)
    parser.add_argument("--epoch", type=int, choices=(1, 3), required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--expected-output", type=Path, required=True)
    parser.add_argument("--valid-bytes", type=int, required=True)
    parser.add_argument("--valid-rows", type=int, required=True)
    parser.add_argument("--valid-row-groups", type=int, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate source, data header and MAIN path binding without exporting",
    )
    return parser


def _require_sha256(value: str, label: str) -> None:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")


def _verify_file(path: Path, expected_sha256: str, label: str) -> Path:
    _require_sha256(expected_sha256, f"{label} expected SHA-256")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: {actual}")
    return resolved


def _validate_bindings(args: argparse.Namespace) -> dict[str, object]:
    project_root = args.project_root.resolve(strict=True)
    trainer_path = _verify_file(args.trainer, args.trainer_sha256, "trainer")
    predictor_path = _verify_file(args.predictor, args.predictor_sha256, "predictor")
    source_bindings_path = _verify_file(
        args.source_bindings, args.source_bindings_sha256, "source bindings"
    )
    input_bindings_path = _verify_file(
        args.input_bindings, args.input_bindings_sha256, "input bindings"
    )

    source_bindings = json.loads(source_bindings_path.read_text(encoding="utf-8"))
    if source_bindings.get("hft_snapshot", {}).get("trainer_sha256") != args.trainer_sha256:
        raise ValueError("source binding does not name the pinned trainer digest")
    export_consumer = source_bindings.get("export_consumer", {})
    if Path(export_consumer.get("script", "")).resolve() != predictor_path:
        raise ValueError("source binding does not name the pinned MAIN predictor path")

    input_bindings = json.loads(input_bindings_path.read_text(encoding="utf-8"))
    if input_bindings.get("project_root") != str(project_root):
        raise ValueError("input bindings refer to a different project root")
    fold = input_bindings.get("fold", {})
    if fold.get("fold") != 1 or fold.get("evaluation_group_count") != 457:
        raise ValueError("input bindings are not the sealed fold-1 export membership")
    if input_bindings.get("split_manifest", {}).get("sha256") != (
        "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae"
    ):
        raise ValueError("input bindings have an unexpected fold-1 split manifest")

    valid_path = project_root / "datasets" / "valid.parquet"
    if not valid_path.is_file() or valid_path.stat().st_size != args.valid_bytes:
        raise ValueError("actual valid.parquet is absent or has an unexpected byte size")
    import pyarrow.parquet as parquet

    metadata = parquet.ParquetFile(valid_path).metadata
    if metadata.num_rows != args.valid_rows or metadata.num_row_groups != args.valid_row_groups:
        raise ValueError("actual valid.parquet footer differs from the bound header")

    expected_output = project_root / "cache" / f"pred_{args.output_name}_{args.split}.npy"
    expected_sidecar = expected_output.with_suffix(".json")
    declared_output = args.expected_output.resolve(strict=False)
    if declared_output != expected_output.resolve(strict=False):
        raise ValueError("declared prediction output differs from the predictor path")
    if expected_output.exists() or expected_sidecar.exists():
        raise FileExistsError("prediction output or sidecar already exists")

    expected_epochs = {
        "oldz4_h230_parent_f1_s3_r4": 3,
        "oldz4_h230_student_f1_s3_r4": 1,
    }
    if args.run not in expected_epochs:
        raise ValueError("run is outside this identified F1 export successor")
    if args.epoch != expected_epochs[args.run] or args.output_name != args.run:
        raise ValueError("run, fixed checkpoint epoch, and output name do not match")
    checkpoint = project_root / "artifacts" / "runs" / args.run / f"model_ep{args.epoch}.pt"
    if not args.preflight_only:
        if args.expected_checkpoint_sha256 is None:
            raise ValueError("root terminal admission must bind the exact checkpoint SHA-256")
        _require_sha256(args.expected_checkpoint_sha256, "checkpoint expected SHA-256")
        if not checkpoint.is_file() or sha256_file(checkpoint) != args.expected_checkpoint_sha256:
            raise ValueError("checkpoint bytes do not match the root-bound producer receipt")

    if "train_rnn" in sys.modules:
        raise RuntimeError("train_rnn was imported before the pinned source check")
    sys.path.insert(0, str(trainer_path.parent))
    train_rnn = importlib.import_module("train_rnn")
    imported_trainer = Path(train_rnn.__file__).resolve(strict=True)
    if imported_trainer != trainer_path:
        raise ValueError(f"imported trainer path differs: {imported_trainer}")
    if train_rnn.ROOT.resolve() != trainer_path.parents[1]:
        raise ValueError("trainer default root no longer matches its frozen source snapshot")

    train_rnn.ROOT = project_root
    if Path(train_rnn.ROOT).resolve(strict=True) != project_root:
        raise ValueError("frozen trainer ROOT did not bind to MAIN")
    sequences = train_rnn.Sequences(args.split, [])
    if sequences.path.resolve(strict=True) != valid_path.resolve(strict=True):
        raise ValueError("bound trainer Sequences path does not resolve to MAIN valid.parquet")
    if sequences._file is not None:
        raise RuntimeError("empty-membership path check unexpectedly opened the dataset")

    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": (None if args.preflight_only else args.expected_checkpoint_sha256),
        "expected_output": str(expected_output),
        "expected_sidecar": str(expected_sidecar),
        "input_bindings_sha256": args.input_bindings_sha256,
        "project_root": str(project_root),
        "predictor_path": str(predictor_path),
        "predictor_sha256": args.predictor_sha256,
        "source_bindings_sha256": args.source_bindings_sha256,
        "trainer_default_root": str(trainer_path.parents[1]),
        "trainer_import_path": str(imported_trainer),
        "trainer_root_after_binding": str(Path(train_rnn.ROOT).resolve()),
        "trainer_sha256": args.trainer_sha256,
        "prediction_output_absent": True,
        "prediction_sidecar_absent": True,
        "valid_bytes": valid_path.stat().st_size,
        "valid_parquet_rows": metadata.num_rows,
        "valid_row_groups": metadata.num_row_groups,
        "valid_sequences_path_after_root_binding": str(sequences.path.resolve()),
        "sequences_file_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    receipt = _validate_bindings(args)
    if args.preflight_only:
        sys.stdout.write(
            json.dumps({"status": "preflight_pass_no_export", **receipt}, sort_keys=True) + "\n"
        )
        return 0

    predictor_path = Path(args.predictor).resolve(strict=True)
    sys.argv = [
        str(predictor_path),
        args.run,
        args.split,
        "--epoch",
        str(args.epoch),
        "--output-name",
        args.output_name,
    ]
    runpy.run_path(str(predictor_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

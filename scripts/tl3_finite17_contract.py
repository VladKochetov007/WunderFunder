"""Frozen recipe, source, and metadata-index checks for the finite TL3 chain."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

from scripts.tl3_chronological_recipes import (
    FINAL_WEIGHTS,
    PREFIX,
    SPLIT_SHA256,
    W18_COEFFICIENTS,
    build_nodes,
)
from wnn.chronological_source_index import verify_source_index_receipt
from wnn.portability import sha256_file
from wnn.splits import load_campaign_split
from wnn.training.environment import TrainingEnvironment
from wnn.training.source_identity import training_source_identity

FIT_LIMIT = 17
MAX_WALL_SECONDS = 168 * 60 * 60
MAX_GPU_SECONDS = 256 * 60 * 60
MAX_CELL_SECONDS = 6 * 60 * 60
MIN_FREE_BYTES = 12 << 30
TRAIN_GROUPS = 10325
VALID_GROUPS = 1873
VALID_SCORED_ROWS = 3528273
SEQUENCE_ROWS = 20000
TARGET_WIDTH = 2
FINAL_RUNS = tuple(PREFIX + name for name in FINAL_WEIGHTS)
ADOPTION_SCHEMA = "wnn.tl3-fit-adoption.v1"
PLAN_SCHEMA = "wnn.tl3-finite17-plan.v1"
STATE_SCHEMA = "wnn.tl3-finite17-state.v1"


class DriverError(RuntimeError):
    """A frozen finite-chain contract or ownership check failed."""


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON used for immutable command and plan identities."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_paths(root: Path) -> list[Path]:
    """Return the complete Python implementation set exercised by fit/export/score."""
    fixed = (
        "pyproject.toml",
        "uv.lock",
        "scripts/train_rnn.py",
        "scripts/predict_run.py",
        "scripts/build_distill_cache.py",
        "scripts/tl3_chronological_recipes.py",
        "scripts/tl3_finite17_contract.py",
        "scripts/tl3_finite17_process.py",
        "scripts/run_tl3_finite17.py",
        "scripts/score_tl3_chronological.py",
        "src/wnn/portability.py",
    )
    paths = {root / name for name in fixed}
    paths.update((root / "src/wnn").rglob("*.py"))
    if any(not path.is_file() for path in paths):
        missing = sorted(path.relative_to(root).as_posix() for path in paths if not path.is_file())
        raise DriverError(f"source identity has missing files: {missing}")
    return sorted(paths)


def source_identity(root: Path) -> dict[str, object]:
    """Hash exact local source files and the training subset recorded in checkpoints."""
    root = root.resolve(strict=True)
    files = {
        path.relative_to(root).as_posix(): sha256_file(path) for path in _source_paths(root)
    }
    training = training_source_identity(TrainingEnvironment(root=root, source_root=root))
    return {
        "schema": "wnn.tl3-source-identity.v1",
        "sha256": canonical_sha256(files),
        "files": files,
        "training_source": training,
        "training_source_sha256": _training_source_sha(training),
    }


def _training_source_sha(value: dict[str, object]) -> str:
    """Accept either historical aggregate field without weakening file identities."""
    for key in ("combined_sha256", "sha256"):
        digest = value.get(key)
        if isinstance(digest, str) and len(digest) == 64:
            return digest
    raise DriverError("training source identity lacks a SHA-256 aggregate")


def _verify_legacy_source_documents(
    adoption: dict[str, object], root: Path
) -> tuple[str, dict[str, str]]:
    """Validate the original R3 identity JSON without applying today's hash recipe."""
    adopted_path = root / "adopted-source.identity.json"
    snapshot_path = root / "source-r3.identity.json"
    for path in (adopted_path, snapshot_path):
        if path.is_symlink() or not path.is_file():
            raise DriverError(f"required immutable R3 source receipt is missing: {path.name}")
    if sha256_file(snapshot_path) != adoption.get("source_identity_file_sha256"):
        raise DriverError("R3 source snapshot JSON bytes differ from the adoption receipt")
    try:
        adopted = json.loads(adopted_path.read_text(encoding="utf-8"))
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DriverError("R3 source identity receipts are unreadable JSON") from error
    files = adopted.get("files") if isinstance(adopted, dict) else None
    snapshot_files = snapshot.get("files") if isinstance(snapshot, dict) else None
    if (
        not isinstance(adopted, dict)
        or adopted.get("schema") != "wnn.training-source.v1"
        or not isinstance(files, dict)
        or not files
        or not isinstance(snapshot, dict)
        or not isinstance(snapshot_files, dict)
    ):
        raise DriverError("R3 source identity receipts do not match the historical schemas")
    if adopted.get("missing_dependency_files") != ["uv.lock"]:
        raise DriverError("R3 source receipt dependency omissions changed")
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for name, digest in files.items()
    ):
        raise DriverError("R3 source manifest contains an invalid path or content digest")
    manifest_sha = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest_sha != adopted.get("sha256") or manifest_sha != adoption.get("source_sha256"):
        raise DriverError("R3 source closure does not reproduce the adopted training SHA")
    missing_snapshot_paths = set(files) - set(snapshot_files)
    if missing_snapshot_paths != {"pyproject.toml"}:
        raise DriverError("R3 source closure does not align with its source snapshot manifest")
    if any(files[name] != snapshot_files[name] for name in set(files) & set(snapshot_files)):
        raise DriverError("R3 closure file hashes disagree with the recorded source snapshot")
    source_root = adoption.get("source_root")
    if (
        not isinstance(source_root, str)
        or not PurePosixPath(source_root).is_absolute()
        or PurePosixPath(source_root).name != "source-r3"
    ):
        raise DriverError("adopted source root is not the pinned R3 directory")
    return source_root, files


def _verify_legacy_source_files(
    source_root: str, files: dict[str, str], root: Path
) -> None:
    """Hash only the bounded historical code closure beneath the pinned R3 directory."""
    root = root.resolve(strict=True)
    source_path = Path(source_root)
    if source_path.is_symlink():
        raise DriverError("adopted R3 source directory cannot be a symlink")
    resolved_source = source_path.resolve(strict=True)
    if resolved_source != (root / "source-r3").resolve(strict=True):
        raise DriverError("adopted R3 source directory is not the runtime root child")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise DriverError(f"R3 source manifest contains an unsafe path: {name}")
        path = resolved_source / relative
        try:
            resolved_path = path.resolve(strict=True)
        except OSError as error:
            raise DriverError(f"R3 source file is missing: {name}") from error
        if (
            path.is_symlink()
            or not path.is_file()
            or not resolved_path.is_relative_to(resolved_source)
            or sha256_file(path) != expected
        ):
            raise DriverError(f"R3 source file bytes changed: {name}")


def _normalized_args(raw: list[str]) -> dict[str, object]:
    """Parse the real trainer CLI and resolve only documented schedule defaults."""
    from wnn.training.checkpoints import configure_checkpoint_policy
    from wnn.training.cli import build_parser

    parser = build_parser()
    try:
        args = parser.parse_args(raw)
    except SystemExit as error:
        raise DriverError("frozen trainer arguments do not parse") from error
    configure_checkpoint_policy(args, parser)
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if values.get("checkpoint_policy") == "fixed" and values.get("checkpoint_epoch") is None:
        values["checkpoint_epoch"] = int(values["epochs"]) - 1
    if values.get("sched") == "onecycle" and values.get("warmup_frac") is None:
        values["warmup_frac"] = 0.05
    return values


def validate_recipe_arguments(nodes: list[dict[str, object]], split: object) -> None:
    """Require every frozen node to pass the canonical train-only primary gate."""
    from wnn.training.setup_lineage import require_chronological_primary_training

    for node in nodes:
        raw = list(node["args"])
        from wnn.training.checkpoints import configure_checkpoint_policy
        from wnn.training.cli import build_parser

        parser = build_parser()
        try:
            args = parser.parse_args(raw)
        except SystemExit as error:
            raise DriverError(f"{node['id']}: frozen trainer arguments do not parse") from error
        configure_checkpoint_policy(args, parser)
        try:
            require_chronological_primary_training(parser, args, split)
        except SystemExit as error:
            raise DriverError(f"{node['id']}: chronological primary gate rejected recipe") from error
        if args.checkpoint_epoch != int(node["endpoint"]):
            raise DriverError(f"{node['id']}: fixed endpoint differs from the recipe")
        if args.train_source != "train" or args.train_seqs != TRAIN_GROUPS:
            raise DriverError(f"{node['id']}: fit membership differs from official train")
        if args.no_eval_during_fit or not args.full_valid or args.eval_source != "dev":
            raise DriverError(f"{node['id']}: every epoch must report full official valid")
        if args.precision != "fp32" or args.checkpoint_policy != "fixed":
            raise DriverError(f"{node['id']}: precision or endpoint policy changed")
        if hasattr(args, "train_eval_seqs"):
            expected_train_eval = 0 if node["id"] == "fin256_s3" else 32
            if args.train_eval_seqs != expected_train_eval:
                raise DriverError(f"{node['id']}: bounded train-eval diagnostic changed")
        if args.propensity or args.prop_input or args.prop_cache or args.mask_weight:
            raise DriverError(f"{node['id']}: forbidden learned or valid-mask weighting")
    if len(nodes) != FIT_LIMIT or sum(int(node["expected_updates"]) for node in nodes) != 257740:
        raise DriverError("the finite TL3 fit/update inventory changed")


def verify_data_contract(root: Path, split_path: Path) -> dict[str, object]:
    """Validate the official source audit, full indices, graph, and exact train/valid split."""
    root, split_path = root.resolve(strict=True), split_path.resolve(strict=True)
    split = load_campaign_split(split_path, root)
    if split.sha256 != SPLIT_SHA256:
        raise DriverError("chronological split differs from the frozen TL3 source pin")
    if len(split.train_groups) != TRAIN_GROUPS or split.valid_groups:
        raise DriverError("official fitting membership is not train-only 10,325 sequences")
    if len(split.evaluation_groups) != VALID_GROUPS:
        raise DriverError("official full-valid membership is not 1,873 sequences")
    indices: dict[str, dict[str, object]] = {}
    for name in ("train", "valid"):
        groups, seq_ix = verify_source_index_receipt(root, name)
        receipt_path = root / "cache" / f"{name}_index.identity.json"
        indices[name] = {
            "index_sha256": sha256_file(root / "cache" / f"{name}_index.npz"),
            "receipt_sha256": sha256_file(receipt_path),
            "index_map_sha256": canonical_index_sha(groups, seq_ix),
            "group_count": len(groups),
        }
    source_audit_path = root / "cache/source_audit.json"
    return {
        "split": split,
        "identity": {
            "split_manifest": str(split_path),
            "split_manifest_sha256": split.sha256,
            "source_audit_sha256": sha256_file(source_audit_path),
            "indices": indices,
            "fit_train_count": len(split.train_groups),
            "fit_valid_count": len(split.valid_groups),
            "evaluation_valid_count": len(split.evaluation_groups),
        },
    }


def canonical_index_sha(groups: np.ndarray, seq_ix: np.ndarray) -> str:
    """Hash the verified little-endian source map without altering its order."""
    digest = hashlib.sha256()
    for values in (groups, seq_ix):
        digest.update(np.asarray(values, dtype="<i8").tobytes())
    return digest.hexdigest()


def ensure_private_train_index(root: Path, groups: np.ndarray) -> dict[str, object]:
    """Create only the row-group field legacy teacher alignment needs; never labels."""
    path = root / "cache/train_targets.npz"
    receipt_path = root / "cache/tl3_private_train_index.identity.json"
    valid_targets = root / "cache/valid_targets.npz"
    if valid_targets.exists():
        raise DriverError("valid target arrays are outside this train-only driver contract")
    groups = np.asarray(groups, dtype=np.int64)
    if path.is_symlink() or receipt_path.is_symlink():
        raise DriverError("private TL3 train index cannot use symlinks")
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"row_group"}:
                raise DriverError("existing train target cache contains labels or non-index arrays")
            current = np.asarray(archive["row_group"])
        if current.dtype.kind not in "iu" or not np.array_equal(current, groups):
            raise DriverError("existing train target index differs from the verified source index")
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".tl3-train-index-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez(stream, row_group=groups)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    receipt: dict[str, object] = {
        "schema": "wnn.tl3-private-row-group-index.v1",
        "path": "cache/train_targets.npz",
        "sha256": sha256_file(path),
        "group_count": len(groups),
        "row_group_sha256": hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest(),
        "fields": ["row_group"],
        "source": "verified cache/train_index.npz only",
        "targets_or_scoring_mask_read": False,
    }
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
            raise DriverError("private train-index receipt changed")
    else:
        write_atomic_json(receipt_path, receipt)
    return receipt


def build_plan(
    root: Path,
    split_path: Path,
    state_dir: Path,
    adoption: dict[str, object],
) -> dict[str, object]:
    """Freeze 17 nodes, source hashes, fixed endpoints, and first-fit adoption."""
    root, split_path = root.resolve(strict=True), split_path.resolve(strict=True)
    data = verify_data_contract(root, split_path)
    split = data["split"]
    nodes = build_nodes(split_path, root)
    validate_recipe_arguments(nodes, split)
    first_node = next(node for node in nodes if node["id"] == "fin256_s3")
    validate_adoption(adoption, first_node, root)
    identity = source_identity(root)
    scorer = root / "scripts/score_tl3_chronological.py"
    if not scorer.is_file():
        raise DriverError("fixed chronological TL3 scorer is missing before chain admission")
    rows, _ = verify_source_index_receipt(root, "train")
    private_index = ensure_private_train_index(root, rows)
    enriched = []
    for node in nodes:
        record = dict(node)
        record["args"] = list(node["args"])
        record["args_sha256"] = canonical_sha256(record["args"])
        record["expected_updates"] = int(node["expected_updates"])
        record["run_directory"] = str(root / "artifacts/runs" / str(node["run"]))
        enriched.append(record)
    w18 = next(node for node in enriched if node["id"] == "w18final_distilled_s5")
    cache_path = Path(str(w18["args"][w18["args"].index("--distill-cache-root") + 1]))
    plan: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "root": str(root),
        "state_dir": str(state_dir.resolve()),
        "split": data["identity"],
        "source_identity": identity,
        "recipe_sha256": identity["files"]["scripts/tl3_chronological_recipes.py"],
        "scorer_sha256": identity["files"]["scripts/score_tl3_chronological.py"],
        "private_train_index": private_index,
        "adoption_sha256": canonical_sha256(adoption),
        "adopted_first_fit": adoption,
        "nodes": enriched,
        "final_runs": list(FINAL_RUNS),
        "w18_cache": str(cache_path),
        "w18_coefficients": {
            name: {"family": family, "coefficient": coefficient, "epoch": epoch}
            for name, (family, coefficient, epoch) in W18_COEFFICIENTS.items()
        },
        "coefficient_sum": sum(value[1] for value in W18_COEFFICIENTS.values()),
        "budgets": {
            "fits": FIT_LIMIT,
            "wall_seconds": MAX_WALL_SECONDS,
            "gpu_seconds": MAX_GPU_SECONDS,
            "cell_seconds": MAX_CELL_SECONDS,
            "minimum_free_disk_bytes": MIN_FREE_BYTES,
            "excluded_gpu_indices": [4],
        },
    }
    plan["sha256"] = canonical_sha256(plan)
    return plan


def validate_adoption(
    adoption: dict[str, object], node: dict[str, object], root: Path
) -> dict[str, object]:
    """Check the existing seed-3 command, source, process group, and fixed recipe."""
    required = {
        "schema",
        "node_id",
        "run",
        "source_sha256",
        "source_identity_file_sha256",
        "source_root",
        "args_sha256",
        "argv",
        "training_args",
        "pid",
        "start_ticks",
        "process_group",
        "gpu_index",
        "gpu_uuid",
        "started_unix",
    }
    if set(adoption) != required or adoption.get("schema") != ADOPTION_SCHEMA:
        raise DriverError("first-fit adoption receipt has missing or unsupported fields")
    if adoption.get("node_id") != "fin256_s3" or adoption.get("run") != node["run"]:
        raise DriverError("adoption receipt does not identify the already-running fin256_s3")
    training_args = adoption.get("training_args")
    argv = adoption.get("argv")
    if not isinstance(training_args, list) or not all(isinstance(v, str) for v in training_args):
        raise DriverError("adoption receipt training arguments are invalid")
    if not isinstance(argv, list) or not all(isinstance(v, str) for v in argv):
        raise DriverError("adoption receipt original argv is invalid")
    if canonical_sha256(training_args) != adoption.get("args_sha256"):
        raise DriverError("adopted training arguments differ from their receipt hash")
    name_index = argv.index("--name") if "--name" in argv else -1
    if name_index < 0 or argv[name_index:] != training_args:
        raise DriverError("adopted complete argv does not end in its pinned training arguments")
    if _normalized_args(training_args) != _normalized_args(list(node["args"])):
        raise DriverError("adopted first fit differs from fin256_s3 after documented defaults")
    source_root, source_files = _verify_legacy_source_documents(adoption, root)
    _verify_legacy_source_files(source_root, source_files, root)
    pid, pgid = adoption.get("pid"), adoption.get("process_group")
    if type(pid) is not int or type(pgid) is not int or pid != pgid:
        raise DriverError("adopted process group is not led by the receipt PID")
    if type(adoption.get("gpu_index")) is not int or adoption["gpu_index"] == 4:
        raise DriverError("adopted fit uses an excluded GPU index")
    if not isinstance(adoption.get("gpu_uuid"), str) or not adoption["gpu_uuid"].startswith("GPU-"):
        raise DriverError("adopted fit lacks a physical GPU UUID")
    return {
        "pid": pid,
        "process_group": pgid,
        "start_ticks": str(adoption["start_ticks"]),
        "session": pgid,
        "gpu_index": adoption["gpu_index"],
        "gpu_uuid": adoption["gpu_uuid"],
        "started_unix": float(adoption["started_unix"]),
        "source_sha256": adoption["source_sha256"],
        "source_identity_file_sha256": adoption["source_identity_file_sha256"],
        "args_sha256": adoption["args_sha256"],
        "argv": argv,
    }


def expected_prediction_bytes(group_count: int) -> int:
    """Return float32 sequence export bytes plus conservative NPY header slack."""
    if group_count < 0:
        raise ValueError("prediction group count must be nonnegative")
    return group_count * SEQUENCE_ROWS * TARGET_WIDTH * 4 + 4096


def write_atomic_json(path: Path, value: object) -> None:
    """Atomically replace one finite-chain JSON receipt and sync its directory."""
    if path.is_symlink():
        raise DriverError(f"refusing to write through a JSON symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)

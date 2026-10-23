"""Stream official scores and paired component intervals for the fixed F0 panel.

This consumer accepts only the reviewed five-arm, epoch-5, feature-only export.
It reads one validation group at a time and writes a create-only local report.
It does not run inference, fit, select epochs, or create a public-transfer claim.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np

SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
SOURCE_PINS = {
    "src/wnn/__init__.py": (63, "2dca10ee7e2e8c1139bed2165b322d2077fdbeac8095428e8c796efe47ea66c6"),
    "src/wnn/data.py": (4000, "a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00"),
    "src/wnn/gmetric.py": (
        8902,
        "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa",
    ),
    "src/wnn/evaluation.py": (
        8080,
        "96cebf2c8d564493f5a3a866b8d879964247f885587a455ebb1449e937230d0e",
    ),
    "src/wnn/portability.py": (
        2239,
        "a35f84ee5c95fd45cd76aa5f34f7013c8e1c5963a6306b0e4049386515589a44",
    ),
    "src/wnn/splits.py": (9000, "24024fbb82d33afdc2c709912404f7c1c26504a0816a050428c9f66e04e951fe"),
}
EXPECTED_BINDING_BYTES = 15822
EXPECTED_BINDING_SHA256 = "fcb3ee7934a19d023f29090e527be1d151dde23ae9fae3e0e3a911cea1bd809c"
EXPECTED_SOURCE_MANIFEST_BYTES = 27569
EXPECTED_SOURCE_MANIFEST_SHA256 = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
EXPECTED_SOURCE_SET_SHA256 = "9c461bcc5883572a071d41a86e18b9a91136099cf8fa87371ce7c3ee41445bd8"
EXPECTED_SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
EXPECTED_VALID_PARQUET = {
    "bytes": 5_615_509_128,
    "device": 64512,
    "inode": 743836257,
    "islink": False,
    "mtime_ns": 1_577_833_200_000_000_000,
    "path": "/home/grinchenkoeg/computer-vision-homework-20261018/data/datasets/valid.parquet",
    "registered_sha256": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
    "resolved": "/home/grinchenkoeg/WunderFunder/datasets/valid.parquet",
}
EXPECTED_DATA_ADMISSION = {
    "absolute_path": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_child_allthree_fit_v1/ADMISSION.json",
    "bytes": 12912,
    "relative_path": "ADMISSION.json",
    "root_key": "fit_receipts_root",
    "sha256": "5312ae5feffc67d7a33051379afabb9c134dd946f1f97315d839c7d6f10d5540",
}
EXPECTED_PORTABLE_REGISTRY_SHA256 = (
    "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
)
EXPECTED_PORTABLE_REGISTRY_BYTES = 2157
EXPECTED_PORTABLE_REGISTRY_RELATIVE_PATH = "support-source/terminal-v1/PORTABLE_SOURCES.json"
EXPECTED_PORTABLE_MANIFEST = {
    "absolute_path": "/home/grinchenkoeg/computer-vision-homework-20261018/support-source/terminal-v1/PORTABLE_SOURCES.json",
    "bytes": 2157,
    "sha256": EXPECTED_PORTABLE_REGISTRY_SHA256,
}
EXPECTED_PORTABLE_SOURCE_NAMES = [
    "cache/chain_folds.npz",
    "cache/chains.npz",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
    "datasets/train.parquet",
    "datasets/valid.parquet",
]
EXPECTED_PARENT_RESOLUTIONS = {
    "cache": "/home/grinchenkoeg/WunderFunder/cache",
    "datasets": "/home/grinchenkoeg/WunderFunder/datasets",
}
EXPECTED_REGISTERED_SOURCES = {
    "cache/chain_folds.npz": {
        "bytes": 462719,
        "device": 64512,
        "inode": 735483549,
        "islink": True,
        "mtime_ns": 1792272476034928524,
        "path": "/home/grinchenkoeg/computer-vision-homework-20261018/data/cache/chain_folds.npz",
        "registered_sha256": "a112f809259f631c2ee7c12480e4ec1d4eef197a980e7ce23c707d842d092567",
        "resolved": "/home/grinchenkoeg/computer-vision-homework-20261018/receipts/chain_folds_derived.npz",
    },
    "cache/chains.npz": {
        "bytes": 650142,
        "device": 64512,
        "inode": 742428360,
        "islink": True,
        "mtime_ns": 1790639004032905239,
        "path": "/home/grinchenkoeg/computer-vision-homework-20261018/data/cache/chains.npz",
        "registered_sha256": "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766",
        "resolved": "/home/grinchenkoeg/WunderFunder/cache/chains.npz",
    },
    "cache/train_targets.npz": {
        "bytes": 1909335790,
        "device": 64512,
        "inode": 742428599,
        "islink": True,
        "mtime_ns": 1790702729565256036,
        "path": "/home/grinchenkoeg/computer-vision-homework-20261018/data/cache/train_targets.npz",
        "registered_sha256": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9",
        "resolved": "/home/grinchenkoeg/WunderFunder/cache/train_targets.npz",
    },
    "cache/valid_targets.npz": {
        "bytes": 337186198,
        "device": 64512,
        "inode": 742428078,
        "islink": True,
        "mtime_ns": 1790638944452596999,
        "path": "/home/grinchenkoeg/computer-vision-homework-20261018/data/cache/valid_targets.npz",
        "registered_sha256": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955",
        "resolved": "/home/grinchenkoeg/WunderFunder/cache/valid_targets.npz",
    },
    "datasets/train.parquet": {
        "bytes": 29240219011,
        "device": 64512,
        "inode": 743836249,
        "islink": False,
        "mtime_ns": 1577833200000000000,
        "path": "/home/grinchenkoeg/computer-vision-homework-20261018/data/datasets/train.parquet",
        "registered_sha256": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
        "resolved": "/home/grinchenkoeg/WunderFunder/datasets/train.parquet",
    },
    "datasets/valid.parquet": EXPECTED_VALID_PARQUET,
}
ARM_MODES = {
    "control": "base",
    "ema": "ema",
    "sparse": "sparse",
    "nll": "base",
    "allthree": "ema_sparse",
}
N_GROUPS = 468
N_ROWS = 20_000
N_OUTPUTS = 2
PREDICTION_SHAPE = (N_GROUPS, N_ROWS, N_OUTPUTS)
PREDICTION_DATA_BYTES = int(np.prod(PREDICTION_SHAPE)) * np.dtype("<f4").itemsize
MAX_SMALL_JSON_BYTES = 2_000_000
MAX_EXPORT_MANIFEST_BYTES = 4_000_000
MAX_SPLIT_BYTES = 2_000_000
MAX_MEMBERSHIP_ARCHIVE_BYTES = 64_000_000
MAX_MEMBERSHIP_ARRAY_BYTES = 64_000_000
MAX_PREDICTION_BYTES = PREDICTION_DATA_BYTES + 4096
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 271828


class SequenceReader(Protocol):
    """Provide one source-qualified sequence with targets and scored mask."""

    def read(self, index: int, *, targets: bool = True, mask: bool = True) -> Any: ...


@dataclass(frozen=True)
class MetricHelpers:
    """Hold canonical metric and source-data APIs after exact source verification."""

    sequence_file: type
    global_accumulator: type
    global_wp: Any
    paired_component_moments: Any
    paired_component_bootstrap: Any
    load_campaign_split: Any
    warmup: int
    metric_id: str


def _stable_identity(before: os.stat_result, after: os.stat_result) -> tuple[int, int, int, int]:
    """Return fields used to detect a file changing during a bounded read."""
    return before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns


def _read_bounded(path: Path, cap: int) -> tuple[bytes, dict[str, Any], tuple[int, int, int, int]]:
    """Read a bounded regular file through a no-follow descriptor and pin its bytes."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
            raise ValueError(f"metadata is not a regular file or exceeds {cap} bytes: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, 1 << 20):
            if total + len(chunk) > cap:
                raise ValueError(f"metadata grew beyond its {cap}-byte bound: {path}")
            chunks.append(chunk)
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        linked = path.lstat()
        identity = _stable_identity(before, after)
        if (
            identity != _stable_identity(after, after)
            or (linked.st_dev, linked.st_ino) != identity[:2]
        ):
            raise ValueError(f"metadata changed while being read: {path}")
        if total != before.st_size:
            raise ValueError(f"metadata length changed while being read: {path}")
        raw = b"".join(chunks)
        return raw, {"path": str(path), "bytes": total, "sha256": digest.hexdigest()}, identity
    finally:
        os.close(descriptor)


def _pin_file(path: Path, cap: int) -> tuple[dict[str, Any], tuple[int, int, int, int]]:
    """Hash one regular no-follow file in bounded chunks."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
            raise ValueError(f"file is not regular or exceeds {cap} bytes: {path}")
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, 1 << 20):
            if total + len(chunk) > cap:
                raise ValueError(f"file grew beyond its {cap}-byte bound: {path}")
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        linked = path.lstat()
        identity = _stable_identity(before, after)
        if (
            identity != _stable_identity(after, after)
            or (linked.st_dev, linked.st_ino) != identity[:2]
        ):
            raise ValueError(f"file changed while being hashed: {path}")
        if total != before.st_size:
            raise ValueError(f"file length changed while being hashed: {path}")
        return {"path": str(path), "bytes": total, "sha256": digest.hexdigest()}, identity
    finally:
        os.close(descriptor)


def _safe_child(root: Path, relative: str) -> Path:
    """Resolve a bundle child while refusing traversal and symlink components."""
    child = PurePosixPath(relative)
    if child.is_absolute() or not child.parts or any(part in (".", "..") for part in child.parts):
        raise ValueError(f"unsafe relative path: {relative}")
    candidate = root
    for part in child.parts[:-1]:
        candidate = candidate / part
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(f"unsafe bundle directory: {candidate}")
    return candidate / child.parts[-1]


def _preflight_split_inputs(
    split_path: Path,
    data_root: Path,
    registered_sources: dict[str, dict[str, Any]],
) -> None:
    """Bound membership and cache inputs before the canonical split validator opens them."""
    raw, _, _ = _read_bounded(split_path, MAX_SPLIT_BYTES)
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SPLIT_SHA256:
        raise ValueError("campaign split bytes differ from the fixed c09f F0 manifest")
    manifest = json.loads(raw)
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("campaign split source identities are malformed")
    for relative, identity in sources.items():
        if not isinstance(relative, str) or not isinstance(identity, dict):
            raise ValueError("campaign split source identity is malformed")
        is_registered = relative in registered_sources
        if is_registered:
            registered = registered_sources[relative]
            observation = observe_registered_source(data_root, relative, registered)
            source_path = Path(str(registered["path"]))
            if (
                identity.get("bytes") != observation["bytes"]
                or identity.get("mtime_ns") != observation["mtime_ns"]
            ):
                raise ValueError(f"split source stat differs from admission: {relative}")
        else:
            source_path = _safe_child(data_root, relative)
        metadata = source_path.stat() if is_registered else source_path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"campaign source is not a regular file: {relative}")
        if not is_registered and "index_sha256" in identity and metadata.st_size > 64_000_000:
            raise ValueError(f"indexed campaign source exceeds the 64 MB preflight cap: {relative}")
        if not is_registered and "sha256" in identity and metadata.st_size > 64_000_000:
            raise ValueError(f"hashed campaign source exceeds the 64 MB preflight cap: {relative}")
        if "index_sha256" in identity:
            _preflight_npz_members(source_path, {"row_group", "seq_ix"}, 2)
    for relative in ("cache/chains.npz", "cache/chain_folds.npz"):
        if relative in registered_sources:
            observe_registered_source(data_root, relative, registered_sources[relative])
            cache_path = Path(str(registered_sources[relative]["path"]))
        else:
            cache_path = _safe_child(data_root, relative)
        metadata = cache_path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_MEMBERSHIP_ARCHIVE_BYTES:
            raise ValueError(f"campaign membership cache is not a bounded regular file: {relative}")
        _preflight_npz_members(cache_path, set(), MAX_MEMBERSHIP_ARRAY_BYTES)


def _preflight_npz_members(path: Path, required: set[str], total_cap: int) -> None:
    """Bound ZIP member metadata before NumPy expands the canonical membership arrays."""
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
    if len(members) > 32:
        raise ValueError(f"membership archive contains too many arrays: {path}")
    names = {Path(item.filename).name.removesuffix(".npy") for item in members}
    if not required.issubset(names):
        raise ValueError(f"membership archive is missing required index arrays: {path}")
    expanded_total = 0
    for item in members:
        if item.flag_bits & 0x1 or item.file_size > total_cap:
            raise ValueError(f"membership archive contains an unsafe expanded member: {path}")
        expanded_total += item.file_size
    if expanded_total > total_cap:
        raise ValueError(f"membership archive exceeds the expanded-size bound: {path}")


def observe_registered_source(
    data_root: Path, relative: str, expected_identity: dict[str, Any]
) -> dict[str, Any]:
    """Stat one exact admitted leaf, allowing only its registered resolved target."""
    path = Path(str(expected_identity.get("path", "")))
    expected_resolved = Path(str(expected_identity.get("resolved", "")))
    if path != data_root / PurePosixPath(relative):
        raise ValueError(f"registered source path differs from the fixed data root: {relative}")
    parent = path.parent
    parent_key = relative.split("/", 1)[0]
    if parent_key not in EXPECTED_PARENT_RESOLUTIONS:
        raise ValueError(f"registered source has an unrecognized parent alias: {relative}")
    expected_parent = Path(EXPECTED_PARENT_RESOLUTIONS[parent_key])
    parent_before = parent.stat()
    if not stat.S_ISDIR(parent_before.st_mode):
        raise ValueError(f"registered source parent is not a directory: {relative}")
    if (
        parent.resolve(strict=True) != expected_parent
        or path.resolve(strict=True) != expected_resolved
    ):
        raise ValueError(f"registered source differs from its exact alias target: {relative}")
    linked_before = path.lstat()
    if stat.S_ISLNK(linked_before.st_mode) is not expected_identity["islink"]:
        raise ValueError(f"registered source symlink status differs from admission: {relative}")
    source_stat = path.stat()
    linked_after = path.lstat()
    source_after = path.stat()
    parent_after = parent.stat()
    resolved_after = path.resolve(strict=True)
    parent_resolved_after = parent.resolve(strict=True)
    if (
        not stat.S_ISREG(source_stat.st_mode)
        or stat.S_ISLNK(linked_after.st_mode) is not expected_identity["islink"]
        or (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns)
        != (
            expected_identity["device"],
            expected_identity["inode"],
            expected_identity["bytes"],
            expected_identity["mtime_ns"],
        )
        or (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns)
        != (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
        )
        or (linked_before.st_dev, linked_before.st_ino, linked_before.st_mtime_ns)
        != (linked_after.st_dev, linked_after.st_ino, linked_after.st_mtime_ns)
        or (parent_before.st_dev, parent_before.st_ino)
        != (parent_after.st_dev, parent_after.st_ino)
        or resolved_after != expected_resolved
        or parent_resolved_after != expected_parent
    ):
        raise ValueError(
            f"registered source identity changed or differs from admission: {relative}"
        )
    return {
        "path": str(path),
        "resolved": str(expected_resolved),
        "bytes": source_stat.st_size,
        "device": source_stat.st_dev,
        "inode": source_stat.st_ino,
        "islink": bool(expected_identity["islink"]),
        "mtime_ns": source_stat.st_mtime_ns,
        "parent": {
            "path": str(parent),
            "resolved": str(expected_parent),
            "device": parent_after.st_dev,
            "inode": parent_after.st_ino,
        },
    }


def _read_json(
    path: Path, cap: int
) -> tuple[dict[str, Any], dict[str, Any], tuple[int, int, int, int]]:
    """Parse a bounded JSON object and retain its exact file pin."""
    raw, pin, identity = _read_bounded(path, cap)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, pin, identity


def _verify_source_pins(source_root: Path) -> None:
    """Verify the direct source modules used by this scorer before importing them."""
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("source root must be a real directory")
    if not sys.dont_write_bytecode:
        raise RuntimeError("launch this scorer with Python -B to avoid source-tree bytecode")
    for relative, (expected_bytes, expected_sha256) in SOURCE_PINS.items():
        path = _safe_child(source_root, relative)
        pin, _ = _pin_file(path, 50_000_000)
        if (pin["bytes"], pin["sha256"]) != (expected_bytes, expected_sha256):
            raise ValueError(f"source pin differs: {relative}")


def _load_helpers(source_root: Path, *, portable_registry: Path | None = None) -> MetricHelpers:
    """Import only exact source-pinned data, scoring, and portable-input helpers."""
    registry_env = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    if portable_registry is None:
        if registry_env:
            raise RuntimeError("a portable registry path must be explicitly bound by the scorer")
    else:
        expected_registry = _safe_child(
            source_root.parent, EXPECTED_PORTABLE_REGISTRY_RELATIVE_PATH
        )
        if portable_registry != expected_registry or registry_env != str(expected_registry):
            raise RuntimeError("portable registry environment differs from the fixed worker path")
        registry_pin, _ = _pin_file(expected_registry, 2_000_000)
        if (registry_pin["bytes"], registry_pin["sha256"]) != (
            EXPECTED_PORTABLE_REGISTRY_BYTES,
            EXPECTED_PORTABLE_REGISTRY_SHA256,
        ):
            raise ValueError("portable source registry differs from the fixed admitted registry")
    _verify_source_pins(source_root)
    if any(name == "wnn" or name.startswith("wnn.") for name in sys.modules):
        raise RuntimeError("WNN modules were preloaded before source pin verification")
    sys.path.insert(0, str(source_root / "src"))
    previous_cache_prefix = sys.pycache_prefix
    cache_prefix = os.path.join(os.devnull, "wunder-first-cell-no-bytecode")
    if Path(cache_prefix).exists():
        raise RuntimeError("isolated bytecode cache path unexpectedly exists")
    sys.pycache_prefix = cache_prefix
    try:
        data = importlib.import_module("wnn.data")
        gmetric = importlib.import_module("wnn.gmetric")
        evaluation = importlib.import_module("wnn.evaluation")
        portability = importlib.import_module("wnn.portability")
        splits = importlib.import_module("wnn.splits")
        package = importlib.import_module("wnn")
    finally:
        sys.pycache_prefix = previous_cache_prefix
    modules = {
        "src/wnn/__init__.py": package,
        "src/wnn/data.py": data,
        "src/wnn/gmetric.py": gmetric,
        "src/wnn/evaluation.py": evaluation,
        "src/wnn/portability.py": portability,
        "src/wnn/splits.py": splits,
    }
    for relative, module in modules.items():
        origin = Path(module.__file__).resolve(strict=True)
        expected = (source_root / relative).resolve(strict=True)
        if origin != expected:
            raise ValueError(f"import origin differs for {relative}: {origin}")
        pin, _ = _pin_file(origin, 50_000_000)
        if (pin["bytes"], pin["sha256"]) != SOURCE_PINS[relative]:
            raise ValueError(f"imported source changed for {relative}")
    if gmetric.WARMUP != 99 or gmetric.METRIC_ID != "global_wp_clip2_v1":
        raise ValueError("official metric identity differs from the fixed panel")
    return MetricHelpers(
        sequence_file=data.SequenceFile,
        global_accumulator=gmetric.GlobalAccumulator,
        global_wp=gmetric.global_wp,
        paired_component_moments=evaluation.paired_component_moments,
        paired_component_bootstrap=evaluation.paired_component_bootstrap,
        load_campaign_split=splits.load_campaign_split,
        warmup=gmetric.WARMUP,
        metric_id=gmetric.METRIC_ID,
    )


def require_ordered_groups(expected: list[int], actual: Any) -> None:
    """Require exact group membership and order from the frozen export contract."""
    if not isinstance(actual, list) or any(type(group) is not int for group in actual):
        raise ValueError("ordered group list must contain strict integer IDs")
    if actual != expected:
        raise ValueError("export group order differs from the canonical F0 evaluation order")


def membership_sha256(groups: list[int]) -> str:
    """Hash the producer's ASCII decimal row-group list, one ID per line."""
    encoded = "".join(f"{group}\n" for group in groups).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def score_stream(
    groups: list[int],
    component_by_group: dict[int, int],
    predictions: dict[str, np.ndarray],
    sequence_reader: SequenceReader,
    helpers: MetricHelpers,
) -> dict[str, Any]:
    """Stream per-arm official scores and four paired component bootstraps."""
    if set(predictions) != set(ARM_MODES):
        raise ValueError("exactly the five fixed comparison arms are required")
    if (
        len(groups) == 0
        or any(type(group) is not int for group in groups)
        or len(set(groups)) != len(groups)
    ):
        raise ValueError("scoring group list must be nonempty and unique")
    if any(group not in component_by_group for group in groups):
        raise ValueError("a selected group lacks its canonical feature-only component")
    if any(type(component_by_group[group]) is not int for group in groups):
        raise ValueError("component IDs must be canonical strict integers")
    for arm, array in predictions.items():
        if array.shape != (len(groups), N_ROWS, N_OUTPUTS) or array.dtype.str != "<f4":
            raise ValueError(f"{arm} prediction array is not aligned FP32 (N, 20000, 2)")

    accumulators = {arm: helpers.global_accumulator() for arm in ARM_MODES}
    contrast_arms = tuple(arm for arm in ARM_MODES if arm != "control")
    paired_candidate = {
        arm: np.zeros((len(groups), 2, 6), dtype=np.float64) for arm in contrast_arms
    }
    paired_control = {arm: np.zeros((len(groups), 2, 6), dtype=np.float64) for arm in contrast_arms}
    steps = np.arange(N_ROWS) >= helpers.warmup

    for position, group in enumerate(groups):
        sequence = sequence_reader.read(group, targets=True, mask=True)
        targets = sequence.targets
        scored = sequence.scored
        if targets is None or scored is None:
            raise ValueError(f"group {group} lacks validation targets or official scored mask")
        if targets.shape != (N_ROWS, 2) or targets.dtype != np.float32:
            raise ValueError(f"group {group} target shape or dtype differs")
        if scored.shape != (N_ROWS,) or scored.dtype.kind != "b":
            raise ValueError(f"group {group} scored mask shape or dtype differs")
        official_mask = np.asarray(scored, dtype=bool) & steps
        targets_batch = targets[None, ...]
        mask_batch = official_mask[None, ...]
        group_predictions = {
            arm: np.asarray(predictions[arm][position], dtype=np.float32)[None, ...]
            for arm in ARM_MODES
        }
        for arm in ARM_MODES:
            if not np.isfinite(group_predictions[arm]).all():
                raise ValueError(f"{arm} has a nonfinite exported prediction in group {group}")
            accumulators[arm].add(targets_batch, group_predictions[arm], mask_batch)
        if official_mask.any():
            for arm in contrast_arms:
                arm_moments, control_moments = helpers.paired_component_moments(
                    targets_batch,
                    group_predictions[arm],
                    group_predictions["control"],
                    mask_batch,
                )
                paired_candidate[arm][position] = arm_moments[0]
                paired_control[arm][position] = control_moments[0]
        del sequence, targets, scored, official_mask, targets_batch, mask_batch, group_predictions

    arm_scores = {arm: accumulators[arm].result() for arm in ARM_MODES}
    components = np.asarray([component_by_group[group] for group in groups], dtype=np.int64)
    contrasts: dict[str, Any] = {}
    for arm in contrast_arms:
        paired = helpers.paired_component_bootstrap(
            paired_candidate[arm],
            paired_control[arm],
            components,
            n_boot=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
            centered=True,
        )
        contrasts[arm] = {"candidate_minus_control": paired}
    return {
        "metric_id": helpers.metric_id,
        "ordered_group_count": len(groups),
        "distinct_component_count": int(np.unique(components).size),
        "official_mask": "is_scored AND (t >= 99)",
        "arms": arm_scores,
        "paired_contrasts_vs_control": contrasts,
        "uncertainty_scope": "paired feature-only successor-component bootstrap conditional on these five fixed epoch-5 models; not fit-seed or public-transfer uncertainty",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "centered_moments": True,
        },
    }

"""Frozen inputs and bounded validators for the B16 offline recovery."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PACKET = Path("research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2")
GRADE_PACKET = Path("research/campaign_2026-10-21/kaggle_h512_b16_f1_endpoint_grade_r2")
FIT_NAME = "r01_h512_f1_s1004_kaggle_fit_r2"
SWA_NAME = "r01_h512_f1_s1004_kaggle_swa345_r2"
SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
SOURCE_PROJECT_SHA = "18d727bb46bbe5e1d09164616555029dd7f13f362128ec4fcc424c76389cd278"
SOURCE_LOCK_SHA = "6776378c6216d219e58e24f006416c11aa0f9d9e1d8e5e080d7e25d67b07e173"
PREDICTOR_SHA = "904ddc9a2cafd194dc8d6ab9bdf1524120d6a089b83e33e2eb731e67abed6754"
TRAINER_ENTRYPOINT_SHA = "74530d3499bba1b5647e459105cb263bfc7205d7f3aa928684dfb279c2677345"
DERIVED_SHA = "548b42f06710a0f7c06377b8f2aaf7241469d75f0dad9de05bd4fbc29448d0c3"
ORIGINAL_SHA = "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae"
REGISTRY_SHA = "2dcd5cee727d1299f1fdab86b71116dc4a70008160c9495be8bf6f5c00b9dc67"
PROPOSAL_SHA = "dc519df5c7f6a182bfe3d34c1d580580b3b8dad55e4144a8117f24e16208ad09"
PLAN_SHA = "2ab85601f8bd7d99c52df036d7f7dbed32310e594dab7bca580564a81249c29e"
WORKER_SHA = "9b94418f06652e78ae7cdd4c02d7113b568c1f7fbece40ff3db4899b1c0dd57f"
COLLECTION_SHA = "217f62251d22c5e93a85f5c6bdc9de874135fe86b9998699a2c0b665aac98783"
MEAN_SOURCE_SHA = "8646ccc12b537a1bf2f2544dffd85d10287076451e5bab86a4b2d6379e39bbb3"
RUN_SOURCE_SHA = "7033ea007b9c7dcf498d3ab1e31aaeacb5725b7009fe719e9071c05b99272a76"
INITIALIZATION_STATUS = "not_emitted_by_pinned_trainer_source"
INITIALIZATION_LIMITATION = (
    "The pinned trainer records no initial_parameter_sha256. Cold-start arguments, "
    "no parent checkpoint, no resume state, and the pinned source are auditable; "
    "the initial tensor digest itself is unavailable and is not reconstructed."
)
EXPECTED_CHECKPOINTS = {
    3: (8_904_103, "9bc7d51731bb0312b2789e9d59e170d042c12c6c01a68ddad2bee36b3d5b0899"),
    4: (8_904_871, "d08c39fb4a93b3d7683aaa183b9e759df8eda04fba730fd46e1aeb241a7ffd14"),
    5: (8_905_703, "82cfa6662d7e8f2b61700253070f38e9f7ff0272aa26380049334306567840a9"),
}
EXPECTED_NORMALIZATION = {
    "mean": {
        "bytes_sha256": "d1c0855fd4fefcdfb2a738ee84f6076f702c7a7c66cce7d59518fb2dcf54a9e4",
        "dtype": "torch.float32",
        "shape": [211],
    },
    "std": {
        "bytes_sha256": "0604dc9301564444842b042a0d2bbb66fdd1c071a76372fdf04159578904d3d3",
        "dtype": "torch.float32",
        "shape": [211],
    },
}
DEFAULT_ARTIFACT_ROOT = Path(
    "/home/vlad/development/wunder/research_tmp/"
    "kaggle-h512-f1-b16-conditional-input-stage-20261021-r2/"
    "B16_TERMINAL_COLLECTION_V2_R3/outputs/results/"
    "h512_f1_b16_seed1004_profile_then_conditional_fit_r2/artifacts"
)
DEFAULT_RECOVERY_ROOT = Path(
    "/home/vlad/development/wunder/research_tmp/h512-f1-b16-offline-recovery-r1"
)
SCRATCH_ROOT = Path("/home/vlad/development/wunder/research_tmp")

# JSON decoded from frozen receipts is heterogeneous; every field is checked at its boundary.
JsonObject = dict[str, Any]


def sha256(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_int64(values: list[int]) -> str:
    """Hash ordered signed 64-bit row-group IDs in the recorded little-endian form."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<q", int(value)))
    return digest.hexdigest()


def read_json(path: Path) -> JsonObject:
    """Read a JSON object and reject non-object roots."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    """Raise one explicit contract error when a frozen invariant fails."""
    if not condition:
        raise ValueError(message)


def verify_pinned(path: Path, expected: str, label: str) -> Path:
    """Require a file to match its source-pinned SHA-256."""
    resolved = path.resolve(strict=True)
    require(sha256(resolved) == expected, f"Pinned {label} hash changed: {resolved}")
    return resolved


def load_frozen_inputs(root: Path = ROOT) -> tuple[JsonObject, JsonObject, JsonObject, JsonObject]:
    """Load and hash-check the predeclared recipe, F1 view, portable registry and score plan."""
    proposal = read_json(
        verify_pinned(
            root / PACKET / "H512_F1_B16_CONDITIONAL_PROFILE_FIT_PROPOSAL_20261021_R2.json",
            PROPOSAL_SHA,
            "proposal",
        )
    )
    derived = read_json(
        verify_pinned(root / PACKET / "fold1_profile_view.json", DERIVED_SHA, "derived F1 view")
    )
    registry = read_json(
        verify_pinned(
            root / PACKET / "fold1_portable_inputs.json", REGISTRY_SHA, "portable input registry"
        )
    )
    original = read_json(
        verify_pinned(
            root / "research/campaign_2026-09-29/artifacts/splits/fold1.json",
            ORIGINAL_SHA,
            "canonical F1 manifest",
        )
    )
    plan = read_json(
        verify_pinned(
            root / GRADE_PACKET / "F1_FIXED_SCREEN_R2.json",
            PLAN_SHA,
            "frozen five-endpoint score plan",
        )
    )
    require(
        {**original, "sources": derived.get("sources")} == derived,
        "Derived F1 view changed fields beyond raw source identities",
    )
    require(
        registry.get("split_manifest_sha256") == ORIGINAL_SHA,
        "Portable registry is not bound to canonical F1",
    )
    return proposal, derived, registry, plan


def validate_worker_receipts(root: Path = ROOT) -> JsonObject:
    """Confirm the recorded worker stopped only at the missing-identity postfit gate."""
    worker = read_json(
        verify_pinned(
            root / PACKET / "B16_WORKER_TERMINAL_R3.json", WORKER_SHA, "B16 worker terminal"
        )
    )
    collection = read_json(
        verify_pinned(
            root / PACKET / "B16_COLLECTION_R3.json", COLLECTION_SHA, "B16 collection receipt"
        )
    )
    require(worker.get("status") == "FAILED", "Expected the preserved failed-worker receipt")
    require(
        worker.get("error") == "Missing fresh initial parameter identity",
        "Worker failure differs from the diagnosed postfit identity gate",
    )
    require(worker.get("error_type") == "ValueError", "Worker terminal error type changed")
    require(worker.get("quality_fit_claim") is None, "Failed worker made a quality-fit claim")
    require(
        collection.get("file_count") == 206 and collection.get("total_bytes") == 95_199_834,
        "Collected artifact inventory changed",
    )
    require(
        collection.get("manifest_sha256")
        == "b45a283af8d55a292d1f211bb389fe945581086c0c45c67437563f601438af86",
        "Collected artifact manifest changed",
    )
    updates = worker.get("profile", {}).get("updates", {})
    require(
        updates.get("optimizer_completed_updates") == 40
        and updates.get("amp_skipped_updates") == 0,
        "B16 profile evidence is incomplete",
    )
    require(
        updates.get("loss_finite") is True and updates.get("gradient_finite") is True,
        "B16 profile finite-update evidence changed",
    )
    precision = worker.get("precision_audit", {})
    require(
        precision.get("nvidia_tf32_override") == "0", "Worker did not record NVIDIA_TF32_OVERRIDE=0"
    )
    flags = precision.get("flags_after_profile", {})
    require(set(flags) == {"matmul_tf32", "cudnn_tf32"}, "Profile TF32 flag receipt is incomplete")
    require(
        all(value is False for value in flags.values()),
        "Profile TF32 flags were not disabled",
    )
    identity = worker.get("input_identity", {})
    require(
        identity.get("original_manifest_sha256") == ORIGINAL_SHA,
        "Worker canonical manifest identity changed",
    )
    require(
        identity.get("derived_manifest_sha256") == DERIVED_SHA,
        "Worker execution manifest identity changed",
    )
    require(
        identity.get("registry_sha256") == REGISTRY_SHA, "Worker portable registry identity changed"
    )
    require(identity.get("proposal_sha256") == PROPOSAL_SHA, "Worker proposal identity changed")
    require(
        identity.get("cache_files")
        == {
            "cache/chain_folds.npz": {
                "bytes": 550_072,
                "sha256": "a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f",
            },
            "cache/chains.npz": {
                "bytes": 650_142,
                "sha256": "39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766",
            },
        },
        "Worker structural-cache identities changed",
    )
    return worker


def _verify_source_commit(project_root: Path, source_files: dict[str, str]) -> None:
    """Compare every trainer implementation digest with the requested clean Git revision."""
    for relative, expected in sorted(source_files.items()):
        result = subprocess.run(
            ["git", "show", f"{SOURCE_COMMIT}:{relative}"],
            cwd=project_root,
            check=True,
            capture_output=True,
        )
        actual = hashlib.sha256(result.stdout).hexdigest()
        require(actual == expected, f"Pinned trainer source differs at {relative}")


def validate_inference_source(source_root: Path) -> JsonObject:
    """Require the unmodified frozen producer checkout and lock for prediction exports."""
    resolved = source_root.expanduser().resolve(strict=True)
    result = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    git_root = Path(result.stdout.strip()).resolve(strict=True)
    require(git_root == resolved, "Inference source path must be the checkout root")
    revision = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(revision == SOURCE_COMMIT, "Inference checkout is not the frozen trainer revision")
    status = subprocess.run(
        ["git", "-C", str(resolved), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(not status, "Inference checkout has tracked or untracked changes")
    project = verify_pinned(
        resolved / "pyproject.toml", SOURCE_PROJECT_SHA, "frozen dependency declaration"
    )
    lock = verify_pinned(resolved / "uv.lock", SOURCE_LOCK_SHA, "frozen dependency lock")
    predictor = verify_pinned(
        resolved / "scripts/predict_run.py", PREDICTOR_SHA, "frozen prediction entrypoint"
    )
    trainer = verify_pinned(
        resolved / "scripts/train_rnn.py",
        TRAINER_ENTRYPOINT_SHA,
        "frozen model and prediction helpers",
    )
    return {
        "source_revision": revision,
        "working_tree_clean": True,
        "project_manifest_path": str(project),
        "project_manifest_sha256": SOURCE_PROJECT_SHA,
        "dependency_lock_path": str(lock),
        "dependency_lock_sha256": SOURCE_LOCK_SHA,
        "predictor_path": str(predictor),
        "predictor_sha256": PREDICTOR_SHA,
        "trainer_entrypoint_path": str(trainer),
        "trainer_entrypoint_sha256": TRAINER_ENTRYPOINT_SHA,
    }


def safe_recovery_root(path: Path) -> Path:
    """Require recovery products to live in the approved physical-disk scratch."""
    resolved = path.expanduser().resolve(strict=False)
    scratch = SCRATCH_ROOT.resolve(strict=True)
    require(resolved.is_relative_to(scratch), "Recovery output must be under physical research_tmp")
    return resolved

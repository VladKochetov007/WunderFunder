from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import cast

PLAN_RELATIVE = Path(
    "research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/"
    "STRONG_PARENT_WARM_I1_PLAN_R1.json"
)
PLAN_SHA256 = "eeec65c1982cda52cbc5260643d9d58bc39bdee264543a85ca6170682f7bc1e1"
PROFILE_RECIPE_RELATIVE = Path(
    "research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/"
    "PROFILE_F0_WARM_R2.json"
)
PROFILE_RECIPE_SHA256 = "ba13ead87fdad5c97d3ea12e7843a733342a7e05630e5fa60548a0995dc71d34"
AMENDMENT_RELATIVE = Path(
    "research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/"
    "PROFILE_F1_SOURCE_AMENDMENT_R4.json"
)
AMENDMENT_SHA256 = "bfec8f4c915495be238820a754cfae851344eb9e918475950ded244689db3008"
SOURCE_ROOT = Path("/home/vlad/development/wunder/research_tmp/i1-warm-source-repaired-20261022")
SOURCE_COMMIT = "81f47a3fd94e4209e47b1f3815af2907e8f0b5d7"
ADDITION_SHA256 = "1c9eed9537e2fbc10173860dbd19ba99d722ad85995db3d7c2e2cf579ba16136"
PROFILE_ARGV_SHA256 = "242364de6fd83db8d174c02890ef0796c853829d7df0e77b0e424fd9d2bfd330"
PRIOR_ADMISSION_RELATIVE = Path(
    "research/campaign_2026-10-21/next_cycle_r1/WARM_F1_PROFILE_ADMISSION_R3.json"
)
PRIOR_ADMISSION_SHA256 = "b37cdb521938de081f4a6238f55bf89669e2815a4fdfe52ce04acacd4b8419a5"
PRIOR_EXECUTION_RELATIVE = Path(
    "research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/"
    "PROFILE_F1_LOCAL_EXECUTION_R3.json"
)
PRIOR_EXECUTION_SHA256 = "5df66a2543355034036c920a91ef184a4ad8063b4da51c2167d9c3c03dbe478e"


def _sha256(path: Path) -> str:
    """Hash a bounded metadata/source file using fixed-size reads."""
    if path.stat().st_size > 4_000_000:
        raise ValueError(f"source amendment input exceeds the 4 MB bound: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    """Read a small JSON object and reject a changing file."""
    size = path.stat().st_size
    if size <= 0 or size > 200_000:
        raise ValueError(f"source amendment JSON has unsupported size: {size}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError("source amendment JSON changed while it was read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("source amendment must be a JSON object")
    return cast(dict[str, object], value)


def _verify_clean_source(source_root: Path) -> None:
    """Require the exact committed and clean frozen source checkout."""
    head = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    if head != SOURCE_COMMIT:
        raise ValueError(f"repaired source HEAD differs from the amendment: {head}")
    if status:
        raise ValueError("repaired source checkout is not clean")


def _plan_profile(plan: dict[str, object]) -> dict[str, object]:
    """Return the frozen F1 profile entry from the immutable plan."""
    profiles = cast(list[dict[str, object]], plan["profiles"])
    matches = [profile for profile in profiles if profile.get("fold") == 1]
    if len(matches) != 1:
        raise ValueError("frozen plan must contain exactly one F1 profile")
    return matches[0]


def verify_f1_source_amendment(
    amendment_path: Path,
    source_sha: str | None,
    source_root: Path,
    project_root: Path,
    plan_path: Path,
    plan: dict[str, object],
) -> dict[str, object]:
    """Verify the F1-only repaired source amendment before canonical imports."""
    amendment_path = amendment_path.resolve()
    source_root = source_root.resolve()
    project_root = project_root.resolve()
    if amendment_path != (project_root / AMENDMENT_RELATIVE).resolve():
        raise ValueError("F1 amendment path differs from the frozen R4 artifact")
    if source_root != SOURCE_ROOT or source_sha != SOURCE_COMMIT:
        raise ValueError("F1 command must select the exact repaired frozen source")
    if plan_path.resolve() != (project_root / PLAN_RELATIVE).resolve():
        raise ValueError("F1 command does not use the frozen warm-history plan path")
    if _sha256(plan_path) != PLAN_SHA256:
        raise ValueError("frozen warm-history plan bytes differ from R1")
    if _sha256(amendment_path) != AMENDMENT_SHA256:
        raise ValueError("F1 source-amendment bytes differ from the prepared R4 artifact")

    amendment = _read_object(amendment_path)
    if amendment.get("schema_version") != 1:
        raise ValueError("unsupported F1 source-amendment schema")
    if amendment.get("amendment_id") != "WARM_F1_PROFILE_SOURCE_AMENDMENT_R4":
        raise ValueError("F1 source-amendment identity differs from R4")
    if amendment.get("profile_id") != "i1warm_profile_f1_seed4_r2":
        raise ValueError("F1 profile identity differs from the frozen plan")

    base_plan = cast(dict[str, object], amendment["base_plan"])
    if base_plan != {"path": str(PLAN_RELATIVE), "sha256": PLAN_SHA256}:
        raise ValueError("amendment base plan does not bind the immutable R1 plan")
    base_recipe = cast(dict[str, object], amendment["base_profile_recipe"])
    expected_recipe = {
        "path": str(PROFILE_RECIPE_RELATIVE),
        "sha256": PROFILE_RECIPE_SHA256,
    }
    recipe_path = project_root / PROFILE_RECIPE_RELATIVE
    if base_recipe != expected_recipe or _sha256(recipe_path) != PROFILE_RECIPE_SHA256:
        raise ValueError("amendment base profile recipe differs from the frozen R2 recipe")

    source = cast(dict[str, object], amendment["source"])
    additions = {"src/wnn/distillation_cache.py": ADDITION_SHA256}
    expected_source = {
        "root": str(SOURCE_ROOT),
        "commit": SOURCE_COMMIT,
        "required_additions_sha256": additions,
        "only_commit_path_changed_from_prior_source": "src/wnn/distillation_cache.py",
        "prior_source_root": "/home/vlad/development/wunder/research_tmp/i1-warm-source-frozen-20261022",
        "prior_source_commit": "66959b920ba25de1d2f2911d8c5508d22d3eb4d0",
        "prior_cache_reader_sha256": "008fd6615ac5b627273f419072933cb95dc85fcd6c30eaffd6f1fb2df53ab0eb",
        "scope": (
            "Only the bounded distillation-cache metadata reader cap changes from "
            "512000 to 1048576 bytes; all 41 frozen source pins and the R1 training "
            "argv remain unchanged."
        ),
    }
    if source != expected_source:
        raise ValueError("amendment does not bind the one approved F1 source repair")
    added_source = source_root / "src/wnn/distillation_cache.py"
    if _sha256(added_source) != ADDITION_SHA256:
        raise ValueError("repaired distillation-cache source bytes differ from the amendment")
    _verify_clean_source(source_root)

    prior = cast(dict[str, object], amendment["prior_attempt"])
    expected_prior = {
        "admission_path": str(PRIOR_ADMISSION_RELATIVE),
        "admission_sha256": PRIOR_ADMISSION_SHA256,
        "execution_path": str(PRIOR_EXECUTION_RELATIVE),
        "execution_sha256": PRIOR_EXECUTION_SHA256,
        "log_path": "research/campaign_2026-10-21/next_cycle_r1/warm_f1_profile_r3.log",
        "log_sha256": "568e41a74ee2f43d889e0ee44833b57755f5c6150a852e516ba674d2c6b1ae37",
        "source_commit": "66959b920ba25de1d2f2911d8c5508d22d3eb4d0",
        "source_root": "/home/vlad/development/wunder/research_tmp/i1-warm-source-frozen-20261022",
        "adapter_sha256": "5625ce22b867812c0103b81963ea0531675e21327330d0a21d162bdf17dc0685",
        "status": "failed",
        "wall_seconds": 5.879722407000372,
        "failure": (
            "metadata read bound rejected the pinned 842417-byte F1 metadata.json "
            "at the old 512000-byte cap before training updates"
        ),
        "metadata_path": "cache/h230_four_family_fold1_source_20261009_r1/metadata.json",
        "metadata_bytes": 842417,
        "metadata_sha256": "8bd3854d867c47eacbeed56528ea9712f7598cc7eb21a46f5c85e7a0f2fba703",
        "training_updates": 0,
    }
    if prior != expected_prior:
        raise ValueError("R3 failed-attempt record differs from the preserved execution evidence")
    for relative, expected in (
        (PRIOR_ADMISSION_RELATIVE, PRIOR_ADMISSION_SHA256),
        (PRIOR_EXECUTION_RELATIVE, PRIOR_EXECUTION_SHA256),
        (
            Path(str(prior["log_path"])),
            "568e41a74ee2f43d889e0ee44833b57755f5c6150a852e516ba674d2c6b1ae37",
        ),
    ):
        if _sha256(project_root / relative) != expected:
            raise ValueError(f"preserved R3 attempt bytes differ: {relative}")

    profile = cast(dict[str, object], amendment["profile"])
    plan_profile = _plan_profile(plan)
    argv = cast(list[str], plan_profile["argv"])
    argv_hash = hashlib.sha256(
        json.dumps(argv[1:], separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    expected_profile = {
        "id": "i1warm_profile_f1_seed4_r2",
        "fold": 1,
        "profile_steps": 100,
        "profile_warmup": 10,
        "unchanged_argv_sha256": PROFILE_ARGV_SHA256,
        "argv_hash_encoding": (
            "SHA-256 of ASCII bytes from json.dumps(plan profile argv[1:], "
            "separators=(',', ':'), ensure_ascii=True); only the training script "
            "path at argv[0] is replaced by the repaired source root."
        ),
        "expected_optimizer_updates": 1100,
        "scientific_fit_performed": False,
    }
    if profile != expected_profile or profile.get("id") != plan_profile.get("profile_id"):
        raise ValueError("R4 profile fields differ from the frozen F1 profile")
    if argv_hash != PROFILE_ARGV_SHA256:
        raise ValueError("frozen F1 profile argv tail no longer matches R4")

    preserved_f0 = cast(dict[str, object], amendment["preserved_f0"])
    expected_f0 = {
        "execution_path": (
            "research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/"
            "PROFILE_F0_LOCAL_EXECUTION_R2.json"
        ),
        "execution_sha256": "b25793c46f4d524a43066db4e80e2cdd3d522baa1e7fb92dcc1b0cd659492cf0",
        "log_path": "research/campaign_2026-10-21/next_cycle_r1/warm_f0_profile_r2.log",
        "log_sha256": "c4447fa4826651ed3c1299b4beeac9b30c8d292bdbf6dfc1ebb014f0461e6f52",
        "source_commit": "66959b920ba25de1d2f2911d8c5508d22d3eb4d0",
        "status": "complete",
        "wall_seconds": 304.76052887600963,
        "profile_batches": 110,
        "optimizer_updates": 1100,
        "scientific_fit_performed": False,
    }
    if preserved_f0 != expected_f0:
        raise ValueError("preserved F0 profile evidence differs from its frozen receipt")
    for key in ("execution", "log"):
        if _sha256(project_root / str(expected_f0[f"{key}_path"])) != expected_f0[f"{key}_sha256"]:
            raise ValueError(f"preserved F0 profile {key} bytes differ from R2")

    return {
        "path": str(amendment_path),
        "sha256": _sha256(amendment_path),
        "amendment_id": amendment["amendment_id"],
        "source_root": str(source_root),
        "source_commit": SOURCE_COMMIT,
        "required_additions_sha256": additions,
        "profile_id": profile["id"],
        "profile_argv_tail_sha256": argv_hash,
        "prior_attempt": expected_prior,
        "preserved_f0": expected_f0,
    }

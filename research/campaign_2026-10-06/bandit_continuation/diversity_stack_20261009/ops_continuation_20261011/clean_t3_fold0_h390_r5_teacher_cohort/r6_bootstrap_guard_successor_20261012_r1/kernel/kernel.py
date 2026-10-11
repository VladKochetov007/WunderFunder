from __future__ import annotations

import hashlib
import math
import json
import os
import signal
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

BUNDLE_SLUG = 'wnn-clean-t3-f0-teachers-20261011-r5'
EXPECTED_ARCHIVE_SHA256 = "88641b3c9fc629a7a2ce8b11cfd98ecf8b53fe943ffb2d7f632e1dfb997fa549"
EXPECTED_EXTERNAL_MOUNTS = {'artifacts/runs/c20260929_prop_f0/model.pt': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7.wnn'}, 'cache/chain_folds.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f.wnn'}, 'cache/chains.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766.wnn'}, 'cache/propcache_c20260929_prop_f0_train.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'b90e502506c4c93903057f5e8f15485f8e3af9148024055ecf77110f1f0155f9.wnn'}, 'cache/propcache_c20260929_prop_f0_train.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '811ce4328be739b039bc62ba1f696386e59b168fbe3733be81783a41417da174.wnn'}, 'cache/propcache_c20260929_prop_f0_valid.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '775ee227b15d0b5510c9eeee5012b23d674fed227a343e4581344568eacfb82a.wnn'}, 'cache/propcache_c20260929_prop_f0_valid.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '9f3e09f04b9fadfd142d4ecbaba8b4d17d6d1ae04ea40e2cb8d4b64ecd0a718c.wnn'}, 'cache/train_targets.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9.wnn'}, 'cache/valid_targets.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955.wnn'}, 'datasets/train.parquet': 'train.parquet', 'datasets/valid.parquet': 'valid.parquet', 'research/campaign_2026-09-29/artifacts/splits/fold0.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d.wnn'}, 'research/campaign_2026-10-04/portable_inputs.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c.wnn'}}
KERNEL_PROCESS_WALL_SECONDS_MAX = 14_400
RUNNER_FINALIZATION_RESERVE_SECONDS = 300
BOOTSTRAP_STARTED_MONOTONIC = time.monotonic()
R3_GATE_RELATIVE_PATH = 'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort/CONDITIONAL_FIT_ADMISSION_R3.json'
R5_GATE_RELATIVE_PATH = 'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r5_teacher_cohort/CONDITIONAL_FIT_ADMISSION_R5.json'
R5_GATE_SHA256 = "dac831855956c5b889af48e0bb64a48aec77aea66c51905c1410fce617de4d43"
R5_ROOT_RELEASE_RECEIPT_RELATIVE_PATH = 'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r5_teacher_cohort/ROOT_CONDITIONAL_RELEASE_R5.json'
R5_ROOT_RELEASE_RECEIPT_SHA256 = "b93ac7d52890949d0a6f1e07574be36f84c872be9a1b975f33b461cbef4f4a0f"
R5_PRESEAL_BINDINGS = {'schema': 'wnn.kaggle.teacher_cohort_r5_preseal_bindings.r1', 'r4_dataset': 'vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r4', 'r4_kernel_id': 'vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r4-runner', 'r4_archive_sha256': 'ae8796a9c0cca1ed64d45341c14c917fc19f6889a1df20ff1873a87a200570dc', 'r4_bundle_manifest_sha256': '5f3f0aa8f3759a5ff594c9414c04ad6c4c8834a22f156482af18b411388dcaf7', 'r4_terminal_receipt_sha256': '833570ab8908a119811bed53a11477ba515db511eb65d55f08de659be61afac0', 'r4_diagnostics_sha256': '121a317b9bd75117939af3901116e496a3466a6d477c6d5095f2be1ad8734ad8', 'r4_actual_failure_fixture_sha256': 'f3770c483ee72c2a0b0fa6f751b2e4072f9d40bcd23d934ac06019230841ae1a', 'r3_teacher_plan_sha256': 'aee76572ef4d2ba50b5d27b91aa9a8bd9ea7056702aace87bd936d535442ccd9', 'r4_failure_classification': {'provider_terminal': 'ERROR', 'paired_profiles_completed': True, 'teacher_fits_started': False, 'error': 'ValueError: Worker reservation exceeds host operational budget', 'exact_post_profile_memory_inventory': 'not persisted', 'file_cache_cause': 'unproven hypothesis', 'inference_only': 'given the recorded 32,212,254,720-byte capacity remained unchanged, the runner error implies budget <8,000,000,000B and therefore available <14,442,450,944B; exact available bytes were not captured'}, 'source_manifest_sha256': '41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa', 'source_binding_sha256': '1895d15df875763921f9bf0ccfbf671a07029261fe7ffaad125275e23708d2e4', 'input_binding_sha256': '2d74435d33df9c24f5c7d5656ee82ff2bbbe0ede095a8c6d6f82f8aa9234ef48', 'producer_preflight_sha256': 'fc16040084e041a8e458f80463f9096bcd9a3c946928580b55aa4468cde35afc', 'profile_guard_contract_sha256': '401ad91a7587894803a6701bf91130da134e290bcd0ebe1f9077b5107622c71d', 'profile_reference_sha256': 'bed34488551ce7798330dcffef8f7f7b9f23fd13419b7d1fa89dade7f25e2790', 'resource_handoff_contract_sha256': '060bfb8b3ad38822ee32c2c148b8ce82b244ade83209b2e9efa0e344d2e6afe5', 'r5_driver_sha256': 'e931157aa1c8be25180d4599fdc51cbb847a77f9c588d5e9245ba0a6255554e1', 'worker_specs_sha256': {'profile_clean_t3_f0_muonall_h384_e3_s3_paired_t4_r5': '717632bba22eb74e0f84ca5b663245b853e7fc7bbbcce5cb9669e62593bcd73f', 'profile_clean_t3_f0_muonall_h384_e3_s4_paired_t4_r5': '32288c8b006df23220621e477dc5dd53330d39b2d2036aa913ff0be263951f80', 'fit_clean_t3_f0_muonall_h384_e3_s3_r5': '82ee17b9b9e6b616c6e713d719a5485cd7a6859b5121d3c6710dcf3b57f923b2', 'fit_clean_t3_f0_muonall_h384_e3_s4_r5': '5e45cb354c215da0a080a7fd85ce6e8b29c6813b9838d6de26986d4e4b35cca8', 'fit_clean_t3_f0_muonall_h384_e3_s5_r5': '834d9a704ef600bcaf4b7554f6e5c5cc27504df97c10490d58d6cb6db1bd3de1', 'export_clean_t3_f0_muonall_h384_e3_s3_r5_fit_train': '6ee3ba26cd3c02dac88fd9512a47e030d14c37e0dc729b2090d8f123aca906ea', 'export_clean_t3_f0_muonall_h384_e3_s3_r5_fit_valid': '095b366bf910defc43d21c8361e3e0c6de32a7125ea3835c2eda1cc14c510be0', 'export_clean_t3_f0_muonall_h384_e3_s3_r5_full_valid': 'cd310dc9c39fb721d2b0f2ab716e83db98a07b5cfd9092f570d07e4b6ec894cd', 'export_clean_t3_f0_muonall_h384_e3_s4_r5_fit_train': 'e24837edf22a282ce9d02832230a6d3a42b0d6904f3f61f7ec66856811b7faa7', 'export_clean_t3_f0_muonall_h384_e3_s4_r5_fit_valid': 'ce535fafa038aca12f8687b63ad3f9bd906a324915b655505161e49430661f2f', 'export_clean_t3_f0_muonall_h384_e3_s4_r5_full_valid': '771e082a83db6fc3d3383795cbcaca13d67f0a7ecffc3a61b25e518c534c5f7d', 'export_clean_t3_f0_muonall_h384_e3_s5_r5_fit_train': '7291ca7a07f9e1b448f409465ede062ae807fcd9145bd67b7f813847cfc81fba', 'export_clean_t3_f0_muonall_h384_e3_s5_r5_fit_valid': '548572ddd3316c95971e1dfac5d553b30c1b1439f207acc81431a9f5966c8613', 'export_clean_t3_f0_muonall_h384_e3_s5_r5_full_valid': '5bbe882ddd767e68032ac50849fbfe6e06669ddbd4115bb0b800b87aabe03ed8'}, 'science_recipe_sha256': 'ef18ec6064252cd13eef021ee644d528d840e7718511e8514901931a57adb17c', 'fit_worker_ram_reservation_bytes': 8000000000, 'allowed_delta': ['embed_exact_root_conditional_release_receipt_r5', 'authorize_only_the_existing_profile_gated_teacher_fit_branch', 'rebind_bootstrap_constants_for_r5_slug_archive_gate_and_release_receipt', 'add_host_and_cgroup_memory_snapshots_around_the_existing_fit_admission', 'advise_dontneed_only_for_hash_verified_declared_raw_and_own_cache_inputs_after_profiles_exit', 'rebind_worker_attempt_names_and_package_identity_without_changing_science_or_resource_limits']}
R5_RELEASE_TARGET = {'dataset': 'vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5', 'kernel_id': 'vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5-runner'}
R5_ALLOWED_DELTA = ['embed_exact_root_conditional_release_receipt_r5', 'authorize_only_the_existing_profile_gated_teacher_fit_branch', 'rebind_bootstrap_constants_for_r5_slug_archive_gate_and_release_receipt', 'add_host_and_cgroup_memory_snapshots_around_the_existing_fit_admission', 'advise_dontneed_only_for_hash_verified_declared_raw_and_own_cache_inputs_after_profiles_exit', 'rebind_worker_attempt_names_and_package_identity_without_changing_science_or_resource_limits']
R3_GATE_SHA256 = '55da771e35bc94a5222a493b097f1633ff8258ef6fe055ae05f6dcfe89780226'
ROOT_RELEASE_RECEIPT_RELATIVE_PATH = 'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort/ROOT_CONDITIONAL_RELEASE_R4.json'
ROOT_RELEASE_RECEIPT_SHA256 = 'bf79983eaf3548c8a3544139b94ad5c9f29baabecfced150b28cf4c27fe4f0ac'
ROOT_PRESEAL_R3_BINDINGS = {'r3_packet_relative_path': 'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort', 'r3_dataset': 'vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r3', 'r3_archive_sha256': 'f6fc9de3b47289bbe7e95dc5cdf446ddd2d4b528ceb6e09cc1437a3f5ce7d2fd', 'r3_bundle_manifest_sha256': '11430722c1f5591e1c00d09b6aeff4c556212d63bb6ccc973dadd0a5813a00a0', 'r3_bootstrap_sha256': 'ca085e79029794b0de92a32780d2a96f215c3c6c724f6164db5bd2e6acb654fa', 'r3_gate_sha256': '14644f9feeb9f248dedae2ee2884e68fa394fb9da5aeaf5a23d16aa6b9f12df7', 'r3_source_manifest_sha256': '41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa', 'r3_source_binding_sha256': '1895d15df875763921f9bf0ccfbf671a07029261fe7ffaad125275e23708d2e4', 'r3_input_binding_sha256': '2d74435d33df9c24f5c7d5656ee82ff2bbbe0ede095a8c6d6f82f8aa9234ef48', 'r3_producer_preflight_sha256': 'fc16040084e041a8e458f80463f9096bcd9a3c946928580b55aa4468cde35afc', 'r3_profile_guard_contract_sha256': '401ad91a7587894803a6701bf91130da134e290bcd0ebe1f9077b5107622c71d', 'r3_profile_reference_sha256': 'bed34488551ce7798330dcffef8f7f7b9f23fd13419b7d1fa89dade7f25e2790', 'r3_plan_sha256': 'aee76572ef4d2ba50b5d27b91aa9a8bd9ea7056702aace87bd936d535442ccd9', 'r3_collection_contract_sha256': '5d0b9347b71d9c817fb7870d41ffea0abc5b3a47b9e038decf406e21e65e4b6b', 'r3_source_delta_sha256': '912205eba101c8539c372206e5ae0e56f2d0a39f8997b271189b3d2f7d601406', 'r3_static_validation_sha256': '6ed41d0e0b434c974d1be97aa0ffec5056088fb60d2c19f2a5894b7bdf8b53ef', 'r3_root_release_gates_sha256': '3839daa57e369bd73848fcd6a9eca2258844fa4f699e5c0701bf14dee20ee8f6', 'r3_kernel_configuration_sha256': '2b54e3259b2eba3bc49b2e6cd0edcdeaf7154dbc483949dc08602f07a54273ef', 'r3_driver_sha256': '08eb2f8e4aab6857dfec12a03a739825cf238819234c4f119494976c9cccf8af', 'r3_job_specs_sha256': {'export_clean_t3_f0_muonall_h384_e3_s3_r3_fit_train.json': '605a75aa95919d7f6e4386791a93b0a0d65046e90bacbb63256e480893d5b2af', 'export_clean_t3_f0_muonall_h384_e3_s3_r3_fit_valid.json': '5d061e864fa626ee989f6c7139b0c3f9d80228193bff6d04a46263c66c5ebe9a', 'export_clean_t3_f0_muonall_h384_e3_s3_r3_full_valid.json': '89479307b87dcb85dc18059c0e92e53cd7885dc298655e9fb11ee9967ab6712f', 'export_clean_t3_f0_muonall_h384_e3_s4_r3_fit_train.json': '12d00e843ddd3c1ac535bb4ba9685a526c34cbd93721029f351abe8bfec314a8', 'export_clean_t3_f0_muonall_h384_e3_s4_r3_fit_valid.json': '0b6e829abe478f6248ee981620c815ff409988e176dfc5e3567ae89122604de2', 'export_clean_t3_f0_muonall_h384_e3_s4_r3_full_valid.json': '3e5c5b4c127b4e2777055e5d0f33e1f3a5ff3fc99078a551a5a97941dafa9d35', 'export_clean_t3_f0_muonall_h384_e3_s5_r3_fit_train.json': 'd8f3e4f95cc1e3523d5af1f3bc02a606244891b80681b4ce2f7017c2cb517535', 'export_clean_t3_f0_muonall_h384_e3_s5_r3_fit_valid.json': '8f4727e2ba09c6949fbc68e4f8ad566edae8fe2ca032ca564669991605b9360b', 'export_clean_t3_f0_muonall_h384_e3_s5_r3_full_valid.json': '5fadc9051d4b99be52e8715d9112ef45ee43e1a16c77495ab7c9df62c3550d2c', 'fit_clean_t3_f0_muonall_h384_e3_s3_r3.json': '529e8eef7de3ef7fe380eadd92f83a06ac1382c793dc374effff229fb520fed1', 'fit_clean_t3_f0_muonall_h384_e3_s4_r3.json': '25d6158597344c9751b6155d858da26eba468e22cbeece2fa91900b02dc66173', 'fit_clean_t3_f0_muonall_h384_e3_s5_r3.json': 'c0537a55ce234d229782c986fd8afdeb533aa5e0dfafe0472a42a4550152370c', 'profile_clean_t3_f0_muonall_h384_e3_s3_paired_t4_r3.json': '433331bc24bfd585b8bbe999afa2bff48ec726c0a0ff57a77c09992e995e7f64', 'profile_clean_t3_f0_muonall_h384_e3_s4_paired_t4_r3.json': 'fe826246c923a92e5ab34d6d2a58e8c482e2e652b77dd5460969727de4ce77d5'}, 'r3_science_recipe_sha256': '2319ffd40d1dfe1ccbdd28241e3acb6977cbc4a43dd480243d1c5630d7c85eda'}
ROOT_ALLOWED_DELTA = ['embed_exact_root_conditional_release_receipt_r4', 'authorize_only_the_existing_r3_profile_gated_fit_branch', 'rebind_bootstrap_constants_for_r4_slug_archive_gate_and_release_receipt', 'reseal_archive_manifest_and_job_gate_descriptors_without_science_changes', 'rebind_dataset_and_kernel_metadata_to_r4_slug']
ROOT_RELEASE_TARGET = {'dataset': 'vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r4', 'kernel_id': 'vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r4-runner'}
def _require_finite_quota_headroom(value: object, field_name: str) -> None:
    if type(value) is int:
        valid = value >= 28800
    elif type(value) is float:
        valid = math.isfinite(value) and value >= 28800
    else:
        valid = False
    if not valid:
        raise PermissionError(f'R4 root receipt has invalid {field_name}')
def _require_fixed_quota_integer(value: object, expected: int, field_name: str) -> None:
    if type(value) is not int or value != expected:
        raise PermissionError(f'R4 root receipt has invalid {field_name}')


def remaining_seconds(deadline_monotonic: float, *, now_monotonic: float | None = None) -> float:
    """Return remaining kernel-process budget from the bootstrap-entry deadline."""
    now = time.monotonic() if now_monotonic is None else now_monotonic
    return max(0.0, deadline_monotonic - now)


def require_remaining(deadline_monotonic: float, phase: str) -> None:
    """Fail closed when synchronous bootstrap work has exhausted its deadline."""
    if remaining_seconds(deadline_monotonic) <= 0:
        raise TimeoutError(f"kernel bootstrap exceeded its deadline during {phase}")


def worker_timeout_for_deadline(
    declared_seconds: int,
    deadline_monotonic: float,
    *,
    now_monotonic: float | None = None,
) -> int:
    """Shrink a worker cap to leave time for runner artifact collection and exit."""
    now = time.monotonic() if now_monotonic is None else now_monotonic
    available = int(
        deadline_monotonic - now - RUNNER_FINALIZATION_RESERVE_SECONDS
    )
    return max(0, min(int(declared_seconds), available))


def terminate_process_tree(process: subprocess.Popen, *, grace_seconds: float = 2.0) -> None:
    """Stop only the runner and its descendants, including workers in new sessions."""
    try:
        import psutil
    except ImportError as error:  # the bundled runner requires psutil as well
        raise RuntimeError("cannot safely enforce kernel deadline without psutil") from error

    try:
        root = psutil.Process(process.pid)
        members = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        process.wait()
        return

    # The runner creates a separate session for each admitted worker. Kill the
    # process tree by identity, not the bootstrap's process group alone.
    for child in members[1:]:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        root.terminate()
    except psutil.NoSuchProcess:
        pass
    _, alive = psutil.wait_procs(members, timeout=grace_seconds)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=grace_seconds)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        # The Popen itself is also in our process tree; this is a final bounded
        # fallback if the process changed state during psutil's snapshot.
        try:
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=grace_seconds)


def write_runtime_spec(source: Path, destination: Path, deadline_monotonic: float) -> int:
    """Copy a sealed recipe spec with a runtime timeout reduced to current headroom."""
    spec = json.loads(source.read_text(encoding="utf-8"))
    limits = spec.get("limits")
    if not isinstance(limits, dict) or not isinstance(limits.get("timeout_seconds"), int):
        raise ValueError("kernel spec has no integer worker timeout")
    worker_seconds = worker_timeout_for_deadline(
        limits["timeout_seconds"], deadline_monotonic
    )
    if worker_seconds < 1:
        raise TimeoutError("kernel deadline leaves no worker time after finalization reserve")
    limits["timeout_seconds"] = worker_seconds
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return worker_seconds


def release_mounts(root: Path, mounts: dict) -> None:
    """Unlink only external input symlinks before Kaggle captures working outputs."""
    for logical in mounts:
        target = root / logical
        if target.is_symlink():
            target.unlink()


def main() -> None:
    """Extract a verified private bundle and mount hashed raw data without copies."""
    kernel_started = BOOTSTRAP_STARTED_MONOTONIC
    kernel_deadline = kernel_started + KERNEL_PROCESS_WALL_SECONDS_MAX
    require_remaining(kernel_deadline, "input discovery")
    matches = list(Path("/kaggle/input").glob(f"**/{BUNDLE_SLUG}/bundle-manifest.json"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one named private bundle")
    manifest = json.loads(matches[0].read_text())
    if manifest.get("private") is not True:
        raise ValueError("Bundle is not marked private")
    if manifest.get("external_mounts") != EXPECTED_EXTERNAL_MOUNTS:
        raise ValueError("External mappings differ from the controller-bound kernel")
    require_remaining(kernel_deadline, "bundle metadata verification")
    archive = matches[0].parent / "bundle.wnn"
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    if (
        digest.hexdigest() != manifest["archive_sha256"]
        or digest.hexdigest() != EXPECTED_ARCHIVE_SHA256
    ):
        raise ValueError("Private bundle archive identity mismatch")
    require_remaining(kernel_deadline, "bundle archive verification")
    root = Path("/kaggle/working/experiment")
    if shutil.disk_usage(root.parent).free < manifest["expanded_bytes"] + 2 * 1024**3:
        raise ValueError("Insufficient working disk to extract bundle with reserve")
    root.mkdir(exist_ok=False)
    allowed = {item["path"] for item in manifest["files"]} | set(manifest["spec_paths"])
    external_allowed = set(EXPECTED_EXTERNAL_MOUNTS)
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            require_remaining(kernel_deadline, "bundle extraction")
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("Archive path escapes experiment directory")
            if not member.isfile() or member.name not in allowed or member.name in seen:
                raise ValueError("Unsupported, undeclared or duplicate archive member")
            seen.add(member.name)
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("Archive member has no file content")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, 8 * 1024**2)
    require_remaining(kernel_deadline, "bundle extraction")
    gate_file = root / R3_GATE_RELATIVE_PATH
    if not gate_file.is_file() or hashlib.sha256(gate_file.read_bytes()).hexdigest() != R3_GATE_SHA256:
        raise ValueError("R3 conditional admission receipt is missing or changed")
    gate_receipt = json.loads(gate_file.read_text())
    if gate_receipt.get("status") != "admitted" or gate_receipt.get("release_authorized") is not True or not gate_receipt.get("root_release_receipt_sha256"):
        raise PermissionError("R3 conditional admission is pending; refusing before external mounts or workers")
    if gate_receipt.get("root_release_receipt_sha256") != ROOT_RELEASE_RECEIPT_SHA256:
        raise PermissionError("R4 gate does not bind the embedded root release receipt")
    if gate_receipt.get("root_release_receipt_path") != ROOT_RELEASE_RECEIPT_RELATIVE_PATH:
        raise PermissionError("R4 gate names an unexpected root release receipt path")
    if gate_receipt.get("profile_guard_contract_sha256") != ROOT_PRESEAL_R3_BINDINGS["r3_profile_guard_contract_sha256"] or gate_receipt.get("profile_reference_sha256") != ROOT_PRESEAL_R3_BINDINGS["r3_profile_reference_sha256"]:
        raise PermissionError("R4 gate does not bind the exact profile guard and reference")
    root_receipt_path = root / ROOT_RELEASE_RECEIPT_RELATIVE_PATH
    if not root_receipt_path.is_file() or hashlib.sha256(root_receipt_path.read_bytes()).hexdigest() != ROOT_RELEASE_RECEIPT_SHA256:
        raise PermissionError("R4 root release receipt bytes are missing or changed")
    root_receipt = json.loads(root_receipt_path.read_text())
    if root_receipt.get("schema") != "wnn.root_conditional_teacher_cohort_release.r4" or root_receipt.get("status") != "admitted" or root_receipt.get("authorized") is not True:
        raise PermissionError("R4 root release receipt is not authorized")
    if root_receipt.get("binding", {}).get("preseal_r3") != ROOT_PRESEAL_R3_BINDINGS:
        raise PermissionError("R4 root receipt does not bind the exact R3 source and jobs")
    if root_receipt.get("target") != ROOT_RELEASE_TARGET:
        raise PermissionError("R4 root receipt does not bind the exact title-derived target")
    if root_receipt.get("allowed_delta") != ROOT_ALLOWED_DELTA:
        raise PermissionError("R4 root receipt allows an unexpected packet delta")
    root_binding = root_receipt.get("binding", {})
    prior_profile = root_binding.get("previous_profile_collection", {})
    if prior_profile.get("file_count") != 77 or not isinstance(prior_profile.get("sha256"), str) or len(prior_profile["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in prior_profile["sha256"]):
        raise PermissionError("R4 root receipt omits the prior 77-file profile collection")
    quota_budget = root_binding.get("quota_budget", {})
    _require_fixed_quota_integer(quota_budget.get("wall_ceiling_seconds"), 14400, "wall ceiling")
    _require_fixed_quota_integer(quota_budget.get("two_device_gpu_seconds_upper_bound"), 28800, "GPU upper bound")
    _require_fixed_quota_integer(quota_budget.get("reserved_gpu_seconds"), 28800, "GPU reservation")
    if quota_budget.get("paid_spending_authorized") is not False:
        raise PermissionError("R4 root receipt quota/budget fields differ from the sealed ceiling")
    _require_finite_quota_headroom(quota_budget.get("campaign_headroom_gpu_seconds"), "campaign GPU headroom")
    _require_finite_quota_headroom(quota_budget.get("provider_headroom_gpu_seconds"), "provider GPU headroom")
    r5_gate_path = root / R5_GATE_RELATIVE_PATH
    if not r5_gate_path.is_file() or hashlib.sha256(r5_gate_path.read_bytes()).hexdigest() != R5_GATE_SHA256:
        raise PermissionError("R5 conditional gate bytes are missing or changed")
    r5_gate = json.loads(r5_gate_path.read_text())
    if r5_gate.get("status") != "admitted" or r5_gate.get("release_authorized") is not True:
        raise PermissionError("R5 root conditional admission is pending; refusing before external mounts")
    if r5_gate.get("root_release_receipt_path") != R5_ROOT_RELEASE_RECEIPT_RELATIVE_PATH or r5_gate.get("root_release_receipt_sha256") != R5_ROOT_RELEASE_RECEIPT_SHA256:
        raise PermissionError("R5 gate does not bind the exact root receipt")
    r5_release_path = root / R5_ROOT_RELEASE_RECEIPT_RELATIVE_PATH
    if not r5_release_path.is_file() or hashlib.sha256(r5_release_path.read_bytes()).hexdigest() != R5_ROOT_RELEASE_RECEIPT_SHA256:
        raise PermissionError("R5 root release receipt bytes are missing or changed")
    r5_release = json.loads(r5_release_path.read_text())
    if r5_release.get("schema") != "wnn.root_conditional_teacher_cohort_release.r5" or r5_release.get("status") != "admitted" or r5_release.get("authorized") is not True:
        raise PermissionError("R5 root release receipt is not authorized")
    if r5_release.get("binding", {}).get("preseal_r5") != R5_PRESEAL_BINDINGS or r5_release.get("target") != R5_RELEASE_TARGET or r5_release.get("allowed_delta") != R5_ALLOWED_DELTA:
        raise PermissionError("R5 root receipt does not bind the exact source, guards, and target")
    quota = r5_release.get("binding", {}).get("quota_budget", {})
    _require_fixed_quota_integer(quota.get("wall_ceiling_seconds"), 14400, "R5 wall ceiling")
    _require_fixed_quota_integer(quota.get("two_device_gpu_seconds_upper_bound"), 28800, "R5 GPU upper bound")
    _require_fixed_quota_integer(quota.get("reserved_gpu_seconds"), 28800, "R5 GPU reservation")
    if quota.get("paid_spending_authorized") is not False:
        raise PermissionError("R5 release may not authorize paid spending")
    _require_finite_quota_headroom(quota.get("campaign_headroom_gpu_seconds"), "R5 campaign GPU headroom")
    _require_finite_quota_headroom(quota.get("provider_headroom_gpu_seconds"), "R5 provider GPU headroom")
    for logical, mapping in manifest["external_mounts"].items():
        require_remaining(kernel_deadline, "external input mounting")
        path = PurePosixPath(logical)
        if (
            not isinstance(logical, str)
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in logical
            or logical not in external_allowed
        ):
            raise ValueError("Unsafe external input mapping")
        dataset = (
            mapping["dataset"] if isinstance(mapping, dict) else "vladkochetov07/wunder-funder"
        )
        filename = mapping["filename"] if isinstance(mapping, dict) else mapping
        slug = dataset.split("/")[1]
        candidates = list(Path("/kaggle/input").glob(f"**/{slug}/{filename}"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one authorized input: {dataset}/{filename}")
        target = root / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(candidates[0])
    command = [sys.executable, str(root / 'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r5_teacher_cohort/kernel_support/teacher_cohort_r5_driver.py'), "--root", str(root)]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    environment = os.environ.copy()
    environment["WNN_KERNEL_DEADLINE_MONOTONIC"] = repr(kernel_deadline)
    environment["WNN_R5_GATE_SHA256"] = R5_GATE_SHA256
    environment["WNN_R5_ROOT_RELEASE_SHA256"] = R5_ROOT_RELEASE_RECEIPT_SHA256
    try:
        require_remaining(kernel_deadline, "teacher cohort driver launch")
        process = subprocess.Popen(command, cwd=root, env=environment, start_new_session=True)
        try:
            returncode = process.wait(timeout=remaining_seconds(kernel_deadline))
        except subprocess.TimeoutExpired as error:
            terminate_process_tree(process)
            raise TimeoutError("R3 profile-gated teacher cohort exceeded 14,400 seconds") from error
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        if remaining_seconds(kernel_deadline) <= 0:
            raise TimeoutError("R3 cohort completed outside the admitted wall ceiling")
    finally:
        release_mounts(root, manifest["external_mounts"])


if __name__ == "__main__":
    main()

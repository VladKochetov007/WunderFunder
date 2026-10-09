from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rolling_driver as driver  # noqa: E402
import seal_sampled_cohort as sealer  # noqa: E402


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def test_grid_keys_are_unique_and_cover_the_already_run_controls() -> None:
    assert set(driver.GRID) >= driver.ALREADY_RUN
    assert len(driver.GRID) == 4 * 2 * 14


def test_rolling_round_seals_boots_and_maps_every_arm(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(driver, "ROUNDS", tmp_path / "rounds")
    state = driver.start_round()
    assert state["tag"] == "r4" and len(state["arms"]) == driver.ARMS_PER_ROUND
    assert not set(state["arms"]) & driver.ALREADY_RUN
    admitted = tmp_path / "rounds/r4/sealed"
    admission = json.loads((tmp_path / "rounds/r4/root_admission.json").read_text())

    input_root = tmp_path / "input"
    _copy_tree(sealer.SOURCE_ASSET, input_root / "datasets" / sealer.SOURCE_DATASET)
    _copy_tree(admitted / "runner_overlay_asset", input_root / "datasets" / state["dataset"])
    monkeypatch.setenv("KAGGLE_INPUT", str(input_root))
    monkeypatch.setenv("KAGGLE_WORKING", str(tmp_path / "working"))
    monkeypatch.delenv("WNN_SAMPLED_COHORT_OFFLINE", raising=False)

    module_path = admitted / "kernel" / sealer.KERNEL_FILENAME
    spec = importlib.util.spec_from_file_location("rolling_bootstrap_fixture", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def boundary():
        raise RuntimeError("reached fold0 mount boundary after admission validation")

    module.discover_fold0_asset_manifest = boundary
    assert module.main() == 1
    failure = json.loads((tmp_path / "working/failure.json").read_text())
    assert failure["error"] == "reached fold0 mount boundary after admission validation"

    repo = tmp_path / "working/wnn"
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo / "src"))
    sys.modules.pop("loss_pair_gpu_worker", None)
    monkeypatch.setenv("WNN_SAMPLED_ROOT_ADMISSION_JSON", json.dumps(admission))
    worker_spec = importlib.util.spec_from_file_location("rolling_worker", repo / "scripts/sampled_recipe_worker.py")
    worker = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker)
    cohort = worker.load_spec()
    packet = json.loads((admitted / "sampled_cohort.packet.json").read_text())
    _, seconds = worker._validate_admission(
        cohort,
        {
            "runner_overlay_manifest_sha256": packet["source"]["runner_overlay_manifest_sha256"],
            "code_manifest_sha256": packet["source"]["code_manifest_sha256"],
        },
    )
    assert seconds == 3300
    for arm in cohort["arms"]:
        argv = worker.build_train_argv(arm, Path("m.json"), Path("v.json"), "0" * 64)
        assert argv[argv.index("--sched") + 1] == arm["scheduler"]
        assert argv[argv.index("--loss") + 1] == arm["loss"]
        if arm["loss"] == "huber":
            assert float(argv[argv.index("--huber-delta") + 1]) == arm["huber_delta"]
        if arm["scheduler"] != "onecycle":
            rows = worker._expected_nononecycle_rows(arm)
            assert len(rows) == 220


def test_fold1_round_installs_fold_files_and_rebinds_worker(tmp_path: Path, monkeypatch) -> None:
    root = driver.ROOT
    mf = "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/multifold_r1"
    split_rel = "research/campaign_2026-09-29/artifacts/splits/fold1.json"
    pins = {
        "SOURCE_KERNEL": "vladkochetov07/wnn-parsed-cache-stage-20261009-f1",
        "EXPECTED_SOURCE_STAGE_RECEIPT_SHA256": "1" * 64,
        "EXPECTED_PHASE_LOG_SHA256": "2" * 64,
        "EXPECTED_CACHE_MANIFEST_SHA256": "3" * 64,
        "EXPECTED_CACHE_KEY": "4" * 64,
        "EXPECTED_CPU_VALIDATION_SHA256": "5" * 64,
        "EXPECTED_CPU_PROFILE_SHA256": "6" * 64,
        "EXPECTED_SOURCE_VERIFICATION_SHA256": "7" * 64,
        "EXPECTED_SPLIT_SHA256": sealer.sha_file(root / split_rel),
        "EXPECTED_REGISTRY_SHA256": sealer.sha_file(root / mf / "portable_inputs_fold1.json"),
        "EXPECTED_MEMBERSHIP_SHA256": sealer.sha_file(root / mf / "membership_fold1_f03_s20261009.json"),
        "SPLIT_RELATIVE": split_rel,
        "REGISTRY_RELATIVE": f"{mf}/portable_inputs_fold1.json",
        "MEMBERSHIP_RELATIVE": f"{mf}/membership_fold1_f03_s20261009.json",
    }
    pin_dir = tmp_path / "fold_pins"
    pin_dir.mkdir()
    (pin_dir / "fold_1.json").write_text(json.dumps({"fold": 1, "pins": pins, "fit_train_count": 349, "evaluation_count": 457}))
    monkeypatch.setattr(driver, "ROUNDS", tmp_path / "rounds")
    monkeypatch.setattr(driver, "FOLD_PINS", pin_dir)
    (tmp_path / "rounds/r4").mkdir(parents=True)  # next tag r5 -> fold 1 with folds [0, 1]
    state = driver.start_round()
    assert state["tag"] == "r5" and state["fold"] == 1
    assert state["arms"][:2] == ["wmse_gelu_onecycle", "wmse_gelu_wsd_d0p2f0p01"]
    admitted = tmp_path / "rounds/r5/sealed"
    manifest = json.loads((admitted / "runner_overlay_asset/assets-manifest.json").read_text())
    assert {row["path"] for row in manifest["files"]} == {
        "scripts/cohort_spec.json", "scripts/sampled_recipe_worker.py",
        pins["SPLIT_RELATIVE"], pins["REGISTRY_RELATIVE"], pins["MEMBERSHIP_RELATIVE"],
    }
    metadata = json.loads((admitted / "kernel/kernel-metadata.json").read_text())
    assert metadata["kernel_sources"][0] == pins["SOURCE_KERNEL"]
    kernel_text = next((admitted / "kernel").glob("*.py")).read_text()
    assert "4" * 64 in kernel_text and "__CACHE_KEY__" not in kernel_text

    admission = json.loads((tmp_path / "rounds/r5/root_admission.json").read_text())
    monkeypatch.setenv("WNN_SAMPLED_ROOT_ADMISSION_JSON", json.dumps(admission))
    sys.modules.pop("sampled_recipe_worker", None)
    worker_spec = importlib.util.spec_from_file_location("rolling_worker_f1", HERE / "sampled_recipe_worker.py")
    worker = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker)
    cohort = worker.load_spec(tmp_path / "rounds/r5/cohort_spec.json")
    assert worker.FOLD == 1 and worker.base.EXPECTED_CACHE_KEY == "4" * 64
    argv = worker.build_train_argv(cohort["arms"][0], Path("m.json"), Path("v.json"), "0" * 64)
    assert argv[argv.index("--valid-fold") + 1] == "1"

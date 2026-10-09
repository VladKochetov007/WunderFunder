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

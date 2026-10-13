from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tests.test_training_state import _assert_equal
from wnn import data
from wnn.training_state import load_training_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_rnn as training  # noqa: E402


def _repository(root: Path, with_sentinels: bool = False, with_quiet_rows: bool = False) -> None:
    (root / "src").symlink_to(Path(__file__).resolve().parents[1] / "src", target_is_directory=True)
    (root / "datasets").mkdir()
    (root / "cache").mkdir()
    rng = np.random.default_rng(71)
    values = rng.normal(size=(5, 8, data.N_FEATURES)).astype("float32")
    if with_quiet_rows:
        values[:, 1::2, :22] = values[:, ::2, :22]
    if with_sentinels:
        values[:, 1::2, -1] = np.float32(-5.19934)
        values[:, ::3, 5] = np.float32(-5.19934)
    targets = np.tile(np.array([[-0.12543, 0.25086]], dtype="float32"), (5, 8, 1))
    targets[:, 4:, :] *= -1
    tables = []
    for group in range(5):
        columns = {name: values[group, :, index] for index, name in enumerate(data.FEATURE_COLUMNS)}
        columns.update(
            seq_ix=np.full(8, group),
            t0=targets[group, :, 0],
            t1=targets[group, :, 1],
            is_scored=np.arange(8) % 2 == 1,
        )
        tables.append(pa.table(columns))
    pq.write_table(pa.concat_tables(tables), root / "datasets/valid.parquet", row_group_size=8)
    np.savez(root / "cache/valid_targets.npz", row_group=np.arange(5), seq_ix=np.arange(5))
    np.save(root / "cache/pred_teacher_valid.npy", targets * 0.7)
    np.savez(
        root / "cache/chains.npz",
        next=np.full(5, -1),
        split=np.array(["valid"] * 5),
        row_group=np.arange(5),
    )


@pytest.mark.parametrize("controlled", [False, True])
def test_real_main_epoch_resume_and_completed_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controlled: bool,
    research_features: str = "none",
    origin_view: str = "off",
    credit_mode: str = "deterministic",
    keep_probability: float = 0.5,
    qat: str = "none",
    recurrent_rank: int = 0,
    sentinel_arm: tuple[str, str] | None = None,
    thin_mode: str | None = None,
) -> None:
    """Execute actual main, parquet reads, normalization, all P0 losses and exports."""
    _repository(
        tmp_path, with_sentinels=sentinel_arm is not None, with_quiet_rows=thin_mode is not None
    )
    device = os.environ.get("WNN_INTEGRATION_DEVICE", "cpu")
    if device == "cuda":
        assert torch.cuda.is_available(), "requested CUDA integration requires a visible device"
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "SENTINEL_PREPROCESSOR", None)
    monkeypatch.setattr(training, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(data, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(training, "WARMUP", 1)
    monkeypatch.setattr(training, "valid_row_groups", lambda path: (np.arange(4), np.array([4])))
    monkeypatch.setattr(training, "dev_halves", lambda path: (np.arange(2), np.arange(2, 4)))
    monkeypatch.setenv("WNN_LOADER_WORKERS", "2" if device == "cuda" else "0")
    monkeypatch.delenv("WNN_PORTABLE_DATA_MANIFEST", raising=False)
    base = [
        "train_rnn.py",
        "--device",
        device,
        "--train-source",
        "valid",
        "--full-valid",
        "--batch",
        "2",
        "--hidden",
        "4",
        "--layers",
        "1",
        "--chunk",
        "4",
        "--epochs",
        "3",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "2",
        "--no-eval-during-fit",
        "--loss",
        "wmse",
        "--distill",
        "teacher",
        "--aux-struct",
        ".3",
        "--aux-next",
        ".3",
        "--ema-teacher",
        "1",
        "--eval-ema",
        "--sync-free",
        "--seed",
        "3",
        "--save-training-state",
        "--research-features",
        research_features,
    ]
    if controlled:
        base += [
            "--chunk",
            "1",
            "--update-rows",
            "4",
            "--detach-rows",
            "2",
            "--detach-boundary-jitter",
            "--distill-boot-h",
            "1",
            "--ema-reference-rows",
            "4",
            "--optim",
            "muon",
            "--optim-switch",
            "0.5:adamw",
            "--grad-noise",
            "1e-6",
            "--batch-order",
            "seeded",
            "--antithetic",
        ]
    if device == "cuda":
        base += ["--fast"] if credit_mode == "deterministic" else ["--pin-memory", "--prop-cache"]
    if origin_view != "off":
        base += [
            "--chunk",
            "1",
            "--update-rows",
            "2",
            "--detach-rows",
            "2",
            "--origin-view",
            origin_view,
            "--ema-reference-rows",
            "2",
        ]
    if credit_mode != "deterministic":
        base += [
            "--chunk",
            "1",
            "--update-rows",
            "4",
            "--detach-rows",
            "4",
            "--credit-mode",
            credit_mode,
            "--credit-keep-probability",
            str(keep_probability),
            "--optim",
            "sgd",
            "--momentum",
            ".9",
            "--precision",
            "fp32",
            "--no-gradient-clip",
        ]
    if qat != "none":
        base += ["--qat", qat, "--precision", "fp32"]
    if recurrent_rank:
        base += [
            "--recurrent-rank",
            str(recurrent_rank),
            "--precision",
            "fp32",
            "--optim",
            "adamw",
            "--optim-switch",
            "",
        ]

    forward_rows: dict[str, int] = {}
    active_run = ""
    original_forward = training.forward_chunk

    def counted_forward(model, inputs, window, state):
        forward_rows[active_run] = forward_rows.get(active_run, 0) + window.stop - window.start
        return original_forward(model, inputs, window, state)

    if origin_view == "prefix-discard":
        monkeypatch.setattr(training, "forward_chunk", counted_forward)

    def run(name: str, extra: list[str]) -> Path:
        nonlocal active_run
        active_run = name
        monkeypatch.setattr(sys, "argv", [*base, "--name", name, *extra])
        training.main()
        return tmp_path / "artifacts/runs" / name

    if qat != "none" or recurrent_rank:
        parent_dir = run("parent", ["--qat", "none", "--recurrent-rank", "0"])
        base += ["--init", "parent"]
        float_dir = run("float_continuation", ["--qat", "none", "--recurrent-rank", "0"])
        float_state = load_training_state(float_dir / "training_state.pt")
    if qat != "none":
        from export_qat_control import export_control

        export_control(parent_dir / "model.pt", tmp_path / "no_finetune.pt", qat)
    if recurrent_rank:
        from export_factorized_control import export_control as export_rank_control

        export_rank_control(parent_dir / "model.pt", tmp_path / "zero_update.pt", recurrent_rank)
    if sentinel_arm is not None:
        from sentinel_prepare import prepare

        parent_dir = run("parent", [])
        directory = tmp_path / "artifacts/runs/sentinel_initializers"
        prepare(parent_dir / "model.pt", directory, tmp_path, sequence_count=3, row_stride=2)
        base += [
            "--init",
            "sentinel_initializers/" + "_".join(sentinel_arm),
            "--inherit-init-preprocessing",
        ]

    if thin_mode is not None:
        run("parent", [])
        base += [
            "--init",
            "parent",
            "--inherit-init-preprocessing",
            "--thin-mode",
            thin_mode,
            "--thin-prob",
            "1",
            "--thin-range",
            "0.4,0.8",
        ]
    reference_dir = run("reference", [])
    if origin_view == "prefix-discard":
        carry_dir = run("carry", ["--origin-view", "off"])
        run("refresh", ["--origin-view", "prefix-fixed"])
        carry = load_training_state(carry_dir / "training_state.pt")
        sham = load_training_state(reference_dir / "training_state.pt")
        for key in (
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "ema",
            "loader",
            "rng",
            "auxiliary",
        ):
            _assert_equal(carry[key], sham[key])
        assert carry["progress"]["step_count"] == sham["progress"]["step_count"]
        assert forward_rows["refresh"] == forward_rows["reference"]
        assert forward_rows["reference"] > forward_rows["carry"]
        active_run = "without_state"
    no_state_args = [argument for argument in base if argument != "--save-training-state"]
    monkeypatch.setattr(sys, "argv", [*no_state_args, "--name", "without_state"])
    training.main()
    without_state = torch.load(
        tmp_path / "artifacts/runs/without_state/model.pt", map_location="cpu", weights_only=False
    )
    reference_model = torch.load(reference_dir / "model.pt", map_location="cpu", weights_only=False)
    _assert_equal(without_state["model"], reference_model["model"])
    if qat != "none":
        _assert_equal(without_state["quantization"], reference_model["quantization"])
    if recurrent_rank:
        _assert_equal(without_state["factorization"], reference_model["factorization"])
    if sentinel_arm is not None:
        _assert_equal(
            without_state["sentinel_preprocessing"], reference_model["sentinel_preprocessing"]
        )
    if credit_mode == "artbp" and keep_probability == 1:
        full_dir = run("full_window", ["--credit-mode", "deterministic"])
        full_model = torch.load(full_dir / "model.pt", map_location="cpu", weights_only=False)
        _assert_equal(full_model["model"], reference_model["model"])
    interrupted_dir = run("interrupted", ["--stop-after-epoch", "0"])
    resumed_dir = run("resumed", ["--resume-state", str(interrupted_dir / "training_state.pt")])
    reference = load_training_state(reference_dir / "training_state.pt")
    resumed = load_training_state(resumed_dir / "training_state.pt")
    if qat != "none":
        from wnn.training_qat import effective_qat_state

        assert float_state["progress"]["step_count"] == reference["progress"]["step_count"]
        assert float_state["contract"]["options"]["precision"] == "fp32"
        assert float_state["contract"]["options"]["init"] == "parent"
        assert reference["contract"]["options"]["init"] == "parent"
        ema_weights = reference["ema"]["state"]
        effective, metadata = effective_qat_state(ema_weights, qat, "ema")
        _assert_equal(effective, reference_model["model"])
        _assert_equal(metadata, reference_model["quantization"])
        assert any(
            not torch.equal(ema_weights[name], effective[name]) for name in metadata["selected"]
        )
    if recurrent_rank:
        from wnn.training_factorized import configure_factorized, factorized_inference_state

        assert float_state["progress"]["step_count"] == reference["progress"]["step_count"]
        ema_weights = reference["ema"]["state"]
        model = training.make_model(
            reference_model["args"],
            reference_model["n_in"],
            reference_model["model"]["head.2.bias"].numel(),
        ).to(device)
        model.load_state_dict(
            {
                name: value
                for name, value in ema_weights.items()
                if not name.startswith("gru.factors.")
            }
        )
        configure_factorized(model, recurrent_rank)
        model.load_state_dict(ema_weights)
        effective, metadata = factorized_inference_state(model, "ema")
        _assert_equal(effective, reference_model["model"])
        _assert_equal(metadata, reference_model["factorization"])
        assert any(name.startswith("gru.factors.") for name in reference["model"])
        assert not any(name.startswith("gru.factors.") for name in effective)
    assert reference["contract"]["runtime"]["threads"] == training.cpu_threads()
    assert reference["model"]["mean"].numel() == (
        data.N_FEATURES
        + training.N_EVENT_FEATURES
        + len(training.research_feature_names(research_features))
        + (112 if sentinel_arm is not None and sentinel_arm[1] != "none" else 0)
    )
    for key in (
        "contract",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "ema",
        "loader",
        "rng",
        "auxiliary",
    ):
        _assert_equal(reference[key], resumed[key])
    expected_steps = 24 if origin_view != "off" else 12
    assert (
        reference["progress"]["step_count"] == resumed["progress"]["step_count"] == expected_steps
    )
    for first, second in zip(
        reference["progress"]["update_diagnostics"],
        resumed["progress"]["update_diagnostics"],
        strict=True,
    ):
        _assert_equal(first, second)
    recovered_dir = run("recovered", ["--resume-state", str(resumed_dir / "training_state.pt")])
    recovered_model = torch.load(recovered_dir / "model.pt", map_location="cpu", weights_only=False)
    expected_model = torch.load(reference_dir / "model.pt", map_location="cpu", weights_only=False)
    _assert_equal(recovered_model["model"], expected_model["model"])
    if qat != "none":
        _assert_equal(recovered_model["quantization"], expected_model["quantization"])
    if recurrent_rank:
        _assert_equal(recovered_model["factorization"], expected_model["factorization"])
    if sentinel_arm is not None:
        _assert_equal(
            recovered_model["sentinel_preprocessing"], expected_model["sentinel_preprocessing"]
        )
        prepared = torch.load(directory / "_".join(sentinel_arm) / "model.pt", weights_only=False)
        _assert_equal(expected_model["model"]["mean"], prepared["model"]["mean"])
        _assert_equal(expected_model["model"]["std"], prepared["model"]["std"])
        if sentinel_arm[1] != "none":
            assert torch.count_nonzero(expected_model["model"]["inp.weight"][:, -112:]) > 0
    if thin_mode is not None:
        parent_model = torch.load(
            tmp_path / "artifacts/runs/parent/model.pt", map_location="cpu", weights_only=False
        )
        _assert_equal(expected_model["model"]["mean"], parent_model["model"]["mean"])
        _assert_equal(expected_model["model"]["std"], parent_model["model"]["std"])
    assert recovered_model["epoch"] == 2
    assert len(json.loads((recovered_dir / "history.json").read_text())) == 3
    assert (
        json.loads((recovered_dir / "provenance.json").read_text())["cumulative_updates"][
            "optimizer_attempts"
        ]
        == expected_steps
    )


def test_main_restores_thread_budget_after_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-time initializer change must not become the saved fit's thread count."""
    original_builder = training.build_optimizer
    initialize_once = iter([True])

    def changing_builder(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
        optimizer = original_builder(model, args)
        if next(initialize_once, False):
            torch.set_num_threads(training.cpu_threads() + 1)
        return optimizer

    monkeypatch.setattr(training, "build_optimizer", changing_builder)
    monkeypatch.setenv("WNN_INTEGRATION_DEVICE", "cpu")
    test_real_main_epoch_resume_and_completed_export(tmp_path, monkeypatch, False)


@pytest.mark.parametrize("research_features", ["volume28", "volume28_current"])
def test_main_research_features_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, research_features: str
) -> None:
    """Expanded normalization and feature checkpoints survive actual trainer resume."""
    test_real_main_epoch_resume_and_completed_export(
        tmp_path, monkeypatch, False, research_features
    )


@pytest.mark.parametrize("origin_view", ["prefix-fixed", "prefix-random", "prefix-discard"])
def test_main_origin_view_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, origin_view: str
) -> None:
    """Full prefix replay with actual row-clock EMA stays exact across epoch resume."""
    test_real_main_epoch_resume_and_completed_export(
        tmp_path, monkeypatch, False, origin_view=origin_view
    )


@pytest.mark.parametrize("thin_mode", ["compress", "retained-mass"])
def test_main_thinning_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, thin_mode: str
) -> None:
    """Quiet-row views keep frozen normalization and exact loader/RNG continuation."""
    test_real_main_epoch_resume_and_completed_export(
        tmp_path, monkeypatch, False, thin_mode=thin_mode
    )


@pytest.mark.parametrize(
    "credit_mode,keep_probability",
    [
        ("random-uncompensated", 0.5),
        ("artbp", 0.5),
        ("artbp", 1.0),
    ],
)
def test_main_random_credit_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    credit_mode: str,
    keep_probability: float,
) -> None:
    """Private credit gates and actual SGD momentum continuation remain deterministic."""
    test_real_main_epoch_resume_and_completed_export(
        tmp_path, monkeypatch, False, credit_mode=credit_mode, keep_probability=keep_probability
    )


@pytest.mark.parametrize(
    "qat", ["recurrent", "recurrent_head", "recurrent_w4", "input_w4_recurrent_w8"]
)
@pytest.mark.parametrize("controlled", [False, True])
def test_main_qat_finetune_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qat: str, controlled: bool
) -> None:
    """Actual FP32 QAT fine-tuning preserves raw continuation and effective EMA export."""
    test_real_main_epoch_resume_and_completed_export(tmp_path, monkeypatch, controlled, qat=qat)


@pytest.mark.parametrize("controlled", [False, True])
def test_main_factorized_finetune_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, controlled: bool
) -> None:
    """Latent factors and their AdamW moments resume exactly with effective EMA export."""
    test_real_main_epoch_resume_and_completed_export(
        tmp_path, monkeypatch, controlled, recurrent_rank=2
    )


@pytest.mark.parametrize(
    "sentinel_arm",
    [
        ("unchanged", "none"),
        ("unchanged", "missing"),
        ("neutral", "none"),
        ("safe", "missing"),
        ("safe", "current"),
    ],
)
def test_main_sentinel_finetune_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sentinel_arm: tuple[str, str]
) -> None:
    """Frozen preprocessing and zero-init feature projections survive actual fine-tune resume."""
    test_real_main_epoch_resume_and_completed_export(
        tmp_path, monkeypatch, False, sentinel_arm=sentinel_arm
    )


def test_research_features_preserve_propensity_input_and_parent_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen propensity sees its original inputs; research columns precede extras."""
    raw = torch.randn(2, 8, data.N_FEATURES)
    extra = torch.randn(2, 8, 3)
    monkeypatch.setattr(training, "RESEARCH_FEATURES", "volume28")
    monkeypatch.setattr(training, "INPUT_PROPENSITY", object())
    expected_propensity = torch.cat((raw, training.event_features_torch(raw), extra), -1)

    def check_propensity(model: object, inputs: torch.Tensor) -> torch.Tensor:
        torch.testing.assert_close(inputs, expected_propensity, rtol=0, atol=0)
        return inputs.new_full(inputs.shape[:2], 0.25)

    monkeypatch.setattr(training, "propensity", check_propensity)
    inputs = training.build_inputs(torch.cat((raw, extra), -1))
    base_width = data.N_FEATURES + training.N_EVENT_FEATURES
    torch.testing.assert_close(
        inputs[..., base_width : base_width + 28],
        training.research_features_torch(raw, "volume28"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(inputs[..., -4:-1], extra, rtol=0, atol=0)
    assert (inputs[..., -1] == 0.25).all()
    args = argparse.Namespace(research_features="none")
    parent = {"args": {"research_features": "volume28"}}
    training.inherit_init_arguments(args, parent, [])
    assert args.research_features == "volume28"
    args.research_features = "volume28_current"
    with pytest.raises(ValueError, match="disagrees"):
        training.inherit_init_arguments(args, parent, ["--research-features"])

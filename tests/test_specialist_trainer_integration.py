from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tests.test_training_state import _assert_equal
from wnn import data
from wnn.specialist import PrefixAdapterGRUModel, SpecialistGRUModel, causal_prefix_mean
from wnn.training_state import load_training_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_rnn as training  # noqa: E402


def _repository(root: Path, sequence_length: int = 104) -> np.ndarray:
    (root / "src").symlink_to(Path(__file__).resolve().parents[1] / "src", target_is_directory=True)
    (root / "datasets").mkdir()
    (root / "cache").mkdir()
    rng = np.random.default_rng(941)
    values = rng.normal(size=(5, sequence_length, data.N_FEATURES)).astype("float32")
    targets = rng.normal(scale=0.2, size=(5, sequence_length, 2)).astype("float32")
    tables = []
    for group in range(5):
        columns = {name: values[group, :, index] for index, name in enumerate(data.FEATURE_COLUMNS)}
        columns.update(
            seq_ix=np.full(sequence_length, group),
            t0=targets[group, :, 0],
            t1=targets[group, :, 1],
            is_scored=np.arange(sequence_length) >= 99,
        )
        tables.append(pa.table(columns))
    pq.write_table(pa.concat_tables(tables), root / "datasets/valid.parquet", row_group_size=sequence_length)
    np.savez(root / "cache/valid_targets.npz", row_group=np.arange(5), seq_ix=np.arange(5))
    np.save(root / "cache/pred_teacher_valid.npy", targets * np.float32(0.7))
    np.savez(
        root / "cache/chains.npz",
        next=np.full(5, -1),
        split=np.array(["valid"] * 5),
        row_group=np.arange(5),
    )
    return values


@pytest.mark.parametrize(
    ("arch", "control"),
    [
        ("specialist", "learned"),
        ("specialist", "hash"),
        ("specialist", "universal"),
        ("prefix-adapter", "conditional"),
        ("prefix-adapter", "unconditional"),
        ("prefix-adapter", "diagonal"),
    ],
)
def test_actual_trainer_resume_checkpoint_and_chunk_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arch: str, control: str
) -> None:
    """Fit each arm through real Parquet/cache loaders, then prove restart and inference parity."""
    sequence_length = 112
    values = _repository(tmp_path, sequence_length)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "SENTINEL_PREPROCESSOR", None)
    monkeypatch.setattr(training, "SEQUENCE_LENGTH", sequence_length)
    monkeypatch.setattr(data, "SEQUENCE_LENGTH", sequence_length)
    monkeypatch.setattr(training, "WARMUP", 99)
    monkeypatch.setattr(training, "valid_row_groups", lambda path: (np.arange(4), np.array([4])))
    monkeypatch.setattr(training, "dev_halves", lambda path: (np.arange(2), np.arange(2, 4)))
    monkeypatch.setenv("WNN_LOADER_WORKERS", "0")
    monkeypatch.delenv("WNN_PORTABLE_DATA_MANIFEST", raising=False)

    base = [
        "train_rnn.py", "--device", "cpu", "--train-source", "valid", "--full-valid",
        "--batch", "4", "--hidden", "4", "--expert-hidden", "3", "--num-experts", "3",
        "--layers", "1", "--chunk", "4", "--update-rows", "8", "--detach-rows", "4",
        "--epochs", "2", "--checkpoint-policy", "fixed", "--checkpoint-epoch", "1",
        "--no-eval-during-fit", "--loss", "wmse", "--distill", "teacher",
        "--distill-boot-h", "1", "--ema-teacher", "1", "--ema-reference-rows", "8",
        "--eval-ema", "--precision", "fp32", "--optim", "muon", "--seed", "19",
        "--save-training-state", "--arch", arch,
    ]
    if arch == "specialist":
        base += ["--specialist-route", control]
    else:
        base += ["--adapter-rank", "2", "--adapter-mode", control]

    route_counts = np.zeros(3, dtype=np.float64)
    expert_gradient_chunks = np.zeros(3, dtype=np.int64)
    capture_training_model = False
    original_make_model = training.make_model

    def traced_make_model(args_dict, n_in, n_out):
        model = original_make_model(args_dict, n_in, n_out)
        if capture_training_model and arch == "specialist":
            assert isinstance(model, SpecialistGRUModel)

            def route_trace(module, args):
                if module is not model:
                    return  # AveragedModel deep-copies module hooks; exclude EMA forwards.
                inputs = args[0]
                state = args[1] if len(args) > 1 else None
                normalized = (inputs - module.mean) / module.std
                prefix_state = None if state is None else (state[-2], state[-1])
                prefix, _ = causal_prefix_mean(normalized, prefix_state, module.prefix_rows)
                prefix_count = 0 if state is None else int(state[-1][0])
                if prefix_count < module.prefix_rows:
                    prefix = prefix[:, max(0, module.prefix_rows - prefix_count):]
                routes = module.route_indices(prefix)
                if module.route == "universal":
                    route_counts[:] += routes.numel() / module.num_experts
                else:
                    valid = routes >= 0
                    if valid.any():
                        route_counts[:] += torch.bincount(
                            routes[valid].detach().cpu(), minlength=module.num_experts
                        ).numpy()

            model.register_forward_pre_hook(route_trace)
            for index, expert in enumerate(model.experts):
                def backward_trace(module, grad_input, grad_output, expert_index=index):
                    if any(torch.count_nonzero(value).item() > 0 for value in grad_output if value is not None):
                        expert_gradient_chunks[expert_index] += 1

                expert.register_full_backward_hook(backward_trace)
        return model

    monkeypatch.setattr(training, "make_model", traced_make_model)

    def run(name: str, extra: list[str] | None = None) -> Path:
        nonlocal capture_training_model
        capture_training_model = name == "reference"
        monkeypatch.setattr(sys, "argv", [*base, "--name", name, *(extra or [])])
        training.main()
        return tmp_path / "artifacts/runs" / name

    reference_dir = run("reference")
    capture_training_model = False
    interrupted_dir = run("interrupted", ["--stop-after-epoch", "0"])
    resumed_dir = run("resumed", ["--resume-state", str(interrupted_dir / "training_state.pt")])

    reference_state = load_training_state(reference_dir / "training_state.pt")
    resumed_state = load_training_state(resumed_dir / "training_state.pt")
    for key in ("contract", "model", "optimizer", "scheduler", "scaler", "ema", "loader", "rng", "auxiliary"):
        _assert_equal(reference_state[key], resumed_state[key])
    expected = torch.load(reference_dir / "model.pt", map_location="cpu", weights_only=False)
    actual = torch.load(resumed_dir / "model.pt", map_location="cpu", weights_only=False)
    _assert_equal(expected["model"], actual["model"])

    n_in = expected["n_in"]
    n_out = expected["model"]["head.2.bias"].numel() if arch == "prefix-adapter" else 2
    reloaded = original_make_model(expected["args"], n_in, n_out)
    reloaded.load_state_dict(expected["model"])
    parameter_count = sum(parameter.numel() for parameter in reloaded.parameters())
    assert parameter_count == sum(expected["model"][name].numel() for name, parameter in reloaded.named_parameters())
    x = torch.from_numpy(values[0:1])
    prediction_4 = training.predict(reloaded, x, 4, torch.device("cpu"))
    prediction_13 = training.predict(reloaded, x, 13, torch.device("cpu"))
    np.testing.assert_allclose(prediction_4, prediction_13, rtol=2e-5, atol=2e-6)

    if arch == "specialist":
        assert route_counts.sum() > 0
        assert expert_gradient_chunks.sum() > 0
        assert isinstance(reloaded, SpecialistGRUModel)
        assert len(reloaded(torch.zeros(2, 4, n_in))[1]) == reloaded.num_experts + 3
        print("TRAIN_ONLY_SPECIALIST_AUDIT", control, parameter_count, route_counts.tolist(), expert_gradient_chunks.tolist())
    else:
        assert isinstance(reloaded, PrefixAdapterGRUModel)
        assert len(reloaded(torch.zeros(2, 4, n_in))[1]) == 3
        print("PREFIX_ADAPTER_PARAMETER_COUNT", control, parameter_count)


@pytest.mark.parametrize(
    "extra",
    [["--rnn-init", "orth"], ["--muon-scope", "hidden"], ["--head-lr-mult", "2"]],
)
def test_specialist_rejects_unsupported_training_options(
    monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["train_rnn.py", "--arch", "specialist", *extra])
    with pytest.raises(SystemExit):
        training.main()

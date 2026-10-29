import pytest
import torch

from wnn.nn import RNNModel
from wnn.tabm_gru import TabMGRUModel
from wnn.training.models import make_model


def _model(k: int = 3) -> TabMGRUModel:
    torch.manual_seed(0)
    model = TabMGRUModel(n_in=7, hidden=16, layers=2, n_out=4, k=k)
    model.mean.normal_()
    model.std.uniform_(0.5, 2.0)
    with torch.no_grad():
        model.member_scale.mul_(torch.rand_like(model.member_scale) + 0.5)
    return model


def test_training_rows_follow_member_index_and_eval_returns_member_mean() -> None:
    model = _model()
    x = torch.randn(2, 5, 7)
    model.train()
    rows, _ = model(x.repeat_interleave(3, dim=0))
    model.eval()
    mean, state = model(x)
    assert rows.shape == (6, 5, 4)
    assert mean.shape == (2, 5, 4)
    assert state.shape == (2, 6, 16)
    torch.testing.assert_close(mean, rows.reshape(2, 3, 5, 4).mean(dim=1))


def test_members_materialize_exactly_as_plain_gru_models_with_carried_state() -> None:
    model = _model().eval()
    x = torch.randn(2, 9, 7)
    first, state = model(x[:, :4])
    second, _ = model(x[:, 4:], state)
    expected = torch.cat([first, second], dim=1)
    outputs = []
    for member in range(3):
        plain = RNNModel(7, 16, 2, 0.0, n_out=4).eval()
        plain.load_state_dict(model.member_state_dict(member))
        head, member_state = plain(x[:, :4])
        tail, _ = plain(x[:, 4:], member_state)
        outputs.append(torch.cat([head, tail], dim=1))
    torch.testing.assert_close(torch.stack(outputs).mean(dim=0), expected, rtol=1e-5, atol=1e-6)


def test_factory_builds_tabm_and_rejects_non_materializable_layouts() -> None:
    args = {"hidden": 16, "layers": 1, "dropout": 0.0, "ensemble_k": 4}
    model = make_model(args, 7, 2)
    assert isinstance(model, TabMGRUModel)
    assert model.state_dict()["head.2.weight"].shape == (2, 4, 16)
    with pytest.raises(ValueError, match="enc"):
        make_model({**args, "enc": "mlp"}, 7, 2)


def test_training_batch_must_hold_whole_member_repeats() -> None:
    model = _model().train()
    with pytest.raises(ValueError, match="whole member repeats"):
        model(torch.randn(4, 3, 7))


def test_factory_passes_adapter_init() -> None:
    args = {"hidden": 8, "layers": 1, "dropout": 0.0, "ensemble_k": 2, "ensemble_init": "ones"}
    assert torch.equal(make_model(args, 5, 2).member_scale, torch.ones(2, 5))

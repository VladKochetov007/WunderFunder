"""Synthetic restricted-loader checks for teacher terminal verifier V2."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import teacher_terminal_check_v2 as check
import torch


class _UnapprovedGlobal:
    """Pickle global that must remain outside the V2 loader allowlist."""


def test_torchversion_and_numpy_rng_load_with_narrow_teacher_allowlist(tmp_path) -> None:
    version_class = torch.torch_version.TorchVersion
    assert isinstance(torch.__version__, version_class)
    allowed = check._teacher_safe_globals(np)
    assert allowed[-1] is version_class

    path = tmp_path / "producer-shaped-checkpoint.pt"
    torch.save(
        {
            "provenance": {"update_counter_implementation": {"torch_version": torch.__version__}},
            "rng": np.random.RandomState(23).get_state(),
        },
        path,
    )
    loaded, _, _ = check._load_torch_file(path, torch, allowlist=allowed)

    version = loaded["provenance"]["update_counter_implementation"]["torch_version"]
    assert type(version) is version_class
    assert version == torch.__version__
    assert loaded["rng"][0] == "MT19937"
    assert loaded["rng"][1].dtype == np.dtype("uint32")


def test_teacher_allowlist_still_rejects_unapproved_globals(tmp_path) -> None:
    path = tmp_path / "unapproved.pt"
    torch.save({"unsupported": _UnapprovedGlobal()}, path)
    with pytest.raises(pickle.UnpicklingError, match="_UnapprovedGlobal"):
        check._load_torch_file(path, torch, allowlist=check._teacher_safe_globals(np))

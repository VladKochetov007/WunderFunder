from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from wnn.training.state import TrainingRun


def compare_parent_parity(
    run: TrainingRun, project_root: Path, source_root: Path, tolerance: float
) -> list[dict[str, object]]:
    """Compare parent and child first-two outputs on complete fitting sequences."""
    import numpy as np
    import torch

    from wnn.data import N_FEATURES
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.inputs import build_inputs

    environment = run.environment
    parent_checkpoint = cast(dict[str, object], run.network.init)
    parent_state = cast(dict[str, torch.Tensor], parent_checkpoint["model"])
    parent_args = cast(dict[str, object], parent_checkpoint["args"])
    parent_model = environment.make_model(parent_args, 211, 23)
    parent_model.load_state_dict(parent_state)
    parent_model.eval()
    child_model = run.network.model
    child_model.eval()
    if run.options.args.chunk != 2000 or environment.sequence_length != 20000:
        raise ValueError("parity must exercise full sequences with 2,000-row chunks")

    parent_environment = TrainingEnvironment(root=project_root, source_root=source_root)
    groups = [
        ("train", int(run.data.train_idx[0])),
        ("valid", int(run.data.extra_valid[0])),
    ]
    results: list[dict[str, object]] = []
    with torch.inference_mode():
        for split, group in groups:
            source = environment.sequence_file(project_root / "datasets" / f"{split}.parquet")
            sequence = source.read(group, targets=False, mask=False)
            if sequence.features.shape != (environment.sequence_length, N_FEATURES):
                raise ValueError("parity input is not a complete feature-only sequence")
            raw = torch.from_numpy(
                np.asarray(sequence.features, dtype=np.float32).copy()
            ).unsqueeze(0)
            parent_input = build_inputs(raw, environment=parent_environment)
            child_input = build_inputs(raw, environment=environment)
            parent_hidden = child_hidden = None
            maximum = 0.0
            for start in range(0, environment.sequence_length, 2000):
                window = slice(start, start + 2000)
                parent_output, parent_hidden = environment.forward_chunk(
                    parent_model, parent_input, window, parent_hidden
                )
                child_output, child_hidden = environment.forward_chunk(
                    child_model, child_input, window, child_hidden
                )
                error = float((parent_output[..., :2] - child_output[..., :2]).abs().max().item())
                maximum = max(maximum, error)
            if maximum > tolerance:
                raise ValueError(f"parent/child parity failed for {split} group {group}: {maximum}")
            results.append({"split": split, "row_group": group, "max_abs_error": maximum})
            del parent_input, child_input, raw
    return results

from __future__ import annotations

import torch
import torch.nn.functional as F


def next_input_losses(
    output: torch.Tensor,
    inputs: torch.Tensor,
    rows: slice,
    head_at: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict the next normalized input row and return Huber, MSE and finite count.

    Output position t is paired with input row t + 1. The final sequence row has
    no successor and is omitted. This objective reads model inputs only; it does
    not use target labels, score masks, or validation/test feedback.
    """
    if rows.start is None or rows.stop is None:
        raise ValueError("next-input loss requires a bounded output slice")
    if output.ndim != 3 or inputs.ndim != 3:
        raise ValueError("next-input loss expects batch-time-feature tensors")
    start, stop = rows.start, rows.stop
    if (
        start < 0
        or stop > inputs.shape[1]
        or output.shape[1] != stop - start
        or output.shape[0] != inputs.shape[0]
    ):
        raise ValueError("next-input loss slice does not match the model output")
    width = inputs.shape[-1]
    if head_at < 0 or head_at + width > output.shape[-1]:
        raise ValueError("next-input output head is narrower than the input feature vector")
    if mean.shape != (width,) or std.shape != (width,):
        raise ValueError("next-input normalization buffers do not match the input width")

    row_count = max(0, min(stop, inputs.shape[1] - 1) - start)
    if row_count == 0:
        zero = output.sum() * 0.0
        return zero, zero.detach(), output.new_zeros((), dtype=torch.long)

    prediction = output[:, :row_count, head_at : head_at + width].float()
    target = inputs[:, start + 1 : start + 1 + row_count].float()
    target = (target - mean.float()) / std.float()
    finite = torch.isfinite(target)

    safe_prediction = torch.where(finite, prediction, 0.0)
    safe_target = torch.where(finite, target, 0.0)
    count = finite.sum()
    count_float = count.clamp_min(1).to(dtype=prediction.dtype)
    huber = F.smooth_l1_loss(safe_prediction, safe_target, beta=1.0, reduction="none") * finite
    mse = ((safe_prediction - safe_target).square() * finite).sum() / count_float
    return huber.sum() / count_float, mse.detach(), count.detach()

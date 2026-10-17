from __future__ import annotations

import torch

from wnn.training.environment import TrainingEnvironment


def thin_rows(
    result: tuple[torch.Tensor, ...],
    thin: tuple[float, float, float] | tuple[float, float, float, str] | None,
    *,
    environment: TrainingEnvironment | None = None,
) -> tuple[torch.Tensor, ...]:
    """Training view of one sequence without some rows whose i0 price codes did not change.

    Targets are constant between quote moves and each target depends only on the future quote path, so deleting a
    row with unchanged i0 price codes (columns 0..21) after the warm-up leaves every kept row's label exact; the
    stream simply looks as if fewer book updates were published. With probability ``thin[0]`` a rate q ~ U(thin[1],
    thin[2]) of those rows is dropped; the sequence is padded back to its length by repeating the last kept row, and
    the kept length is appended so the loop can exclude padding from every loss.
    The retained-mass control uses the identical random keep map but leaves all observations,
    targets and cached values on their original time axis, appending the supervised-row mask.
    """
    environment = environment if environment is not None else TrainingEnvironment()
    prob, low, high = thin[:3]
    mode = thin[3] if len(thin) == 4 else "compress"
    if mode not in ("compress", "retained-mass"):
        raise ValueError("unknown row-thinning mode")
    x = result[0]
    n = x.shape[0]
    if torch.rand(()) >= prob:
        return result + (
            torch.ones(n, dtype=torch.bool) if mode == "retained-mass" else torch.tensor(n),
        )
    q = low + (high - low) * torch.rand(())
    quiet = torch.zeros(n, dtype=torch.bool)
    quiet[environment.warmup :] = (
        x[environment.warmup :, :22] == x[environment.warmup - 1 : -1, :22]
    ).all(dim=1)
    keep = ~(quiet & (torch.rand(n) < q))
    if mode == "retained-mass":
        return result + (keep,)
    kept = int(keep.sum())
    index = torch.cat(
        [torch.nonzero(keep).squeeze(1), torch.full((n - kept,), int(torch.nonzero(keep)[-1]))]
    )
    thinned = []
    for j, part in enumerate(result):
        part = part[index]
        if j == 2:
            part = part.clone()
            part[kept:] = False
        thinned.append(part)
    return tuple(thinned) + (torch.tensor(kept),)

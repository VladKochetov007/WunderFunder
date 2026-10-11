"""Fast sequence renderer: aggregate 20k rows to pixel columns before drawing.

Lines are drawn as per-pixel min/max envelopes (``envelope``), event streams as
per-pixel counts, and wide feature blocks as ``imshow`` heatmaps, so a full
sequence renders in well under a second with the Agg backend.
"""

from __future__ import annotations

import numpy as np


def pixel_bins(n: int, width: int) -> np.ndarray:
    """Start index of each pixel column (reduceat offsets)."""
    width = max(1, min(width, n))
    return np.linspace(0, n, width + 1).astype(np.int64)[:-1]


def envelope(y: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x_centers, min, max) per pixel column; exact extrema of each bin."""
    y = np.asarray(y, dtype=np.float64)
    starts = pixel_bins(len(y), width)
    lo = np.minimum.reduceat(y, starts)
    hi = np.maximum.reduceat(y, starts)
    ends = np.r_[starts[1:], len(y)]
    return (starts + ends - 1) / 2.0, lo, hi


def counts(flags: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    flags = np.asarray(flags, dtype=np.float64)
    starts = pixel_bins(len(flags), width)
    ends = np.r_[starts[1:], len(flags)]
    return (starts + ends - 1) / 2.0, np.add.reduceat(flags, starts)


def heat(block: np.ndarray, width: int) -> np.ndarray:
    """(columns, width) mean per pixel column for imshow."""
    block = np.asarray(block, dtype=np.float64)
    starts = pixel_bins(len(block), width)
    sums = np.add.reduceat(block, starts, axis=0)
    sizes = np.diff(np.r_[starts, len(block)])[:, None]
    return (sums / sizes).T


def render_sequence(seq, out_path, *, predictions=None, window=None, width=1600, title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wnn.data import TARGET_QUANTA, market_class

    x = seq.features
    lo, hi = window if window else (0, len(x))
    x = x[lo:hi]
    y = seq.targets[lo:hi]
    d = np.diff(x, axis=0, prepend=x[:1]) != 0
    ev_px = d[:, 0] & d[:, 11]
    ev_trade = d[:, 48]
    ev_i1 = d[:, 52] | d[:, 63]
    level = np.cumsum(np.sign(np.diff(x[:, 0], prepend=x[0, 0])))
    c = market_class(seq.targets)
    q = TARGET_QUANTA[c] if c >= 0 else 1.0

    fig, ax = plt.subplots(
        7,
        1,
        figsize=(16, 15),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.2, 1.2, 2, 2, 1.5, 1]},
    )
    for k, color in ((0, "tab:blue"), (1, "tab:orange")):
        xc, a, b = envelope(y[:, k] / q + (0.08 if k else -0.08), width)
        ax[0].vlines(
            xc + lo, a - 0.04, b + 0.04, color=color, lw=1.0, alpha=0.8, label=f"t{k} / quantum"
        )
        if predictions is not None:
            xc, a, b = envelope(predictions[lo:hi, k] / q, width)
            ax[0].vlines(xc + lo, a - 0.02, b + 0.02, color="k", lw=0.6, alpha=0.5)
    if seq.scored is not None:
        xc, n = counts(seq.scored[lo:hi], width)
        ax[5].vlines(xc + lo, 0, n, color="tab:green", lw=1.0)
        ax[5].set_ylabel("scored/px")
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel("targets (quanta)")
    xc, a, b = envelope(level, width)
    ax[1].vlines(xc + lo, a - 0.3, b + 0.3, color="tab:purple", lw=1.0)
    ax[1].set_ylabel("sum sign dp0")
    for name, flags, color in (
        ("i0 px ev", ev_px, "tab:red"),
        ("i0 trade", ev_trade, "tab:brown"),
        ("i1 px ev", ev_i1, "tab:gray"),
    ):
        xc, n = counts(flags, width)
        ax[2].plot(xc + lo, n, lw=0.8, color=color, label=name)
    ax[2].legend(loc="upper right")
    ax[2].set_ylabel("events/px")
    ax[3].imshow(
        heat(x[:, 22:44], width),
        aspect="auto",
        cmap="coolwarm",
        extent=(lo, hi, 43.5, 21.5),
        vmin=-1,
        vmax=1,
    )
    ax[3].set_ylabel("i0 vols 22-43")
    ax[4].imshow(
        heat(x[:, 0:22], width), aspect="auto", cmap="viridis", extent=(lo, hi, 21.5, -0.5)
    )
    ax[4].set_ylabel("i0 prices 0-21")
    ax[6].imshow(
        heat(x[:, 104:112].clip(-3, 3), width),
        aspect="auto",
        cmap="magma",
        extent=(lo, hi, 111.5, 103.5),
    )
    ax[6].set_ylabel("a0-a7")
    ax[6].set_xlabel("step")
    fig.suptitle(title or f"seq {seq.seq_ix} class {c}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)

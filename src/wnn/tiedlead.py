from __future__ import annotations

from typing import TYPE_CHECKING

from wnn.data import N_FEATURES
from wnn.features import N_EVENT_FEATURES

if TYPE_CHECKING:
    import torch

TL3_LEAD_WIDTH = 52
TIEDLEAD_BASE_INPUT_WIDTH = N_FEATURES + N_EVENT_FEATURES
TIEDLEAD_INPUT_WIDTH = TIEDLEAD_BASE_INPUT_WIDTH + TL3_LEAD_WIDTH
TIEDLEAD_INSTRUMENT_WIDTH = 52
TIEDLEAD_BOOK_WIDTH = 2 * TIEDLEAD_INSTRUMENT_WIDTH
TIEDLEAD_HIDDEN = 256
TIED_ENCODER_HIDDEN = 128
TIED_ENCODER_WIDTH = 64


def tied_book_deltas(
    raw_books: torch.Tensor, previous_book: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build row-to-row book deltas and retain the last raw book for the next chunk.

    Args:
        raw_books: Raw i0 and i1 books with shape ``(batch, time, 104)``.
        previous_book: The preceding chunk's final raw book, or ``None`` at a sequence start.

    Returns:
        Per-row deltas with the same shape as ``raw_books`` and a cloned ``(batch, 104)`` tail.

    Raises:
        ValueError: If the sequence or carried state has an incompatible shape, dtype, or device.
    """
    import torch

    if raw_books.ndim != 3 or raw_books.shape[1] == 0 or raw_books.shape[2] != TIEDLEAD_BOOK_WIDTH:
        raise ValueError("tiedlead book deltas require a nonempty (batch,time,104) tensor")
    batch_size = raw_books.shape[0]
    if previous_book is None:
        predecessor = raw_books[:, :1]
    else:
        if (
            previous_book.shape != (batch_size, TIEDLEAD_BOOK_WIDTH)
            or previous_book.dtype != raw_books.dtype
            or previous_book.device != raw_books.device
        ):
            raise ValueError("tiedlead carried book must match the batch, dtype, and device")
        predecessor = previous_book[:, None]
    earlier_rows = torch.cat((predecessor, raw_books[:, :-1]), dim=1)
    deltas = raw_books - earlier_rows
    return deltas, raw_books[:, -1].clone()

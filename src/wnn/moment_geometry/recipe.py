"""Shared fixed budgets and channel schema for the registered prepared-array pilots."""

from __future__ import annotations

SEQ, WIDTH, RAW, GEO, CURRENT, BATCH, CHUNK, WARMUP = 20_000, 231, 211, 20, 4, 4, 512, 99
MAX_NORM, BASE_LR, MIN_LR, WD, LOG_EVERY = 1.0, 1e-3, 1e-5, 0.01, 40
ARMS = ("control", "moments", "scaled", "dictionary", "combined")

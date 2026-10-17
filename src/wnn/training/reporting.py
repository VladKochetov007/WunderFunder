from __future__ import annotations

import logging
import sys

LOGGER = logging.getLogger("wnn.training")


class TrainingLogHandler(logging.StreamHandler):
    """Follow current stdout when a caller runs several CLIs in one process."""

    def emit(self, record: logging.LogRecord) -> None:
        """Write one record to the current caller-owned stdout stream."""
        self.stream = sys.stdout
        super().emit(record)


def emit(*values: object, flush: bool = False) -> None:
    """Emit unchanged machine-readable records through standard logging.

    StreamHandler flushes each record, preserving legacy CLI output timing.
    """
    if not LOGGER.handlers:
        handler = TrainingLogHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("{message}", style="{"))
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False
    LOGGER.info(" ".join(str(value) for value in values))

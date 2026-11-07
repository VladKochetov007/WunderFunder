"""Frozen final-fit teacher contract shared by cache admission and drift tests."""

from __future__ import annotations

import math
from collections.abc import Mapping

FINAL_RUN_PREFIX = "tl3final_"
FINAL_W18_HOMOGENEOUS_FAMILY = "strong"
FINAL_W18_TEACHERS = (
    ("tl3final_fbc1k_s5", "strong", 0.3, 3),
    ("tl3final_ffull_s3", "full_finetuned", 0.1, 1),
    ("tl3final_ffull_s4", "full_finetuned", 0.1, 1),
    ("tl3final_frawW_s3ft", "raw_wmse", 0.125, 1),
    ("tl3final_frawW_s4ft", "raw_wmse", 0.125, 1),
    ("tl3final_orgbase", "samefold_baseline", 0.0625, 3),
)


def validate_exact_final_w18_teachers(metadata: Mapping[str, object]) -> None:
    """Require the exact six ordered W18 sources and their homogeneous family."""
    coefficients, sources = metadata.get("source_coefficients"), metadata.get("sources")
    if (
        not isinstance(coefficients, list)
        or not isinstance(sources, list)
        or len(coefficients) != len(FINAL_W18_TEACHERS)
        or len(sources) != len(FINAL_W18_TEACHERS)
        or metadata.get("homogeneous_family") != FINAL_W18_HOMOGENEOUS_FAMILY
    ):
        raise ValueError("final W18 cache differs from the frozen six-teacher recipe")
    actual_coefficients: list[float] = []
    for index, (expected, coefficient, source) in enumerate(
        zip(FINAL_W18_TEACHERS, coefficients, sources, strict=True)
    ):
        if not isinstance(coefficient, Mapping) or not isinstance(source, Mapping):
            raise ValueError("final W18 cache teacher records are malformed")
        run, family, coefficient_value, epoch = expected
        for record in (coefficient, source):
            value = record.get("coefficient")
            if (
                record.get("run") != run
                or record.get("family") != family
                or record.get("epoch") != epoch
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or float(value) != coefficient_value
            ):
                raise ValueError(f"final W18 teacher {index} differs from {run}")
        actual_coefficients.append(float(coefficient["coefficient"]))
    if math.fsum(actual_coefficients) != 0.8125 or metadata.get("coefficient_sum") != 0.8125:
        raise ValueError("final W18 cache coefficient sum differs from the frozen 0.8125 total")

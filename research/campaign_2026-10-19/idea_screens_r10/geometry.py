"""Causal rolling box and convex-hull summaries for one fixed 2D coordinate pair."""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

WINDOW_ROWS = 256
SEQUENCE_ROWS = 20_000
ROW_STRIDE = 40
PAIR_COLUMNS = (0, 52)
BOUNDARY_TOLERANCE = 1e-12
Category = Literal["box_outside", "box_inside_hull_outside", "hull_inside"]
HullKind = Literal["point", "line", "area"]


class WindowGeometry(NamedTuple):
    """Fixed scalar description of one query against its causal window."""

    category: Category
    hull_vertices: int
    hull_kind: HullKind
    hull_box_area_ratio: float | None


def query_rows(scored: np.ndarray) -> np.ndarray:
    """Return fixed stride-40 fit queries after a complete 256-row history."""
    mask = np.asarray(scored)
    if mask.dtype != np.dtype(np.bool_) or mask.shape != (SEQUENCE_ROWS,):
        raise ValueError("expected the original boolean scored mask for one 20,000-row sequence")
    stride_rows = np.arange(0, SEQUENCE_ROWS, ROW_STRIDE, dtype=np.int32)
    selected = stride_rows[(stride_rows >= WINDOW_ROWS) & mask[stride_rows]]
    if selected.shape != (493,):
        raise ValueError(f"expected 493 eligible stride rows, got {selected.size}")
    return selected


def classify_at(coordinates: np.ndarray, row: int) -> WindowGeometry:
    """Classify one current pair using only its immediately preceding 256 rows."""
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or row < WINDOW_ROWS or row >= len(points):
        raise ValueError("expected a valid row and a (T,2) coordinate sequence")
    return classify_window(points[row - WINDOW_ROWS : row], points[row])


def classify_window(history: np.ndarray, query: np.ndarray) -> WindowGeometry:
    """Compare a finite query with the same window's axis box and convex hull."""
    points = np.asarray(history, dtype=np.float64)
    current = np.asarray(query, dtype=np.float64)
    if points.shape != (WINDOW_ROWS, 2) or current.shape != (2,):
        raise ValueError("expected a 256-by-2 history and one 2D query")
    if not np.isfinite(points).all() or not np.isfinite(current).all():
        raise ValueError("selected support coordinates must be finite")

    low = points.min(axis=0)
    spans = points.max(axis=0) - low
    normalized = np.zeros_like(points)
    normalized_query = np.zeros(2, dtype=np.float64)
    in_box = True
    for axis, span in enumerate(spans):
        if span == 0.0:
            if current[axis] != low[axis]:
                in_box = False
        else:
            normalized[:, axis] = (points[:, axis] - low[axis]) / span
            normalized_query[axis] = (current[axis] - low[axis]) / span
            if (
                normalized_query[axis] < -BOUNDARY_TOLERANCE
                or normalized_query[axis] > 1.0 + BOUNDARY_TOLERANCE
            ):
                in_box = False

    hull = _convex_hull(normalized)
    hull_kind: HullKind = "point" if len(hull) == 1 else "line" if len(hull) == 2 else "area"
    area_ratio = _polygon_area(hull) if bool(np.all(spans > 0.0)) else None
    inside_hull = _inside_hull(normalized_query, hull)
    category: Category
    if not in_box:
        category = "box_outside"
    elif inside_hull:
        category = "hull_inside"
    else:
        category = "box_inside_hull_outside"
    return WindowGeometry(category, len(hull), hull_kind, area_ratio)


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Return the signed 2D area for the ordered triple."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    """Build a monotone-chain hull, removing fixed-tolerance collinear turns."""
    ordered = sorted({(float(x), float(y)) for x, y in points})
    if len(ordered) <= 1:
        return ordered
    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= BOUNDARY_TOLERANCE:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= BOUNDARY_TOLERANCE:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return hull if hull else ordered[:1]


def _inside_hull(query: np.ndarray, hull: list[tuple[float, float]]) -> bool:
    """Test inclusive point, segment, or counter-clockwise polygon membership."""
    qx, qy = float(query[0]), float(query[1])
    if len(hull) == 1:
        return max(abs(qx - hull[0][0]), abs(qy - hull[0][1])) <= BOUNDARY_TOLERANCE
    if len(hull) == 2:
        a, b = hull
        cross = _cross(a, b, (qx, qy))
        projection = (qx - a[0]) * (b[0] - a[0]) + (qy - a[1]) * (b[1] - a[1])
        length_squared = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
        return (
            abs(cross) <= BOUNDARY_TOLERANCE
            and -BOUNDARY_TOLERANCE <= projection <= length_squared + BOUNDARY_TOLERANCE
        )
    point = (qx, qy)
    return all(
        _cross(hull[index], hull[(index + 1) % len(hull)], point) >= -BOUNDARY_TOLERANCE
        for index in range(len(hull))
    )


def _polygon_area(hull: list[tuple[float, float]]) -> float:
    """Return the convex hull area in the normalized unit box."""
    if len(hull) < 3:
        return 0.0
    twice_area = sum(
        hull[index][0] * hull[(index + 1) % len(hull)][1]
        - hull[(index + 1) % len(hull)][0] * hull[index][1]
        for index in range(len(hull))
    )
    return 0.5 * abs(twice_area)

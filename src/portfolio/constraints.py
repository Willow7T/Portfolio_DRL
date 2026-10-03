"""Fast deterministic portfolio-weight projections."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def project_bounded_simplex(
    values: np.ndarray,
    caps: np.ndarray,
    total: float = 1.0,
) -> np.ndarray:
    """Project non-negative scores proportionally onto a capped simplex.

    This is a deterministic water-filling projection. It preserves relative
    scores among unconstrained entries and is inexpensive enough to call at
    every environment step.
    """
    scores = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    limits = np.asarray(caps, dtype=np.float64).reshape(-1)
    if scores.shape != limits.shape:
        raise ValueError("Values and caps must have the same shape.")
    if total < 0 or np.any(limits < 0):
        raise ValueError("Total and caps must be non-negative.")
    if limits.sum() < total - 1e-10:
        raise ValueError("Weight caps are infeasible for the requested total.")

    result = np.zeros_like(scores)
    active = limits > 1e-12
    remaining = float(total)
    while active.any() and remaining > 1e-12:
        active_scores = scores[active]
        if active_scores.sum() <= 1e-12:
            proposed = np.full(active.sum(), remaining / active.sum())
        else:
            proposed = active_scores / active_scores.sum() * remaining
        active_indices = np.flatnonzero(active)
        room = limits[active_indices] - result[active_indices]
        saturated = proposed >= room - 1e-12
        if not saturated.any():
            result[active_indices] += proposed
            remaining = 0.0
            break
        saturated_indices = active_indices[saturated]
        result[saturated_indices] = limits[saturated_indices]
        remaining = total - result.sum()
        active[saturated_indices] = False

    if remaining > 1e-8:
        raise RuntimeError("Unable to satisfy portfolio weight constraints.")
    correction = total - result.sum()
    if abs(correction) > 1e-12:
        candidates = np.flatnonzero(result < limits - 1e-10)
        if candidates.size:
            result[candidates[0]] += correction
    return result


def project_portfolio_weights(
    values: np.ndarray,
    max_weight: float = 1.0,
    sector_labels: Sequence[str] | None = None,
    max_sector_weight: float = 1.0,
) -> np.ndarray:
    """Normalise scores subject to individual and optional sector caps."""
    scores = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    if scores.size == 0:
        raise ValueError("At least one asset is required.")
    if not 0 < max_weight <= 1 or not 0 < max_sector_weight <= 1:
        raise ValueError("Weight caps must be in (0, 1].")
    if scores.sum() <= 1e-12:
        scores.fill(1.0)

    if not sector_labels:
        return project_bounded_simplex(
            scores,
            np.full(scores.size, max_weight),
        ).astype(np.float32)
    if len(sector_labels) != scores.size:
        raise ValueError("Sector labels must align with asset scores.")

    sectors = list(dict.fromkeys(sector_labels))
    group_indices = [
        np.asarray([i for i, label in enumerate(sector_labels) if label == sector])
        for sector in sectors
    ]
    group_scores = np.asarray([scores[index].sum() for index in group_indices])
    group_caps = np.asarray(
        [min(max_sector_weight, len(index) * max_weight) for index in group_indices]
    )
    group_weights = project_bounded_simplex(group_scores, group_caps)

    result = np.zeros_like(scores)
    for target, indices in zip(group_weights, group_indices, strict=True):
        result[indices] = project_bounded_simplex(
            scores[indices],
            np.full(len(indices), max_weight),
            total=float(target),
        )
    return result.astype(np.float32)

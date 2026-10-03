"""Regime-level reports and dependence-aware paired bootstrap utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..portfolio.metrics import extended_performance_summary, sharpe_ratio


def regime_performance_table(
    returns: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Calculate every strategy metric within each causal threshold regime."""
    aligned_regimes = regimes.reindex(returns.index)
    records = []
    for regime in aligned_regimes.dropna().unique():
        mask = aligned_regimes == regime
        for strategy in returns.columns:
            metrics = extended_performance_summary(
                returns.loc[mask, strategy].to_numpy()
            )
            records.append({"regime": regime, "strategy": strategy, **metrics})
    return pd.DataFrame(records).set_index(["regime", "strategy"]).sort_index()


def paired_block_bootstrap(
    strategy_returns: pd.Series,
    baseline_returns: pd.Series,
    block_size: int = 20,
    samples: int = 2_000,
    random_seed: int = 42,
) -> dict[str, float]:
    """Bootstrap paired return and Sharpe differences with moving blocks."""
    paired = pd.concat([strategy_returns, baseline_returns], axis=1).dropna()
    values = paired.to_numpy(dtype=float)
    if len(values) < block_size * 2:
        raise ValueError("Return history is too short for the selected block size.")
    generator = np.random.default_rng(random_seed)
    maximum_start = len(values) - block_size
    number_of_blocks = int(np.ceil(len(values) / block_size))
    mean_differences = np.empty(samples)
    sharpe_differences = np.empty(samples)
    for sample_index in range(samples):
        starts = generator.integers(0, maximum_start + 1, size=number_of_blocks)
        indices = np.concatenate(
            [np.arange(start, start + block_size) for start in starts]
        )[: len(values)]
        sampled = values[indices]
        mean_differences[sample_index] = np.mean(sampled[:, 0] - sampled[:, 1])
        sharpe_differences[sample_index] = (
            sharpe_ratio(sampled[:, 0]) - sharpe_ratio(sampled[:, 1])
        )

    return {
        "daily_mean_difference": float(np.mean(values[:, 0] - values[:, 1])),
        "daily_mean_ci_lower": float(np.quantile(mean_differences, 0.025)),
        "daily_mean_ci_upper": float(np.quantile(mean_differences, 0.975)),
        "sharpe_difference": float(
            sharpe_ratio(values[:, 0]) - sharpe_ratio(values[:, 1])
        ),
        "sharpe_ci_lower": float(np.quantile(sharpe_differences, 0.025)),
        "sharpe_ci_upper": float(np.quantile(sharpe_differences, 0.975)),
    }

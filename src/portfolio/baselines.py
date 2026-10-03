"""Classical portfolio-allocation baselines."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .constraints import project_portfolio_weights


def equal_weight_weights(number_of_assets: int) -> np.ndarray:
    """Return a long-only equal-weight portfolio."""
    if number_of_assets < 1:
        raise ValueError("At least one asset is required.")
    return np.full(number_of_assets, 1 / number_of_assets)


def markowitz_weights(
    train_returns: pd.DataFrame,
    risk_aversion: float = 5.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Fit a long-only mean-variance portfolio using training data only."""
    if not 0 < max_weight <= 1:
        raise ValueError("Maximum weight must be in (0, 1].")
    if max_weight * train_returns.shape[1] < 1 - 1e-12:
        raise ValueError("Maximum weight is infeasible for this asset count.")
    mean_returns = train_returns.mean().to_numpy(dtype=float) * 252
    covariance = train_returns.cov().to_numpy(dtype=float) * 252
    covariance += np.eye(len(mean_returns)) * 1e-8

    weights = cp.Variable(len(mean_returns))
    objective = cp.Maximize(
        mean_returns @ weights
        - risk_aversion * cp.quad_form(weights, cp.psd_wrap(covariance))
    )
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if weights.value is None:
        raise RuntimeError("Markowitz optimisation did not converge.")

    result = np.maximum(np.asarray(weights.value).reshape(-1), 0)
    return result / result.sum()


def minimum_variance_weights(
    train_returns: pd.DataFrame,
    max_weight: float = 1.0,
    sector_labels: tuple[str, ...] = (),
    max_sector_weight: float = 1.0,
) -> np.ndarray:
    """Estimate shrinkage minimum-variance weights from past returns."""
    covariance = LedoitWolf().fit(train_returns.to_numpy()).covariance_ * 252
    inverse = np.linalg.pinv(covariance) @ np.ones(train_returns.shape[1])
    return project_portfolio_weights(
        np.maximum(inverse, 0),
        max_weight=max_weight,
        sector_labels=sector_labels,
        max_sector_weight=max_sector_weight,
    )


def inverse_volatility_weights(
    train_returns: pd.DataFrame,
    max_weight: float = 1.0,
    sector_labels: tuple[str, ...] = (),
    max_sector_weight: float = 1.0,
) -> np.ndarray:
    """Return a transparent inverse-volatility risk-allocation baseline."""
    volatility = train_returns.std().to_numpy(dtype=float)
    scores = 1 / np.maximum(volatility, 1e-8)
    return project_portfolio_weights(
        scores,
        max_weight=max_weight,
        sector_labels=sector_labels,
        max_sector_weight=max_sector_weight,
    )


def shrinkage_markowitz_weights(
    train_returns: pd.DataFrame,
    risk_aversion: float = 5.0,
    max_weight: float = 1.0,
    sector_labels: tuple[str, ...] = (),
    max_sector_weight: float = 1.0,
) -> np.ndarray:
    """Use covariance shrinkage and proportional constrained projection."""
    mean_returns = train_returns.mean().to_numpy(dtype=float) * 252
    covariance = LedoitWolf().fit(train_returns.to_numpy()).covariance_ * 252
    raw = np.linalg.pinv(covariance) @ mean_returns / max(risk_aversion, 1e-8)
    return project_portfolio_weights(
        np.maximum(raw, 0),
        max_weight=max_weight,
        sector_labels=sector_labels,
        max_sector_weight=max_sector_weight,
    )


def momentum_weights(
    train_returns: pd.DataFrame,
    max_weight: float = 1.0,
    sector_labels: tuple[str, ...] = (),
    max_sector_weight: float = 1.0,
) -> np.ndarray:
    """Allocate to positive 12-month momentum scaled by recent volatility."""
    history = train_returns.tail(252)
    momentum = (1 + history).prod().to_numpy(dtype=float) - 1
    volatility = history.tail(63).std().to_numpy(dtype=float) * np.sqrt(252)
    scores = np.maximum(momentum, 0) / np.maximum(volatility, 1e-8)
    if scores.sum() <= 1e-12:
        scores = 1 / np.maximum(volatility, 1e-8)
    return project_portfolio_weights(
        scores,
        max_weight=max_weight,
        sector_labels=sector_labels,
        max_sector_weight=max_sector_weight,
    )


ROLLING_METHODS = {
    "Minimum Variance": minimum_variance_weights,
    "Inverse Volatility": inverse_volatility_weights,
    "Shrinkage Markowitz": shrinkage_markowitz_weights,
    "Momentum": momentum_weights,
}


def rolling_allocation_backtest(
    train_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    method: str,
    lookback: int = 252,
    rebalance_frequency: int = 5,
    transaction_cost: float = 0.0,
    max_weight: float = 1.0,
    sector_labels: tuple[str, ...] = (),
    max_sector_weight: float = 1.0,
) -> pd.Series:
    """Backtest a rolling estimator without using future evaluation rows."""
    if method not in ROLLING_METHODS:
        raise ValueError(f"Unknown rolling baseline: {method}")
    if rebalance_frequency < 1 or lookback < 2:
        raise ValueError("Lookback and rebalance frequency must be positive.")
    combined = pd.concat([train_returns, test_returns]).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    evaluation_start = test_returns.index.min()
    current = equal_weight_weights(combined.shape[1])
    realised: list[float] = []
    dates: list[pd.Timestamp] = []
    turnovers: list[float] = []
    costs: list[float] = []
    concentrations: list[float] = []
    days_since_rebalance = rebalance_frequency

    for index, (date, row) in enumerate(combined.iterrows()):
        cost = 0.0
        day_turnover = 0.0
        if (
            date >= evaluation_start
            and index >= lookback
            and days_since_rebalance >= rebalance_frequency
        ):
            estimation = combined.iloc[max(0, index - lookback) : index]
            target = ROLLING_METHODS[method](
                estimation,
                max_weight=max_weight,
                sector_labels=sector_labels,
                max_sector_weight=max_sector_weight,
            )
            day_turnover = float(np.abs(target - current).sum())
            cost = transaction_cost * day_turnover
            current = target
            days_since_rebalance = 0

        asset_returns = row.to_numpy(dtype=float)
        net_return = float(current @ asset_returns - cost)
        gross_positions = current * (1 + asset_returns)
        gross_total = gross_positions.sum()
        current = (
            gross_positions / gross_total if gross_total > 0 else current
        )
        if date >= evaluation_start:
            realised.append(net_return)
            dates.append(date)
            turnovers.append(day_turnover)
            costs.append(cost)
            concentrations.append(float(np.square(current).sum()))
            days_since_rebalance += 1

    result = pd.Series(realised, index=dates, name="portfolio_return")
    result.attrs = {
        "mean_daily_turnover": float(np.mean(turnovers)),
        "total_trading_cost": float(np.sum(costs)),
        "mean_concentration": float(np.mean(concentrations)),
    }
    return result


def buy_and_hold_equal_weight(test_returns: pd.DataFrame) -> pd.Series:
    """Buy equal weights once and allow them to drift without rebalancing."""
    current = equal_weight_weights(test_returns.shape[1])
    realised = []
    concentrations = []
    for row in test_returns.to_numpy(dtype=float):
        realised.append(float(current @ row))
        concentrations.append(float(np.square(current).sum()))
        gross_positions = current * (1 + row)
        current = gross_positions / gross_positions.sum()
    result = pd.Series(realised, index=test_returns.index, name="portfolio_return")
    result.attrs = {
        "mean_daily_turnover": 0.0,
        "total_trading_cost": 0.0,
        "mean_concentration": float(np.mean(concentrations)),
    }
    return result


def fixed_weight_backtest(
    returns: pd.DataFrame,
    weights: np.ndarray,
    transaction_cost: float = 0.0,
    rebalance_frequency: int = 1,
) -> pd.Series:
    """Backtest daily rebalancing to fixed target weights."""
    target = np.asarray(weights, dtype=float)
    if len(target) != returns.shape[1]:
        raise ValueError("Weight count must match the number of assets.")

    current = equal_weight_weights(len(target))
    realised_returns = []
    turnovers = []
    costs = []
    concentrations = []
    for day, asset_returns in enumerate(returns.to_numpy(dtype=float)):
        rebalance = day % rebalance_frequency == 0
        turnover = float(np.abs(target - current).sum()) if rebalance else 0.0
        if rebalance:
            current = target
        net_return = float(current @ asset_returns - transaction_cost * turnover)
        realised_returns.append(net_return)
        turnovers.append(turnover)
        costs.append(transaction_cost * turnover)
        concentrations.append(float(np.square(current).sum()))

        gross_positions = current * (1 + asset_returns)
        gross_value = gross_positions.sum()
        current = (
            gross_positions / gross_value
            if gross_value > 0
            else current.copy()
        )

    result = pd.Series(
        realised_returns,
        index=returns.index,
        name="portfolio_return",
    )
    result.attrs = {
        "mean_daily_turnover": float(np.mean(turnovers)),
        "total_trading_cost": float(np.sum(costs)),
        "mean_concentration": float(np.mean(concentrations)),
    }
    return result

import numpy as np


def _clean_returns(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate the annualised Sharpe ratio from daily returns."""
    values = _clean_returns(returns)
    if values.size < 2:
        return float("nan")

    daily_risk_free_rate = risk_free_rate / 252
    excess_returns = values - daily_risk_free_rate
    volatility = np.std(excess_returns, ddof=1)

    if volatility == 0:
        return float("nan")

    return float(np.mean(excess_returns) / volatility * np.sqrt(252))


def annualised_return(returns: np.ndarray) -> float:
    """Calculate geometric annualised return from daily returns."""
    values = _clean_returns(returns)
    if values.size == 0:
        return float("nan")
    growth = np.prod(1 + values)
    return float(growth ** (252 / values.size) - 1)


def annualised_volatility(returns: np.ndarray) -> float:
    """Calculate annualised standard deviation of daily returns."""
    values = _clean_returns(returns)
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) * np.sqrt(252))


def maximum_drawdown(returns: np.ndarray) -> float:
    """Return the largest peak-to-trough loss as a positive fraction."""
    values = _clean_returns(returns)
    if values.size == 0:
        return float("nan")
    wealth = np.cumprod(1 + values)
    peaks = np.maximum.accumulate(wealth)
    drawdowns = 1 - wealth / peaks
    return float(np.max(drawdowns))


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate annualised Sortino ratio using downside deviation."""
    values = _clean_returns(returns)
    if values.size < 2:
        return float("nan")
    daily_risk_free_rate = risk_free_rate / 252
    excess = values - daily_risk_free_rate
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("nan")
    downside_deviation = np.sqrt(np.mean(np.square(downside)))
    if downside_deviation == 0:
        return float("nan")
    return float(np.mean(excess) / downside_deviation * np.sqrt(252))


def performance_summary(returns: np.ndarray) -> dict[str, float]:
    """Create the principal out-of-sample performance statistics."""
    values = _clean_returns(returns)
    total_return = float(np.prod(1 + values) - 1) if values.size else float("nan")
    return {
        "total_return": total_return,
        "annualised_return": annualised_return(values),
        "annualised_volatility": annualised_volatility(values),
        "sharpe_ratio": sharpe_ratio(values),
        "sortino_ratio": sortino_ratio(values),
        "maximum_drawdown": maximum_drawdown(values),
    }


def extended_performance_summary(returns: np.ndarray) -> dict[str, float]:
    """Add Calmar ratio for the expanded research reports."""
    summary = performance_summary(returns)
    drawdown = summary["maximum_drawdown"]
    summary["calmar_ratio"] = (
        float(summary["annualised_return"] / drawdown)
        if np.isfinite(drawdown) and drawdown > 0
        else float("nan")
    )
    return summary

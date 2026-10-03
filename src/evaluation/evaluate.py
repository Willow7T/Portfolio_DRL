"""Evaluate trained DRL agents and classical baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN, PPO

from ..environments.portfolio_env import DiscretePortfolioEnv, PortfolioEnv
from ..portfolio.baselines import (
    equal_weight_weights,
    fixed_weight_backtest,
    markowitz_weights,
)
from ..portfolio.metrics import performance_summary
from ..utils.config import DEFAULT_CONFIG, ExperimentConfig


def _environment_kwargs(config: ExperimentConfig) -> dict[str, float | int]:
    return {
        "lookback_window": config.lookback_window,
        "initial_balance": config.initial_balance,
        "transaction_cost": config.transaction_cost,
        "slippage_cost": config.slippage_cost,
        "turnover_penalty": config.turnover_penalty,
        "max_weight": config.max_weight,
        "drawdown_limit": config.drawdown_limit,
        "drawdown_penalty": config.drawdown_penalty,
        "sharpe_reward_weight": config.sharpe_reward_weight,
        "rebalance_frequency": config.rebalance_frequency,
        "sector_labels": config.sector_labels,
        "max_sector_weight": config.max_sector_weight,
        "action_temperature": config.action_temperature,
        "action_smoothing": config.action_smoothing,
        "minimum_rebalance_turnover": config.minimum_rebalance_turnover,
        "maximum_rebalance_turnover": config.maximum_rebalance_turnover,
        "continuous_action_mode": config.continuous_action_mode,
    }


def evaluate_agent(
    model: PPO | DQN,
    environment: PortfolioEnv | DiscretePortfolioEnv,
) -> pd.Series:
    """Run a deterministic agent through one complete test episode."""
    observation, _ = environment.reset()
    terminated = False

    while not terminated:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = environment.step(action)
        terminated = terminated or truncated

    return pd.Series(
        environment.portfolio_returns,
        index=environment.return_dates,
        name="portfolio_return",
    )


def compare_strategies(
    train_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    config: ExperimentConfig = DEFAULT_CONFIG,
    test_features: pd.DataFrame | None = None,
    include_dqn: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate PPO, DQN, equal weight, and Markowitz out of sample."""
    ppo = PPO.load(config.model_directory / "ppo_portfolio")
    environment_kwargs = _environment_kwargs(config)

    strategy_returns: dict[str, pd.Series] = {
        "PPO": evaluate_agent(
            ppo,
            PortfolioEnv(
                test_returns,
                features=test_features,
                **environment_kwargs,
            ),
        ),
        "Equal Weight": fixed_weight_backtest(
            test_returns,
            equal_weight_weights(test_returns.shape[1]),
            config.transaction_cost + config.slippage_cost,
        ),
        "Markowitz": fixed_weight_backtest(
            test_returns,
            markowitz_weights(
                train_returns,
                max_weight=config.max_weight,
            ),
            config.transaction_cost + config.slippage_cost,
        ),
    }
    if include_dqn:
        dqn = DQN.load(config.model_directory / "dqn_portfolio")
        strategy_returns["DQN"] = evaluate_agent(
            dqn,
            DiscretePortfolioEnv(
                test_returns,
                features=test_features,
                **environment_kwargs,
            ),
        )
    returns_frame = pd.concat(strategy_returns, axis=1).dropna()
    metrics = pd.DataFrame(
        {
            name: performance_summary(series.to_numpy())
            for name, series in returns_frame.items()
        }
    ).T
    return returns_frame, metrics


def save_evaluation(
    returns_frame: pd.DataFrame,
    metrics: pd.DataFrame,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> dict[str, Path]:
    """Save tables and a cumulative-performance comparison figure."""
    config.report_directory.mkdir(parents=True, exist_ok=True)
    config.figure_directory.mkdir(parents=True, exist_ok=True)

    returns_path = config.report_directory / "out_of_sample_returns.csv"
    metrics_path = config.report_directory / "performance_metrics.csv"
    figure_path = config.figure_directory / "strategy_comparison.png"
    returns_frame.to_csv(returns_path)
    metrics.to_csv(metrics_path)

    growth = (1 + returns_frame).cumprod()
    axis = growth.plot(figsize=(10, 6))
    axis.set_title("Out-of-Sample Portfolio Performance")
    axis.set_ylabel("Growth of £1")
    axis.set_xlabel("Date")
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()
    return {
        "returns": returns_path,
        "metrics": metrics_path,
        "figure": figure_path,
    }

"""Evaluation for the expanded multi-regime equity experiment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN, PPO

from ..data.splits import expanding_walk_forward_splits
from ..environments.portfolio_env import DiscretePortfolioEnv, PortfolioEnv
from ..portfolio.baselines import (
    ROLLING_METHODS,
    buy_and_hold_equal_weight,
    equal_weight_weights,
    fixed_weight_backtest,
    rolling_allocation_backtest,
)
from ..portfolio.metrics import extended_performance_summary
from ..utils.config import ExperimentConfig
from .evaluate import _environment_kwargs, evaluate_agent
from .statistics import paired_block_bootstrap, regime_performance_table


def _expanded_baselines(
    train_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, pd.Series]:
    total_cost = config.transaction_cost + config.slippage_cost
    strategies: dict[str, pd.Series] = {
        "Equal Weight": fixed_weight_backtest(
            test_returns,
            equal_weight_weights(test_returns.shape[1]),
            total_cost,
            rebalance_frequency=config.rebalance_frequency,
        ),
        "Buy and Hold Equal Weight": buy_and_hold_equal_weight(test_returns),
    }
    for method in ROLLING_METHODS:
        strategies[method] = rolling_allocation_backtest(
            train_returns,
            test_returns,
            method=method,
            lookback=252,
            rebalance_frequency=config.rebalance_frequency,
            transaction_cost=total_cost,
            max_weight=config.max_weight,
            sector_labels=config.sector_labels,
            max_sector_weight=config.max_sector_weight,
        )
    if "SPY" in test_returns.columns:
        strategies["SPY Market Proxy"] = test_returns["SPY"].rename(
            "portfolio_return"
        )
        strategies["SPY Market Proxy"].attrs = {
            "mean_daily_turnover": 0.0,
            "total_trading_cost": 0.0,
            "mean_concentration": 1.0,
        }
    return strategies


def compare_expanded_strategies(
    train_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    test_features: pd.DataFrame,
    test_regimes: pd.Series,
    config: ExperimentConfig,
    include_dqn: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare PPO with strong rolling baselines and statistical evidence."""
    model = PPO.load(
        config.model_directory / "ppo_portfolio",
        device=config.ppo_device or config.device,
    )
    environment = PortfolioEnv(
        test_returns,
        features=test_features,
        **_environment_kwargs(config),
    )
    strategies = {"PPO": evaluate_agent(model, environment)}
    agent_environments: dict[str, PortfolioEnv] = {"PPO": environment}
    if include_dqn:
        dqn = DQN.load(
            config.model_directory / "dqn_portfolio",
            device=config.dqn_device or config.device,
        )
        dqn_environment = DiscretePortfolioEnv(
            test_returns,
            features=test_features,
            **_environment_kwargs(config),
        )
        strategies["DQN"] = evaluate_agent(dqn, dqn_environment)
        agent_environments["DQN"] = dqn_environment
    strategies.update(_expanded_baselines(train_returns, test_returns, config))
    diagnostics = {
        name: series.attrs.copy() for name, series in strategies.items()
    }
    for agent_name, agent_environment in agent_environments.items():
        diagnostics[agent_name] = {
            "mean_daily_turnover": float(
                pd.Series(agent_environment.turnover_history).mean()
            ),
            "total_trading_cost": float(sum(agent_environment.cost_history)),
            "mean_concentration": float(
                pd.DataFrame(agent_environment.target_weight_history)
                .pow(2)
                .sum(axis=1)
                .mean()
            ),
        }
    returns_frame = pd.concat(strategies, axis=1).dropna()
    metrics = pd.DataFrame(
        {
            name: extended_performance_summary(series.to_numpy())
            for name, series in returns_frame.items()
        }
    ).T.sort_values("sharpe_ratio", ascending=False)
    for column in (
        "mean_daily_turnover",
        "total_trading_cost",
        "mean_concentration",
    ):
        metrics[column] = pd.Series(
            {name: values.get(column, float("nan")) for name, values in diagnostics.items()}
        )
    regimes = regime_performance_table(returns_frame, test_regimes)

    baseline_metrics = metrics.drop(index=["PPO", "DQN"], errors="ignore")
    best_baseline = baseline_metrics["sharpe_ratio"].idxmax()
    bootstrap_records = []
    for agent_name in agent_environments:
        bootstrap_records.append(
            {
                "strategy": agent_name,
                "baseline": best_baseline,
                **paired_block_bootstrap(
                    returns_frame[agent_name],
                    returns_frame[best_baseline],
                ),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_records).set_index(
        ["strategy", "baseline"]
    )
    return returns_frame, metrics, regimes, bootstrap


def baseline_walk_forward_report(
    returns: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Evaluate strong baselines across the four predeclared development folds."""
    records = []
    for fold_index, (train, validation, test) in enumerate(
        expanding_walk_forward_splits(returns),
        start=1,
    ):
        estimation = pd.concat([train, validation])
        strategies = _expanded_baselines(estimation, test, config)
        aligned = pd.concat(strategies, axis=1).dropna()
        for strategy, series in aligned.items():
            records.append(
                {
                    "fold": fold_index,
                    "strategy": strategy,
                    "test_start": test.index.min(),
                    "test_end": test.index.max(),
                    **extended_performance_summary(series.to_numpy()),
                }
            )
    return pd.DataFrame(records).set_index(["fold", "strategy"]).sort_index()


def cost_sensitivity_report(
    train_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    test_features: pd.DataFrame,
    config: ExperimentConfig,
    costs: tuple[float, ...] = (0.0005, 0.0015, 0.0025),
    include_dqn: bool = True,
) -> pd.DataFrame:
    """Re-evaluate the frozen PPO and baselines under declared cost levels."""
    model = PPO.load(
        config.model_directory / "ppo_portfolio",
        device=config.ppo_device or config.device,
    )
    dqn = (
        DQN.load(
            config.model_directory / "dqn_portfolio",
            device=config.dqn_device or config.device,
        )
        if include_dqn
        else None
    )
    records = []
    for cost in costs:
        cost_config = replace(config, transaction_cost=cost, slippage_cost=0.0)
        environment = PortfolioEnv(
            test_returns,
            features=test_features,
            **_environment_kwargs(cost_config),
        )
        strategies = {"PPO": evaluate_agent(model, environment)}
        if dqn is not None:
            dqn_environment = DiscretePortfolioEnv(
                test_returns,
                features=test_features,
                **_environment_kwargs(cost_config),
            )
            strategies["DQN"] = evaluate_agent(dqn, dqn_environment)
        strategies.update(
            _expanded_baselines(train_returns, test_returns, cost_config)
        )
        aligned = pd.concat(strategies, axis=1).dropna()
        for strategy, series in aligned.items():
            records.append(
                {
                    "total_cost": cost,
                    "strategy": strategy,
                    **extended_performance_summary(series.to_numpy()),
                }
            )
    return pd.DataFrame(records).set_index(["total_cost", "strategy"])


def save_expanded_evaluation(
    returns: pd.DataFrame,
    metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    walk_forward: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, Path]:
    """Save every table used by the expanded experiment conclusion."""
    config.report_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "returns": config.report_directory / "out_of_sample_returns.csv",
        "metrics": config.report_directory / "performance_metrics.csv",
        "regimes": config.report_directory / "regime_metrics.csv",
        "bootstrap": config.report_directory / "paired_bootstrap.csv",
        "walk_forward": config.report_directory / "walk_forward_baselines.csv",
        "costs": config.report_directory / "cost_sensitivity.csv",
    }
    returns.to_csv(paths["returns"])
    metrics.to_csv(paths["metrics"])
    regime_metrics.to_csv(paths["regimes"])
    bootstrap.to_csv(paths["bootstrap"])
    walk_forward.to_csv(paths["walk_forward"])
    cost_sensitivity.to_csv(paths["costs"])
    return paths

"""End-to-end DRL portfolio experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.agents.train_agents import train_dqn, train_ppo
    from src.data.splits import chronological_split
    from src.evaluation.evaluate import compare_strategies, save_evaluation
    from src.utils.config import DEFAULT_CONFIG
else:
    from .agents.train_agents import train_dqn, train_ppo
    from .data.splits import chronological_split
    from .evaluation.evaluate import compare_strategies, save_evaluation
    from .utils.config import DEFAULT_CONFIG


def run_experiment(skip_training: bool = False) -> None:
    """Train agents and produce an out-of-sample benchmark report."""
    config = DEFAULT_CONFIG
    returns = pd.read_csv(
        config.returns_path,
        index_col=0,
        parse_dates=True,
    ).dropna()
    train, validation, test = chronological_split(
        returns,
        train_end=config.train_end,
        validation_end=config.validation_end,
    )

    print(
        f"Observations — train: {len(train)}, validation: {len(validation)}, "
        f"test: {len(test)}"
    )

    if not skip_training:
        train_ppo(train, validation, config)
        train_dqn(train, validation, config)

    strategy_returns, metrics = compare_strategies(train, test, config)
    paths = save_evaluation(strategy_returns, metrics, config)
    print(metrics.round(4))
    print("Saved evaluation outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Evaluate previously trained PPO and DQN models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_experiment(skip_training=arguments.skip_training)

"""Expanded, regime-aware DRL portfolio experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .agents.train_agents import train_dqn, train_ppo
from .data.download_data import download_market_data
from .data.feature_engineering import (
    RobustFeatureScaler,
    create_robust_market_features,
)
from .data.preprocess import preprocess_market_data
from .data.splits import chronological_split
from .evaluation.evaluate import compare_strategies, save_evaluation
from .utils.config import (
    END_DATE,
    ROBUST_CONFIG,
    ROBUST_START_DATE,
    ROBUST_TICKERS,
)

RAW_PRICES_PATH = Path("data/raw/robust_asset_prices.csv")
FEATURES_PATH = Path("data/processed/robust_features.csv")
SCALER_PATH = Path("results/models/robust/feature_scaler.json")


def prepare_robust_data() -> None:
    """Download the fixed ETF universe and create past-only market features."""
    download_market_data(
        ROBUST_TICKERS,
        ROBUST_START_DATE,
        END_DATE,
        RAW_PRICES_PATH,
    )
    returns = preprocess_market_data(
        RAW_PRICES_PATH,
        ROBUST_CONFIG.returns_path,
    )
    features = create_robust_market_features(returns)
    aligned_returns = returns.loc[features.index]
    aligned_returns.to_csv(ROBUST_CONFIG.returns_path)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH)
    print(
        f"Robust dataset: {len(aligned_returns):,} observations, "
        f"{aligned_returns.shape[1]} assets, {features.shape[1]} features."
    )


def _save_scaler(
    scaler: RobustFeatureScaler,
    feature_columns: list[str],
) -> None:
    if scaler.median is None or scaler.scale is None:
        raise RuntimeError("Cannot save an unfitted feature scaler.")
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_value": scaler.clip_value,
        "feature_columns": feature_columns,
        "median": scaler.median.to_dict(),
        "scale": scaler.scale.to_dict(),
        "fit_end": ROBUST_CONFIG.train_end,
    }
    SCALER_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_robust_experiment(
    prepare_data: bool = False,
    prepare_only: bool = False,
    skip_training: bool = False,
    ppo_only: bool = False,
) -> None:
    """Train robust candidates and evaluate once on the held-out test period."""
    if prepare_data or prepare_only:
        prepare_robust_data()
    if prepare_only:
        return

    returns = pd.read_csv(
        ROBUST_CONFIG.returns_path,
        index_col=0,
        parse_dates=True,
    ).dropna()
    features = pd.read_csv(
        FEATURES_PATH,
        index_col=0,
        parse_dates=True,
    ).dropna()
    common_index = returns.index.intersection(features.index)
    returns = returns.loc[common_index]
    features = features.loc[common_index]

    train, validation, test = chronological_split(
        returns,
        train_end=ROBUST_CONFIG.train_end,
        validation_end=ROBUST_CONFIG.validation_end,
    )
    train_features = features.loc[train.index]
    validation_features = features.loc[validation.index]
    test_features = features.loc[test.index]

    scaler = RobustFeatureScaler(clip_value=5.0)
    train_features = scaler.fit_transform(train_features)
    validation_features = scaler.transform(validation_features)
    test_features = scaler.transform(test_features)
    _save_scaler(scaler, list(train_features.columns))

    print(
        f"Observations — train: {len(train)}, validation: {len(validation)}, "
        f"test: {len(test)}; features: {train_features.shape[1]}"
    )
    print(
        "Validation selection uses yearly Sharpe stability, drawdown, and "
        "turnover; the test period is not used for checkpoint selection."
    )

    if not skip_training:
        train_ppo(
            train,
            validation,
            ROBUST_CONFIG,
            train_features=train_features,
            validation_features=validation_features,
        )
        if not ppo_only:
            train_dqn(
                train,
                validation,
                ROBUST_CONFIG,
                train_features=train_features,
                validation_features=validation_features,
            )

    strategy_returns, metrics = compare_strategies(
        train,
        test,
        ROBUST_CONFIG,
        test_features=test_features,
        include_dqn=not ppo_only,
    )
    paths = save_evaluation(strategy_returns, metrics, ROBUST_CONFIG)
    print(metrics.round(4))
    print("Saved robust evaluation outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Download the expanded universe and regenerate robust features.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare robust data and exit without training or evaluation.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Evaluate existing robust PPO and DQN models.",
    )
    parser.add_argument(
        "--ppo-only",
        action="store_true",
        help="Train PPO only; DQN remains an optional discrete baseline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_robust_experiment(
        prepare_data=arguments.prepare_data,
        prepare_only=arguments.prepare_only,
        skip_training=arguments.skip_training,
        ppo_only=arguments.ppo_only,
    )

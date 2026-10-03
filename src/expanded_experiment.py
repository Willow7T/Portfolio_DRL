"""Expanded-equity, multi-regime PPO research experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .agents.train_agents import train_dqn, train_ppo
from .data.download_data import download_market_data
from .data.feature_engineering import (
    RobustFeatureScaler,
    create_expanded_market_features,
)
from .data.preprocess import preprocess_market_data
from .data.regimes import create_historical_regime_labels
from .data.splits import chronological_split
from .data.universe import EXPANDED_TICKERS, write_universe_manifest
from .evaluation.expanded import (
    baseline_walk_forward_report,
    compare_expanded_strategies,
    cost_sensitivity_report,
    save_expanded_evaluation,
)
from .utils.config import END_DATE, EXPANDED_CONFIG, ExperimentConfig


RAW_PRICES_PATH = Path("data/raw/expanded_prices.csv")
FEATURES_PATH = Path("data/processed/expanded_features.csv")
REGIMES_PATH = Path("data/processed/expanded_regimes.csv")
UNIVERSE_MANIFEST_PATH = Path("data/processed/expanded_universe.json")
SCALER_PATH = Path("results/models/expanded/feature_scaler.json")
EXPANDED_START_DATE = "2007-01-01"


def prepare_expanded_data(config: ExperimentConfig = EXPANDED_CONFIG) -> None:
    """Download the fixed research universe and build train-fitted regimes."""
    write_universe_manifest(UNIVERSE_MANIFEST_PATH)
    download_market_data(
        EXPANDED_TICKERS,
        EXPANDED_START_DATE,
        END_DATE,
        RAW_PRICES_PATH,
    )
    returns = preprocess_market_data(RAW_PRICES_PATH, config.returns_path)
    features, regimes, transformer = create_expanded_market_features(
        returns,
        train_end=config.train_end,
        random_seed=config.seed,
    )
    aligned_returns = returns.loc[features.index]
    aligned_returns.to_csv(config.returns_path)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH)
    regimes.to_csv(REGIMES_PATH, header=True)
    transformer.save(config.model_directory)
    print(
        f"Expanded dataset: {len(aligned_returns):,} observations, "
        f"{aligned_returns.shape[1]} assets, {features.shape[1]} features."
    )


def load_expanded_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load and strictly align returns, observations, and regime labels."""
    returns = pd.read_csv(
        EXPANDED_CONFIG.returns_path,
        index_col=0,
        parse_dates=True,
    ).dropna()
    features = pd.read_csv(
        FEATURES_PATH,
        index_col=0,
        parse_dates=True,
    ).dropna()
    common_index = returns.index.intersection(features.index)
    returns = returns.loc[common_index, EXPANDED_TICKERS]
    features = features.loc[common_index]
    # Derive fixed historical labels from dates on every load so datasets
    # prepared under an older labelling scheme cannot silently retain it.
    regimes = create_historical_regime_labels(common_index)
    expected_features = (
        len(EXPANDED_TICKERS) * EXPANDED_CONFIG.per_asset_feature_count
        + EXPANDED_CONFIG.global_feature_count
    )
    if features.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} ordered features, found "
            f"{features.shape[1]}. Regenerate the expanded dataset."
        )
    return returns, features, regimes


def _scale_features(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
    config: ExperimentConfig = EXPANDED_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = RobustFeatureScaler(clip_value=5.0)
    train_scaled = scaler.fit_transform(train_features)
    validation_scaled = scaler.transform(validation_features)
    test_scaled = scaler.transform(test_features)
    if scaler.median is None or scaler.scale is None:
        raise RuntimeError("Expanded feature scaler was not fitted.")
    scaler_path = config.model_directory / "feature_scaler.json"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.write_text(
        json.dumps(
            {
                "clip_value": scaler.clip_value,
                "feature_columns": list(train_scaled.columns),
                "median": scaler.median.to_dict(),
                "scale": scaler.scale.to_dict(),
                "fit_end": config.train_end,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return train_scaled, validation_scaled, test_scaled


def smoke_config(config: ExperimentConfig = EXPANDED_CONFIG) -> ExperimentConfig:
    """Return a cheap integration profile that cannot be mistaken for final."""
    return replace(
        config,
        model_directory=Path("results/models/expanded_v2_smoke"),
        report_directory=Path("results/reports/expanded_v2_smoke"),
        figure_directory=Path("results/figures/expanded_v2_smoke"),
        ppo_timesteps=50_000,
        dqn_timesteps=50_000,
        training_seeds=(42,),
        validation_frequency=5_000,
        validation_patience=5,
        minimum_training_timesteps=20_000,
        selection_start_timestep=5_000,
        dqn_buffer_size=20_000,
        dqn_learning_starts=2_000,
    )


def run_expanded_experiment(
    prepare_data: bool = False,
    prepare_only: bool = False,
    skip_training: bool = False,
    smoke: bool = False,
    evaluate_final: bool = False,
    ppo_only: bool = False,
) -> None:
    """Train expanded PPO; open the locked test only with explicit consent."""
    config = smoke_config() if smoke else EXPANDED_CONFIG
    if prepare_data or prepare_only:
        prepare_expanded_data(config=EXPANDED_CONFIG)
    if prepare_only:
        return

    returns, features, regimes = load_expanded_data()
    train, validation, test = chronological_split(
        returns,
        train_end=config.train_end,
        validation_end=config.validation_end,
    )
    train_features, validation_features, test_features = _scale_features(
        features.loc[train.index],
        features.loc[validation.index],
        features.loc[test.index],
        config=config,
    )
    print(
        f"Observations — train: {len(train)}, validation: {len(validation)}, "
        f"locked test: {len(test)}; assets: {train.shape[1]}; "
        f"features: {train_features.shape[1]}"
    )

    walk_forward = baseline_walk_forward_report(
        returns.loc[: config.validation_end],
        config,
    )
    config.report_directory.mkdir(parents=True, exist_ok=True)
    walk_forward.to_csv(
        config.report_directory / "walk_forward_baselines.csv"
    )

    if not skip_training:
        train_ppo(
            train,
            validation,
            config,
            train_features=train_features,
            validation_features=validation_features,
            train_regimes=regimes.loc[train.index],
            validation_regimes=regimes.loc[validation.index],
        )
        if not ppo_only:
            train_dqn(
                train,
                validation,
                config,
                train_features=train_features,
                validation_features=validation_features,
                train_regimes=regimes.loc[train.index],
                validation_regimes=regimes.loc[validation.index],
            )

    if not evaluate_final:
        print(
            "Training/development run complete. The 2022-2025 test remains "
            "locked. Re-run with --evaluate-final only after freezing the "
            "protocol."
        )
        return

    returns_frame, metrics, regime_metrics, bootstrap = (
        compare_expanded_strategies(
            train,
            test,
            test_features,
            regimes.loc[test.index],
            config,
            include_dqn=not ppo_only,
        )
    )
    cost_sensitivity = cost_sensitivity_report(
        train,
        test,
        test_features,
        config,
        include_dqn=not ppo_only,
    )
    paths = save_expanded_evaluation(
        returns_frame,
        metrics,
        regime_metrics,
        bootstrap,
        walk_forward,
        cost_sensitivity,
        config,
    )
    print(metrics.round(4))
    print("Saved expanded final evaluation outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one seed and 50,000 steps for integration testing.",
    )
    parser.add_argument(
        "--evaluate-final",
        action="store_true",
        help="Explicitly unlock the frozen 2022-2025 final test.",
    )
    parser.add_argument(
        "--ppo-only",
        action="store_true",
        help="Skip DQN when only the continuous PPO candidate is required.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_expanded_experiment(
        prepare_data=arguments.prepare_data,
        prepare_only=arguments.prepare_only,
        skip_training=arguments.skip_training,
        smoke=arguments.smoke,
        evaluate_final=arguments.evaluate_final,
        ppo_only=arguments.ppo_only,
    )

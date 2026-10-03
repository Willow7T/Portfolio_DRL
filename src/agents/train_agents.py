"""Train reproducible PPO and DQN deployment candidates."""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Type

import gymnasium
import numpy as np
import pandas as pd
import stable_baselines3
import torch
from gymnasium import Env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..environments.portfolio_env import DiscretePortfolioEnv, PortfolioEnv
from ..portfolio.metrics import performance_summary
from ..utils.config import DEFAULT_CONFIG, ExperimentConfig
from .feature_extractors import SharedAssetFeatureExtractor


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


def _vector_environment(
    environment_class: Type[Env],
    returns: pd.DataFrame,
    config: ExperimentConfig,
    features: pd.DataFrame | None = None,
    regimes: pd.Series | None = None,
) -> DummyVecEnv:
    return DummyVecEnv(
        [
            lambda: Monitor(
                environment_class(
                    returns,
                    features=features,
                    random_start=config.training_episode_length is not None,
                    episode_length=config.training_episode_length,
                    regime_labels=regimes,
                    balance_regime_sampling=config.balance_regime_sampling,
                    **_environment_kwargs(config),
                )
            )
        ]
    )


def _validation_run(
    model: BaseAlgorithm,
    environment_class: Type[Env],
    returns: pd.DataFrame,
    config: ExperimentConfig,
    features: pd.DataFrame | None = None,
) -> tuple[pd.Series, float]:
    environment = environment_class(
        returns,
        features=features,
        **_environment_kwargs(config),
    )
    observation, _ = environment.reset(seed=config.seed)
    terminated = False
    while not terminated:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = environment.step(action)
        terminated = terminated or truncated
    portfolio_returns = pd.Series(
        environment.portfolio_returns,
        index=environment.return_dates,
        dtype=np.float64,
    )
    mean_turnover = (
        float(np.mean(environment.turnover_history))
        if environment.turnover_history
        else 0.0
    )
    return portfolio_returns, mean_turnover


def validation_summary(
    returns: pd.Series,
    mean_turnover: float,
    config: ExperimentConfig,
    period_labels: pd.Series | None = None,
) -> dict[str, float]:
    """Score validation performance while penalising regime instability."""
    pooled = performance_summary(returns.to_numpy())
    if config.validation_aggregation == "pooled":
        return {
            **pooled,
            "mean_turnover": mean_turnover,
            "robust_score": float(pooled["sharpe_ratio"]),
            "median_period_sharpe": float(pooled["sharpe_ratio"]),
            "worst_period_sharpe": float(pooled["sharpe_ratio"]),
            "period_sharpe_std": 0.0,
        }
    if config.validation_aggregation != "yearly_robust":
        raise ValueError(
            "validation_aggregation must be 'pooled' or 'yearly_robust'."
        )

    period_sharpes = []
    if period_labels is None:
        grouped_returns = returns.groupby(returns.index.year)
    else:
        aligned_labels = period_labels.reindex(returns.index)
        if aligned_labels.isnull().any():
            raise ValueError("Validation regime labels must align with returns.")
        grouped_returns = returns.groupby(aligned_labels)
    for _, period_returns in grouped_returns:
        if len(period_returns) >= 40:
            period_sharpes.append(
                performance_summary(period_returns.to_numpy())["sharpe_ratio"]
            )
    finite_sharpes = np.asarray(period_sharpes, dtype=np.float64)
    finite_sharpes = finite_sharpes[np.isfinite(finite_sharpes)]
    if finite_sharpes.size == 0:
        median_sharpe = worst_sharpe = -np.inf
        sharpe_std = np.inf
        robust_score = -np.inf
    else:
        median_sharpe = float(np.median(finite_sharpes))
        worst_sharpe = float(np.min(finite_sharpes))
        sharpe_std = float(np.std(finite_sharpes))
        worst_weight = config.validation_worst_period_weight
        if not 0 <= worst_weight <= 1:
            raise ValueError("validation_worst_period_weight must be in [0, 1].")
        robust_score = (
            (1 - worst_weight) * median_sharpe
            + worst_weight * worst_sharpe
            - config.validation_dispersion_penalty * sharpe_std
            - config.validation_drawdown_penalty
            * float(pooled["maximum_drawdown"])
            - config.validation_turnover_penalty * mean_turnover
        )
    return {
        **pooled,
        "mean_turnover": mean_turnover,
        "robust_score": float(robust_score),
        "median_period_sharpe": median_sharpe,
        "worst_period_sharpe": worst_sharpe,
        "period_sharpe_std": sharpe_std,
    }


class ValidationCallback(BaseCallback):
    """Select and stop training using a strictly separate validation period."""

    def __init__(
        self,
        environment_class: Type[Env],
        validation_returns: pd.DataFrame,
        validation_features: pd.DataFrame | None,
        config: ExperimentConfig,
        checkpoint_path: Path,
        history_path: Path,
        validation_regimes: pd.Series | None = None,
    ) -> None:
        super().__init__(verbose=1)
        self.environment_class = environment_class
        self.validation_returns = validation_returns
        self.validation_features = validation_features
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.history_path = history_path
        self.validation_regimes = validation_regimes
        self.best_score = -np.inf
        self.evaluations_without_improvement = 0
        self.history: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.config.validation_frequency != 0:
            return True

        returns, mean_turnover = _validation_run(
            self.model,
            self.environment_class,
            self.validation_returns,
            self.config,
            self.validation_features,
        )
        metrics = validation_summary(
            returns,
            mean_turnover,
            self.config,
            period_labels=self.validation_regimes,
        )
        score = float(metrics[self.config.selection_metric])
        if not np.isfinite(score):
            score = -np.inf
        eligible_for_selection = (
            self.num_timesteps >= self.config.selection_start_timestep
        )

        self.history.append({"timesteps": self.num_timesteps, **metrics})
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.history).to_csv(self.history_path, index=False)

        if eligible_for_selection and score > self.best_score:
            self.best_score = score
            self.evaluations_without_improvement = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.checkpoint_path)
        elif eligible_for_selection:
            self.evaluations_without_improvement += 1

        self.logger.record(
            f"validation/{self.config.selection_metric}",
            score,
        )
        self.logger.record("validation/best_score", self.best_score)
        if self.verbose:
            print(
                f"Validation at {self.num_timesteps:,} steps: "
                f"{self.config.selection_metric}={score:.4f}, "
                f"best={self.best_score:.4f}, "
                f"eligible={'yes' if eligible_for_selection else 'no'}"
            )

        may_stop = (
            self.num_timesteps >= self.config.minimum_training_timesteps
            and self.evaluations_without_improvement
            >= self.config.validation_patience
        )
        if may_stop and self.verbose:
            print("Early stopping: validation performance stopped improving.")
        return not may_stop


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def _write_manifest(
    algorithm: str,
    canonical_path: Path,
    selected_seed: int,
    validation_score: float,
    train_returns: pd.DataFrame,
    validation_returns: pd.DataFrame,
    feature_columns: list[str] | None,
    config: ExperimentConfig,
) -> None:
    manifest = {
        "algorithm": algorithm,
        "artifact": str(canonical_path.with_suffix(".zip")),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_seed": selected_seed,
        "selection_metric": config.selection_metric,
        "validation_score": validation_score,
        "assets": list(train_returns.columns),
        "feature_count": len(feature_columns or []),
        "feature_columns": feature_columns or [],
        "train_start": str(train_returns.index.min().date()),
        "train_end": str(train_returns.index.max().date()),
        "validation_start": str(validation_returns.index.min().date()),
        "validation_end": str(validation_returns.index.max().date()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gymnasium.__version__,
        "config": _json_ready(asdict(config)),
    }
    manifest_path = canonical_path.with_name(
        f"{canonical_path.name}_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _train_candidates(
    algorithm_name: str,
    model_class: type[PPO] | type[DQN],
    environment_class: Type[Env],
    train_returns: pd.DataFrame,
    validation_returns: pd.DataFrame,
    train_features: pd.DataFrame | None,
    validation_features: pd.DataFrame | None,
    config: ExperimentConfig,
    total_timesteps: int,
    model_kwargs: dict[str, Any],
    train_regimes: pd.Series | None = None,
    validation_regimes: pd.Series | None = None,
    device: str | None = None,
) -> Path:
    if validation_returns.empty:
        raise ValueError("Validation returns are required for model selection.")
    valid_metrics = set(performance_summary(np.asarray([0.0, 0.0])))
    valid_metrics.add("robust_score")
    if config.selection_metric not in valid_metrics:
        raise ValueError(f"Unknown selection metric: {config.selection_metric}")
    if not config.training_seeds:
        raise ValueError("At least one training seed is required.")

    config.model_directory.mkdir(parents=True, exist_ok=True)
    run_directory = config.model_directory / "candidates" / algorithm_name
    report_directory = config.report_directory / "training"
    canonical_path = config.model_directory / f"{algorithm_name}_portfolio"
    best_candidate: Path | None = None
    best_score = -np.inf
    best_seed = config.seed

    for seed in config.training_seeds:
        environment = _vector_environment(
            environment_class,
            train_returns,
            config,
            train_features,
            train_regimes,
        )
        seed_directory = run_directory / f"seed_{seed}"
        checkpoint_path = seed_directory / "best_validation_model"
        callback = ValidationCallback(
            environment_class=environment_class,
            validation_returns=validation_returns,
            validation_features=validation_features,
            config=config,
            checkpoint_path=checkpoint_path,
            history_path=report_directory
            / f"{algorithm_name}_seed_{seed}_validation.csv",
            validation_regimes=validation_regimes,
        )
        model = model_class(
            "MlpPolicy",
            environment,
            verbose=1,
            seed=seed,
            device=device or config.device,
            tensorboard_log=str(config.report_directory / "tensorboard"),
            **model_kwargs,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=f"{algorithm_name}_seed_{seed}",
        )
        seed_directory.mkdir(parents=True, exist_ok=True)
        model.save(seed_directory / "last_model")
        environment.close()

        candidate_zip = checkpoint_path.with_suffix(".zip")
        if candidate_zip.exists() and callback.best_score > best_score:
            best_score = callback.best_score
            best_candidate = candidate_zip
            best_seed = seed

    if best_candidate is None:
        raise RuntimeError(
            f"No valid {algorithm_name.upper()} validation checkpoint was produced."
        )

    canonical_zip = canonical_path.with_suffix(".zip")
    shutil.copy2(best_candidate, canonical_zip)
    _write_manifest(
        algorithm=algorithm_name.upper(),
        canonical_path=canonical_path,
        selected_seed=best_seed,
        validation_score=best_score,
        train_returns=train_returns,
        validation_returns=validation_returns,
        feature_columns=(
            list(train_features.columns) if train_features is not None else None
        ),
        config=config,
    )
    return canonical_zip


def train_ppo(
    train_returns: pd.DataFrame,
    validation_returns: pd.DataFrame,
    config: ExperimentConfig = DEFAULT_CONFIG,
    train_features: pd.DataFrame | None = None,
    validation_features: pd.DataFrame | None = None,
    train_regimes: pd.Series | None = None,
    validation_regimes: pd.Series | None = None,
) -> Path:
    """Train PPO candidates and deploy the best validation checkpoint."""
    return _train_candidates(
        algorithm_name="ppo",
        model_class=PPO,
        environment_class=PortfolioEnv,
        train_returns=train_returns,
        validation_returns=validation_returns,
        train_features=train_features,
        validation_features=validation_features,
        config=config,
        total_timesteps=config.ppo_timesteps,
        model_kwargs={
            "learning_rate": config.ppo_learning_rate,
            "n_steps": config.ppo_n_steps,
            "batch_size": config.ppo_batch_size,
            "n_epochs": config.ppo_n_epochs,
            "gamma": config.ppo_gamma,
            "gae_lambda": config.ppo_gae_lambda,
            "clip_range": config.ppo_clip_range,
            "ent_coef": config.ppo_ent_coef,
            "policy_kwargs": {
                "net_arch": {
                    "pi": list(config.policy_hidden_layers),
                    "vf": list(config.policy_hidden_layers),
                },
                "optimizer_kwargs": {
                    "weight_decay": config.policy_weight_decay,
                },
                **(
                    {
                        "features_extractor_class": SharedAssetFeatureExtractor,
                        "features_extractor_kwargs": {
                            "n_assets": train_returns.shape[1],
                            "lookback_window": config.lookback_window,
                            "per_asset_feature_count": (
                                config.per_asset_feature_count
                            ),
                            "global_feature_count": config.global_feature_count,
                            "asset_embedding_dim": config.asset_embedding_dim,
                            "context_embedding_dim": (
                                config.context_embedding_dim
                            ),
                        },
                    }
                    if config.use_shared_asset_extractor
                    else {}
                ),
            },
        },
        train_regimes=train_regimes,
        validation_regimes=validation_regimes,
        device=config.ppo_device,
    )


def train_dqn(
    train_returns: pd.DataFrame,
    validation_returns: pd.DataFrame,
    config: ExperimentConfig = DEFAULT_CONFIG,
    train_features: pd.DataFrame | None = None,
    validation_features: pd.DataFrame | None = None,
    train_regimes: pd.Series | None = None,
    validation_regimes: pd.Series | None = None,
) -> Path:
    """Train DQN candidates and deploy the best validation checkpoint."""
    return _train_candidates(
        algorithm_name="dqn",
        model_class=DQN,
        environment_class=DiscretePortfolioEnv,
        train_returns=train_returns,
        validation_returns=validation_returns,
        train_features=train_features,
        validation_features=validation_features,
        config=config,
        total_timesteps=config.dqn_timesteps,
        model_kwargs={
            "learning_rate": config.dqn_learning_rate,
            "buffer_size": config.dqn_buffer_size,
            "learning_starts": config.dqn_learning_starts,
            "batch_size": config.dqn_batch_size,
            "gamma": config.dqn_gamma,
            "train_freq": config.dqn_train_frequency,
            "gradient_steps": config.dqn_gradient_steps,
            "target_update_interval": config.dqn_target_update_interval,
            "exploration_fraction": config.dqn_exploration_fraction,
            "exploration_final_eps": (
                config.dqn_exploration_final_epsilon
            ),
            "optimize_memory_usage": config.dqn_optimize_memory_usage,
            **(
                {
                    "replay_buffer_kwargs": {
                        "handle_timeout_termination": False,
                    }
                }
                if config.dqn_optimize_memory_usage
                else {}
            ),
            "policy_kwargs": {
                "net_arch": list(config.policy_hidden_layers),
                "optimizer_kwargs": {
                    "weight_decay": config.policy_weight_decay,
                },
                **(
                    {
                        "features_extractor_class": SharedAssetFeatureExtractor,
                        "features_extractor_kwargs": {
                            "n_assets": train_returns.shape[1],
                            "lookback_window": config.lookback_window,
                            "per_asset_feature_count": (
                                config.per_asset_feature_count
                            ),
                            "global_feature_count": config.global_feature_count,
                            "asset_embedding_dim": config.asset_embedding_dim,
                            "context_embedding_dim": (
                                config.context_embedding_dim
                            ),
                        },
                    }
                    if config.use_shared_asset_extractor
                    else {}
                ),
            },
        },
        train_regimes=train_regimes,
        validation_regimes=validation_regimes,
        device=config.dqn_device,
    )

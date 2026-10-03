"""Leakage-safe Gymnasium environments for dynamic portfolio allocation."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..portfolio.constraints import project_portfolio_weights


class PortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous, long-only portfolio environment intended for PPO."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame | None = None,
        lookback_window: int = 30,
        initial_balance: float = 1_000_000.0,
        transaction_cost: float = 0.001,
        slippage_cost: float = 0.0,
        turnover_penalty: float = 0.0,
        max_weight: float = 1.0,
        drawdown_limit: float = 0.20,
        drawdown_penalty: float = 2.0,
        sharpe_reward_weight: float = 0.05,
        rebalance_frequency: int = 1,
        sector_labels: tuple[str, ...] | None = None,
        max_sector_weight: float = 1.0,
        random_start: bool = False,
        episode_length: int | None = None,
        regime_labels: pd.Series | None = None,
        balance_regime_sampling: bool = False,
        action_temperature: float = 1.0,
        action_smoothing: float = 1.0,
        minimum_rebalance_turnover: float = 0.0,
        maximum_rebalance_turnover: float = 2.0,
        continuous_action_mode: str = "scores",
    ) -> None:
        super().__init__()
        if returns.isnull().any().any():
            raise ValueError("Returns must not contain missing values.")
        if len(returns) <= lookback_window:
            raise ValueError("Returns must be longer than the lookback window.")
        if transaction_cost < 0:
            raise ValueError("Transaction cost cannot be negative.")
        if slippage_cost < 0:
            raise ValueError("Slippage cost cannot be negative.")
        if turnover_penalty < 0:
            raise ValueError("Turnover penalty cannot be negative.")
        if not 0 < max_weight <= 1:
            raise ValueError("Maximum weight must be in (0, 1].")
        if rebalance_frequency < 1:
            raise ValueError("Rebalance frequency must be at least one day.")
        if not 0 < max_sector_weight <= 1:
            raise ValueError("Maximum sector weight must be in (0, 1].")
        if action_temperature <= 0:
            raise ValueError("Action temperature must be positive.")
        if not 0 < action_smoothing <= 1:
            raise ValueError("Action smoothing must be in (0, 1].")
        if minimum_rebalance_turnover < 0:
            raise ValueError("Minimum rebalance turnover cannot be negative.")
        if not 0 < maximum_rebalance_turnover <= 2:
            raise ValueError("Maximum rebalance turnover must be in (0, 2].")
        if minimum_rebalance_turnover > maximum_rebalance_turnover:
            raise ValueError("Minimum turnover cannot exceed maximum turnover.")
        if continuous_action_mode not in {"scores", "softmax"}:
            raise ValueError(
                "Continuous action mode must be 'scores' or 'softmax'."
            )

        self.returns = returns.astype(np.float32)
        if features is None:
            features = returns
        features = features.reindex(returns.index)
        if features.isnull().any().any():
            raise ValueError("Features must align with returns without gaps.")
        self.features = features.astype(np.float32)
        self._returns_array = self.returns.to_numpy(dtype=np.float32)
        self._features_array = self.features.to_numpy(dtype=np.float32)
        self.dates = returns.index
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage_cost = slippage_cost
        self.turnover_penalty = turnover_penalty
        self.max_weight = max_weight
        self.drawdown_limit = drawdown_limit
        self.drawdown_penalty = drawdown_penalty
        self.sharpe_reward_weight = sharpe_reward_weight
        self.n_assets = returns.shape[1]
        self.rebalance_frequency = int(rebalance_frequency)
        self.sector_labels = tuple(sector_labels or ())
        self.max_sector_weight = max_sector_weight
        self.random_start = random_start
        self.episode_length = episode_length
        self.balance_regime_sampling = balance_regime_sampling
        self.action_temperature = action_temperature
        self.action_smoothing = action_smoothing
        self.minimum_rebalance_turnover = minimum_rebalance_turnover
        self.maximum_rebalance_turnover = maximum_rebalance_turnover
        self.continuous_action_mode = continuous_action_mode

        if self.sector_labels and len(self.sector_labels) != self.n_assets:
            raise ValueError("Sector labels must align with return columns.")
        if self.sector_labels:
            feasible_sector_capacity = sum(
                min(
                    self.max_sector_weight,
                    self.sector_labels.count(sector) * self.max_weight,
                )
                for sector in set(self.sector_labels)
            )
            if feasible_sector_capacity < 1 - 1e-12:
                raise ValueError("Sector and individual caps are infeasible.")

        if self.max_weight * self.n_assets < 1 - 1e-12:
            raise ValueError(
                "Maximum weight is infeasible for the number of assets."
            )
        observation_size = (
            self.lookback_window * self.features.shape[1]
            + self.n_assets
            + 1
            + self.lookback_window
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=(-5.0 if self.continuous_action_mode == "softmax" else 0.0),
            high=(5.0 if self.continuous_action_mode == "softmax" else 1.0),
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        self._step_index = self.lookback_window
        self._episode_end = len(self.returns)
        self._regime_labels: np.ndarray | None = None
        if regime_labels is not None:
            aligned_regimes = regime_labels.reindex(self.returns.index)
            if aligned_regimes.isnull().any():
                raise ValueError("Regime labels must align with returns.")
            self._regime_labels = aligned_regimes.astype(str).to_numpy()
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.weights = np.full(self.n_assets, 1 / self.n_assets, dtype=np.float32)
        self.portfolio_returns: list[float] = []
        self.value_history: list[float] = []
        self.weight_history: list[np.ndarray] = []
        self.target_weight_history: list[np.ndarray] = []
        self.turnover_history: list[float] = []
        self.cost_history: list[float] = []
        self.return_dates: list[pd.Timestamp] = []
        self._recent_returns: deque[float] = deque(
            maxlen=self.lookback_window
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        episode_length = self.episode_length or (
            len(self.returns) - self.lookback_window
        )
        maximum_start = len(self.returns) - episode_length
        if self.random_start and maximum_start > self.lookback_window:
            candidates = np.arange(self.lookback_window, maximum_start + 1)
            if self.balance_regime_sampling and self._regime_labels is not None:
                candidate_labels = self._regime_labels[candidates]
                unique_labels = np.unique(candidate_labels)
                selected_label = self.np_random.choice(unique_labels)
                candidates = candidates[candidate_labels == selected_label]
            self._step_index = int(self.np_random.choice(candidates))
        else:
            self._step_index = self.lookback_window
        self._episode_end = min(
            self._step_index + episode_length,
            len(self.returns),
        )
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.weights = np.full(self.n_assets, 1 / self.n_assets, dtype=np.float32)
        self.portfolio_returns = []
        self.value_history = [self.initial_balance]
        self.weight_history = [self.weights.copy()]
        self.target_weight_history = [self.weights.copy()]
        self.turnover_history = []
        self.cost_history = []
        self.return_dates = []
        self._recent_returns = deque(
            [0.0] * self.lookback_window,
            maxlen=self.lookback_window,
        )
        return self._get_observation(), self._get_info(turnover=0.0)

    def _cap_weights(self, weights: np.ndarray) -> np.ndarray:
        """Project non-negative weights onto a capped unit simplex."""
        return project_portfolio_weights(
            weights,
            max_weight=self.max_weight,
            sector_labels=self.sector_labels,
            max_sector_weight=self.max_sector_weight,
        )

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        if self.continuous_action_mode == "scores":
            scores = np.clip(np.asarray(action, dtype=np.float64), 0, 1)
            if scores.sum() < 1e-12:
                scores = np.ones(self.n_assets, dtype=np.float64)
            return self._cap_weights(scores / scores.sum())
        logits = np.clip(np.asarray(action, dtype=np.float64), -20, 20)
        logits = logits / self.action_temperature
        logits -= logits.max()
        scores = np.exp(logits)
        weights = (scores / scores.sum()).astype(np.float32)
        return self._cap_weights(weights)

    def _stabilise_target_weights(self, proposed: np.ndarray) -> np.ndarray:
        """Smooth continuous allocations and suppress uneconomic small trades."""
        blended = (
            (1 - self.action_smoothing) * self.weights
            + self.action_smoothing * proposed
        )
        target = self._cap_weights(blended)
        turnover = float(np.abs(target - self.weights).sum())
        if turnover < self.minimum_rebalance_turnover:
            return self.weights.copy()
        if turnover > self.maximum_rebalance_turnover:
            fraction = self.maximum_rebalance_turnover / turnover
            target = self.weights + fraction * (target - self.weights)
        return target

    def _rolling_sharpe(self) -> float:
        values = np.asarray(self._recent_returns, dtype=np.float64)
        if len(values) < 2 or np.std(values, ddof=1) < 1e-12:
            return 0.0
        sharpe = np.mean(values) / np.std(values, ddof=1) * np.sqrt(252)
        return float(np.clip(sharpe, -5.0, 5.0))

    @property
    def drawdown(self) -> float:
        return float(1 - self.portfolio_value / self.peak_value)

    def _get_observation(self) -> np.ndarray:
        market_window = self._features_array[
            self._step_index - self.lookback_window : self._step_index
        ]
        observation = np.concatenate(
            [
                market_window.reshape(-1),
                self.weights,
                np.asarray([self.drawdown], dtype=np.float32),
                np.asarray(self._recent_returns, dtype=np.float32),
            ]
        )
        return observation.astype(np.float32)

    def _get_info(self, turnover: float) -> dict[str, Any]:
        date_index = min(self._step_index, len(self.dates) - 1)
        return {
            "date": self.dates[date_index],
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.copy(),
            "drawdown": self.drawdown,
            "turnover": turnover,
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target_weights = self._stabilise_target_weights(
            self._action_to_weights(action)
        )
        interval_turnover = float(np.abs(target_weights - self.weights).sum())
        interval_reward = 0.0
        days_to_run = min(
            self.rebalance_frequency,
            self._episode_end - self._step_index,
        )

        for day in range(days_to_run):
            day_turnover = interval_turnover if day == 0 else 0.0
            day_cost = (
                (self.transaction_cost + self.slippage_cost) * day_turnover
            )
            start_weights = target_weights if day == 0 else self.weights.copy()
            asset_returns = self._returns_array[self._step_index].astype(
                np.float64,
                copy=False,
            )
            gross_return = float(np.dot(start_weights, asset_returns))
            net_return = max(gross_return - day_cost, -0.999999)

            self.portfolio_value *= 1 + net_return
            self.peak_value = max(self.peak_value, self.portfolio_value)
            post_return_values = start_weights.astype(np.float64) * (
                1 + asset_returns
            )
            post_return_total = post_return_values.sum()
            self.weights = (
                (post_return_values / post_return_total).astype(np.float32)
                if post_return_total > 1e-12
                else start_weights.astype(np.float32)
            )
            self.portfolio_returns.append(net_return)
            self._recent_returns.append(net_return)
            self.value_history.append(self.portfolio_value)
            self.target_weight_history.append(start_weights.copy())
            self.weight_history.append(self.weights.copy())
            self.turnover_history.append(day_turnover)
            self.cost_history.append(day_cost)
            self.return_dates.append(self.dates[self._step_index])

            excess_drawdown = max(self.drawdown - self.drawdown_limit, 0.0)
            interval_reward += (
                100 * np.log1p(net_return)
                + self.sharpe_reward_weight * self._rolling_sharpe() / 252
                - self.drawdown_penalty * excess_drawdown
                - self.turnover_penalty * day_turnover
            )
            self._step_index += 1

        terminated = self._step_index >= self._episode_end
        observation = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated
            else self._get_observation()
        )
        return (
            observation,
            float(interval_reward),
            terminated,
            False,
            self._get_info(interval_turnover),
        )


class DiscretePortfolioEnv(PortfolioEnv):
    """Discrete allocation environment intended for vanilla DQN."""

    def __init__(self, returns: pd.DataFrame, **kwargs: Any) -> None:
        super().__init__(returns, **kwargs)
        self.action_space = spaces.Discrete(self.n_assets + 1)

    def _action_to_weights(self, action: int | np.ndarray) -> np.ndarray:
        action_index = int(np.asarray(action).item())
        if action_index == 0:
            return np.full(
                self.n_assets,
                1 / self.n_assets,
                dtype=np.float32,
            )
        if self.n_assets == 1:
            return np.ones(1, dtype=np.float32)
        if self.max_weight >= 1.0:
            weights = np.zeros(self.n_assets, dtype=np.float32)
            weights[action_index - 1] = 1.0
            return weights
        selected_weight = min(
            self.max_weight,
            1 - (self.n_assets - 1) * 1e-6,
        )
        remaining_weight = (1 - selected_weight) / (self.n_assets - 1)
        weights = np.full(
            self.n_assets,
            remaining_weight,
            dtype=np.float32,
        )
        weights[action_index - 1] = selected_weight
        return self._cap_weights(weights)

    def _stabilise_target_weights(self, proposed: np.ndarray) -> np.ndarray:
        """Keep DQN's discrete tilt semantics unchanged for comparison."""
        return proposed

"""Custom Stable-Baselines3 feature extractors."""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SharedAssetFeatureExtractor(BaseFeaturesExtractor):
    """Encode every asset with shared weights before policy/value heads.

    The environment observation starts with a time-by-feature window. Per-asset
    features occupy contiguous blocks, followed by global market/regime
    features. Portfolio weights, drawdown, and recent returns follow the market
    window.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_assets: int,
        lookback_window: int,
        per_asset_feature_count: int,
        global_feature_count: int,
        asset_embedding_dim: int = 32,
        context_embedding_dim: int = 64,
    ) -> None:
        if min(
            n_assets,
            lookback_window,
            per_asset_feature_count,
            asset_embedding_dim,
            context_embedding_dim,
        ) < 1:
            raise ValueError("Extractor dimensions must be positive.")
        self.n_assets = n_assets
        self.lookback_window = lookback_window
        self.per_asset_feature_count = per_asset_feature_count
        self.global_feature_count = global_feature_count
        self.features_per_date = (
            n_assets * per_asset_feature_count + global_feature_count
        )
        self.market_window_size = lookback_window * self.features_per_date
        expected_observation_size = (
            self.market_window_size + n_assets + 1 + lookback_window
        )
        if observation_space.shape[0] != expected_observation_size:
            raise ValueError(
                "Observation size does not match shared asset extractor layout: "
                f"expected {expected_observation_size}, received "
                f"{observation_space.shape[0]}."
            )

        output_size = n_assets * asset_embedding_dim + context_embedding_dim
        super().__init__(observation_space, features_dim=output_size)
        asset_input_size = lookback_window * per_asset_feature_count
        context_input_size = (
            lookback_window * global_feature_count
            + n_assets
            + 1
            + lookback_window
        )
        self.asset_encoder = nn.Sequential(
            nn.LayerNorm(asset_input_size),
            nn.Linear(asset_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, asset_embedding_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_input_size),
            nn.Linear(context_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, context_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        window = observations[:, : self.market_window_size].reshape(
            batch_size,
            self.lookback_window,
            self.features_per_date,
        )
        asset_feature_size = self.n_assets * self.per_asset_feature_count
        asset_window = window[:, :, :asset_feature_size].reshape(
            batch_size,
            self.lookback_window,
            self.n_assets,
            self.per_asset_feature_count,
        )
        asset_inputs = asset_window.permute(0, 2, 1, 3).reshape(
            batch_size * self.n_assets,
            self.lookback_window * self.per_asset_feature_count,
        )
        asset_embeddings = self.asset_encoder(asset_inputs).reshape(
            batch_size,
            -1,
        )

        global_window = window[:, :, asset_feature_size:].reshape(
            batch_size,
            -1,
        )
        portfolio_state = observations[:, self.market_window_size :]
        context = self.context_encoder(
            torch.cat([global_window, portfolio_state], dim=1)
        )
        return torch.cat([asset_embeddings, context], dim=1)

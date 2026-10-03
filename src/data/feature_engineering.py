from pathlib import Path

import numpy as np
import pandas as pd

from .regimes import (
    CausalRegimeTransformer,
    create_causal_regime_features,
    create_historical_regime_labels,
)

INPUT_PATH = Path("data/processed/returns.csv")
OUTPUT_PATH = Path("data/processed/features.csv")


def create_features() -> None:
    """Create simple rolling mean and volatility features."""
    data = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    features = data.copy()

    for column in data.columns:
        features[f"{column}_ma5"] = data[column].rolling(5).mean()
        features[f"{column}_vol5"] = data[column].rolling(5).std()

    features = features.dropna()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH)

    print("Features created.")


def create_robust_market_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Build past-only momentum, risk, drawdown, and regime features."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("Returns must use a DatetimeIndex.")
    returns = returns.sort_index().astype(float)
    features: dict[str, pd.Series] = {}

    for asset in returns.columns:
        asset_returns = returns[asset]
        wealth = (1 + asset_returns).cumprod()
        rolling_peak = wealth.rolling(252, min_periods=63).max()
        downside = asset_returns.clip(upper=0)

        features[f"{asset}__return_1"] = asset_returns
        for window in (5, 21, 63, 252):
            features[f"{asset}__momentum_{window}"] = (
                (1 + asset_returns).rolling(window).apply(np.prod, raw=True) - 1
            )
        for window in (21, 63):
            features[f"{asset}__volatility_{window}"] = (
                asset_returns.rolling(window).std() * np.sqrt(252)
            )
        features[f"{asset}__downside_volatility_63"] = (
            downside.pow(2).rolling(63).mean().pow(0.5) * np.sqrt(252)
        )
        features[f"{asset}__drawdown_252"] = wealth / rolling_peak - 1

    market_return = returns.mean(axis=1)
    market_wealth = (1 + market_return).cumprod()
    features["market__return_1"] = market_return
    features["market__volatility_21"] = (
        market_return.rolling(21).std() * np.sqrt(252)
    )
    features["market__volatility_63"] = (
        market_return.rolling(63).std() * np.sqrt(252)
    )
    features["market__momentum_63"] = (
        (1 + market_return).rolling(63).apply(np.prod, raw=True) - 1
    )
    features["market__momentum_252"] = (
        (1 + market_return).rolling(252).apply(np.prod, raw=True) - 1
    )
    features["market__drawdown_252"] = (
        market_wealth / market_wealth.rolling(252, min_periods=63).max() - 1
    )

    feature_frame = pd.DataFrame(features, index=returns.index)
    return feature_frame.replace([np.inf, -np.inf], np.nan).dropna()


def create_expanded_market_features(
    returns: pd.DataFrame,
    train_end: str,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, CausalRegimeTransformer]:
    """Build expanded features and fit regime states on training rows only."""
    asset_features = create_robust_market_features(returns)
    regime_statistics = create_causal_regime_features(returns)
    common_index = asset_features.index.intersection(regime_statistics.index)
    asset_features = asset_features.loc[common_index]
    regime_statistics = regime_statistics.loc[common_index]

    training_statistics = regime_statistics.loc[:train_end]
    transformer = CausalRegimeTransformer(random_seed=random_seed).fit(
        training_statistics
    )
    regime_features, _causal_state_labels = transformer.transform(
        regime_statistics
    )
    labels = create_historical_regime_labels(common_index)
    combined = pd.concat([asset_features, regime_features], axis=1)
    if combined.isnull().any().any():
        raise RuntimeError("Expanded features contain missing values.")
    return combined.astype(np.float32), labels, transformer


class RobustFeatureScaler:
    """Median/IQR scaler fitted only on training observations."""

    def __init__(self, clip_value: float = 5.0) -> None:
        self.clip_value = clip_value
        self.median: pd.Series | None = None
        self.scale: pd.Series | None = None

    def fit(self, features: pd.DataFrame) -> "RobustFeatureScaler":
        if features.empty:
            raise ValueError("Cannot fit a feature scaler on empty data.")
        self.median = features.median()
        lower = features.quantile(0.25)
        upper = features.quantile(0.75)
        scale = upper - lower
        self.scale = scale.mask(scale.abs() < 1e-12, 1.0)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.median is None or self.scale is None:
            raise RuntimeError("The feature scaler must be fitted first.")
        missing = self.median.index.difference(features.columns)
        if not missing.empty:
            raise ValueError(f"Missing feature columns: {list(missing)}")
        transformed = (
            features.loc[:, self.median.index] - self.median
        ) / self.scale
        return transformed.clip(-self.clip_value, self.clip_value).astype(
            np.float32
        )

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)


if __name__ == "__main__":
    create_features()

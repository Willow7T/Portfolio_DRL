"""Causal market-regime features and train-only regime fitting."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


REGIME_STATISTIC_COLUMNS = [
    "regime__market_volatility_21",
    "regime__market_volatility_63",
    "regime__market_momentum_63",
    "regime__market_momentum_252",
    "regime__market_drawdown_252",
    "regime__cross_sectional_dispersion_21",
    "regime__average_correlation_63",
]
THRESHOLD_REGIMES = ("calm_bull", "volatile_bull", "bear", "crisis")

HISTORICAL_REGIMES = (
    {
        "key": "regime_1",
        "label": "Regime 1 — GFC Aftermath and Euro-Area Stress",
        "start": "2008-01-01",
        "end": "2013-12-31",
        "description": "Elevated volatility and the sovereign-debt crisis.",
    },
    {
        "key": "regime_2",
        "label": "Regime 2 — Post-Crisis Expansion",
        "start": "2014-01-01",
        "end": "2019-12-31",
        "description": "Low rates and comparatively steady growth.",
    },
    {
        "key": "regime_3",
        "label": "Regime 3 — COVID Crash and Recovery",
        "start": "2020-01-01",
        "end": "2021-12-31",
        "description": "Extreme drawdown followed by a rapid rebound.",
    },
    {
        "key": "regime_4",
        "label": "Regime 4 — Inflation Shock and Rate Hikes",
        "start": "2022-01-01",
        "end": "2025-12-31",
        "description": "Aggressive Fed tightening and a bond sell-off.",
    },
)


def create_historical_regime_labels(index: pd.DatetimeIndex) -> pd.Series:
    """Assign the four predeclared, human-interpretable regime windows."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Historical regime labels require a DatetimeIndex.")
    labels = pd.Series(pd.NA, index=index, dtype="object", name="regime")
    for regime in HISTORICAL_REGIMES:
        mask = (
            (index >= pd.Timestamp(regime["start"]))
            & (index <= pd.Timestamp(regime["end"]))
        )
        labels.loc[mask] = regime["label"]
    if labels.isnull().any():
        missing = labels.index[labels.isnull()]
        raise ValueError(
            "Dates fall outside the declared 2008-2025 regime windows: "
            f"{missing.min().date()} to {missing.max().date()}."
        )
    return labels


def historical_regime_splits(
    frame: pd.DataFrame | pd.Series,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Split a dated object into the four fixed historical regimes."""
    labels = create_historical_regime_labels(frame.index)
    return {
        regime["key"]: frame.loc[labels == regime["label"]].copy()
        for regime in HISTORICAL_REGIMES
    }


def _rolling_average_correlation(
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """Calculate mean off-diagonal correlation using trailing windows."""
    values = returns.to_numpy(dtype=np.float64)
    result = np.full(len(returns), np.nan, dtype=np.float64)
    for end in range(window - 1, len(returns)):
        sample = values[end - window + 1 : end + 1]
        correlations = np.corrcoef(sample, rowvar=False)
        if correlations.ndim != 2:
            result[end] = 0.0
            continue
        upper = correlations[np.triu_indices_from(correlations, k=1)]
        finite = upper[np.isfinite(upper)]
        result[end] = float(finite.mean()) if finite.size else 0.0
    return pd.Series(result, index=returns.index)


def create_causal_regime_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Create market-state statistics using observations at or before each date."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("Returns must use a DatetimeIndex.")
    returns = returns.sort_index().astype(float)
    market_return = returns.mean(axis=1)
    market_wealth = (1 + market_return).cumprod()
    rolling_peak = market_wealth.rolling(252, min_periods=63).max()

    features = pd.DataFrame(index=returns.index)
    features[REGIME_STATISTIC_COLUMNS[0]] = (
        market_return.rolling(21).std() * np.sqrt(252)
    )
    features[REGIME_STATISTIC_COLUMNS[1]] = (
        market_return.rolling(63).std() * np.sqrt(252)
    )
    features[REGIME_STATISTIC_COLUMNS[2]] = (
        (1 + market_return).rolling(63).apply(np.prod, raw=True) - 1
    )
    features[REGIME_STATISTIC_COLUMNS[3]] = (
        (1 + market_return).rolling(252).apply(np.prod, raw=True) - 1
    )
    features[REGIME_STATISTIC_COLUMNS[4]] = market_wealth / rolling_peak - 1
    features[REGIME_STATISTIC_COLUMNS[5]] = (
        returns.std(axis=1).rolling(21).mean() * np.sqrt(252)
    )
    features[REGIME_STATISTIC_COLUMNS[6]] = _rolling_average_correlation(
        returns,
        window=63,
    )
    return features.replace([np.inf, -np.inf], np.nan).dropna()


class CausalRegimeTransformer:
    """Fit threshold states and mixture probabilities on training data only."""

    def __init__(self, n_components: int = 4, random_seed: int = 42) -> None:
        self.n_components = n_components
        self.random_seed = random_seed
        self.thresholds: dict[str, float] | None = None
        self.scaler: RobustScaler | None = None
        self.model: GaussianMixture | None = None
        self.component_order: np.ndarray | None = None

    def fit(self, features: pd.DataFrame) -> "CausalRegimeTransformer":
        training = features.loc[:, REGIME_STATISTIC_COLUMNS].dropna()
        if len(training) < max(100, self.n_components * 10):
            raise ValueError("Insufficient training observations for regime fitting.")

        self.thresholds = {
            "volatility_75": float(training.iloc[:, 1].quantile(0.75)),
            "volatility_90": float(training.iloc[:, 1].quantile(0.90)),
            "momentum_25": float(training.iloc[:, 2].quantile(0.25)),
            "drawdown_25": float(training.iloc[:, 4].quantile(0.25)),
            "drawdown_10": float(training.iloc[:, 4].quantile(0.10)),
        }
        self.scaler = RobustScaler().fit(training)
        scaled = self.scaler.transform(training)
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            reg_covar=1e-5,
            n_init=5,
            random_state=self.random_seed,
        ).fit(scaled)

        # Canonicalise mixture labels from calmest to most stressed. This
        # prevents arbitrary component permutations across fitted folds.
        centres = self.scaler.inverse_transform(self.model.means_)
        risk_score = (
            centres[:, 1]
            - centres[:, 2]
            - centres[:, 4]
            + centres[:, 6]
        )
        self.component_order = np.argsort(risk_score)
        return self

    def transform(
        self,
        features: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if (
            self.thresholds is None
            or self.scaler is None
            or self.model is None
            or self.component_order is None
        ):
            raise RuntimeError("Regime transformer must be fitted before transform.")
        values = features.loc[:, REGIME_STATISTIC_COLUMNS]
        volatility = values.iloc[:, 1]
        momentum = values.iloc[:, 2]
        drawdown = values.iloc[:, 4]

        labels = pd.Series("calm_bull", index=values.index, dtype="object")
        labels.loc[volatility > self.thresholds["volatility_75"]] = (
            "volatile_bull"
        )
        labels.loc[
            (momentum < self.thresholds["momentum_25"])
            | (drawdown < self.thresholds["drawdown_25"])
        ] = "bear"
        labels.loc[
            (volatility > self.thresholds["volatility_90"])
            & (drawdown < self.thresholds["drawdown_10"])
        ] = "crisis"

        output = values.copy()
        for regime in THRESHOLD_REGIMES:
            output[f"regime_state__{regime}"] = (labels == regime).astype(float)

        probabilities = self.model.predict_proba(self.scaler.transform(values))
        probabilities = probabilities[:, self.component_order]
        for index in range(self.n_components):
            output[f"regime_probability__{index}"] = probabilities[:, index]
        return output.astype(np.float32), labels.rename("regime")

    def fit_transform(
        self,
        training_features: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        return self.fit(training_features).transform(training_features)

    def save(self, directory: Path) -> dict[str, Path]:
        if self.thresholds is None or self.model is None:
            raise RuntimeError("Cannot save an unfitted regime transformer.")
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / "regime_transformer.json"
        model_path = directory / "regime_transformer.joblib"
        metadata_path.write_text(
            json.dumps(
                {
                    "n_components": self.n_components,
                    "random_seed": self.random_seed,
                    "thresholds": self.thresholds,
                    "statistic_columns": REGIME_STATISTIC_COLUMNS,
                    "component_order": self.component_order.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        joblib.dump(self, model_path)
        return {"metadata": metadata_path, "model": model_path}

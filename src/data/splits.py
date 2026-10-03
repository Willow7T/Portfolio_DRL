"""Chronological dataset splitting utilities."""

from __future__ import annotations

import pandas as pd


WALK_FORWARD_FOLDS = (
    ("2008-01-01", "2012-12-31", "2013-12-31", "2015-12-31"),
    ("2008-01-01", "2014-12-31", "2015-12-31", "2017-12-31"),
    ("2008-01-01", "2016-12-31", "2017-12-31", "2019-12-31"),
    ("2008-01-01", "2018-12-31", "2019-12-31", "2021-12-31"),
)


def expanding_walk_forward_splits(
    returns: pd.DataFrame,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Return the four predeclared development folds through 2021."""
    folds = []
    for start, train_end, validation_end, test_end in WALK_FORWARD_FOLDS:
        sample = returns.loc[start:test_end]
        train = sample.loc[:train_end].copy()
        validation = sample.loc[
            (sample.index > pd.Timestamp(train_end))
            & (sample.index <= pd.Timestamp(validation_end))
        ].copy()
        test = sample.loc[
            (sample.index > pd.Timestamp(validation_end))
            & (sample.index <= pd.Timestamp(test_end))
        ].copy()
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError("A walk-forward fold contains an empty period.")
        folds.append((train, validation, test))
    return folds


def chronological_split(
    returns: pd.DataFrame,
    train_end: str,
    validation_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split returns chronologically to prevent look-ahead leakage."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("Returns must use a DatetimeIndex.")
    returns = returns.sort_index()
    train = returns.loc[:train_end].copy()
    validation = returns.loc[
        (returns.index > pd.Timestamp(train_end))
        & (returns.index <= pd.Timestamp(validation_end))
    ].copy()
    test = returns.loc[returns.index > pd.Timestamp(validation_end)].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "Train, validation, and test periods must all contain observations."
        )
    return train, validation, test

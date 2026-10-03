from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RETURNS_PATH = Path("data/processed/returns.csv")
FIGURE_PATH = Path("results/figures/backtest.png")


def run_backtest() -> None:
    """Run an equal-weight baseline backtest."""
    returns = pd.read_csv(RETURNS_PATH, index_col=0, parse_dates=True)
    equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
    portfolio_returns = (returns * equal_weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cumulative_returns.plot()
    plt.title("Portfolio Backtest")
    plt.ylabel("Growth of £1")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH)
    plt.show()


if __name__ == "__main__":
    run_backtest()

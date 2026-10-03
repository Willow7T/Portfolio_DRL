from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/raw/stock_prices.csv")
OUTPUT_PATH = Path("data/processed/returns.csv")


def prices_to_returns(
    prices: pd.DataFrame,
    maximum_forward_fill_days: int = 3,
) -> pd.DataFrame:
    """Create aligned returns without silently filling long data gaps."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Prices must use a DatetimeIndex.")
    prices = prices.sort_index().astype(float)
    prices = prices.ffill(limit=maximum_forward_fill_days)
    returns = prices.pct_change(fill_method=None).replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )
    returns = returns.dropna(how="any")
    if returns.empty:
        raise ValueError("No complete return observations remain.")
    return returns.astype(float)


def preprocess_market_data(
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Convert a price file into aligned daily returns."""
    data = pd.read_csv(input_path, index_col=0, parse_dates=True)
    returns = prices_to_returns(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(output_path)

    print("Preprocessing complete.")
    print(returns.head())
    return returns


def preprocess_data() -> None:
    """Convert baseline prices into daily percentage returns."""
    preprocess_market_data(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    preprocess_data()

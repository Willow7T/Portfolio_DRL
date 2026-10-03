from pathlib import Path
import sys

import yfinance as yf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.config import END_DATE, START_DATE, TICKERS
else:
    from ..utils.config import END_DATE, START_DATE, TICKERS

OUTPUT_PATH = Path("data/raw/stock_prices.csv")


def download_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    output_path: Path,
) -> None:
    """Download adjusted closing prices for a specified market universe."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    data = downloaded["Adj Close"]
    if data.empty:
        raise RuntimeError("No market data was returned by yfinance.")
    missing_tickers = sorted(set(tickers) - set(data.columns))
    if missing_tickers:
        raise RuntimeError(f"Missing downloaded tickers: {missing_tickers}")
    data = data.loc[:, tickers]
    data.to_csv(output_path)

    print("Data downloaded successfully.")
    print(data.head())


def download_data() -> None:
    """Download adjusted closing prices for the baseline assets."""
    download_market_data(TICKERS, START_DATE, END_DATE, OUTPUT_PATH)


if __name__ == "__main__":
    download_data()

"""Fetch S&P 500 tickers from alternative open datasets.

This script is intended as a fallback when the Wikipedia scraper is
blocked (for example, returning HTTP 403). It downloads the latest
constituent list from mirrored open-data sources and writes the tickers to
``training_stocks.txt`` in the project root.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Iterable, List

import csv
import requests

DATA_SOURCES: List[str] = [
    # Primary source mirrors Wikipedia data on GitHub via the datasets project.
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
    # Secondary mirror hosted by DataHub.
    "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
]

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class FetchResult:
    url: str
    tickers: List[str]


def fetch_tickers_from_csv(url: str) -> FetchResult:
    """Download and parse a CSV file of S&P 500 constituents."""

    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
    response.raise_for_status()

    handle = StringIO(response.text)
    reader = csv.DictReader(handle)

    tickers: List[str] = []
    for row in reader:
        symbol = row.get("Symbol") or row.get("symbol") or row.get("Ticker")
        if symbol:
            tickers.append(symbol.strip())

    if not tickers:
        raise ValueError(f"No tickers parsed from {url}")

    return FetchResult(url=url, tickers=tickers)


def get_sp500_tickers_from_alternatives() -> FetchResult:
    """Try each alternative source until a list of tickers is retrieved."""

    errors: List[str] = []
    for url in DATA_SOURCES:
        try:
            return fetch_tickers_from_csv(url)
        except Exception as exc:  # pragma: no cover - defensive logging only
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Failed to download S&P 500 tickers from alternative sources. "
        + " | ".join(errors)
    )


def save_tickers_to_file(tickers: Iterable[str], filename: str = "training_stocks.txt") -> None:
    tickers_list = list(tickers)
    if not tickers_list:
        raise ValueError("Ticker list is empty; nothing to write.")

    ticker_string = ",".join(tickers_list)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(ticker_string)

    print(f"Saved {len(tickers_list)} tickers to {filename}")


if __name__ == "__main__":
    result = get_sp500_tickers_from_alternatives()
    save_tickers_to_file(result.tickers)
    print(f"Fetched {len(result.tickers)} tickers from {result.url}")
    print(f"First 10 tickers: {', '.join(result.tickers[:10])}")
    print(f"Last 10 tickers: {', '.join(result.tickers[-10:])}")

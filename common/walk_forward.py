"""Walk-forward validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train: pd.DataFrame
    test: pd.DataFrame


def _parse_window(window: str) -> pd.DateOffset:
    if not isinstance(window, str) or not window.strip():
        raise ValueError("Window must be a non-empty string like '2y' or '1m'.")
    text = window.strip().lower()
    number = ""
    unit = ""
    for char in text:
        if char.isdigit():
            number += char
        else:
            unit += char
    if not number or not unit:
        raise ValueError("Window must include a number and a unit (d, w, m, y).")
    count = int(number)
    if unit == "d":
        return pd.DateOffset(days=count)
    if unit == "w":
        return pd.DateOffset(weeks=count)
    if unit == "m":
        return pd.DateOffset(months=count)
    if unit == "y":
        return pd.DateOffset(years=count)
    raise ValueError("Unsupported window unit. Use d, w, m, or y.")


def parse_window(window: str) -> pd.DateOffset:
    """Public wrapper for parsing window strings like '2y' or '1m'."""
    return _parse_window(window)


def walk_forward_splits(
    data: pd.DataFrame,
    train_window: str,
    test_window: str,
    embargo: int,
    step: Optional[str] = None,
    min_train_rows: int = 30,
) -> Generator[WalkForwardSplit, None, None]:
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must be indexed by DatetimeIndex.")

    if data.empty:
        return

    train_offset = _parse_window(train_window)
    test_offset = _parse_window(test_window)
    step_offset = _parse_window(step or test_window)
    embargo_rows = int(embargo)
    if embargo_rows < 0:
        raise ValueError("embargo must be non-negative.")

    index = data.index.sort_values()
    first_date = index.min()
    last_date = index.max()

    train_end = first_date + train_offset
    fold = 0

    while True:
        train_start = train_end - train_offset

        train_mask = (index > train_start) & (index <= train_end)
        train_df = data.loc[train_mask]
        if train_df.empty:
            train_end = train_end + step_offset
            if train_end >= last_date:
                break
            continue

        # Use row-based embargo so target horizons (which are row-based shifts)
        # cannot overlap the first test row, including holiday-affected calendars.
        train_end_pos = int(index.searchsorted(train_df.index.max(), side="right") - 1)
        first_test_pos = train_end_pos + embargo_rows + 1
        if first_test_pos >= len(index):
            break

        test_start = index[first_test_pos]
        test_end = test_start + test_offset
        if test_end > last_date:
            break

        test_mask = (index >= test_start) & (index <= test_end)
        test_df = data.loc[test_mask]

        if test_df.empty:
            break

        if len(train_df) >= min_train_rows:
            fold += 1
            yield WalkForwardSplit(
                fold=fold,
                train_start=train_df.index.min(),
                train_end=train_df.index.max(),
                test_start=test_df.index.min(),
                test_end=test_df.index.max(),
                train=train_df,
                test=test_df,
            )

        train_end = train_end + step_offset
        if train_end >= last_date:
            break

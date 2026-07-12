import unittest

import pandas as pd

from common.backtesting import Order, OrderRequest, OrderSide, OrderStatus, OrderType
from rl_portfolio_management.extract_ticker_attribution import fifo_trade_stats, reconcile_rows


def fill(order_id, side, qty, price, timestamp):
    request = OrderRequest("AAPL", side, qty, OrderType.MARKET)
    ts = pd.Timestamp(timestamp)
    return Order(order_id, request, ts - pd.Timedelta(days=1), ts - pd.Timedelta(days=1),
                 OrderStatus.FILLED, qty, price, ts)


class TickerAttributionTest(unittest.TestCase):
    def test_fifo_reversal_and_trading_bar_holding(self):
        index = pd.DatetimeIndex(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"])
        orders = [
            fill(1, OrderSide.BUY, 2, 100, index[0]),
            fill(2, OrderSide.BUY, 1, 110, index[1]),
            fill(3, OrderSide.SELL, 4, 120, index[2]),
            fill(4, OrderSide.BUY, 1, 115, index[3]),
        ]
        result = fifo_trade_stats(orders, index)["AAPL"]
        self.assertEqual(result["completed_trade_count"], 3)
        self.assertAlmostEqual(result["completed_trade_avg_pnl"], (40 + 10 + 5) / 3)
        self.assertAlmostEqual(result["completed_trade_median_holding_bars"], 1)
        self.assertAlmostEqual(result["long_matched_contribution"], 50)
        self.assertAlmostEqual(result["short_matched_contribution"], 5)

    def test_reconciliation(self):
        rows = [{"total_contribution": 30.25}, {"total_contribution": -10.0}]
        self.assertAlmostEqual(reconcile_rows(rows, 10020.25), 0)
        with self.assertRaises(AssertionError):
            reconcile_rows(rows, 10020.0)


if __name__ == "__main__":
    unittest.main()

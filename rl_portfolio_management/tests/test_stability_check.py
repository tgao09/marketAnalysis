import unittest

from rl_portfolio_management.stability_check import classify_isolated_spike


class IsolatedSpikeTests(unittest.TestCase):
    def test_flags_statistical_spike(self):
        self.assertTrue(classify_isolated_spike(3.0, [1.0, 1.1, 0.9, 1.0]))

    def test_flags_majority_negative_or_invalid(self):
        self.assertTrue(classify_isolated_spike(0.5, [-1000.0, -0.2, -0.1, 0.4]))

    def test_accepts_supported_optimum(self):
        self.assertFalse(classify_isolated_spike(1.1, [0.8, 1.0, 1.2, 0.9]))


if __name__ == "__main__":
    unittest.main()

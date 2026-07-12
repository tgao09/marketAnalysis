"""Common helpers shared across strategies."""

from .yfinance_utils import (
    DEFAULT_SECTOR_ETF_MAP,
    canonicalize_sector_name,
    configure_yfinance_cache,
    get_history,
    get_info,
    get_sector_etf_metrics,
    get_sector_holdings,
    get_ticker,
    resolve_sector_etf_for_symbol,
)
from .walk_forward import WalkForwardSplit, parse_window, walk_forward_splits
from .sanity_checks import SanityConfig, SanityReport, print_report, run_sanity_checks, write_report_json
from .pca_utils import PCATransformer, load_pca_json, save_pca_json, select_k_by_variance
from .significance import (
    SignificanceConfig,
    SignificanceReport,
    print_significance_report,
    run_significance,
    write_significance_json,
)

__all__ = [
    "DEFAULT_SECTOR_ETF_MAP",
    "canonicalize_sector_name",
    "configure_yfinance_cache",
    "get_history",
    "get_info",
    "get_sector_etf_metrics",
    "get_sector_holdings",
    "get_ticker",
    "resolve_sector_etf_for_symbol",
    "WalkForwardSplit",
    "parse_window",
    "walk_forward_splits",
    "SanityConfig",
    "SanityReport",
    "run_sanity_checks",
    "print_report",
    "write_report_json",
    "PCATransformer",
    "select_k_by_variance",
    "save_pca_json",
    "load_pca_json",
    "SignificanceConfig",
    "SignificanceReport",
    "run_significance",
    "print_significance_report",
    "write_significance_json",
]

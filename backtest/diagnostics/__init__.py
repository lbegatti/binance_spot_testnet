"""
backtest/diagnostics/
---------------------
Standalone diagnostic tools for the backtesting framework.

These scripts are **not** wired into ``backtest/runner.py`` and must be
run independently.  They are kept in a separate sub-package to make it
immediately clear that they are one-off checks, not part of the main
backtesting pipeline.

Available tools
---------------
regime_validation.py
    Offline long-horizon HMM regime validation.
    Tests whether the regime labels from ``RegimeDirector`` remain
    statistically meaningful on a fully out-of-sample 3-day test set.

    Run with:
        python -m backtest.diagnostics.regime_validation
"""

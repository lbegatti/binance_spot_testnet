"""Tier B tests for strategy/param_loader.py — the best_params.json consumer."""

import json

import pytest

import strategy.param_loader as pl
from strategy.param_loader import load_best_params_for_backtest, rows_to_lookback


@pytest.mark.parametrize(
    "rows, expected",
    [
        (120, "10 hours ago UTC"),  # 600 min, exact hours
        (12, "1 hour ago UTC"),  # 60 min, singular
        (24, "2 hours ago UTC"),  # 120 min
        (1, "5 minutes ago UTC"),  # < 60 min
        (13, "65 minutes ago UTC"),  # non-exact hour → minutes form
    ],
)
def test_rows_to_lookback(rows, expected):
    """rows_to_lookback converts a row count into Binance's human lookback
    string, using the "hours" form on exact hours and "minutes" otherwise.
    The @parametrize cases above each exercise one branch of that rule."""
    assert rows_to_lookback(rows) == expected


def test_load_best_params_for_backtest_missing_returns_empty(tmp_path, monkeypatch):
    """When best_params.json is absent the loader returns {} (the backtest
    then falls back to config defaults) rather than raising. monkeypatch
    redirects BEST_PARAMS_PATH to a non-existent temp file for the test."""
    monkeypatch.setattr(pl, "BEST_PARAMS_PATH", tmp_path / "nope.json")
    assert load_best_params_for_backtest() == {}


def test_load_best_params_for_backtest_reads_and_casts(tmp_path, monkeypatch):
    """A valid file is read, only the whitelisted keys are kept (extras like
    generated_at are dropped), and each value is cast to its declared type
    (ints stay int, thresholds become float)."""
    p = tmp_path / "best.json"
    p.write_text(
        json.dumps(
            {
                "hmm_lookback_rows": 120,
                "hmm_max_regimes": 3,
                "vwap_window": 20,
                "vwap_threshold": 0.002,
                "fee_rate": 0.001,
                "generated_at": "2026-01-01T00:00:00+00:00",  # extra key → ignored
            }
        )
    )
    monkeypatch.setattr(pl, "BEST_PARAMS_PATH", p)
    result = load_best_params_for_backtest()
    assert result == {
        "hmm_lookback_rows": 120,
        "hmm_max_regimes": 3,
        "vwap_window": 20,
        "vwap_threshold": 0.002,
        "fee_rate": 0.001,
    }
    assert isinstance(result["hmm_lookback_rows"], int)
    assert isinstance(result["vwap_threshold"], float)


def test_load_best_params_for_backtest_corrupt_returns_empty(tmp_path, monkeypatch):
    """Malformed JSON is caught and treated as "no params": the loader
    returns {} instead of crashing the backtest at startup."""
    p = tmp_path / "bad.json"
    p.write_text("{ not valid json ")
    monkeypatch.setattr(pl, "BEST_PARAMS_PATH", p)
    assert load_best_params_for_backtest() == {}

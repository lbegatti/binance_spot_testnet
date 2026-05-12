"""
strategy/param_loader.py
------------------------
Loads the best parameter set produced by ``backtest/sensitivity.py`` and
applies it to the live trading system before ``RegimeDirector()`` is
instantiated.

Why ``strategy/`` and not ``backtest/``?
    ``backtest/`` contains the pipeline that *produces* ``best_params.json``.
    This module is the *consumer* used by the live system — it belongs in
    ``strategy/`` alongside the other runtime components.

    ``backtest/runner.py`` has its own independent loader
    (``_load_best_params_for_backtest()``) that passes the values as kwargs
    to ``run_signals()`` and ``simulate_pnl()`` — no module patching needed
    there because the backtest functions accept explicit overrides.

Usage
-----
    # In websocket_main.py — call AFTER logging.basicConfig(), BEFORE RegimeDirector():
    from strategy.param_loader import load_best_params
    load_best_params()

How it patches the live system
-------------------------------
``regime_director.py`` binds its constants at import time via::

    from config_parameters import HMM_MAX_REGIMES, HMM_LOOKBACK, ...

Patching ``config_parameters`` *after* the module is loaded has no effect
because Python has already copied the values into ``regime_director``'s own
namespace.  The only correct approach is to patch
``strategy.regime_director`` directly::

    import strategy.regime_director as rd_mod
    rd_mod.HMM_MAX_REGIMES = 2     # takes effect on the next RegimeDirector()

Fields NOT overridden
----------------------
- ``vwap_window``  — backtest-only constant (``backtest/signals.py``).
                     The live system does not use ``VWAP_WINDOW``.
- ``fee_rate``     — Binance charges its own fees regardless of this value.
                     Stored in ``best_params.json`` for *reference only* — it
                     shows the fee level at which the sensitivity sweep was
                     optimal, which is useful diagnostic information but should
                     NOT be used to override the simulation fee.  The optimizer
                     converging on the lowest tested fee (e.g. 0.00025) is a
                     signal that strategy alpha is thin at realistic fees
                     (0.001 = standard Binance Spot taker); simulating with the
                     lower value produces ~4× over-optimistic P&L.
                     ``runner.py`` deliberately ignores this field and
                     always uses ``BACKTEST_FEE_RATE`` from ``config_parameters.py``.

Why patching ``config_parameters`` directly would NOT work
-----------------------------------------------------------
A common misconception: if ``config_parameters.HMM_LOOKBACK = "2 hours ago UTC"``
is the source, why not just do::

    import config_parameters
    config_parameters.HMM_LOOKBACK = "1 hour ago UTC"   # ← has NO EFFECT on RegimeDirector

When Python executes::

    # inside regime_director.py (top of file, at import time)
    from config_parameters import HMM_LOOKBACK

it **copies** the string value ``"2 hours ago UTC"`` into a new name living
inside the ``strategy.regime_director`` module namespace.  After that point,
``config_parameters.HMM_LOOKBACK`` and ``strategy.regime_director.HMM_LOOKBACK``
are **two independent variables** that happen to hold the same string.
Patching one does not affect the other.

The correct target is therefore the copy that ``regime_director.py`` actually
reads at runtime::

    import strategy.regime_director as rd_mod
    rd_mod.HMM_LOOKBACK = "1 hour ago UTC"   # ← patches the copy RegimeDirector reads

Step-by-step: how ``"2 hours ago UTC"`` becomes ``"1 hour ago UTC"``
---------------------------------------------------------------------
1. **At import time** — ``regime_director.py`` is first imported:

   * ``from config_parameters import HMM_LOOKBACK`` runs.
   * Python creates ``strategy.regime_director.HMM_LOOKBACK = "2 hours ago UTC"``
     (a separate copy).
   * ``config_parameters.HMM_LOOKBACK`` is also ``"2 hours ago UTC"`` and
     stays that way **forever**.

2. **``load_best_params()`` is called** (before ``RegimeDirector()``):

   * ``import strategy.regime_director as rd_mod`` fetches the already-imported
     module from ``sys.modules`` — no re-execution, just a reference.
   * ``rd_mod.HMM_LOOKBACK = "1 hour ago UTC"`` overwrites the **copy**
     inside ``strategy.regime_director``'s namespace.
   * ``config_parameters.HMM_LOOKBACK`` is still ``"2 hours ago UTC"``.

3. **``RegimeDirector()`` is instantiated**:

   * ``__init__`` receives no ``lookback`` argument, so the ``None`` sentinel
     is in effect.
   * The body executes::

         self.lookback = lookback if lookback is not None else HMM_LOOKBACK

   * Python resolves the bare name ``HMM_LOOKBACK`` by looking it up in
     ``regime_director``'s **own** module namespace (the LEGB rule: Local →
     Enclosing → **Global** → Built-in, where "Global" means the module).
   * That namespace now holds ``"1 hour ago UTC"`` (from step 2).
   * Result: ``self.lookback = "1 hour ago UTC"``.

Why the ``None`` sentinel matters
----------------------------------
If ``__init__`` were written as::

    def __init__(self, lookback: str = HMM_LOOKBACK, ...):

Python evaluates the default expression ``HMM_LOOKBACK`` **once, at the
``def`` statement** (still import time).  The frozen default is
``"2 hours ago UTC"`` and never changes — even if
``rd_mod.HMM_LOOKBACK`` is later patched.  The ``None`` sentinel forces
Python to re-read the module-level name **each time** ``__init__`` is called,
making the patch visible.

HMM_LOOKBACK conversion
------------------------
``best_params.json`` stores ``hmm_lookback_rows`` (``int``) because
``sensitivity.py`` is a backtest tool that counts rows.  The live system
uses ``HMM_LOOKBACK`` (a dateutil string, e.g. ``"40 minutes ago UTC"``).
Since 1 candle = 1 minute, the conversion is: ``rows_to_lookback(n)``
which produces ``"N minutes ago UTC"`` for any positive integer N.
This means any value Optuna discovers (30, 40, 70, …) is applied correctly
without a static lookup table.
"""

import json
import logging
import pathlib

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Fixed — always resolves relative to this file's location (strategy/)
BEST_PARAMS_PATH = (
    pathlib.Path(__file__).parent.parent / "backtest" / "results" / "best_params.json"
)


# Converts hmm_lookback_rows (int, 1-minute candles) → HMM_LOOKBACK dateutil
# string used by the live system.  1 row = 1 minute, so the conversion is
# direct: 40 rows → "40 minutes ago UTC".
# A static lookup table was used previously but broke whenever Optuna discovered
# a value outside the hand-coded set (e.g. 40). The function below handles any
# positive integer correctly.


def rows_to_lookback(rows: int) -> str:
    """Convert a candle count to a dateutil lookback string (1 row = 1 minute)."""
    if rows < 60:
        return f"{rows} minutes ago UTC"
    hours, mins = divmod(rows, 60)
    if mins == 0:
        return f"{hours} hour{'s' if hours != 1 else ''} ago UTC"
    return f"{rows} minutes ago UTC"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_best_params() -> None:
    """
    If ``best_params.json`` exists, patch ``strategy.regime_director``'s
    module namespace with the tuned HMM parameters before
    ``RegimeDirector()`` is instantiated.

    This function is idempotent — safe to call multiple times (each call
    simply re-reads the file and re-applies the overrides).

    Call this **after** ``logging.basicConfig()`` and **before** the line::

        regime_director = RegimeDirector()

    Parameters overridden in ``strategy.regime_director``:

    +-----------------------+-----------------------------+
    | JSON field            | Module attribute patched    |
    +=======================+=============================+
    | ``hmm_max_regimes``   | ``HMM_MAX_REGIMES``         |
    +-----------------------+-----------------------------+
    | ``hmm_lookback_rows`` | ``HMM_LOOKBACK`` (string,   |
    |                       | via ``ROWS_TO_LOOKBACK``)   |
    +-----------------------+-----------------------------+

    Additionally patches ``strategy.analysis.VWAP_THRESHOLD_MULTIPLIER``
    when ``vwap_threshold`` is present in ``best_params.json``.

    Falls back to ``config_parameters.py`` defaults silently when:

    - ``best_params.json`` is absent (WARNING logged).
    - The file cannot be parsed (WARNING logged).
    - ``hmm_lookback_rows`` is not in ``ROWS_TO_LOOKBACK`` (WARNING logged,
      ``HMM_LOOKBACK`` left unchanged).
    """
    import strategy.regime_director as rd_mod  # deferred — avoids import cycle
    import strategy.analysis as analysis_mod  # deferred — same reason

    if not BEST_PARAMS_PATH.exists():
        logging.warning(
            "param_loader: no best_params.json at %s — "
            "using config_parameters.py defaults.",
            BEST_PARAMS_PATH,
        )
        return

    try:
        with BEST_PARAMS_PATH.open() as fh:
            best = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning(
            "param_loader: could not read %s (%s) — using defaults.",
            BEST_PARAMS_PATH,
            exc,
        )
        return

    # ── Override HMM_MAX_REGIMES ──────────────────────────────────────────
    if "hmm_max_regimes" in best:
        rd_mod.HMM_MAX_REGIMES = int(best["hmm_max_regimes"])
        logging.info("param_loader: HMM_MAX_REGIMES = %d", rd_mod.HMM_MAX_REGIMES)

    # ── Override HMM_LOOKBACK (convert int rows → dateutil string) ────────
    if "hmm_lookback_rows" in best:
        rows = int(best["hmm_lookback_rows"])
        lookback_str = rows_to_lookback(rows)
        rd_mod.HMM_LOOKBACK = lookback_str
        logging.info(
            "param_loader: HMM_LOOKBACK = '%s' (%d rows)",
            lookback_str,
            rows,
        )

    # ── Override VWAP_THRESHOLD_MULTIPLIER in analysis.py ────────────────
    if "vwap_threshold" in best:
        analysis_mod.VWAP_THRESHOLD_MULTIPLIER = float(best["vwap_threshold"])
        logging.info(
            "param_loader: VWAP_THRESHOLD_MULTIPLIER = %.5f (%.3f %%)",
            analysis_mod.VWAP_THRESHOLD_MULTIPLIER,
            analysis_mod.VWAP_THRESHOLD_MULTIPLIER * 100,
        )

    logging.info(
        "param_loader: best_params applied (generated %s, %s=%.4f).",
        best.get("generated_at", "unknown"),
        best.get("source_metric", "?"),
        best.get("source_value", float("nan")),
    )


def load_best_params_for_backtest() -> dict:
    """
    Load tuned parameters from ``best_params.json`` for use in the backtest.

    Unlike :func:`load_best_params` (which patches ``strategy.regime_director``
    for the live system), this function simply reads the JSON and returns the
    relevant fields as a plain ``dict``.  The caller passes them as keyword
    arguments to ``run_signals()`` and ``simulate_pnl()``, both of which
    already accept ``None`` values and fall back to ``config_parameters.py``
    defaults automatically.

    No module-namespace patching is needed here because ``run_signals()``
    already accepts explicit overrides (``hmm_lookback_rows``,
    ``hmm_max_regimes``, ``vwap_window``) as function kwargs.

    Returns
    -------
    dict
        Zero or more of the following keys:

        +-----------------------+----------+
        | Key                   | Type     |
        +=======================+==========+
        | ``hmm_lookback_rows`` | ``int``  |
        +-----------------------+----------+
        | ``hmm_max_regimes``   | ``int``  |
        +-----------------------+----------+
        | ``vwap_window``       | ``int``  |
        +-----------------------+----------+
        | ``vwap_threshold``    | ``float``|
        +-----------------------+----------+
        | ``fee_rate``          | ``float``|
        +-----------------------+----------+

        Returns ``{}`` (empty dict) when ``best_params.json`` is absent or
        unreadable — the caller's ``dict.get()`` calls will return ``None``,
        which triggers the default-fallback path in ``run_signals()`` and
        ``simulate_pnl()``.
    """
    if not BEST_PARAMS_PATH.exists():
        logging.warning(
            "param_loader: no best_params.json at %s — "
            "using config_parameters.py defaults for backtest.",
            BEST_PARAMS_PATH,
        )
        return {}

    try:
        with BEST_PARAMS_PATH.open() as fh:
            best = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning(
            "param_loader: could not read %s (%s) — using defaults.",
            BEST_PARAMS_PATH,
            exc,
        )
        return {}

    logging.info(
        "param_loader: backtest params loaded from %s  (generated %s, %s=%.4f)",
        BEST_PARAMS_PATH,
        best.get("generated_at", "unknown"),
        best.get("source_metric", "?"),
        best.get("source_value", float("nan")),
    )

    # Return only the keys that the backtest pipeline consumes.
    # Unknown / extra keys (e.g. generated_at) are intentionally excluded.
    result = {}
    for key, cast in (
        ("hmm_lookback_rows", int),
        ("hmm_max_regimes", int),
        ("vwap_window", int),
        ("vwap_threshold", float),
        ("fee_rate", float),
    ):
        if key in best:
            result[key] = cast(best[key])
    return result

"""
strategy/position_store.py
--------------------------
Lightweight persistence for the live session's open position, so a carried
position's true cost basis survives a restart (used only when
FLATTEN_ON_START = False).

This is a DECORATION around the strategy — it never changes trade logic. Both
functions are fail-safe: ``save_position`` never lets an I/O error propagate
into session shutdown, and ``load_position`` returns ``None`` on any problem
(missing file, corrupt JSON, schema/symbol mismatch) so the caller simply falls
back to its default behaviour. Deleting the state file reverts the bot to the
session-start-price anchor exactly as if persistence did not exist.

State file (default ``state/live_position.json``)::

    {
      "schema": 1,
      "symbol": "BTCUSDT",
      "position_open": true,
      "avg_entry_price": 58966.06,
      "btc_qty": 0.40412,
      "saved_at": "2026-06-27T16:35:56Z"
    }
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# Bump if the on-disk shape changes incompatibly; load() rejects other values.
_SCHEMA_VERSION = 1


def save_position(
    path: str,
    *,
    position_open: bool,
    avg_entry_price: float,
    btc_qty: float,
    symbol: str,
) -> None:
    """
    Atomically write the current open-position state to ``path``.

    Writes a temp file in the same directory then ``os.replace``s it into place,
    so a crash mid-write cannot leave a corrupt file. Any exception is logged
    and swallowed — persistence must never break session shutdown.

    Args:
        path: Destination JSON path (parent dirs are created).
        position_open: ``AnalysisEngine._position_open`` at shutdown.
        avg_entry_price: ``AnalysisEngine._avg_entry_price`` (cost-basis anchor).
        btc_qty: Total BTC held at shutdown (free + locked).
        symbol: Trading pair, stored so a stale file for a different pair is
            rejected on load.
    """
    payload = {
        "schema": _SCHEMA_VERSION,
        "symbol": symbol,
        "position_open": bool(position_open),
        "avg_entry_price": float(avg_entry_price),
        "btc_qty": float(btc_qty),
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, path)  # atomic
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        log.info(
            "Position state saved: open=%s avg_entry=%.2f qty=%.8f → %s",
            payload["position_open"],
            payload["avg_entry_price"],
            payload["btc_qty"],
            path,
        )
    except Exception as exc:  # never fatal
        log.warning("Could not save position state to %s (%s) — non-fatal.", path, exc)


def load_position(path: str, *, symbol: str | None = None) -> dict | None:
    """
    Read a previously saved position state.

    Returns the parsed dict, or ``None`` if the file is missing, unreadable, not
    valid JSON, has an unexpected ``schema``, or (when ``symbol`` is given) was
    saved for a different pair. Never raises — the caller treats ``None`` as
    "no usable state, use defaults".

    Args:
        path: JSON path to read.
        symbol: If provided, reject a file whose stored ``symbol`` differs.

    Returns:
        dict | None: keys ``position_open``, ``avg_entry_price``, ``btc_qty``,
        ``saved_at``, ``symbol`` on success; ``None`` otherwise.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception as exc:
        log.warning("Could not read position state %s (%s) — ignoring.", path, exc)
        return None
    if data.get("schema") != _SCHEMA_VERSION:
        log.warning(
            "Position state %s has schema %s (expected %s) — ignoring.",
            path, data.get("schema"), _SCHEMA_VERSION,
        )
        return None
    if symbol is not None and data.get("symbol") != symbol:
        log.warning(
            "Position state %s is for %s, not %s — ignoring.",
            path, data.get("symbol"), symbol,
        )
        return None
    return data

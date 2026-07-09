"""
tests/conftest.py — shared pytest configuration for the Tier B suite.

Responsibilities:
  * Make the project root importable so ``from strategy... import ...`` works no
    matter which directory pytest is launched from.
  * Seed the RNGs before every test for reproducibility (belt-and-braces — none
    of the Tier B tests currently rely on randomness, but the guarantee is cheap).

Determinism guarantees for this suite:
  * No test makes a network call (Binance clients are mocked via pytest-mock).
  * No test writes outside pytest's built-in ``tmp_path`` fixture.
  * Tests pin the *current* behaviour of the code — a test that would require a
    source change to pass is a finding, not a silent edit.

Run against the canonical environment (.venv314):
    .venv314/bin/python -m pytest tests/ -q
"""

import pathlib
import random
import sys

import numpy as np
import pytest

# Make the project root (parent of tests/) importable.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(autouse=True)
def _seed_rngs():
    """Seed numpy + the stdlib RNG before every test."""
    np.random.seed(0)
    random.seed(0)
    yield

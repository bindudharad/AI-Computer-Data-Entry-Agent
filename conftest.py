"""Pytest fixtures and path setup for the ATLAS AI test suite."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

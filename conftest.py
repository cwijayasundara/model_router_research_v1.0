"""Pytest bootstrap: always test the live source tree in src/, not the installed copy.

This keeps the tests independent of how (or whether) the package is installed.
"""

import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

"""Ensures the repository root is importable so ``import app`` works under pytest.

pytest's default "prepend" import mode adds the test file's directory (tests/) to
sys.path, not the repo root. Placing a conftest.py at the root makes pytest add the
root to sys.path, so the tests can import app.py regardless of the runner's CWD.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

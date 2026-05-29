"""Make a sibling VFE_2.0 checkout importable for golden equivalence tests.

Resolution order for the 2.0 root:
  1. env var VFE2_ROOT
  2. sibling directory ../VFE_2.0 relative to this repo root

If the 2.0 checkout cannot be found or imported, golden tests are skipped
(not failed) so the suite still runs in environments without 2.0 present.
"""

import os
import sys
from pathlib import Path

import pytest

# repo root = parents[2] of this file (tests/golden/ -> tests/ -> repo root)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _vfe2_root() -> Path | None:
    env = os.environ.get("VFE2_ROOT")
    if env:
        p = Path(env)
        return p if p.exists() else None
    sibling = _REPO_ROOT.parent / "VFE_2.0"
    return sibling if sibling.exists() else None


@pytest.fixture(scope="session")
def vfe2_kl():
    """Return the 2.0 kl_computation module, or skip if unavailable."""
    root = _vfe2_root()
    if root is None:
        pytest.skip("VFE_2.0 checkout not found (set VFE2_ROOT)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from transformer.core import kl_computation
    except ImportError as exc:  # genuinely missing -> skip; other errors surface
        pytest.skip(f"could not import VFE_2.0 kl_computation: {exc}")
    return kl_computation

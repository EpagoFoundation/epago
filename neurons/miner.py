"""Model-miner entrypoint — a thin shim over :mod:`epago.miner.cli`.

Usage:

    python neurons/miner.py preflight --model-dir ./challenger ...

The same commands are available as ``epago miner ...`` on an installed
package; this file exists only so the conventional ``neurons/miner.py`` path
works from a bare checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from epago.miner.cli import app  # noqa: E402

if __name__ == "__main__":
    app()

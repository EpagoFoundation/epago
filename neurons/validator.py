"""Validator neuron entrypoint — a thin shim over ``epago.cli``.

Usage:

    python neurons/validator.py run --netuid 42 --network finney \
        --state-dir ~/.epago/validator --corpus /data/corpus.sqlite

All real wiring (chain client, eval deps, taskgen, state) lives in
:mod:`epago.cli`; this file exists only so the conventional
``neurons/validator.py`` path works.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a bare checkout (``python neurons/validator.py ...``)
# without an editable install: put the repo root on sys.path.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from epago.cli import app  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    app(["validator", *(argv or ["--help"])])


if __name__ == "__main__":
    main()

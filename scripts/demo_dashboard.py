#!/usr/bin/env python
"""Generate a self-contained demo dashboard from a synthetic subnet history.

Runs an extended sandbox soak (many miners, several dethrones, near-misses,
spam, calibration) to produce realistic validator artifacts, exports
``dashboard.json`` from them, and writes ``demo/index.html`` with the data
inlined — one file, openable from disk, no server.

Usage:
    .venv/bin/python scripts/demo_dashboard.py [--iterations 30] [--out demo/]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "demo")
    args = parser.parse_args()

    import subprocess
    import tempfile

    from epago.config import load_config
    from epago.dashboard.export import export_dashboard, load_dashboard_inputs

    soak_out = Path(tempfile.mkdtemp(prefix="epago-demo-soak-"))
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "sandbox_soak.py"),
            "--iterations",
            str(args.iterations),
            "--artifacts-dir",
            str(soak_out),
        ],
        check=True,
    )
    state_dir = sorted(p for p in soak_out.iterdir() if p.is_dir())[-1]

    # The soak writes flat artifacts; the exporter expects the validator layout.
    staged = args.out / "_state"
    (staged / "audit").mkdir(parents=True, exist_ok=True)
    (staged / "state.json").write_bytes((state_dir / "state.json").read_bytes())
    (staged / "audit" / "audit.jsonl").write_bytes((state_dir / "audit.jsonl").read_bytes())

    cfg = load_config()
    data = export_dashboard(load_dashboard_inputs(staged, cfg))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dashboard.json").write_text(json.dumps(data, indent=1, sort_keys=True))

    site = (REPO_ROOT / "leaderboard" / "index.html").read_text()
    slot = "<!-- EPAGO_DATA_SLOT -->"
    if slot not in site:
        raise SystemExit("EPAGO_DATA_SLOT marker not found in leaderboard/index.html")
    payload = json.dumps(data, sort_keys=True).replace("<", "\\u003c")
    inline = "<script>window.EPAGO_DATA = " + payload + ";</script>"
    (args.out / "index.html").write_text(site.replace(slot, inline, 1))

    print(f"demo dashboard: {args.out / 'index.html'}")
    print(f"duels={len(data['duels'])} coronations={len(data['lineage'])} miners={len(data['miners'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

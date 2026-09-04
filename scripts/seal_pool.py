"""Seal a minted pool: write its manifest and print the digests to commit.

A sealed pool is only a commitment once two digests are fixed in the contract
*before* a round opens. This computes both and writes the manifest, so the
operator's job is reduced to pasting four lines and publishing one file.

The two artifacts exist for different readers:

``<pool>``
    Every task with its answer. Stays sealed while the pool is in service —
    publishing it would hand every miner the answer key to rounds that have not
    happened yet.

``<pool>-manifest.json``
    The pool's task ids and nothing else. Published immediately, because an
    opaque id list is useless for training but sufficient to prove which tasks
    a round drew: selection runs over the sorted ids alone.

Order matters, and no later correction undoes getting it wrong. Commit both
digests, publish the manifest, and only then open a round. Publishing the pool
file, or opening a round before the digests are committed, gives miners an exam
they can train on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epago.taskgen.sealed_pool import (  # noqa: E402
    Manifest,
    load_pool,
    pool_digest,
    write_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="the minted pool, JSONL")
    ap.add_argument("--manifest", default="", help="output path (default: <pool>-manifest.json)")
    ap.add_argument("--release", default="POOL1", help="release name for the contract snippet")
    ap.add_argument(
        "--n-pub-tasks",
        type=int,
        default=0,
        help="exam size, to report how many rounds this pool can serve",
    )
    args = ap.parse_args()

    pool_path = Path(args.pool)
    raw = pool_path.read_bytes()
    digest = pool_digest(raw)

    # Loaded through the same path a validator uses, so a pool that would be
    # refused at duel time is refused here instead -- while it can still be
    # re-minted, rather than after its digest is committed on chain.
    tasks = load_pool(pool_path, digest)

    manifest_path = Path(args.manifest) if args.manifest else pool_path.with_name(
        pool_path.stem + "-manifest.json"
    )
    manifest = Manifest.from_pool(tasks, digest)
    manifest_digest = write_manifest(manifest, manifest_path)

    print(f"tasks              {len(tasks)}")
    print(f"manifest           {manifest_path}")
    if args.n_pub_tasks > 0:
        rounds = len(tasks) // args.n_pub_tasks
        print(f"rounds served      {rounds} at {args.n_pub_tasks} tasks each")
        if rounds < 2:
            print("  WARNING: a pool this size serves barely one round")
    print()
    print("Commit these in the contract BEFORE opening a round:")
    print()
    print(f'taskgen_release             = "{args.release}"')
    print(f'public_pool_path            = "{pool_path.resolve()}"')
    print(f'public_pool_digest          = "{digest}"')
    print(f'public_pool_manifest_path   = "{manifest_path.resolve()}"')
    print(f'public_pool_manifest_digest = "{manifest_digest}"')
    print()
    print("Then publish the manifest. Keep the pool file sealed until it retires.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

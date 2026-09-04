"""Mint two-anchor intersection tasks and write them as JSONL.

The question describes two papers, says what kind of thing each one names, and
asks for the single paper that involves both. The answer is never described, so
there is no sentence in the question that a solver can lift out and search.

Stages:

1. draw ``(gold, bridge_x, bridge_y)`` triples from the entity index
2. prove ``docs(X) & docs(Y) == {gold}`` and pick two anchors that carry one
   bridge each and are not the answer
3. verify both anchors are reachable and the question does not surface the gold
4. verbalize offline: type both bridges, reject coincidences, describe the
   anchors
5. re-verify on the assembled question
6. emit

Minting is offline. It runs ahead of a rotation and its output is committed by
digest before any duel uses it.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import re

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epago.core.types import TaskOrigin  # noqa: E402
from epago.environment.corpus import SqliteCorpus  # noqa: E402
from epago.taskgen.chain import (  # noqa: E402
    INTERSECTION_TEMPLATE_NAME,
    INTERSECTION_TIERS,
    IntersectionMinter,
    nameable_anchors,
    tiers_available,
    verify_intersection_route,
)
from epago.taskgen.entities import EntityIndex  # noqa: E402
from epago.taskgen.templates import content_task_id  # noqa: E402
from epago.taskgen.verbalize import (  # noqa: E402
    VERBALIZER_VERSION,
    assemble_intersection_question,
    verbalize_intersection,
)


def _template_question(skeleton) -> str:
    """A stand-in for the pre-verification pass, before wording exists."""
    return (
        f"One study is about {', '.join(skeleton.anchor_a_clues)}. "
        f"Another is about {', '.join(skeleton.anchor_b_clues)}. "
        "Exactly one study involves something named in each. What is its title?"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/corpus-science-big/corpus.db")
    ap.add_argument("--index", default="data/corpus-science-big/entities-v1.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--search-workers", type=int, default=24)
    ap.add_argument("--candidate-factor", type=int, default=12)
    ap.add_argument(
        "--tier-mix",
        default="named_both:0.35,named_one:0.35,described_both:0.30",
        help=(
            "how much of the exam each length should be. An exam of only the "
            "hardest form cannot rank models: 78% of its items went unsolved "
            "and the two arms differed by less than the noise."
        ),
    )
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    index = EntityIndex.read(Path(args.index))
    conn = sqlite3.connect(args.corpus)
    docs = {
        row[0]: (row[1], row[2])
        for row in conn.execute("SELECT doc_id, title, text FROM docs")
    }
    titles = {k: v[0] for k, v in docs.items()}

    # One SQLite handle per worker thread: connections are not shareable, and
    # the snapshot is immutable for the whole run.
    _local = threading.local()

    def corpus() -> SqliteCorpus:
        existing = getattr(_local, "corpus", None)
        if existing is None:
            existing = SqliteCorpus(args.corpus)
            _local.corpus = existing
        return existing

    minter = IntersectionMinter(index, titles, rng=np.random.default_rng(args.seed))

    candidates = minter.intersection_candidates(args.n * args.candidate_factor)
    reasons: Counter[str] = Counter()
    proved = [
        s
        for s in (minter.build_intersection(*t, reasons=reasons) for t in candidates)
        if s is not None
    ]
    print(f"[mint] candidates {len(candidates)}", flush=True)
    print(f"[mint] proved     {len(proved)}", flush=True)
    for key, count in reasons.most_common(8):
        print(f"         reject {key:32s} {count}", flush=True)

    def _preverify(skeleton):
        return skeleton, verify_intersection_route(
            skeleton,
            corpus(),
            _template_question(skeleton),
            anchor_a_query=" ".join(skeleton.anchor_a_clues),
            anchor_b_query=" ".join(skeleton.anchor_b_clues),
        )

    routed, pre_reasons = [], Counter()
    with ThreadPoolExecutor(args.search_workers) as pool:
        for skeleton, report in pool.map(_preverify, proved):
            if report.ok:
                routed.append(skeleton)
            else:
                pre_reasons[report.failure or "?"] += 1
    routed.sort(key=lambda s: (s.gold_doc_id, s.bridge_x, s.bridge_y))
    print(f"[mint] route-pre  {len(routed)}", flush=True)
    for key, count in pre_reasons.most_common(6):
        print(f"         reject {key:32s} {count}", flush=True)

    if args.no_llm:
        print(f"[mint] stopping before the model stage ({time.time() - t0:.0f}s)")
        return 0
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[mint] OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    # Four wording calls per task wanted. Two was the original guess and it is
    # not enough now: roughly a third of worded candidates survive the sense
    # check, the label guard and the per-tier route check, so a 2x batch cannot
    # fill three tier quotas and one of them silently comes out empty. A call
    # costs about $0.002, so over-drawing is far cheaper than a missing tier.
    batch = routed[: max(args.n * 4, args.n)]
    with ThreadPoolExecutor(args.workers) as pool:
        verbalizations = list(
            pool.map(
                lambda s: verbalize_intersection(s, docs, model=args.model), batch
            )
        )

    # Parse the requested mixture and turn it into a target count per tier.
    mix: dict[str, float] = {}
    for part in args.tier_mix.split(","):
        name, _, share = part.partition(":")
        name = name.strip()
        if name not in INTERSECTION_TIERS:
            print(f"[mint] unknown tier {name!r}", file=sys.stderr)
            return 2
        mix[name] = float(share)
    total = sum(mix.values()) or 1.0
    want = {name: max(1, round(args.n * share / total)) for name, share in mix.items()}
    got: Counter[str] = Counter()

    emitted, post_reasons = [], Counter()
    emitted_ids: set[str] = set()
    for skeleton, verbal in zip(batch, verbalizations):
        if not verbal.ok:
            post_reasons[verbal.reject_reason or "?"] += 1
            continue

        # Spend each skeleton on the SCARCEST tier it can serve, not the
        # hardest. Every skeleton can serve `described_both`, because
        # describing an anchor needs nothing; naming one needs a title that
        # spells neither hidden code and a label its own paper supports. Filling
        # the hardest tier first therefore consumed the candidates that were the
        # only ones able to fill the others, and `named_both` came out empty
        # while the other two hit quota exactly.
        #
        # `tiers_available` returns scarcest first, so the first still-wanted
        # option is the right one.
        options = [t for t in tiers_available(skeleton, titles) if got[t] < want.get(t, 0)]
        if not options:
            post_reasons["tier_quota_full"] += 1
            continue
        tier = options[0]

        # Only pass a title the checks cleared. An empty string tells the
        # assembler to describe that anchor instead of printing it.
        a_ok, b_ok = nameable_anchors(skeleton, titles)
        a_title = titles.get(skeleton.anchor_a_doc_id, "") if a_ok else ""
        b_title = titles.get(skeleton.anchor_b_doc_id, "") if b_ok else ""
        question = assemble_intersection_question(
            verbal, tier=tier, anchor_a_title=a_title, anchor_b_title=b_title
        )
        # A named anchor is reached by its own title, a described one by its
        # description. Verifying the wrong one would pass tasks whose route the
        # solver cannot actually take.
        report = verify_intersection_route(
            skeleton,
            corpus(),
            question,
            anchor_a_query=(a_title if a_title and tier in ("named_both", "named_one") else verbal.anchor_a_description),
            anchor_b_query=(b_title if b_title and tier in ("named_both", "named_one") else verbal.anchor_b_description),
        )
        if not report.ok:
            post_reasons[f"route:{report.failure}"] += 1
            continue
        # Last line of defence: the assembled question must not spell either
        # hidden term. Everything above is a reason to believe it does not;
        # this is the check that it does not.
        # Leaked means the question spells the whole term. Sharing one common
        # word with the frame is not a leak, and rejecting on it would throw
        # away sound tasks.
        q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        bx = set(re.findall(r"[a-z0-9]+", skeleton.bridge_x.lower()))
        by = set(re.findall(r"[a-z0-9]+", skeleton.bridge_y.lower()))
        if (bx and bx <= q_tokens) or (by and by <= q_tokens):
            post_reasons["term_leaked_into_assembled_question"] += 1
            continue

        answer = titles[skeleton.gold_doc_id]
        task_id = content_task_id(
            question,
            answer,
            (skeleton.gold_doc_id, skeleton.anchor_a_doc_id, skeleton.anchor_b_doc_id),
        )
        # Last line of defence against a repeated task. Distinct triples make
        # this rare, but two wordings of one skeleton can still collide, and a
        # pool that repeats an id is refused at load: selection runs over the
        # sorted id list and duplicates would desynchronise it from the
        # manifest an auditor redraws with.
        if task_id in emitted_ids:
            post_reasons["duplicate_task_id"] += 1
            continue
        emitted_ids.add(task_id)

        got[tier] += 1
        emitted.append(
            {
                "task_id": task_id,
                "question": question,
                "answer": answer,
                "aliases": [],
                # Both anchors are evidence: they are where the two bridges are
                # read, and a replay that kept only the gold could not show the
                # route was takeable.
                "evidence_doc_ids": [
                    skeleton.gold_doc_id,
                    skeleton.anchor_a_doc_id,
                    skeleton.anchor_b_doc_id,
                ],
                "masked_doc_ids": [],
                "origin": TaskOrigin.GENERATED_PRIVATE.value,
                "template": INTERSECTION_TEMPLATE_NAME,
                "hops": {"named_both": 1, "named_one": 2}.get(tier, 3),
                "meta": {
                    "tier": tier,
                    "bridge_x": skeleton.bridge_x,
                    "bridge_y": skeleton.bridge_y,
                    "bridge_x_type": verbal.bridge_x_type,
                    "bridge_y_type": verbal.bridge_y_type,
                    "bridge_x_df": skeleton.bridge_x_df,
                    "bridge_y_df": skeleton.bridge_y_df,
                    "proof": skeleton.proof,
                    "route": {
                        "anchor_a_rank": report.anchor_rank,
                        "anchor_b_rank": report.gold_rank_on_own_clues,
                        "gold_rank_from_pair": report.gold_rank_via_bridge,
                    },
                    "verbalizer": VERBALIZER_VERSION,
                    "verbalizer_model": verbal.model,
                    "entity_index_digest": index.digest()[:16],
                },
            }
        )
        if len(emitted) >= args.n:
            break

    print(f"[mint] verbalized {len(batch)}", flush=True)
    for key, count in post_reasons.most_common(8):
        print(f"         reject {key:32s} {count}", flush=True)
    print(f"[mint] emitted    {len(emitted)}", flush=True)
    for tier in INTERSECTION_TIERS:
        print(f"         tier {tier:16s} {got[tier]:4d}  (wanted {want.get(tier, 0)})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for task in emitted:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"[mint] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

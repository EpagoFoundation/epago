"""Where does the wobble start — the model, or the search?

Runs the SAME model on a handful of tasks TWICE, capturing every action the
agent emits (search query, browse id, final answer) in order. Then it prints
the two transcripts side by side and marks the first turn where they diverge.

Because the search tool is deterministic (bm25 over a frozen corpus — the same
query always returns the same papers), any divergence must originate in the
model's own token generation. If the FIRST search query already differs, the
model is non-deterministic from its very first token; if the first query
matches and a later turn differs, it is still the model, just later.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epago.environment.corpus import SqliteCorpus  # noqa: E402
from epago.environment.services import ResearchEnvironment  # noqa: E402
from epago.eval.harness import Episode  # noqa: E402
from epago.taskgen.generator import generate_tasks  # noqa: E402

_ACTION = re.compile(r"<(search|browse|answer)>(.*?)</\1>", re.DOTALL)


def actions_of(model, task, env) -> list[tuple[str, str]]:
    """Drive one episode and return its ordered [(kind, payload), ...]."""
    from epago import constants

    ep = Episode(task, env.tools_for_task(task), max_turns=constants.ROLLOUT_MAX_TURNS,
                 timeout_s=constants.ROLLOUT_TIMEOUT_S)
    seq: list[tuple[str, str]] = []
    while ep.begin_turn():
        out = model.generate(ep.prompt(), 512, ["</search>", "</browse>", "</answer>"])
        m = _ACTION.search(out)
        seq.append((m.group(1), m.group(2).strip()[:60]) if m else ("malformed", out.strip()[:60]))
        ep.advance(out)
    return seq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--release", default="SCI2")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    corpus = SqliteCorpus(args.corpus)
    env = ResearchEnvironment(corpus)
    tasks = generate_tasks(seed=args.seed, release=args.release, corpus=corpus,
                           n=args.n, king_probe=None)

    from epago.eval.backend import VllmBackend
    model = VllmBackend(args.model)

    first_query_diffs = 0
    any_diffs = 0
    for i, task in enumerate(tasks, 1):
        a = actions_of(model, task, env)
        b = actions_of(model, task, env)
        print(f"\n{'='*72}\nTASK {i}: {task.question[:90]}")
        first_div = None
        for turn in range(max(len(a), len(b))):
            xa = a[turn] if turn < len(a) else ("—", "")
            xb = b[turn] if turn < len(b) else ("—", "")
            mark = "  " if xa == xb else "≠≠"
            if xa != xb and first_div is None:
                first_div = turn
            print(f"  {mark} turn {turn}: A[{xa[0]}] {xa[1]!r}")
            print(f"           B[{xb[0]}] {xb[1]!r}")
        if first_div is None:
            print("  -> IDENTICAL transcript")
        else:
            any_diffs += 1
            kind = a[first_div][0] if first_div < len(a) else "—"
            print(f"  -> first divergence at turn {first_div} ({kind})")
            if first_div == 0:
                first_query_diffs += 1

    print(f"\n{'='*72}")
    print(f"tasks with any divergence     : {any_diffs}/{len(tasks)}")
    print(f"tasks diverging at FIRST turn : {first_query_diffs}/{len(tasks)}")
    print("→ divergence at turn 0 = the model is non-deterministic from its first token")
    print("→ search is deterministic, so the wobble is the MODEL, not the retrieval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

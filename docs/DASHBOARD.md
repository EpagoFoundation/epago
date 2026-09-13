# 👁 Dashboard

*A static audit surface — one HTML file, one JSON file, rendered entirely from a validator's published artifacts.*

The Epago dashboard is not a website with a backend. There is no server, no API, no
analytics, and no external request of any kind — anyone can host it, mirror it, or
open it from disk, and anyone can regenerate the JSON from a validator's published
state and diff it against what that validator serves.

What it renders is the live state of the competition for the frontier open deep-research
crown in one chain generation — the all-science corpus and the `SCI4` task release, in
the genesis generation. It is where the claim "provably better every round" is checked:
champion accuracy over time, every duel's
LCB against the floor it had to clear, and the external benchmark anchor that would
expose internal accuracy climbing while real capability stood still.

## What it shows

| Section | Data | Source |
|---|---|---|
| KPI row | Champion accuracy (EMA + delta), reign age & decay, duel counts, organic dethrones, verdict p95 vs the 48 h target, queue depth | `state.json`, audit log |
| Round running | Shown only while a round is being evaluated: its number, the block it opened at, and its entrants. No progress is shown; results appear when the round ends | `state.json` (`round_in_progress`) |
| Model improvement | Champion accuracy after each scored round, with coronation markers; per-duel LCB vs the adaptive floor δ | `state.json` (`king_acc_history`), audit log |
| Duel feed | Every duel: miner, checkpoint, μ public/private, LCB, δ, judge reliance, reveal→verdict latency, outcome | audit log |
| Miners | Leaderboard: attempts, crowns, near misses, best LCB, arena credit, last active | audit log + `state.json` |
| Submission pipeline | Funnel of where submissions ended, cheapest gate first | `state.json` |
| Refused | Every submission turned away before a duel, newest first, with a plain reason. The reason is a fixed sentence per refusal code, never the validator's own error text | `state.json` (`failure_memory`, `statuses`, `intake_log`) |
| Emission split | Effective king/arena/burn split: under a fixed burn, the king's flat share; otherwise its position in the 90-85% band, plus the former-king roster | derived from chain + `chain.toml` |
| Quorum | θ, bootstrap threshold, verdict timeout, pending candidates | `state.json` + `chain.toml` |
| SLA | p50/p95 latency, queue, breaker threshold | `state.json` |
| Task ecosystem | Tasks per duel, generator release, private-pool epoch/digest/rotation, LLM-judge reliance per duel | audit log |
| Score determinism | Calibration noise floor and the δ clamp derived from it | `state.json` |
| Benchmark anchor | Internal EMA gain vs external benchmark accuracy per anchor run, with the divergence alert | `state.json` (`anchor_history`) |

**Champion accuracy is measured, never assumed.** A new validator has not yet
measured its king, so it holds a 0.5 stand-in to compute the first round's floor.
That value is never shown: the first scored round replaces it outright and
becomes the genesis value, and later rounds move the average
(`KING_ACC_EMA_K`). Each round's measurement is also published in its round file
(`king_acc`). A duel's audit record keeps the value its own floor was computed
from, which is the one from before its round. Until a round is scored the page
says so instead of showing a number.

Duel outcomes come from the duel's own verdict record, never from the submission's
later lifecycle — a near miss that goes stale after the next dethrone still shows as a
near miss in the feed.

The page also links to the raw files published next to it: **Audit log**
(`../audit/audit.jsonl`) and **All published files** (`../index.json`). The links are
relative, so they work wherever the files are hosted.

## Generating it

```bash
# one-shot export from a validator state dir (or a published mirror of one)
epago dashboard export --state-dir ~/.epago/validator --out ./dashboard

# keep it fresh while the validator runs
epago dashboard watch --state-dir ~/.epago/validator --out ./dashboard

# publish it: each `epago publish watch` pass rebuilds it into the state dir and ships it
epago publish watch --state-dir ~/.epago/validator --repo-id <org>/<audit-repo>
```

`--out` receives `dashboard.json` plus `index.html`; serve the directory with any
static file server (or the `dashboard` service in `docker/docker-compose.yml`). The
page fetches `dashboard.json` beside it; if a `window.EPAGO_DATA` payload is inlined
instead, it renders that — which is how the demo works:

```bash
# self-contained demo from a synthetic subnet history (no chain, no GPU)
.venv/bin/python scripts/demo_dashboard.py --iterations 36 --out demo/
```

## Trust properties

| Property | Why it holds |
|---|---|
| **Reproducible** | `epago dashboard export` is deterministic over the same state + audit inputs; two parties exporting from the same published artifacts get byte-identical `dashboard.json` |
| **Replayable** | Every number traces to audit records that `scripts/replay_verdict.py` verifies against chain commitments — seeds, task sets, digests, LCB arithmetic and signature, all exactly and without a GPU (a sealed-pool exam is redrawn from its pre-committed task-id manifest instead of regenerated from the seed). Re-scoring the models is a separate statistical check against the measured harness noise floor; inference is not bit-reproducible on any GPU stack |
| **Self-contained** | The HTML makes zero external requests; hosting it cannot observe validators, miners, or readers |

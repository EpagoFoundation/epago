# 🔑 Competition rounds

Epago evaluates in **rounds**. Submissions queue continuously, but nothing is
scored until the round authority opens a competition. Then the whole queued
field answers one exam against the king and the best entrant is crowned.

A round is therefore the unit in which the deep-research crown changes hands, and the
cadence below is the cadence at which the model can provably improve.

## What this costs

This is a **privileged, owner-held key and a liveness dependency.** It reverses
two requirements the rest of the mechanism is built around — R1 "no owner API,
no privileged operator" and R2 "zero human-intervention paths". Concretely:

- If the key is lost or the holder goes quiet, **the subnet stops improving.**
  Submissions pile up, the king keeps earning its share, and no fallback opens a
  round without the authority.
- The authority chooses *when* each round opens, so it can wait for a
  particular miner's submission to land before triggering. Validators cannot
  detect or prevent this.

What validators *do* enforce is everything mechanical: only the configured
hotkey is honoured, round numbers must strictly increase, and — if a minimum gap
is set — starts must be at least `ROUND_MIN_INTERVAL_BLOCKS` apart. And the
authority cannot rig the exam —
it is minted from the block hash of its own trigger, which nobody chooses.

With neither an authority nor a local API key configured, no round ever opens.
There is no configuration that restores continuous evaluation.

## Setup

```toml
# chain.toml
[chain]
round_authority_hotkey = "5Your...AuthorityHotkey"
```

Every validator needs the same value — a round is chain state, so all boxes must
agree on which competition is running.

## Opening a round

```bash
epago chain start-round --wallet-name owner --wallet-hotkey authority
```

The wallet **is** the credential: the payload is signed by the hotkey, and
validators check the signer against `round_authority_hotkey`. There is no
separate shared secret to distribute, leak, or rotate.

Useful flags:

| Flag | Effect |
|---|---|
| `--dry-run` | Print the `er1` payload and exit; touches no chain. |
| `--round N` | Set the round number explicitly (default: last on chain + 1). |
| `--force` | Skip the *local* interval pre-check. Cannot make validators accept an early round — they run the same check. |
| `--mock` | Publish to an in-memory chain for rehearsal. |

If a minimum gap is set, the command refuses to publish before it has passed,
and tells you how long is left.

### With the local API

A contract can use a local API key instead of an on-chain authority
(`[chain] round_api_bind`, key in `EPAGO_ROUND_API_KEY`). The owner opens a round
from the validator box:

```bash
curl -X POST -H "X-Epago-Round-Key: $EPAGO_ROUND_API_KEY" http://127.0.0.1:8919/round/start
```

Each accepted request opens one round. Requests sent while a round is running
collapse into one, which opens as soon as that round ends. The exam still comes
from the hash of the block the request lands on, which the owner cannot choose.

## What happens next

1. The `er1` reveals ~5 blocks later. The chain stamps its block; the hash there
   mints the exam.
2. Entrants are every admitted challenge revealed **strictly before** that
   block, ordered by `(reveal_block, digest)` and capped at
   `ROUND_MAX_ENTRANTS` (32). Anything revealed later has already seen the hash,
   so it waits for the next round.
3. The king answers the exam once. Every entrant answers the same exam.
4. The highest LCB among entrants that clear both halves wins — with LCBs
   inside one calibrated noise floor treated as the same measurement, where the
   **earlier reveal** wins (digest as the final deterministic tie-break).
5. The provisional winner is re-dueled once on a fresh confirmation exam and
   must clear the floor again. Confirmed, it gets the round's only `ACCEPT`;
   unconfirmed, it settles as a near-miss. Runners-up that beat the king are
   near-misses too — one re-duel on a fresh exam, no penalty. Every entrant's
   hotkey is spent either way: one submission per hotkey, permanently.
6. Quorum crowns the winner as usual.

The round runs in the background. While it is scored, new submissions still join
the queue, the upload mailbox keeps refreshing, and the dashboard shows the round as
running, with its entrants. There is no live progress; the results, the audit
records and the round's tasks publish when it ends.

## Cadence

Each request opens one round. There is no minimum gap between rounds
(`ROUND_MIN_INTERVAL_BLOCKS = 0`): the owner opens the next round once the
previous field has been scored.

**We aim for one round a day.** That is a goal, not a fixed rule. It depends on
how fast a field is scored, and we are adding evaluation GPUs to keep that pace;
it may change as the subnet grows. A validator can still set a floor with
`EPAGO_ROUND_MIN_INTERVAL_BLOCKS`.

Note that reveal-to-verdict latency now includes waiting for the next round, so
the 48h `SLA_TARGET_HOURS` measures queue wait plus evaluation, not evaluation
alone. See [DESIGN.md](DESIGN.md) §9.

## Running late

There is no penalty and no catch-up: a round opened late is just a round. The
field is whatever has accumulated, subject to the entrant cap; the overflow
keeps its place in the queue for the round after. If you skip a cycle entirely,
miners simply wait longer; the king keeps its share and no round opens.

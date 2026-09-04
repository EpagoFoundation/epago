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
- The authority chooses *when* inside the allowed window, so it can wait for a
  particular miner's submission to land before triggering. Validators cannot
  detect or prevent this.

What validators *do* enforce is everything mechanical: only the configured
hotkey is honoured, round numbers must strictly increase, and starts must be at
least `ROUND_MIN_INTERVAL_BLOCKS` apart. And the authority cannot rig the exam —
it is minted from the block hash of its own trigger, which nobody chooses.

Set `[chain] round_authority_hotkey = ""` to disable rounds entirely. There is
no configuration that restores continuous evaluation.

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

The command refuses to publish if fewer than `ROUND_MIN_INTERVAL_BLOCKS` have
passed, and tells you how long is left.

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

## Cadence

`ROUND_MIN_INTERVAL_BLOCKS` defaults to 14400 blocks (~2 days at 12s blocks).
Override with `EPAGO_ROUND_MIN_INTERVAL_BLOCKS` for testnets.

Note that reveal-to-verdict latency now includes waiting for the next round — up
to ~2 days on its own — so the 48h `SLA_TARGET_HOURS` no longer describes what
it measures. See [DESIGN.md](DESIGN.md) §9.

## Running late

There is no penalty and no catch-up: a round opened late is just a round. The
field is whatever has accumulated, subject to the entrant cap; the overflow
keeps its place in the queue for the round after. If you skip a cycle entirely,
miners simply wait longer; the king keeps its share and no round opens.

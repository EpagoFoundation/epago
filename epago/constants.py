"""Mechanism constants.

Values here are mechanism defaults, overridable via ``EPAGO_<NAME>`` environment
variables for testnets and soaks. Anything that can be self-calibrated at
runtime (the noise floor, the difficulty band) is calibrated by pinned formulas
in code; the numbers below are only the starting points and hard bounds.
"""

from __future__ import annotations

import os


def _env(name: str, default):
    raw = os.environ.get(f"EPAGO_{name}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


# --- duel -----------------------------------------------------------------
# Exam size is set by NOISE, not by the clock. Measured on the pinned corpus:
# the same checkpoint re-scored on the same 400 tasks disagreed on 21% of them
# (batch composition perturbs kernel numerics, and one flipped token sends a
# multi-turn agent down a different trajectory). At 200 paired public tasks the
# standard error is 0.034, so the 99.9% LCB alone demands a 10.6pp win while
# delta contributes 1.05pp -- the crown was priced by noise, not by the effect
# floor. Standard error falls as 1/sqrt(N), so quadrupling the public half cuts
# the bar to roughly 6pp. A 32-entrant round still fits the ~48h interval, and
# fits comfortably once a validator scores on more than one GPU.
N_PUB_TASKS: int = _env("N_PUB_TASKS", 800)
N_PRIV_TASKS: int = _env("N_PRIV_TASKS", 200)
BOOTSTRAP_B: int = _env("BOOTSTRAP_B", 10_000)
EVAL_ALPHA: float = _env("EVAL_ALPHA", 0.001)          # one-sided 99.9% LCB
DELTA_C: float = _env("DELTA_C", 0.05)                 # delta = c * (1 - king_acc_ema)
# Coronation compares the 99.9% LCB (EVAL_ALPHA) against delta, so the LCB is
# already the primary noise guard -- it sits ~3 standard errors below the mean.
# This clamp only keeps delta from dropping below one standard error of the
# score gap; a larger multiplier double-counts the noise the LCB already removed
# and pushes the real winning bar out of reach of a genuinely better model.
DELTA_NOISE_MULTIPLIER: float = _env("DELTA_NOISE_MULTIPLIER", 1.0)
# A provisional round winner is re-dueled once on a fresh exam and must clear
# the floor again before it is committed as ACCEPT. Per attempt, a true-zero
# checkpoint (a noise-perturbed copy of the king) clears one 99.9% LCB above
# delta rarely; requiring two independent clears squares that probability, so
# a fleet of lucky lottery tickets stays harmless without raising the bar a
# genuinely better model must beat. 0 disables confirmation.
CORONATION_CONFIRMATION_DUELS: int = _env("CORONATION_CONFIRMATION_DUELS", 1)
KING_ACC_EMA_K: int = _env("KING_ACC_EMA_K", 10)
#: Where a template's king solve rate is informative. The ceiling is MEASURED
#: against the pinned reference model on the real corpus, not chosen: with the
#: eval path fixed (chat-templated prompts, last-match action parsing, the
#: unanswerable comparison template retired) the reference king scores
#: 260/350 = 74% over seven 50-task replicates of SCI3 on
#: data/corpus-science-5792 (per replicate 68-82%; per template
#: described_finding 73%, cross_doc_join 76%). The old 0.65 ceiling was
#: calibrated when the same model measured 46% through the broken path, so
#: post-fix it would put EVERY live template above the band permanently — and
#: a penalty applied to every template cancels in the mixture normalization,
#: which is worse than no controller at all because it looks like one is
#: running. 0.85 still fires on a genuinely saturated template (SCI1's
#: answer-leaking shapes measured 90%+). The floor is unchanged: nothing in
#: this measurement speaks to it.
KING_SOLVE_BAND_LOW: float = _env("KING_SOLVE_BAND_LOW", 0.45)
KING_SOLVE_BAND_HIGH: float = _env("KING_SOLVE_BAND_HIGH", 0.85)

# --- rollout harness --------------------------------------------------------
ROLLOUT_MAX_TURNS: int = _env("ROLLOUT_MAX_TURNS", 40)
# Hang safety net ONLY, never the binding budget: measured on 1,600 episodes,
# wall time under batching is mostly queue time (38.6 s/turn at concurrency 32,
# 100% of timeout deaths below the turn cap at a median of 21 turns), so a
# tight wall cap makes the verdict depend on the validator's load — two honest
# validators could score the same miner differently under an identical digest.
# The binding budgets are ROLLOUT_MAX_TURNS and TRANSCRIPT_MAX_CHARS, both
# pure functions of the transcript, identical at any load on any hardware.
ROLLOUT_TIMEOUT_S: int = _env("ROLLOUT_TIMEOUT_S", 3600)
ROLLOUT_CONTEXT_TOKENS: int = _env("ROLLOUT_CONTEXT_TOKENS", 32_768)
ROLLOUT_TEMPERATURE: float = 0.0
ROLLOUT_SEED: int = 42
#: Mild multiplicative penalty on already-generated tokens. Greedy decoding on
#: the reference MoE checkpoint degenerates into verbatim repetition loops on
#: long <think> blocks (probe: every malformed SCI4 death was such a loop);
#: 1.05 is the gentlest standard setting that breaks loops while leaving
#: exact-title reproduction intact (verified by the r4 paired run).
ROLLOUT_REPETITION_PENALTY: float = _env("ROLLOUT_REPETITION_PENALTY", 1.05)
# 32 measured 1.23x faster than 16 on a 32 GB card with no accuracy change; 64
# regresses because the KV cache (~105k tokens) leaves too little per sequence
# and vLLM starts preempting. Scale with KV, not with card count: keep at least
# ~3,300 KV tokens per concurrent episode.
ROLLOUT_CONCURRENCY: int = _env("ROLLOUT_CONCURRENCY", 32)
ANSWER_MAX_CHARS: int = _env("ANSWER_MAX_CHARS", 200)

# --- research tools ---------------------------------------------------------
# A tool's output is prompt bytes the moment it enters a transcript, so these
# live here with the other protocol constants and are folded into
# ``epago.eval.harness.harness_digest``. Two validators whose tools differ are
# not scoring the same exam, and the digest is what makes that visible.
#: ``v2`` OR-matches query terms instead of AND-ing them. Under implicit AND
#: every term had to appear in a document, so a question-shaped query returned
#: nothing at all: measured on the 24,082-doc FRAMES corpus, 300/300 natural
#: questions scored zero hits, against recall@10 = 0.937 once OR-ed.
#: ``v3-web`` reshapes the surface to the reference model's NATIVE convention:
#: tools declared as JSON signatures, called via <tool_call> JSON, results
#: returned inside <tool_response> as a web-style results page. Measured under
#: the bespoke v2 surface on SCI4: 48% of episodes died to malformed actions
#: (the model does not speak an invented protocol), and the model scored WORSE
#: with tools than closed-book (7.8% vs 10.0%).
TOOL_VERSION: str = "epago-tools-v3-web"
SEARCH_K: int = _env("SEARCH_K", 10)
BROWSE_PAGE_CHARS: int = _env("BROWSE_PAGE_CHARS", 6000)
SEARCH_SNIPPET_TOKENS: int = _env("SEARCH_SNIPPET_TOKENS", 24)
SEARCH_SNIPPET_FALLBACK_CHARS: int = _env("SEARCH_SNIPPET_FALLBACK_CHARS", 300)
SEARCH_MAX_QUERY_TERMS: int = _env("SEARCH_MAX_QUERY_TERMS", 32)
#: A term in more than this fraction of documents carries no ranking signal
#: (bm25 already weights it to near zero) but, under OR matching, still scans
#: that fraction of the index. Measured on the FRAMES corpus: dropping them
#: costs nothing (recall@10 0.937 both ways) and runs 3x faster.
SEARCH_COMMON_TERM_DF: float = _env("SEARCH_COMMON_TERM_DF", 0.25)

#: Native-surface caps: one <tool_call> may carry several complementary search
#: queries or open several documents; these bound the observation size so a
#: single call cannot blow the context window.
SEARCH_MAX_QUERIES: int = _env("SEARCH_MAX_QUERIES", 3)
VISIT_MAX_DOCS: int = _env("VISIT_MAX_DOCS", 2)
#: The date line the native system prompt ends with. Pinned — a wall-clock
#: date would make the prompt bytes differ between validators.
HARNESS_PROMPT_DATE: str = _env("HARNESS_PROMPT_DATE", "2026-08-01")
#: Transcript size (chars) at which the harness closes the tools and demands
#: an answer; ~90k chars is ~23k tokens, safely inside the 32k context.
# Sized against the 32,768-token context for MAX_ACTION_TOKENS=2048:
# 100k chars is ~25k tokens typical / ~30.3k worst-case (3.3 chars/token on
# number-dense scientific text), + 2,048 generation <= 32.4k with margin.
# (110k overflowed when the generation cap was briefly 4,096; 84k was sized
# for that cap and, measured over 1,600 episodes, starved the constrained
# retrieval family — the exam's least bluffable — which ran exactly at its
# wall while 83% of its budget went to search-results pages.)
TRANSCRIPT_MAX_CHARS: int = _env("TRANSCRIPT_MAX_CHARS", 100_000)

#: Everything that decides the bytes a research tool writes into a transcript.
TOOL_SURFACE: tuple[str, ...] = (
    TOOL_VERSION,
    str(SEARCH_K),
    str(BROWSE_PAGE_CHARS),
    str(SEARCH_SNIPPET_TOKENS),
    str(SEARCH_SNIPPET_FALLBACK_CHARS),
    str(SEARCH_MAX_QUERY_TERMS),
    repr(SEARCH_COMMON_TERM_DF),
    str(SEARCH_MAX_QUERIES),
    str(VISIT_MAX_DOCS),
    HARNESS_PROMPT_DATE,
    str(TRANSCRIPT_MAX_CHARS),
)

# --- submissions ------------------------------------------------------------
#: ``e2`` supersedes ``e1``: the author hotkey is no longer carried in the
#: payload. It was self-declared and unverified, so anyone could submit a junk
#: checkpoint under a competitor's identity and collect that competitor an
#: escalating intake cooldown. The author is now the hotkey that signed the
#: commitment, which the chain already attests. ``e1`` payloads are dropped.
REVEAL_VERSION: str = "e2"
#: ``ev3`` carries the round a duel belongs to, on top of ``ev2``'s adaptive
#: floor (``delta_e6``). Without the floor, near-miss classification — and
#: therefore arena emission — is not derivable from chain state; without the
#: round, a verdict cannot be attributed to the competition that produced it.
VERDICT_VERSION: str = "ev3"
#: ``er1`` starts a competition round. Only the configured round authority can
#: publish one, and nothing is evaluated until it lands.
ROUND_START_VERSION: str = "er1"
POOL_COMMIT_VERSION: str = "ep1"
#: ``ek1`` is the king pointer published by the coronation authority at every
#: crowning. It is what lets a validator start from an empty state directory:
#: without it the only king a fresh box could name was the genesis seed, which
#: made every live challenge look stale and left the box permanently stuck.
KING_POINTER_VERSION: str = "ek1"
STATUS_VERSION: str = "es1"

# --- competition rounds -------------------------------------------------------
#: Minimum blocks between two round starts (~2 days at 12s blocks). A trigger
#: that arrives sooner is refused, so the cadence is a property of the chain
#: rather than of how often the owner happens to run the command.
ROUND_MIN_INTERVAL_BLOCKS: int = _env("ROUND_MIN_INTERVAL_BLOCKS", 14_400)
#: Upper bound on challengers evaluated in one round. Every entrant costs a full
#: sweep of the exam, so the field is capped and the overflow waits for the next
#: round rather than blowing the SLA. Cut entrants are logged, never dropped
#: silently.
ROUND_MAX_ENTRANTS: int = _env("ROUND_MAX_ENTRANTS", 32)
BLOCKS_UNTIL_REVEAL: int = _env("BLOCKS_UNTIL_REVEAL", 5)
VERDICT_REVEAL_BLOCKS: int = _env("VERDICT_REVEAL_BLOCKS", 5)
MAX_CHALLENGER_SIZE_RATIO: float = _env("MAX_CHALLENGER_SIZE_RATIO", 1.05)
PRIVATE_POOL_ROTATION_BLOCKS: int = _env("PRIVATE_POOL_ROTATION_BLOCKS", 43_200)  # ~6 days

# --- probes -----------------------------------------------------------------
FORMAT_PROBE_TASKS: int = _env("FORMAT_PROBE_TASKS", 20)
#: How many tasks the probe *supply* mints before
#: :func:`epago.eval.probes.probe_task_set` picks the FORMAT_PROBE_TASKS it
#: actually runs. Minting exactly FORMAT_PROBE_TASKS made the selection a
#: no-op and handed the format gate a random draw of the full difficulty
#: mixture — including multi-hop synthesis tasks the incumbent king itself
#: cannot finish — so the gate measured research luck, not protocol
#: compliance. A wider pool is what gives the selection anything to select.
FORMAT_PROBE_POOL_TASKS: int = _env("FORMAT_PROBE_POOL_TASKS", 100)
#: Fraction of the probe's GENERATIONS that must carry a parseable action.
#: Replaces the old FORMAT_PROBE_PASS = 18-of-20 *episodes-ending-in-an-answer*,
#: which was unreachable by construction: it asked a cheap gate to solve 90% of
#: a real research exam. Renamed rather than re-tuned so an operator who pinned
#: the old knob gets an error instead of a silently different gate.
#:
#: MEASURED on the pinned reference model, the pinned corpus and the shipped
#: probe (sequential rollouts, the 20-task probe set, MAX_ACTION_TOKENS = 512):
#: 432 of 569 generations over NINE replicates carried a well-formed action —
#: 75.9% compliance, per replicate 73.0-79.4%, replicate SD 2.3pp. The floor
#: sits ~9 replicate-SDs below that, so an honest challenger cannot trip it on
#: greedy-decoding noise, while a checkpoint that cannot drive the loop at all
#: is still rejected before it burns a duel's GPU budget. About half the
#: measured non-compliance is the harness's own doing: at MAX_ACTION_TOKENS =
#: 2048 the same model measures ~89%, because 512 tokens truncates a reasoning
#: checkpoint mid-scratchpad. The floor stays valid if that budget grows.
FORMAT_PROBE_MIN_COMPLIANCE: float = _env("FORMAT_PROBE_MIN_COMPLIANCE", 0.55)
NORM_SANITY_MAX_LAYER_RATIO: float = _env("NORM_SANITY_MAX_LAYER_RATIO", 20.0)
NORM_SANITY_MAX_GLOBAL_RATIO: float = _env("NORM_SANITY_MAX_GLOBAL_RATIO", 5.0)

# --- SLA / queue ------------------------------------------------------------
SLA_TARGET_HOURS: int = _env("SLA_TARGET_HOURS", 48)
QUEUE_BREAKER_HOURS: float = _env("QUEUE_BREAKER_HOURS", 36.0)
CHALLENGE_BOND_BASE_ALPHA: float = _env("CHALLENGE_BOND_BASE_ALPHA", 1.0)
BOND_BURN_LCB_THRESHOLD: float = _env("BOND_BURN_LCB_THRESHOLD", -0.05)
NEAR_MISS_RETRIES: int = _env("NEAR_MISS_RETRIES", 1)

# --- weights ----------------------------------------------------------------
WEIGHT_INTERVAL_BLOCKS: int = _env("WEIGHT_INTERVAL_BLOCKS", 300)
COMMIT_REVEAL_REQUIRED: bool = _env("COMMIT_REVEAL_REQUIRED", True)

# --- emission activation (deterministic phase gate) -------------------------
PHASE_B_MIN_CLEAN_DUELS: int = _env("PHASE_B_MIN_CLEAN_DUELS", 50)
PHASE_B_MIN_DETHRONES: int = _env("PHASE_B_MIN_DETHRONES", 1)
PHASE_B_MIN_BLOCKS: int = _env("PHASE_B_MIN_BLOCKS", 100_800)  # ~14 days

# --- external anchor ---------------------------------------------------------
ANCHOR_INTERVAL_BLOCKS: int = _env("ANCHOR_INTERVAL_BLOCKS", 50_400)  # ~7 days
ANCHOR_DIVERGENCE_ALERT: float = _env("ANCHOR_DIVERGENCE_ALERT", 0.10)

# --- audit -------------------------------------------------------------------
AUDIT_PUBLISH_DELAY_BLOCKS: int = _env("AUDIT_PUBLISH_DELAY_BLOCKS", 50_400)  # public tasks after ~7 days
AUDIT_CHAIN_COMMIT_EVERY: int = _env("AUDIT_CHAIN_COMMIT_EVERY", 100)
# Cold-start floor, used only until this validator has run its own calibration
# duel. 2/400 assumed two runs of one checkpoint differ on half a percent of
# tasks; measured, they differ on 21% (84/400), for a score-gap standard error
# of ~0.030 at n=128. A fallback that is too low lets a fresh validator crown on
# noise before it has measured anything, while one that is too high only delays
# a legitimate coronation until the first calibration duel replaces it — so this
# errs high deliberately. Falls as 1/sqrt(n), so it is conservative at the
# shipped exam size.
CROSS_GPU_NOISE_BUDGET: float = _env("CROSS_GPU_NOISE_BUDGET", 0.03)

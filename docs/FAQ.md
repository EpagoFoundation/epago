# FAQ

*Short answers to the questions everyone asks — each grounded in the [mechanism spec](DESIGN.md).*

### What is Epago actually building?

The world's frontier open deep-research model — small enough to run and fine-tune on
modest hardware, provably better every round. Across all of scientific literature.

The current generation is built on Alibaba's Tongyi-DeepResearch-30B-A3B: a 30.5B
mixture-of-experts with **~3.3B active parameters per token** (not a dense 3B model).
Its authors' published results put it ahead of OpenAI o3 and DeepSeek-V3.1 (671B) on 5
of 7 agentic deep-research benchmarks — HLE 32.9 (24.9 / 29.8), FRAMES 90.6 (84.0 /
83.7), xbench-DeepSearch 75.0 (67.0 / 71.0), WebWalkerQA 72.2, GAIA 70.9, BrowseComp-ZH
46.7 — while **o3 leads BrowseComp, 49.7 to 43.4**.

Those are the *base model's* numbers, quoted as the floor Epago starts from. No Epago
checkpoint can claim them until it wins a coronation. And the claim they support is
narrow: better **at deep research, per unit of compute** — never general intelligence.

### Why would a small model win at this?

Because **deep research is a procedure, not a knowledge store.** The facts live in the
documents being read, not in the weights; what decides quality is decomposing the
question, searching, opening sources, reading, cross-checking conflicting studies, and
attributing each claim. Procedures distill into small models. Memorized world-knowledge
does not — but in grounded research the corpus supplies the recall a large parameter
count would otherwise buy, so parameters stop being the deciding variable.

That is why the argument is domain-limited on purpose. It says nothing about a small
model beating a large one at open-ended reasoning; it says that at *this* task, per unit
of compute, size stops paying.

### Then what is left to compete over?

Quite a lot. On Epago's own deliberately harder exam — the question never identifies
its sources; answers are located by eliminating a crowd of candidates or computed
across documents, so they exist verbatim in no search result and often in no document
at all — the base model measures **33.9%** with full agentic tooling, 56.1% with
perfect retrieval handed to it, and 17.3% closed book. Most of its lost points die in
research episodes abandoned at the turn, clock, or context budget rather than in wrong
answers; finishing more episodes is the first thing a challenger can profitably fix.

That headroom is published rather than buried, because it is the answer: the open
frontier is small but **static** — a lab ships a checkpoint and moves on, and nothing
proves the next version is genuinely better rather than benchmark-tuned. Epago is the
mechanism that makes it compound, with every step proven.

### Which fields does this cover, and what comes after?

All of scientific literature. The pinned duel corpus is **50,420 papers across ~135 fields in
all four OpenAlex domains** — Life 1,450, Physical 1,449, Health 1,449, Social 1,444 —
and the current task release, `SCI4`, mints by one rule — hard to find, easy to check:
constrained search, cross-study comparison, and computed evidence over any field,
so a materials-science, economics or CS abstract mints the same task shapes a clinical
one does. The engine applies wherever claims trace to sources and answers are
mechanically checkable. Health science is one of the four domains it covers, not the
thing it is limited to; `SCI2`, the frozen medicine-tuned predecessor release, still
exists so older pins are not silently re-judged.

Everything after it is configuration rather than code. A corpus *is* a chain generation
(corpus + task templates + architecture pins in one `chains/*.toml`), and generations run
in parallel with their own kings: law, patents, regulatory filings. Three other axes grow
the same way — the task ladder above extraction (screening, cross-study synthesis,
risk-of-bias appraisal, drafted review sections), modality (tables and figures, where most
quantitative results live and every current model is weak), and corpus reach (pinned
snapshot → live retrieval → the customer's own private store; the learned behavior is
retriever-agnostic).

### What does the network produce besides the model?

**Verified reward data.** Every duel emits `(task, answer, correct/incorrect)` records
over documents published after the models were trained, mechanically verified rather than
human-labeled. That is the scarcest input in AI post-training, and it cannot be scraped
because it does not exist anywhere else — it already falls out of the audit records and
the private pools that get published in full at rotation. Fresh documents → generated
tasks → duels → verified reward data → better models → a more valuable crown → more
miners → more duels.

### Why a quorum instead of one evaluator?

A single evaluator is an owner-shaped hole: whoever runs it controls coronation, and
its private test set is a secret no one can ever audit — it could rig half of every
verdict undetectably.

In Epago every validator runs the identical box and evaluates independently;
coronation happens only when accepting verdicts cover ≥ 51% of active-evaluator
stake, as a pure function of on-chain state. A corrupted validator can at worst
withhold its own stake from quorum, and its dissent (or fabrication) is permanently
on record.

### Why do private pools get published?

Privacy and auditability pull in opposite directions; delayed transparency gets
both. A private pool is only useful against overfitting while it is *live* — once it
rotates out (~every 6 days), publishing it costs nothing defensively, because fresh
pools have already replaced it.

And since every verdict committed the pool's digest and epoch on-chain, publication
makes every past private verdict retroactively, cryptographically replayable.
Validators are kept honest on the exact half of the duel that would otherwise be
unauditable.

### If public tasks get published, won't everyone just overfit them?

Fitting the public half is expected, and mostly it is the point: these tasks ask a
model to find a study that two other studies both point at, so getting better at them
*is* getting better at multi-hop research. What must not happen is a model scoring
well by remembering answers instead, and two separate things prevent that.

**A published task is never asked again.** The moment a round's tasks are staged for
release they are retired from the pool, so memorising them buys nothing. Retirement is
by task id and ids are content-addressed, so the same task re-minted into a later pool
keeps the id it was published under and stays retired.

**The private half is the backstop.** Acceptance needs both `lcb_pub > delta` *and*
`mu_priv > 0`. A model that learned the public half's surface patterns rather than the
skill shows up as a public/private gap and does not take the crown. That is the same
mechanism that has always guarded against generator overfitting; a sealed pool does not
change it.

What a miner can legitimately take from published rounds is the shape of the exam —
which is public information anyway, since the task family is documented. The answers
move; the skill does not.

### Will the subnet run out of tasks?

Not for a long time, and the number is known rather than hoped for. Rounds retire
the tasks they ask, so the pinned corpus has a real ceiling: 50,420 documents
yield 21,578 papers usable as answers and 99,831 distinct anchor pairings, which
at the measured 28% acceptance rate is about 28,000 tasks — roughly nine months
of rounds at 800 tasks every two days.

That ceiling is a stop, not a target. Pools are minted a few thousand at a time
and rotated, and the corpus is being extended with more literature as the subnet
runs, which raises the ceiling faster than rounds consume it. Adding papers is
how supply grows; loosening the checks that make a task sound is not, because a
task that is not provably unique is not a task.

Each corpus snapshot is pinned by digest in the contract, so extending it starts
a new generation with its own pins rather than silently changing the exam under a
verdict that has already been recorded.

### Why doesn't the winner take all?

Winner-takes-all checkpoint competition rewards hoarding: leaders sit on
improvements, challengers who narrowly lose get nothing and leave, and payout
variance drives out honest participants.

Epago pays the king 90%, falling to 85% over about three days and then holding
(an unchallenged incumbent bleeds toward the floor but keeps a defined
majority), and pays the three most recent former kings from the 10-15% arena
pool, plus whatever the king's decay releases. The king still earns the large majority —
but only while defending against a live challenger ecosystem, which the schedule itself
funds.

### What happens if the king's repo disappears?

Nothing. Every validator materialized the king's snapshot to duel it and re-pins
those weights at coronation, so the network holds as many mirrors as there are
validators. The king is identified by digest, not by repo availability; weight
setting and dueling continue uninterrupted. There is no `king_lost` operational
path because the failure mode does not exist.

### How is the 48h SLA enforced without an operator?

By pricing, not paging. Every audit record carries reveal, intake, and verdict block
numbers (the verdict block is chain-stamped by the reveal channel, so it cannot be
faked), and per-validator latency is publicly computable.

A hotkey gets one submission, permanently. Whatever happens to it — crowned, near-miss, or beaten — that hotkey is spent, and trying again means registering a fresh one and paying the registration burn.

That price is the mechanism. If attempts were free the cheapest strategy would be to upload many mediocre checkpoints and let the duels find one that got lucky on its holdout, and every one of those costs validators a full rollout sweep. Paying per attempt pushes the spend back into training. A near-miss keeps its one re-duel on fresh tasks, because that is the same submission being re-judged rather than a second one.

### How do I audit a verdict?

Run `scripts/replay_verdict.py` against any `ev3` commitment — no permission, no
special role, and no GPU. Against the audit record bound by the verdict's `audit16`
digest, it re-derives the task-selection and bootstrap seeds from the reveal block
hash, regenerates the public tasks from the pinned corpus and generator release and
checks their ids digest, recomputes the bootstrap LCB from the per-task difference
vector recorded in the audit record, recomputes the `audit16`, verifies the
validator's signature, and cross-checks the commitment on chain. All of that is exact
arithmetic: it matches or it does not.

It does not re-run the models, because no one can do that exactly. GPU inference is
not bit-reproducible — the same checkpoint re-scored on the same 400 tasks disagrees
with itself on about 21% of them, on one card as much as on eight. Checking the scores
themselves is therefore a statistical step you run separately: re-score the regenerated
tasks and compare the paired gap against the measured noise floor (≈0.030 standard
error at n = 128). Fabricating a difference vector wholesale shows up far outside that
band; fabricating a task or two hides inside it and is also too small to change a
verdict, because acceptance requires a 99.9% confidence bound to clear a floor clamped
above that very noise. The protocol was built assuming the scorer is noisy — that is
why the audit still binds.

For the private half, wait for that pool epoch's rotation: the pool is published in
full, verifiable against the digest already committed in the verdict, and checks the
same way.

When the public half is served from a sealed pool, that one step works differently:
the questions were worded by a model, so they cannot be regenerated from a seed.
Instead the replay redraws the round from the pool's task-id manifest — whose digest
was committed before the round — skipping the ids that earlier published rounds
retired, and checks the result against the round file the validator published. Point
`EPAGO_PUBLIC_POOL_MANIFEST` and `EPAGO_PUBLIC_ROUNDS` at those files. Without them the
`tasks` check reports SKIP; it never reports a pass it did not earn.

### How do I run this on testnet so it behaves like mainnet?

Follow the validator guide end to end — [VALIDATING.md](VALIDATING.md) covers
`--network test`, the required hyperparameters (commit-reveal weights is mandatory),
genesis artifact pinning, preflight, and validator bring-up. The one deliberate
difference from defaults: mainnet burns emissions until the deterministic Phase A→B
gate fires, and on testnet you zero the gate's thresholds so it fires at genesis:

```bash
export EPAGO_PHASE_B_MIN_CLEAN_DUELS=0
export EPAGO_PHASE_B_MIN_DETHRONES=0
export EPAGO_PHASE_B_MIN_BLOCKS=0
```

That is the supported "no burn" configuration — the gate stays a pure function of
chain state, it just passes immediately, so emissions, the reign band, arena and
arena pool all behave exactly as they will on mainnet from the first block.
Set the same values on every validator (see `testnet.env.example`).

### Can other miners see my model?

Not unless you win.

A private submission is uploaded into a prefix only you can write and only the
validator can read. The credentials the validator seals to your hotkey are
write-only — they cannot read or list anything, including your own upload — so
nothing any miner holds can reach another's checkpoint. Losing costs you the
duel and nothing else.

Winning makes it public. A crowned model is republished where anyone can fetch
it, because the model collecting emissions has to be re-scorable by anyone.

Submitting publicly through Hugging Face is still supported and needs no
credentials. The only difference is that everyone can read your weights the
moment you reveal.

### Why does private submission need an Ed25519 hotkey?

Because the validator has to encrypt your upload credentials to your hotkey,
and Bittensor's default sr25519 keys sign but do not encrypt. Ed25519 converts
to an X25519 key-exchange key, which sr25519 does not.

Create one with `btcli wallet new-hotkey --key-type ed25519`. Submitting
publicly needs no credential, so it works with either key type.

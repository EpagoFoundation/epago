# Epago: A Decentralized Protocol for a Continuously Improving Open Deep-Research Model

**Whitepaper · Version 1.0**

*Epago builds the world's frontier open deep-research model — small enough to run and
fine-tune on modest hardware, provably better every round, across all of scientific
literature. It runs as a Bittensor subnet: anyone may submit an improved model; a
stake-weighted quorum of independent validators proves the improvement statistically on
questions no one could have prepared for; the best model earns until it is beaten. The
engine applies wherever claims trace to sources and answers are mechanically checkable.
There is no company, no training team, no privileged operator, and no verdict that
cannot be reproduced by a stranger from public data.*

---

## Contents

**Part I — For everyone.** [1. The problem](#1-the-problem) · [2. What Epago is](#2-what-epago-is) · [3. How it works, as a story](#3-how-it-works-as-a-story) · [4. Why it matters](#4-why-it-matters)

**Part II — The model.** [5. What deep research means here](#5-what-deep-research-means-here) · [6. The evaluation environment](#6-the-evaluation-environment) · [7. The genesis model](#7-the-genesis-model)

**Part III — The mechanism.** [8. King-of-the-hill duels](#8-king-of-the-hill-duels) · [9. The two-slice exam](#9-the-two-slice-exam) · [10. The statistical verdict](#10-the-statistical-verdict) · [11. Decentralized validation](#11-decentralized-validation) · [12. Determinism and replay](#12-determinism-and-replay) · [13. Emissions and game theory](#13-emissions-and-game-theory) · [14. Threat model](#14-threat-model)

**Part IV — The network.** [15. Roles](#15-roles) · [16. On-chain protocol](#16-on-chain-protocol) · [17. Deployment and throughput](#17-deployment-and-throughput)

**Part V.** [18. Roadmap](#18-roadmap) · [19. Foundations and references](#19-foundations-and-references) · [20. Conclusion](#20-conclusion)

---

# Part I — For everyone

## 1. The problem

Four problems sit at the intersection of AI and trust, and Epago is built to solve all
four at once.

**The open frontier is static.** The best open deep-research models are already small
and already very good — but a lab ships a checkpoint and moves on. No mechanism makes a
released model keep improving, and no mechanism proves that the next version is
genuinely better rather than tuned to the benchmark it is reported against. Progress
arrives in occasional donated jumps, from whoever happens to feel like donating.

**Benchmarks lie.** When an AI model tops a public leaderboard, you cannot easily tell
whether it is genuinely capable or whether it was quietly trained on the answer key.
Public test sets leak, get memorized, and get gamed. The number goes up; the capability
may not. This is not a fringe concern — it is the central credibility problem of the
entire field.

**Evaluation requires trusting someone.** Every AI competition has an operator you must
trust: a company API, a private test server, a human judge behind a curtain. If that
operator is careless, biased, or compromised, every result downstream is worthless — and
you usually cannot check.

**AI progress is centralized.** The best models are trained by a handful of large labs,
behind closed doors, at costs only they can afford. If you have a better idea for how to
train a deep-research model, you have almost no way to prove it and be rewarded for it.

Epago's thesis is that these four problems have a single shared solution: **make the
competition permissionless, make the exam unpredictable, and make every result
reproducible by anyone.** Do that, and "who do you trust?" stops being a question — you
don't trust, you verify. And a frontier that no longer depends on one lab's release
schedule stops being static.

## 2. What Epago is

Epago produces one thing: the **world's frontier open deep-research model** — small
enough to run and fine-tune on modest hardware, provably better every round, across all
of scientific literature.

**Why "small beats big" is true here and not everywhere.** Deep research is a
**procedure, not a knowledge store.** The facts live in the documents being read, not
in the model's weights. What determines quality is the procedure: decompose the
question, search, open sources, read, cross-check conflicting studies, and attribute
each claim. Procedures can be distilled into small models; memorized world-knowledge
cannot. A larger parameter count mostly buys recall — and in grounded research the
recall is supplied by the corpus, so parameters stop being the deciding variable. That
is why the argument does not generalize into a claim about intelligence: Epago's claim
is superiority **at deep research, per unit of compute**, and nothing wider.

This is already demonstrated rather than hypothesized. The genesis base model runs with
about **3.3 billion active parameters per token** of a 30.5-billion-parameter
mixture-of-experts, and outperforms OpenAI o3 and DeepSeek-V3.1 — a 671-billion-parameter
model — on **5 of 7** agentic deep-research benchmarks in its authors' published results
(§7, where the head-to-head is given in full, including the one benchmark o3 leads).
Those numbers belong to the base model. They are the floor Epago starts from, not
something Epago has earned; no result becomes Epago's own until a challenger is crowned.

**What the model actually does.** It doesn't guess from memory: it searches a corpus of
documents, opens the sources, reads them, cross-checks findings across multiple
documents, and returns a short, precise answer you can trace back to the source it came
from. The pinned corpus is **all of scientific literature** — 50,420 papers across ~135 fields
across the four OpenAlex domains (life, physical, health and social sciences) —
because evidence extraction is the hardest grounding test and the most valuable
verifiable workflow wherever it happens (§4). The current model runs quantized on a
single consumer GPU.

The model is not trained by a team. It improves through **open competition**:

- Anyone can take the current best model — *the king* — fine-tune it however they like,
  and submit their improved version as a **challenger**.
- Independent operators called **validators** run a **duel**: the king and the
  challenger answer the exact same set of freshly generated deep-research tasks.
- If the challenger provably beats the king — at 99.9% statistical confidence, agreed by
  a majority of validators — the challenger becomes the new king and starts earning the
  network's rewards.
- Then everyone tries to beat *it*.

The result is a model that structurally cannot stop improving, scores that structurally
cannot be faked, and a full audit trail that lets anyone re-check any decision the
network ever made.

Epago runs as a **subnet** on Bittensor, a blockchain network that rewards useful AI
work. The blockchain is used for exactly what blockchains are good at: proving *who did
what, when,* and *in what order* — without any central authority keeping the books.

## 3. How it works, as a story

Meet **Maya**, a machine-learning engineer with a good idea and a few GPUs.

She looks at the network's public dashboard: the current champion model scores about 34%
on the network's internal deep-research exam — the exam is deliberately hard, built so
answers can never be copied out of a search result, and even perfect retrieval would only
lift the champion to ~56%; that gap is exactly why there is something to win. She spends four days fine-tuning it. Then she
uploads her improved model — into storage only she and the validator can read, so
rivals cannot copy the work she just spent four days on — and submits it to the
blockchain. Here is the first clever part: her submission is **sealed** for a few
minutes before it becomes visible.

The moment it unseals, at a specific block on the chain, that block's cryptographic
fingerprint is used to **generate the exam questions**. This is the heart of the
anti-cheating design: *the questions did not exist until after Maya's model was frozen.*
She could not possibly have trained on them. Nobody could — not Maya, not the validators,
not the network's creators. And yet, because the questions come from a public, fixed
recipe, anyone can regenerate the exact same questions forever afterward and re-check the
result.

Three validators — **Vera, Victor, and Vlad** — each independently see Maya's
submission. Each one, on its own machine with no communication between them, runs the
same duel: Maya's model and the king each answer ~1,000 fresh deep-research tasks. 800
questions are public (anyone can regenerate them); 200 come from each validator's own
**private question set** that no miner has ever seen — a trap for anyone who tried to
memorize the public style of question.

Both models answer each question by actually reading the literature — searching, opening
papers, cross-checking. Every answer is graded by a simple, un-foolable rule: does it match a
pre-verified correct answer? No opinion, no judge to sweet-talk.

Maya's model wins more questions than it loses. But "winning more" isn't enough — she
could have gotten a lucky exam. So each validator runs a statistical test that asks:
*even in the unluckiest reading of this data, is she still clearly better?* Her margin is
big enough. Vera votes to crown her. So does Victor. Together they hold more than half
the network's voting stake.

**At that moment — not by anyone's decision, but as a mathematical fact derived from the
blockchain — Maya is the new king.** Every observer computes the same result at the same
block. Vlad, whose machine was still finishing its duel, sees the coronation and adopts
Maya's model as the king too. Every validator also saves a copy of Maya's model, so even
if she deletes it tomorrow, the network is unharmed.

Maya's wallet begins to fill with rewards. But her reign is not permanent: the longer she
holds the crown without being challenged, the more her rewards slowly bleed into a pool
that pays *other* promising challengers — so there is always fresh incentive to come for
her crown. Somewhere, three more miners are reading the dashboard and warming up their
GPUs.

Every secret used along the way — each validator's private question set — is
**published in full a week later**, so anyone can go back and verify that the private
half of every verdict was honest. Nothing stays hidden; it is only *delayed*.

## 4. Why it matters

If it works, Epago is three things at once:

- **A frontier open deep-research model that keeps getting better on its own** — owned
  by no one, usable by everyone, small enough to self-host, improving as long as anyone
  in the world has a better idea.
- **A benchmark that cannot be gamed** — because the exam is generated after models are
  frozen, drawn partly from secret question sets, and every verdict is independently
  re-derivable from public data — exactly for the protocol, and to within a measured
  noise floor for the model scores themselves.
- **A new way to fund AI research** — a global, permissionless competition where a better
  idea and a GPU are enough to earn, with no gatekeeper deciding who is allowed to
  compete.

### Why scientific literature

Scientific literature is where a grounded-answer engine can be *proven*: every claim in a
paper is attributable to a source and every reported number is mechanically checkable.
Three properties make the whole of it — not one field of it — the right proving ground,
which is why the pinned corpus spans all four OpenAlex domains:

- **The hardest grounding test.** A fabricated number is a research-integrity failure in
  every field and a patient-safety event in some, so the task forces exactly the
  discipline the model must learn. The failure of the parametric approach here is
  measured, not asserted: published audits have found the majority of a general chatbot's
  medical references to be non-existent or erroneous, and surveys of clinicians report
  near-universal encounters with medical hallucinations, a large share of them judged
  capable of causing patient harm. Health science is simply where this has been audited
  most thoroughly — nothing about the failure mode is peculiar to it.
- **The highest-value verifiable workflow.** Evidence synthesis is slow and expensive in
  every discipline; the best-measured case is medicine, where published estimates put
  systematic reviews near 29,000 a year, each running to roughly 1.72 scientist-years of
  labor and well over a year from protocol to publication. The work is expensive
  *because* every claim must be traced to a source — which is precisely the work that can
  be graded mechanically.
- **Buyers who cannot use hosted APIs.** Data protection, GDPR special-category rules,
  high-risk classification under the EU AI Act, confidentiality over unpublished results,
  and reproducibility duties in reporting standards all push toward models that run inside
  the customer's own perimeter. That is the structural opening for an open, self-hostable
  model, and it is not a science-only property — law, patents, and regulatory filings
  share it.

The engine is field-agnostic; §18 sets out the axes it grows along.

The rest of this document explains, in increasing technical depth, exactly how each of
these guarantees is enforced.

---

# Part II — The model

## 5. What deep research means here

A deep-research agent is not a chatbot answering from memory. It is an agent that,
given a question about a body of documents, plans and executes a multi-step
investigation over a corpus, using two tools:

- **Search(query)** — submit a natural-language query, receive a ranked list of matching
  papers and snippets (BM25 full-text search over the corpus).
- **Browse(doc_id)** — open a specific paper and read its abstract and contents.

The agent alternates *reasoning* and *action* until it can answer — and every answer must
be **grounded**: a specific value or a specific study that exists in the corpus, not a
recollection. To see what this demands, consider the kind of task the exam actually poses.
The study is *described*, never named, and the quoted context is tiny with every other
number masked out:

> *"A study in this corpus concerns: prevalence, screening, adolescents. In that study,
> what value belongs in the blank? '... of N participants, ____ met the criteria ...'"*

There is no title to paste into search and no verbatim sentence to match on, so the agent
cannot shortcut to the source. It searches the handful of shared topic words, gets back a
*field* of candidate papers, opens them, decides by reading which one is the described
study, and only then reads out the missing value. A harder task spans several papers at
once:

> *"Among the studies in this corpus that report accuracy as a percentage, which one
> reports the highest accuracy? Answer with the study title."*

No single search answers this: the agent must find every study reporting that quantity,
read each reported value, and compare them. This is the evidence-synthesis job in
miniature — **the answer exists only as the result of evidence the agent itself gathered
and cross-checked.** That is the capability Epago selects for and rewards.

The **SCI4** task release the exam is built from obeys one rule — *hard to find, easy to
check* — and exercises three abilities: **constrained search** (find the one study
satisfying three to five constraints that each match a crowd of papers; only the
combination is unique, so candidates must be opened and eliminated), **cross-study
comparison** (open every named study and rank a stated quantity — which one wins is
written nowhere), and **computed evidence** (count a result across named studies — a
number that appears verbatim in no document at all). Answers are graded against keys the
generator re-derives mechanically from the corpus — never an opinion — and every mint
self-checks against the live search backend: a question that surfaces its own source near
the top of the results is discarded. An earlier corpus-wide comparison shape ("of every
study reporting X, which reports the highest X?") was retired: it cannot
be answered with the tools the agent has — top-k search gives no way to
enumerate a field defined over the whole corpus — and the reference model measured 0% on
it under every protocol tried. It returned answerably in SCI4's named-set shapes (with the
comparison universe named in the question), as a new release.

**Extraction is rung one of a ladder, not the whole job.** It comes first because it is
the rung that can be graded without a human in the loop, which is what makes a trustless
competition possible at all. The rungs above it — screening, cross-study synthesis,
risk-of-bias appraisal, drafted review sections — are task families in the same
environment rather than a different product, and §18 sets out the order they arrive in.
The same is true of modality: most quantitative results actually live in **tables and
figures**, where every current model is weak, and that is a capability frontier the
task ladder is built to reach rather than a formatting concern.

## 6. The evaluation environment

The environment in which duels run — the corpus and the search/browse tools — is Epago's
own **document corpus**. The genesis generation pins an **all-science** snapshot: 50,420
paper abstracts across ~135 fields in the four OpenAlex domains (Life, Physical, Health, Social).py` and assembled by
`scripts/build_corpus.py`. The papers are stored in a single local **SQLite** database
with an **FTS5** full-text index and **BM25** ranking (see `epago.environment.corpus`,
class `SqliteCorpus`), and the whole snapshot is pinned by a content digest
(`sha256:ceee9ec5abd5a755bc7524ddcd4036c9d8f90121f5f6fb562a3614b55ddc4043`) so every
validator searches a byte-identical world. There is no live internet, no external search
API, and no vector service in the loop — search is a local, deterministic function of the
pinned database, which is what makes Epago's duels fast, cheap, and posed identically on
every machine:

- **A local search tool** — `Search(query)` runs BM25 full-text retrieval over the pinned
  corpus and returns a ranked list of papers with snippets, at zero marginal cost and with
  identical results on every machine.
- **A local browse tool** — `Browse(doc_id)` opens a specific paper and returns its title,
  abstract, and text.

Two properties of this environment are load-bearing for Epago specifically:

1. **Determinism and cost.** Because the corpus is local and fixed, a duel of hundreds of
   multi-turn rollouts costs no API fees and — critically — runs against an identical
   world on every validator's machine. This is the half of reproducibility Epago can make
   exact, and it does: same bytes, same rankings, same tasks, same keys. The other half —
   the model's own behavior — is not exact on any GPU stack, which §10 measures rather
   than assumes and the acceptance test prices.
2. **Source masking.** When a task could be trivially solved by looking at the very paper
   it was minted from, that origin paper is *hidden* from the search index and browse tool
   for that task's rollouts. The agent cannot look the answer up where it came from; it
   must re-derive it through other papers. This turns a retrieval test into a genuine
   deep-research test — and, as we will see, it is also one of Epago's
   anti-overfitting defenses.

A pinned corpus is a scoring requirement, not a limit on where the model can be used.
This determinism is what lets independent validators pose the *identical exam*, which is
why it must run against a byte-identical world. The *behavior* the exam selects for is
retriever-agnostic: the agent learns to decompose, search, read and cross-check, and
the search backend behind `Search`/`Browse` is an interface. §18 sets out the path from
the pinned snapshot to live retrieval and, ultimately, to a customer's own private
document store — swapping the backend is plumbing, not retraining.

The question-generation pipeline is Epago's own, and what makes a task sound is
**mechanical in every case**: a task is admitted only when its answer is proved unique
over the corpus index before any wording exists, when the route to that answer is proved
takeable from the question a solver actually sees, and when the answer is proved to exist
in the corpus. Numbers inside any quoted context window are masked so a task cannot be
solved by pasting the phrase back into search, and tasks that would leak their own answer
are excluded by construction. Grading is later a mechanical comparison rather than a
judgment.

Two ways of producing tasks sit behind those checks. **Template releases** mint directly
from literal sentences by deterministic rule, with no language model in the loop, so
generation is a pure seeded function of *(corpus, release, seed)* that every validator
reproduces exactly. **Sealed-pool releases** word their questions with a language model,
because the task families worth asking cannot be phrased by rule — and a model's wording
is not reproducible from a seed, since no promise about temperature survives a provider
changing hardware or model version.

A sealed pool therefore buys the same two guarantees by sequencing rather than by
recomputation. The pool's digest is fixed on-chain before any duel uses it, so the exam
existed before any challenger's weights were frozen; the pool's task-id manifest is
published up front, so anyone can verify which questions a round drew; and each round's
questions are published in full afterwards, so any verdict can be re-graded. The
uniqueness and route proofs above are unchanged — the model chooses words, never which
task is admitted.

Task releases are frozen once pinned. **`SCI4`** is current: three families built on the
hard-to-find-easy-to-check rule, minting from a physics, CS, economics or materials paper
as readily as from a clinical one through a field-neutral vocabulary. Its predecessor
**`SCI3`** was replaced after instrumented ablations proved it was a cloze test — its
questions retrieved their own source at rank 1 for 88.3% of tasks, and a harness change
proven better on an external benchmark moved it by nothing. `SCI3` and the medicine-tuned
`SCI2` still exist unchanged — a release's template set and vocabulary are a determinism
contract, so contracts pinned to them are never silently re-judged.

## 7. The genesis model

Epago's genesis king is **Tongyi-DeepResearch-30B-A3B**, the open agentic-search model
released by Alibaba under the **Apache-2.0** license and pinned at an immutable revision.
It is a **Mixture-of-Experts** model — **30.5 billion total parameters with about 3.3
billion active per token** (128 experts, 8 active per token) — so it carries the capacity
of a large model while serving at close to small-model cost. "Small" here means *active*
parameters; this is not a dense 3B model. Run **quantized (AWQ 4-bit)** under vLLM, it
fits on a single consumer GPU.

**The floor Epago starts from.** Its authors report the following against OpenAI o3 and
DeepSeek-V3.1 (671B) on agentic deep-research benchmarks:

| Benchmark | Tongyi-DeepResearch-30B-A3B | OpenAI o3 | DeepSeek-V3.1 (671B) |
|---|---|---|---|
| Humanity's Last Exam | **32.9** | 24.9 | 29.8 |
| FRAMES | **90.6** | 84.0 | 83.7 |
| xbench-DeepSearch | **75.0** | 67.0 | 71.0 |
| WebWalkerQA | **72.2** | — | — |
| GAIA | **70.9** | — | — |
| BrowseComp-ZH | **46.7** | — | — |
| BrowseComp | 43.4 | **49.7** | — |

A dash marks a comparator figure not reproduced here; the "5 of 7" count is the base
model's authors' own head-to-head claim and their published table is the authoritative
source for it.

Five of seven, at roughly 3.3B active parameters against a 671B model. **o3 leads
BrowseComp, 49.7 to 43.4** — stated here deliberately: selective reporting is precisely
the gaming this protocol exists to make impossible, and a whitepaper that hid its one
loss would be doing it. Three further boundaries apply to every number above. They are
the **base model's** published results, not Epago's: no Epago-trained checkpoint can
claim them until a coronation actually occurs. They are results at **deep research**,
per unit of compute — not a claim of general intelligence over frontier models. And
they are external benchmarks, not Epago's own exam.

**On Epago's own exam the picture is much less flattering, and that is the point.** The
internal exam is built to be harder than any published benchmark: the question never
identifies its sources, and the answer must be reached by eliminating a crowd of
candidates or computed across documents. Measured with full agentic tooling the base
model sits at **33.9%**; handing it perfect retrieval lifts it only to **56.1%**, and
removing the corpus collapses it to the **17.3%** guessing floor. The exam is validated,
not assumed: a harness change proven better on an external deep-research benchmark moved
this exam and the benchmark *together*, and a same-size model without research training
is ranked below the base by both. The numbers are published rather than buried: the
34-to-56 gap is the headroom the competition exists to close, and per-episode telemetry
shows the cheapest first target — most lost points die in research episodes abandoned at
the turn, clock, or context budget.

A model that reaches frontier deep-research capability while still fitting and fine-tuning
on modest hardware is the crucial enabler for a *decentralized* competition: it is cheap
enough to fine-tune and to evaluate that permissionless, high-volume competition is
economically feasible. The genesis model is a general agentic-search checkpoint that Epago
specializes, through competition, toward grounded deep research across all of scientific
literature. It is the *floor*, not the ceiling — under Epago's mechanism it is only ever
replaced by something provably better.

The genesis model is a starting point that the network improves; Epago's own contribution
is not the model but the **mechanism that makes a global, trustless competition to
improve it possible.** That mechanism is the subject of Part III.

---

# Part III — The mechanism

This is Epago's novel contribution. Each of the following sections presents one mechanism,
the specific failure mode it prevents, and the exact parameters as implemented.

## 8. King-of-the-hill duels

**The primitive.** At any time there is exactly one **king** — the reigning champion
checkpoint. A **challenger** is evaluated head-to-head against the king in a **paired
duel**: both models answer the *identical* set of questions, and only the per-question
difference is scored:

```
d_i = (challenger correct on task i) − (king correct on task i)  ∈ {−1, 0, +1}
```

**Why paired, not a rating ladder.** Absolute scores depend on how hard the particular
exam was — 70% on a hard exam beats 80% on an easy one, and every duel draws a different
fresh exam. But the *difference* on the *same* questions cancels difficulty entirely: a
hard exam lowers both models together, leaving `d_i` untouched. This is precisely what
lets Epago throw a brand-new random exam at every duel and still compare fairly — it never
compares across exams, only the two models within one exam. Rating systems and
multi-metric panels were considered and rejected: they inject cross-exam rank-noise and
invite metric-gaming, and they answer a question ("how good is everyone?") the network
does not need answered. One scalar decides.

## 9. The two-slice exam

Each duel poses ~1,000 questions to both models (800 public + 200 private), drawn from two sources that defend
against two different attacks.

**Slice 1 — Public tasks (default 200): unpredictable, then reproducible.**
### Where a challenger lives

A challenger uploads into a prefix only its author can write and only the
validator can read. Losing therefore costs a miner the duel and nothing else:
the checkpoint is never handed to the rivals that beat it, which is what makes
a second attempt worth training for rather than worth copying.

Credentials for that upload travel over a public channel — there is no private
channel to a miner that does not reintroduce an operator — so each is encrypted
to that miner's own hotkey and all of them are published in one file anyone can
read. They are write-only and scoped to one prefix: a miner cannot read, list
or overwrite anything, including its own upload.

**Winning reverses it.** A crowned model is republished where anyone can fetch
it, because the model actually collecting emissions must be re-scorable by
anyone — that is the difference between trusting a validator and checking one.
A miner who loses and believes it was scored unfairly can publish its own
weights and let anyone verify; the burden sits with the party making the claim.

Submitting publicly through a Hugging Face repository remains supported and
needs no credentials. It differs in exactly one respect: everyone can read the
weights immediately.

A challenger's submission is committed to the chain under a **timelock** and unsealed 5
blocks later at its *reveal block*. The public tasks are generated deterministically from
that reveal block's hash:

```
blake2b(block_hash ‖ author_hotkey ‖ "public")  →  PCG64 seed  →  task selection
```

The block hash does not exist until after the challenger's weights are immutably frozen —
so **no one can train on these questions**, not even the submitter (*failure mode
prevented: training on the test set*). That holds for both release kinds: a template
release generates the questions from the seed, a sealed-pool release draws them from a
pool whose contents were committed earlier, and in neither case can anyone know beforehand
which questions a round will ask.

Reproducibility follows the same split. A template release is a pure function of public
data, so anyone regenerates the identical questions forever after. A sealed-pool release
is checked instead against artifacts pinned before the round — the task-id manifest and
the published round file — which an auditor needs to obtain rather than recompute. That is
a real reduction in independence, and it is the price of asking questions no rule can
phrase; the protocol states it rather than hiding it, and the replay tool reports a
missing artifact as a skip, never as a pass.

Because a published question becomes training data the moment it is released, a round's
questions are **retired from the pool** once published and never asked again.

**Slice 2 — Private tasks (default 200): the overfitting tripwire.**
Public tasks alone have a weakness: the generator is open-source, so a miner could
overfit the *generator's distribution* rather than getting genuinely better. So each duel
also draws questions from that validator's **private pool** — questions the validator
minted from its own sampling, that no miner has ever seen. That pool is drawn from a
**dated, private feed** — a fresh multi-source harvest (OpenAlex, Crossref, Europe PMC,
PubMed) across all four scientific domains, kept private while it is live and revealed
after it rotates — so freshness does the work alongside secrecy: a paper published this
week cannot be in any miner's training data. Acceptance requires the private half to also
favor the challenger on *each* accepting validator. To overfit the network, a miner would
have to overfit every validator's independent, secret question set simultaneously
(*failure mode prevented: overfitting the generator*).

Private evaluation normally costs auditability — "trust my secret test set" is exactly the
trust Epago refuses to require. The resolution is **delayed transparency**: each pool's
digest is committed on-chain with every verdict that used it, and the pool is **published
in full when it rotates** (~6 days). Every private verdict then becomes retroactively,
cryptographically checkable — a validator that fabricated private results is caught by
anyone, permanently. Secrecy while it matters; accountability forever after.

**How a question is answered and graded.** Each model runs an agent rollout (up to 40
turns, 32K context, greedy decoding) with the origin document masked from search. Grading
is a cascade that is programmatic first and un-foolable: normalized exact-match →
alias-match → numeric tolerance → (optional, disabled by default) a pinned, sanitized,
injection-hardened LLM judge as a last resort. The judge's invocation rate is published
per duel; near zero means the exam is cleanly gradeable, and a rise is a public alarm
(*failure mode prevented: gaming or injecting an LLM judge*). Because the answer key is
verified to literally exist in the corpus at task-generation time, "correct" is never an
opinion — it is a string comparison against a pre-proven key.

## 10. The statistical verdict

A duel produces a difference vector `d = [d_1, …, d_N]`. The question is not "was the mean
positive?" — it is "is the challenger *reliably* better, or did it draw a friendly exam?"

**Bootstrap lower confidence bound.** The mean `μ̂ = mean(d)` is a point estimate that
ignores uncertainty. Instead, Epago resamples the *already-collected* difference vector
with replacement B = 10,000 times (pure arithmetic — no model is re-run), computes each
resample's mean, and takes the 0.1st percentile: a one-sided **99.9% lower confidence
bound**, `lcb_pub`. Intuitively:

```
lcb  ≈  μ̂  −  (a penalty for how uncertain μ̂ is)
```

Consistent wins on a large sample → small penalty → the bound stays near the mean.
Scattered results on a small sample → large penalty → the bound collapses toward zero.
A false coronation requires a 1-in-1,000 statistical fluke *per validator*, compounded
across the quorum (*failure mode prevented: lucky coronations, and — decisively — copies
winning by chance*, §14) — and then it must happen **twice**: every provisional winner
is re-dueled once on a fresh confirmation exam before any validator commits an ACCEPT,
which squares the tail. This is why a perturbed copy of the king, whose true edge is
zero but whose scatter is occasionally positive, can never clear the bar: its wide
confidence interval always dips below the threshold, and a lucky draw does not repeat.

**The adaptive floor.** The bound must clear a threshold δ that scales with the king's
remaining headroom:

```
δ = max( 0.05 × (1 − king_accuracy_EMA) ,  1 × noise_floor )
```

A *fixed* floor is wrong at both ends: trivial against a weak early king (say 55%, where
δ = 0.0225), unwinnable against a strong late king (say 95%, where a fixed 2% bar is
mathematically impossible but the adaptive bar asks a fair 0.25%). Scaling to remaining
headroom keeps the *relative* bar constant across the king's entire life. And the floor
cannot be pushed down by miners — it moves only with the king's measured accuracy, and
lowering that means losing duels, which costs the crown (*failure mode prevented:
unwinnable late-game and floor manipulation*).

**The noise clamp.** Model inference is not bit-reproducible, and no setting on this
stack makes it so. Measured: the same prompt through the same engine, at batch of one,
greedy, seeded, with `enforce_eager` on and prefix caching off, decodes differently on
the second call — the fused MoE kernels reduce in a nondeterministic order, below the
level any determinism flag reaches. This is not a multi-GPU artifact; a single-GPU box
has it too. At duel scale it is not a rounding error either: re-scoring the *same
checkpoint* on the *same* 400 tasks flipped correctness on **84 of them (21%)**, and
the paired score gap — the quantity the verdict actually turns on — has a standard
error of **≈0.030 at n = 128**.

This is the environment the mechanism was designed for, and every layer of the design
answers it. The duel is *paired*, so exam difficulty cancels exactly and only the
per-task difference survives. Automated **king-versus-king calibration duels** run the
reigning champion against itself and measure this box's own floor directly, in the same
unit and through the same graded path a scored duel uses. δ is clamped at or above that
measured floor. And acceptance is a 99.9% *lower confidence bound* on the mean
difference, not a point estimate — the bootstrap prices the scatter that the noise
creates, and a challenger whose edge is indistinguishable from jitter cannot clear the
bar however lucky its sampled mean. A challenger must beat the king by more than the
king disagrees with itself (*failure mode prevented: coin-flip verdicts*).

The design therefore never assumed bit-identical inference; it assumed a noisy scorer
and made the verdict robust to one. That is the stronger claim, and it is the one that
survives measurement.

**Acceptance:** `lcb_pub > δ` **and** the private-half mean `> 0`.

## 11. Decentralized validation

**The failure mode: the trusted-operator hole.** A single evaluation server — however
well audited — is an owner-shaped hole in a "decentralized" system: one entity controls
verdicts, and its private holdout is a single unauditable secret.

**The mechanism: stake-quorum coronation.** Every validator runs the identical
open-source software, holds its own private pool, and publishes its own signed verdict
on-chain. A challenger becomes king at the first block where **accepting verdicts cover
≥ θ (default 51%) of active-evaluator stake.** Coronation is a *pure function of public
chain state*: every honest party independently derives the same king at the same block,
with no coordination channel and no coronation message that could be forged. No single
validator can crown or block. A dissenting validator follows the quorum mechanically (its
weight-setting must converge) but its dissent stays permanently on record — and becomes
checkable once its private pool publishes.

**A powerful consequence: weight-copying by validators becomes harmless.** On Bittensor,
a validator's rewards depend on agreeing with consensus, which historically tempts lazy
validators to *copy* other validators' weight vectors rather than do the evaluation work.
In Epago this is neutralized: the correct weight vector is a deterministic function of
chain state, so a copier changes nothing about which model wins — coronation depends on
*verdicts*, which require actually running duels. A weights-only free-rider draws
dividends without contributing, but it cannot corrupt model selection, and it is publicly
visible (a validator with stake but no on-chain verdict trail is a passenger who does not
count toward quorum). Because honest validators all derive the *same* vector, Bittensor's
Yuma consensus finds near-perfect agreement and clips essentially nothing — validator
trust scores sit near 1.0 for everyone honest, with only hour-scale coronation-race jitter
and sub-percent tail differences, both self-healing.

## 12. Determinism and replay

Trust in Epago is not asked for; it is *checked*. Every duel writes a signed **audit
record** binding every input that produced the verdict: the reveal block hash, all derived
seeds, the task-set digests, the full per-task difference vector, the private-pool digest
and epoch, the thresholds, and the harness version digest. The compact verdict committed
on-chain contains a hash of this record.

This makes verdicts replayable at two levels, and the boundary between them is exact.

- **Level 1 (exact, no GPU required):** everything from the reveal block hash to the
  published verdict re-derives bit-for-bit. `scripts/replay_verdict.py` recomputes both
  seeds from the block hash, regenerates the public task set and checks its ids digest,
  verifies the corpus and private-pool digests, recomputes the bootstrap LCB from the
  stored per-task difference vector, recomputes the `audit16` digest, verifies the
  validator's sr25519 signature, and cross-checks the on-chain `ev3` commitment. It
  needs no model, no GPU, and no torch. Each check passes or fails outright; there is
  no tolerance band anywhere in this level, and a validator that got the mathematics,
  the task selection or the commitment wrong is exposed with certainty.
- **Level 2 (statistical):** re-running the actual model rollouts checks that the
  difference vector describes real behavior. This level *cannot* be exact, because
  inference is not reproducible: an honest re-scoring of the same checkpoint on the same
  tasks disagrees with the original on ~21% of them (§10). What it detects is a
  distribution, not a mismatch — a validator that fabricated its difference vector
  wholesale diverges far past the measured floor and is caught, while fudging a task or
  two hides inside the floor and is also, by construction, too small to flip a verdict
  that cleared δ with margin.

Stating the boundary is the point rather than an apology for it. The quantities a
dishonest validator would have to falsify to steal a coronation — the seeds, the task
set, the pool it committed to, the arithmetic, the signature — are all in Level 1, where
replay is exact and unforgiving. The one quantity that can only be checked statistically
is the one the acceptance test already treats as noisy, and prices with a 99.9%
confidence bound over an adaptive floor calibrated to the measured noise.

The public leaderboard is itself a *pure export* of these audit artifacts — anyone can
regenerate it from a validator's published state and diff it against what that validator
serves. A dashboard that disagrees with the audit trail is impossible by construction.
This is how "no trusted operator" is enforced in practice: not by trust, but by
replayability — exact for the protocol, statistical for the scores, and specified
either way.

## 13. Emissions and game theory

**The failure mode: hoarding and dead subnets.** Pure winner-takes-all provably drives
rational miners to hoard improvements and everyone else to quit, collapsing the
competition. Epago's emission schedule is engineered to keep the arena alive. Miner
rewards split three ways:

| Pool | Share | Recipient | Purpose |
|---|---|---|---|
| **King** | 90% falling to 85% over ~3 days | reigning champion | the prize |
| **Arena** | 10% rising to 15% | the three most recent former kings | a dethroned champion is not discarded |

The two shares are the whole budget: the arena receives exactly what the king
does not take, so the weight vector always sums to one.

Four properties make *genuine improvement the only winning strategy*:

- **A bounded reign.** The king's share is a band, not a decay toward nothing. It
  starts at 90% and falls linearly to 85% over roughly three days, then holds. An
  undefended crown bleeds toward the floor — so it is always worth attacking — but an
  incumbent keeps a defined majority, and both ends of the schedule are checkable
  against a block number rather than being an asymptote nobody can verify.
- **The arena is a short succession, not a pension.** Each coronation seats the
  displaced king; the fourth displacement retires the oldest. Being beaten costs the
  crown, not everything, which is what makes attacking a strong king rational for the
  incumbent's rivals *and* survivable for the incumbent.
- **Coronation bonus** (∝ measured improvement, capped, ~24 h) holds a fresh king at
  the top of the band, never above it. Revealing a *big* improvement at once pays more
  than dribbling it out, and the arena's floor is never absorbed.
- **Self-dethrone inherits the reign clock:** crowning yourself with a sliced +δ
  improvement does not reset the bleed, and does not seat you in your own arena — so
  salami-slicing a large improvement into many small coronations buys nothing
  (*failure mode prevented: slicing*).

Before the first coronation the arena is empty and its share is burned, so nothing is
paid out until something has actually been crowned.

## 14. Threat model

Each attack maps to a structural defense. The full 15-row table is in the mechanism spec;
the essentials:

| Attack | Defense |
|---|---|
| Train on the test set | Tasks generated from a block hash that exists only after weights are frozen |
| Overfit the question generator | 50% private holdout per validator; pools published only after rotation |
| "Trust me" evaluation | Chain-stamped, signed verdicts: seeds, tasks, digests, arithmetic and signature replay exactly (§12); the scores replay statistically against a measured floor |
| Copy the king | A copy's true edge is zero; its LCB cannot clear δ — copying is *priced at zero*, not detected (no fragile hash arms-race) |
| Copy a **challenger** | Structurally impossible: a challenger uploads into a prefix only it can write and only the validator can read, so a losing checkpoint is never exposed to the rivals it lost to. Only a crowned model becomes public, and copying the king is already priced at zero |
| Steal a checkpoint by re-revealing it | Digest ownership belongs to the first on-chain reveal; timelock prevents mempool sniping |
| Architecture smuggling | Config lock, safetensors-only (no executable code), 1.05× size cap — all at intake, zero GPU |
| Spam | **One submission per hotkey, permanently.** Whatever happens to it — crowned, near-miss or beaten — that hotkey is spent, and another attempt means registering a new one and paying the registration burn. Attempts are priced rather than free, so flooding the queue with speculative checkpoints costs real TAO instead of costing validators a rollout sweep each |
| A single corrupt validator | Coronation requires ≥ θ stake; dissent is recorded and later checkable |
| Silent benchmark drift | Scheduled anchoring of the king against an external public benchmark; internal-vs-external divergence published with an alarm |

The unifying principle: **each mechanism removes exactly one way to win without building a
better model.** What survives all of them is a genuinely better deep-research model.

---

# Part IV — The network

## 15. Roles

**Validators** run the "validator-in-a-box": chain client, task generator, private pool,
duel engine, audit log, publisher, and weight-setter. There is no lead validator and no
manual operation — corpus ingestion, pool rotation, calibration, anchoring, audit
publication, and weight-setting are all autonomous. A validator is a machine you power,
not a job you perform. Hardware is a CPU coordination box plus a GPU evaluation box (which
may be the same machine); validators size their GPU empirically and tune throughput via
batched rollouts, an optional low-memory mode, and an optional remote-eval split.

**Model miners** train challengers by any method — the protocol never inspects *how* a
checkpoint was produced, only how it performs. Lifecycle: resolve the king → prepare →
train → run the exact validator intake checks locally (preflight, so a submission is never
wasted) → upload a digest-pinned checkpoint → reveal via timelock → duel → crowned,
near-miss, or lost.

## 16. On-chain protocol

Epago runs on the Bittensor subtensor chain, which carries only *fingerprints* — never
data. Models (gigabytes) and the corpus live off-chain, pinned by content hashes.
Everything on-chain is a short string, using two facilities matched to their constraints:

- **The timelock-reveal channel** (multi-entry per participant, chain-stamped block
  numbers) carries everything that must accumulate or be trustlessly ordered: challenge
  submissions, validator verdicts, private-pool digests, and the king pointer. A
  validator cannot backdate a verdict, and because the chain also stamps the
  signer, no payload carries its own claim of authorship.
- **The plaintext commitment slot** (one string per participant, 128-byte cap) carries
  only the compact audit-log checkpoint.

The chain provides exactly the three things it is uniquely good at — unforgeable ordering
(who revealed first), unforgeable timing (no backdated verdicts), and binding tiny
commitments to huge off-chain objects (a 64-character hash pins a multi-gigabyte model
immutably). Final rewards are set through the chain's commit-reveal weights extrinsic and
allocated by its native Yuma consensus.

## 17. Deployment and throughput

Every validator evaluates every candidate (quorum requires it), so per-validator
throughput sets the network's verdict latency. The duel is embarrassingly parallel across
tasks; validators run batched rollouts to keep the evaluation GPU saturated, turning what
would be an hours-long serial duel into a much shorter batched one. Demand is structurally
bounded — one live submission per participant, a hard cap on registered participants,
one submission per hotkey, and an automatic queue circuit-breaker that prices intake in
time when latency approaches the service target. The result is a system that delivers
verdicts well inside its target window on a single evaluation GPU, and scales by adding
GPUs with no mechanism change. All hardware sizing is done empirically by an included
measurement tool rather than assumed.

---

# Part V

## 18. Roadmap

Epago is organized around **chain generations.** Each generation pins a base architecture,
a corpus, and an evaluation environment in a single contract; the competition runs until
the community advances to the next generation. The genesis generation is a 30.5B MoE
deep-research model (base: Tongyi-DeepResearch-30B-A3B, ~3.3B active per token) over an
all-science literature corpus — 5,792 papers balanced across the four OpenAlex domains,
read by the `SCI4` task release. Because the model, corpus, and environment are all
**configuration, not code**, future generations are contract changes, not rewrites. The
near-term path runs from testnet validation (multi-validator quorum, real GPU timing,
live chain behavior) to mainnet launch with the full economic schedule active.

Beyond that, the system scales along four independent axes.

**Axis 1 — Corpora as generations.** A corpus *is* a chain contract: documents, task
templates, and architecture pins in one file. Adding one is configuration, never a
mechanism rewrite, and generations run in parallel — each with its own king, its own
exam, and its own emission share. The genesis generation already covers all of scientific
literature; the order beyond it is law (case law, contracts) → patents → regulatory
filings. Each of those shares the decisive property science has: the answer is in the
documents, it must be attributed, and correctness can be checked mechanically.

**Axis 2 — The task ladder.** Extraction is rung one, chosen because it grades without
a human. Above it: screening, cross-study synthesis, risk-of-bias appraisal, and full
drafted review sections. Each rung is a new task family generated by the same pipeline
and graded by the same programmatic-first cascade, so climbing the ladder is a task
release (§6), not a new protocol.

**Axis 3 — Modality.** Most quantitative results live in **tables and figures**, where
every current model is weak. Table and figure extraction is therefore a defensible capability
frontier rather than a formatting detail, and it is where a specialized small model has
the most room to beat a general large one.

**Axis 4 — Corpus reach.** Pinned snapshot (what deterministic scoring requires) → live
retrieval against public sources such as OpenAlex, Crossref and PubMed → the customer's
own private document store. The behavior the exam selects for is
**retriever-agnostic**; swapping the search
backend is plumbing, not retraining. This is what makes the model deployable inside a
regulated buyer's perimeter, which is the commercial reason the model is open and small
in the first place.

### The compounding asset

The competition emits a second output that is first-class, not a byproduct. Every duel
produces verified `(task, answer, correct/incorrect)` records over documents published
after the models were trained — **verified reward data**, the scarcest input in AI
post-training and the leading bottleneck in scaling reinforcement learning. It cannot be
scraped, because it does not exist anywhere else; a competitor would have to run the
machine for a year to accumulate the equivalent. The flywheel closes: fresh documents →
generated tasks → duels → verified reward data → better models → a more valuable crown →
more miners → more duels.

The infrastructure for this already exists in the protocol as specified — it is the
audit trail (§12) and the published private pools (§9) read as a dataset rather than as
an accountability record.

## 19. Foundations and references

Epago's **evaluation environment** is its own: a pinned document corpus (the genesis
snapshot is 5,792 paper abstracts balanced across the four OpenAlex domains, harvested
from OpenAlex and Crossref), a local SQLite/FTS5 search-and-browse environment
with BM25 ranking, and a task pipeline whose admission checks — proved answer uniqueness,
a proved-takeable route, source masking and difficulty filtering — are mechanical
regardless of whether a release words its questions by rule or by model. Its **genesis model** is **Tongyi-DeepResearch-30B-A3B**, Alibaba's
open agentic-search model (Apache-2.0), adopted as the starting checkpoint and pinned by
its immutable revision; the benchmark results in §7 are that model's published results,
reported by its authors, and are cited here as Epago's starting floor rather than as
Epago's own. Epago's **novel contribution** is the decentralized competitive
*mechanism* that surrounds this environment — the paired-duel selection test,
block-hash-seeded task generation, distributed private holdouts with delayed transparency,
bootstrap-LCB acceptance over an adaptive noise-clamped floor, stake-quorum coronation, the
anti-hoarding emission schedule, and the fully replayable audit trail.

Supporting infrastructure: the Bittensor network (Yuma consensus, timelock commit-reveal,
on-chain commitments); local full-text retrieval via SQLite FTS5 with BM25 ranking;
inference via vLLM, quantized. The complete normative specification, including the full
threat model and every constant, is in the mechanism specification ([DESIGN.md](DESIGN.md));
the complete component-by-component handbook is in [subnet.md](subnet.md).

## 20. Conclusion

The open frontier of deep research is already small. It is also already static: it moves
when a lab chooses to move it, and nothing proves that each move is real. Epago's claim
is that this is fixable by mechanism rather than by scale — that if you make the
competition permissionless and the exam ungameable, a ~3.3B-active model can be pushed
past the frontier by everyone at once, forever, with every step proven.

That turns "which AI model is best?" from a claim into a **theorem** — one that is
re-proved, publicly and reproducibly, every time a challenger is submitted. By generating
the exam after models are frozen, splitting it across independent secret holdouts,
grading it by un-foolable rules, accepting only statistically proven improvements,
crowning by stake quorum, and publishing every input for anyone to re-check, the protocol
removes every path to winning except the one it wants to reward: building a genuinely
better open deep-research model. All of scientific literature is where that is proven;
law, patents and regulatory filings are where it goes next.

There is no company to trust, no operator to bribe, no test set to memorize, and no
verdict that cannot be reproduced by a stranger from public data. The crown is always
worth taking — and always up for grabs.

---

*Epago is open-source under the MIT license. This whitepaper describes the protocol as
implemented; parameters cited are defaults and are configurable per chain generation.
Nothing herein is financial advice.*

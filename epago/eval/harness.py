"""The single rollout loop.

This module is the only place the agent loop exists; probes, duels, and the
server all call :func:`run_rollout`. The system prompt and every protocol
constant are pinned and folded into :func:`harness_digest` so an audit record
binds the exact harness a verdict was produced under. Determinism constraints:
greedy decoding is the backend's job, prompt construction here is pure string
concatenation over the transcript — no wall clock, no randomness, no dict
iteration order enters the prompt. Prompt *delivery* is also the backend's
job: a real engine wraps this text in the checkpoint's own chat template
(:func:`epago.eval.backend.chat_prompt`), which is why nothing here knows a
single turn marker. The wall clock is read only to enforce the
rollout timeout and to report ``wall_time_s``, neither of which feeds scoring
math.
"""

from __future__ import annotations

import hashlib
import json as _json
import re
import time
from typing import TYPE_CHECKING

from epago import constants
from epago.core.types import RolloutResult, Task
from epago.eval.backend import ModelBackend
from epago.eval.judge import LlmJudge, judge_answer

if TYPE_CHECKING:
    from epago.environment.services import ToolSession

#: ``v2`` fixes two defects that changed what the model actually sees and how
#: its output is read: the prompt now reaches a chat/RL-trained checkpoint
#: through its own chat template (:func:`epago.eval.backend.chat_prompt`), and
#: an action is parsed from the LAST tag pair outside any closed think block
#: rather than the first (:func:`parse_action`). Both change scored behaviour,
#: so the version — and therefore :func:`harness_digest` — moves with them; a
#: v1 verdict is not comparable to a v2 verdict.
#:
#: ``v3`` folds the tool surface (:data:`epago.constants.TOOL_SURFACE`) into
#: the digest. Search output is prompt bytes, so a validator running different
#: retrieval semantics was producing different transcripts under an identical
#: digest — the one thing the digest exists to make impossible.
#:
#: ``v4`` replaces the bespoke ``<search>/<browse>`` protocol with the
#: reference model's NATIVE agentic convention: tools declared as JSON
#: signatures in the system prompt, called via ``<tool_call>`` JSON, results
#: returned inside ``<tool_response>`` user turns, and the transcript carried
#: as real chat messages rather than one flat string. Measured under v3 on the
#: SCI4 families: 48% of episodes died to malformed actions and the model
#: scored WORSE with tools than closed-book (7.8% vs 10.0%) — an invented
#: protocol is out of distribution for an agentic-RL-trained checkpoint, and
#: the eval was measuring protocol compliance instead of research.
#: ``v4.1`` demotes the wall clock to a hang safety net (3600 s). Measured on
#: 1,600 recorded episodes: 100% of timeout deaths happened below the turn cap
#: at a median of 21 turns, and per-turn wall time was dominated by batch
#: queueing — the 900 s cap punished a miner for the validator's load, so two
#: honest validators could disagree under an identical digest. The binding
#: budgets are now exclusively the deterministic ones: the turn cap and the
#: transcript character budget, both pure functions of the transcript.
#: v4.3 closes the turn cap the way the character budget was already closed:
#: tools shut and the model is asked for its answer, instead of the episode
#: ending unasked.
#:
#: v4.2 answers a repeated search with a pointer instead of a second copy of
#: the same results page. Tool output was 85% of the context in episodes that
#: overflowed and a quarter of all searches were exact repeats, so the repeat
#: was buying nothing and costing a page. Bumped rather than edited in place:
#: the version is hashed into `harness_digest`, and a duel replays against the
#: digest it was run under.
HARNESS_VERSION = "epago-harness-v4.3"

#: The two tools, declared exactly the way the reference model was trained to
#: read them. Wording is domain-neutral (the corpus spans all of science) and
#: the shape is the model's, not ours.
TOOL_SPECS: tuple[dict, ...] = (
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Run a full-text search over the pinned research corpus — a "
                "fixed local library of scientific papers — and return the top "
                "matching documents with their title, url and snippet. This is "
                "the only literature available: there is no web access. Accepts "
                "multiple complementary queries in one call; if a query returns "
                "off-topic documents, search again with different terms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string", "description": "The search query."},
                        "minItems": 1,
                        "description": "The list of search queries.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": (
                "Open one or more documents from the corpus by the url (or id) "
                "shown in search results and return their full text, so you can "
                "read exact reported values. Only documents surfaced by "
                "`search` can be opened."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "array",
                        "items": {"type": "string", "description": "The url or id to open."},
                        "minItems": 1,
                        "description": "The urls (or ids) of documents to open.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "What you want to find in these documents.",
                    },
                },
                "required": ["url"],
            },
        },
    },
)

SYSTEM_PROMPT = (
    "You are a deep research assistant. Your core function is to conduct "
    "thorough, multi-source investigations into any topic using the provided "
    "corpus tools. When you have gathered sufficient information and are ready "
    "to provide the definitive response, you must enclose the entire final "
    "answer within <answer></answer> tags.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    + "\n".join(_json.dumps(spec, sort_keys=True) for spec in TOOL_SPECS)
    + "\n</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "# Answer rules\n\n"
    "Your evidence is this corpus only: never use outside knowledge and never "
    "invent a document or a number. The final answer must be SHORT and "
    f"LITERAL (at most {constants.ANSWER_MAX_CHARS} characters): when asked "
    "for a title answer with the exact title alone, when asked for a number "
    "or percentage answer with that value alone — no explanation inside the "
    "<answer></answer> tags.\n\n"
    "Current date: " + constants.HARNESS_PROMPT_DATE
)

#: The model yields after every tool call (it was trained to let the runtime
#: write the ``<tool_response>``), and ``</answer>`` cuts the final turn.
STOP_SEQUENCES: tuple[str, ...] = ("\n<tool_response>", "<tool_response>", "</answer>")
#: Room for one turn: a reasoning checkpoint writes its scratchpad before the
#: action, so a tight cap truncates it mid-thought and the turn yields no
#: parseable action at all. Measured on the reference model over the pinned
#: corpus: 512 gives 78% well-formed / 74% accuracy, 2048 gives 89% / 84%, at
#: roughly 3x wall time per rollout. A duel is minutes against a round interval
#: of days, so the accuracy is worth the time.
#: 2048. Raising to 4096 was measured (SCI4, paired, n=900): malformed
#: deaths barely moved (224 -> 202 — they are greedy repetition loops, which
#: fill any budget) while timeouts doubled (61 -> 142) because every turn ran
#: twice as long; accuracy fell 31.8% -> 28.7% (p = 0.043). The loop pathology
#: is addressed by ROLLOUT_REPETITION_PENALTY instead.
MAX_ACTION_TOKENS = 2048
#: One <tool_response> may carry up to VISIT_MAX_DOCS full documents
#: (BROWSE_PAGE_CHARS each) plus headers; the transcript char budget, not this
#: cap, is what bounds the whole context.
OBSERVATION_MAX_CHARS = 13_000

#: Turns held back so a model at the turn cap is asked to commit rather than
#: cut off. Two is enough to read the demand and answer it, and matches the
#: grace the character path already allowed.
_TURN_RESERVE = 2

_NUDGE = (
    "Your previous output contained no tool call and no answer. Either call a "
    "tool with <tool_call>{\"name\": ..., \"arguments\": ...}</tool_call> or "
    "give the final answer inside <answer></answer> tags."
)

#: Appended once when the transcript reaches its char budget: the same
#: close-the-tools move the reference model was trained to obey.
_BUDGET_MSG = (
    "You have now reached the research budget for this question. The tools are "
    "closed and no further tool call will be executed. Based on all the "
    "information above, give what you consider the most likely answer, inside "
    "<answer></answer> tags."
)

#: One action turn. The payload deliberately cannot swallow another opening
#: tag: without that guard, "Provide the answer in <answer> tags. Final:
#: <answer>0.8</answer>" parses as the payload " tags. Final: <answer>0.8",
#: which no judge can ever mark correct even though the model answered.
#: (v3 syntax — kept for replaying old transcripts and for tests.)
_ACTION_RE = re.compile(
    r"<(search|browse|answer)>((?:(?!<(?:search|browse|answer)>).)*?)</\1>", re.DOTALL
)

#: A native tool call: JSON between <tool_call> tags.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
#: The final answer. ``</answer>`` is a stop sequence, so the closing tag may
#: be missing from the generation — the answer then runs to end-of-text.
_ANSWER_RE = re.compile(r"<answer>((?:(?!<answer>).)*?)(?:</answer>|\Z)", re.DOTALL)

#: Reasoning-model scratchpad. Anything inside a CLOSED think block is the
#: model talking to itself, not an action, so it is removed before parsing.
#: An UNCLOSED block is left alone on purpose: the stop sequences cut a
#: generation at the first closing action tag, so a model that starts an
#: action while still inside <think> has its block truncated open, and
#: dropping that text would throw away the only action the turn contains.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_action(raw: str) -> tuple[str, str] | None:
    """The model's action for one turn, or ``None`` when it emitted none.

    Two rules beyond "find a tag pair", both of which only ever matter when
    the generation contains more than the bare action:

    1. closed ``<think>...</think>`` blocks are dropped (see :data:`_THINK_RE`);
    2. the LAST action wins, not the first. Models routinely narrate the
       protocol before obeying it ("I should put this in <answer> tags"), and
       grading the narration instead of the answer silently scores a correct
       episode wrong. With exactly one action present — the overwhelming
       common case, since the stop sequences truncate at the first closing
       tag — last and first are the same match, so behaviour is unchanged.
    """
    matches = list(_ACTION_RE.finditer(_THINK_RE.sub(" ", raw)))
    if not matches:
        return None
    last = matches[-1]
    return last.group(1), last.group(2).strip()


def parse_native_turn(raw: str) -> tuple[str, object] | None:
    """One native-protocol turn: ``("answer", text)``, ``("call", (name, args))``
    or ``None`` when the generation contains neither.

    An answer beats a tool call when both appear (the model has finished);
    among tool calls the LAST syntactically valid JSON object wins, mirroring
    :func:`parse_action`'s narrate-then-obey rule. Closed ``<think>`` blocks
    are dropped first, exactly as in v3.
    """
    text = _THINK_RE.sub(" ", raw)
    answers = list(_ANSWER_RE.finditer(text))
    if answers:
        answer = answers[-1].group(1).strip()
        if answer:
            return ("answer", answer)
    for m in reversed(list(_TOOL_CALL_RE.finditer(text))):
        try:
            obj = _json.loads(m.group(1))
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments")
            return ("call", (obj["name"], args if isinstance(args, dict) else {}))
    return None


def _string_list(value: object) -> list[str]:
    """Normalize a tool argument that may be a string or a list of strings."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


def harness_digest() -> str:
    """sha256 over the pinned prompt, every protocol constant, and the tools.

    Recorded in audit records: two validators produce comparable verdicts only
    if their harness digests match, so anything that changes prompt bytes or
    loop limits must change this digest. That includes the tool surface —
    a search result is prompt bytes the moment it enters the transcript.
    """
    payload = "\n".join(
        (
            HARNESS_VERSION,
            SYSTEM_PROMPT,
            *STOP_SEQUENCES,
            str(MAX_ACTION_TOKENS),
            str(OBSERVATION_MAX_CHARS),
            str(constants.ROLLOUT_MAX_TURNS),
            str(constants.ROLLOUT_TIMEOUT_S),
            repr(constants.ROLLOUT_TEMPERATURE),
            str(constants.ROLLOUT_SEED),
            repr(constants.ROLLOUT_REPETITION_PENALTY),
            str(constants.ANSWER_MAX_CHARS),
            *constants.TOOL_SURFACE,
        )
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def initial_messages(task: Task) -> list[dict]:
    """The conversation as the checkpoint's chat template will render it."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {task.question}"},
    ]


class Episode:
    """One agent episode as a resumable state machine.

    :func:`run_rollout` drives a single episode; :func:`run_rollouts_batched`
    keeps many in flight and feeds every active episode's next prompt to the
    backend as one batch step. Both paths execute the identical per-turn
    logic below, so batching changes throughput and nothing else.
    """

    __slots__ = (
        "task", "session", "max_turns", "timeout_s",
        "start", "messages", "answer", "error", "nudged", "turns", "done",
        "malformed", "budget_closed", "post_budget_turns",
    )

    def __init__(
        self,
        task: Task,
        session: "ToolSession",
        max_turns: int = constants.ROLLOUT_MAX_TURNS,
        timeout_s: float = constants.ROLLOUT_TIMEOUT_S,
    ) -> None:
        self.task = task
        self.session = session
        self.max_turns = max_turns
        self.timeout_s = timeout_s
        self.start = time.monotonic()
        self.messages: list[dict] = initial_messages(task)
        self.answer: str | None = None
        self.error: str | None = None
        self.nudged = False
        self.turns = 0
        self.done = False
        self.malformed = 0
        self.budget_closed = False
        self.post_budget_turns = 0

    def _transcript_chars(self) -> int:
        return sum(len(str(m.get("content", ""))) for m in self.messages)

    def begin_turn(self) -> bool:
        """Advance the episode clock; False when the episode just ended."""
        if self.done:
            return False
        if self.turns >= self.max_turns:
            self.error = "turn_cap"
            self.done = True
            return False
        # Running out of turns is closed the same way as running out of
        # characters: tools shut, and the model is asked for its best answer.
        # Only the character path used to do this, so an episode that hit the
        # turn cap died with no chance to commit however much it had found.
        # That asymmetry decided real episodes -- with cheaper search pages the
        # binding limit moved from characters to turns, and 51% of episodes
        # ended at the cap holding 71% of their evidence, unasked.
        if not self.budget_closed and self.turns >= self.max_turns - _TURN_RESERVE:
            self.budget_closed = True
            self.messages.append({"role": "user", "content": _BUDGET_MSG})
        if time.monotonic() - self.start > self.timeout_s:
            self.error = "timeout"
            self.done = True
            return False
        if self.budget_closed:
            self.post_budget_turns += 1
            if self.post_budget_turns > 2:
                self.error = "context_budget"
                self.done = True
                return False
        elif self._transcript_chars() > constants.TRANSCRIPT_MAX_CHARS:
            # Close the tools and demand an answer, once; the model gets two
            # turns to comply before the episode ends unanswered.
            self.budget_closed = True
            self.messages.append({"role": "user", "content": _BUDGET_MSG})
        self.turns += 1
        return True

    def prompt(self) -> list[dict]:
        return self.messages

    def _dispatch(self, name: str, args: dict) -> str:
        """Run one native tool call; every failure is an observation the model
        can recover from, never an exception that aborts a scored rollout."""
        if self.budget_closed:
            return (
                "The tools are closed. Provide your final answer now inside "
                "<answer></answer> tags."
            )
        try:
            if name == "search":
                queries = _string_list(args.get("query"))
                if not queries:
                    return "search error: 'query' must be a string or a list of strings."
                return self.session.search_page(queries)
            if name == "visit":
                targets = _string_list(args.get("url"))
                if not targets:
                    return "visit error: 'url' must be a string or a list of strings."
                return self.session.visit_pages(targets)
        except Exception as exc:  # noqa: BLE001 - tools must not abort scored rollouts
            return f"tool error: {type(exc).__name__}"
        return f"Unknown tool {name!r}. Available tools: search, visit."

    def advance(self, raw: str) -> None:
        """Consume one generation: parse the turn, run the tool, or finish.

        One malformed turn earns a single nudge retry; a second terminates the
        episode with ``answer=None`` — the harness never guesses an action on
        the model's behalf.
        """
        self.messages.append({"role": "assistant", "content": raw})
        turn = parse_native_turn(raw)
        if turn is None:
            self.malformed += 1
            if self.nudged:
                self.error = "malformed_action"
                self.done = True
                return
            self.nudged = True
            self.messages.append({"role": "user", "content": _NUDGE})
            return
        kind, payload = turn
        if kind == "answer":
            self.answer = str(payload)[: constants.ANSWER_MAX_CHARS]
            self.done = True
            return
        name, args = payload
        observation = self._dispatch(name, args)[:OBSERVATION_MAX_CHARS]
        self.messages.append(
            {"role": "user", "content": f"<tool_response>\n{observation}\n</tool_response>"}
        )

    def result(self, llm_judge: LlmJudge | None = None) -> RolloutResult:
        correct, tier = judge_answer(self.task, self.answer, llm_judge)
        return RolloutResult(
            task_id=self.task.task_id,
            answer=self.answer,
            correct=correct,
            turns=self.turns,
            wall_time_s=time.monotonic() - self.start,
            judge_tier=tier,
            error=self.error,
            malformed_actions=self.malformed,
        )


def run_rollout(
    backend: ModelBackend,
    task: Task,
    session: "ToolSession",
    *,
    max_turns: int = constants.ROLLOUT_MAX_TURNS,
    timeout_s: float = constants.ROLLOUT_TIMEOUT_S,
    llm_judge: LlmJudge | None = None,
) -> RolloutResult:
    """Run one agent episode and judge the final answer."""
    ep = Episode(task, session, max_turns=max_turns, timeout_s=timeout_s)
    while ep.begin_turn():
        ep.advance(backend.generate(ep.prompt(), MAX_ACTION_TOKENS, list(STOP_SEQUENCES)))
    return ep.result(llm_judge)


def run_rollouts_batched(
    backend: ModelBackend,
    tasks: list[Task],
    session_factory,
    *,
    concurrency: int | None = None,
    max_turns: int = constants.ROLLOUT_MAX_TURNS,
    timeout_s: float = constants.ROLLOUT_TIMEOUT_S,
    llm_judge: LlmJudge | None = None,
    on_result=None,
) -> list[RolloutResult]:
    """Run many episodes with up to ``concurrency`` in flight.

    Each step gathers every active episode's next prompt into ONE
    ``generate_many`` call, so a continuous-batching engine stays saturated
    instead of decoding one rollout at a time. Per-episode logic is exactly
    :class:`Episode`; results come back in ``tasks`` order. An episode whose
    session construction or judging crashes scores incorrect — one hostile
    task can never abort the batch.
    """
    from epago.eval.backend import generate_many

    conc = max(int(concurrency or constants.ROLLOUT_CONCURRENCY), 1)
    results: list[RolloutResult | None] = [None] * len(tasks)
    queue = list(enumerate(tasks))
    queue.reverse()  # pop() consumes in task order
    active: list[tuple[int, Episode]] = []

    def finalize(index: int, ep: Episode) -> None:
        try:
            res = ep.result(llm_judge)
        except Exception as exc:  # noqa: BLE001 - judging must not abort the batch
            res = RolloutResult(
                ep.task.task_id, None, False, ep.turns, 0.0, "none",
                f"judge_crash: {type(exc).__name__}",
            )
        results[index] = res
        if on_result is not None:
            on_result(index, res)

    while queue or active:
        while queue and len(active) < conc:
            index, task = queue.pop()
            try:
                active.append(
                    (index, Episode(task, session_factory(task), max_turns=max_turns, timeout_s=timeout_s))
                )
            except Exception as exc:  # noqa: BLE001 - a crashed session is a wrong answer
                results[index] = RolloutResult(
                    task.task_id, None, False, 0, 0.0, "none",
                    f"rollout_crash: {type(exc).__name__}: {exc}",
                )
                if on_result is not None:
                    on_result(index, results[index])
        stepping: list[tuple[int, Episode]] = []
        for index, ep in active:
            if ep.begin_turn():
                stepping.append((index, ep))
            else:
                finalize(index, ep)
        if not stepping:
            active = []
            continue
        try:
            outputs = generate_many(
                backend, [ep.prompt() for _, ep in stepping], MAX_ACTION_TOKENS, list(STOP_SEQUENCES)
            )
        except Exception as exc:  # noqa: BLE001 - one oversized episode must not kill the sweep
            # A batch engine rejects the whole call when any one prompt is
            # over-long. Retry each episode alone so only the offender dies;
            # measured on SCI4 r3, the batch-level failure silently dropped
            # 593/900 episodes from the record.
            outputs = []
            for _, ep in stepping:
                try:
                    outputs.append(
                        generate_many(backend, [ep.prompt()], MAX_ACTION_TOKENS, list(STOP_SEQUENCES))[0]
                    )
                except Exception as inner:  # noqa: BLE001
                    ep.error = f"generate_failed: {type(inner).__name__}"
                    ep.done = True
                    outputs.append("")
            del exc
        survivors: list[tuple[int, Episode]] = []
        for (index, ep), raw in zip(stepping, outputs):
            try:
                if not ep.done:
                    ep.advance(raw)
            except Exception as exc:  # noqa: BLE001
                ep.error = f"rollout_crash: {type(exc).__name__}"
                ep.done = True
            if ep.done:
                finalize(index, ep)
            else:
                survivors.append((index, ep))
        active = survivors

    return results  # type: ignore[return-value]

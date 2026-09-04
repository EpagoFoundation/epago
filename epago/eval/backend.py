"""Model inference backends.

The rollout harness talks to a minimal generate/close protocol so validators
can run a real engine while tests and soaks run deterministic scripted
doubles. All backends share one contract detail the harness depends on: the
returned completion is truncated at the first stop sequence hit and the stop
sequence itself is **included**, so a well-formed action always arrives with
its closing tag and parsing stays a single regex.

Prompt delivery is the second shared contract: a real engine renders the
harness prompt through the *model's own* chat template before decoding (see
:func:`chat_prompt`). The harness builds one plain-text prompt; how that text
is framed for a particular checkpoint is the backend's business, because it is
the backend that owns the tokenizer.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar, Protocol

from epago import constants

logger = logging.getLogger(__name__)


class ModelBackend(Protocol):
    """Greedy, seeded text completion.

    Implementations pin everything they can: temperature 0, a fixed seed, fixed
    caps. That is a best-effort contract, not a guarantee — the production vLLM
    backend does NOT return the same completion for a repeated prompt (see
    VllmBackend below), because the fused MoE kernels reduce in a
    nondeterministic order. Scripted backends used by tests and soaks ARE exactly
    deterministic, which is what makes the suite's equivalence assertions
    (batched vs sequential, remote vs local, low-VRAM vs default) meaningful.
    Callers must not assume repeat-call identity from a real engine; the duel
    math measures the residual disagreement instead."""

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str: ...

    def close(self) -> None: ...


def generate_many(backend: "ModelBackend", prompts: list[str], max_tokens: int, stop: list[str]) -> list[str]:
    """Batch dispatch: use the backend's native batch call when it has one,
    else fall back to sequential single generations (scripted/test backends)."""
    native = getattr(backend, "generate_many", None)
    if callable(native):
        return native(prompts, max_tokens, stop)
    return [backend.generate(p, max_tokens, stop) for p in prompts]


def flatten_messages(messages: list[dict]) -> str:
    """A chat conversation as one plain-text document, for template-less
    checkpoints and for scripted test policies that match on prompt text."""
    return "\n\n".join(str(m.get("content", "")) for m in messages)


def chat_prompt(tokenizer, prompt: "str | list[dict]") -> dict | None:
    """Render one harness prompt through the checkpoint's own chat template.

    Returns a vLLM ``TokensPrompt``-shaped ``{"prompt_token_ids": [...]}``, or
    ``None`` when the checkpoint ships no chat template — the caller then feeds
    the raw prompt through unchanged.

    Why this exists: the harness prompt is instruction-shaped text, and a
    chat/RL-trained checkpoint has only ever seen instructions inside its
    template's turn markers. Handing it to ``engine.generate`` as a raw
    completion puts the model off-distribution: it continues the document
    instead of answering it, so protocol compliance collapses. Measured on the
    pinned reference model, the pinned corpus and the SCI3 release, 50 tasks
    per replicate, identical syntax/tasks/budget on both sides: 55% of episodes
    well-formed and 46% correct as a raw completion, against 78% / 74% through
    the template (5 and 7 replicates; per-task sign test over the replicate
    rates p = 5e-4 for well-formed, p = 7e-5 for accuracy).

    Model-agnostic by construction:

    * the template comes from the tokenizer in the pinned snapshot — no vendor
      string, role name, or special token is written down here;
    * the whole harness prompt travels as ONE ``user`` message with
      ``add_generation_prompt=True``. Nothing is assumed about whether the
      checkpoint supports a system role, tool roles, or multi-turn
      alternation, so a challenger is never penalized for a template that
      differs from the incumbent's;
    * no template at all (a plain base checkpoint) is a supported case, not an
      error: it falls back to the raw prompt so a base-model challenger is
      scored exactly as it was before this change.

    Determinism: ``chat_template.jinja`` / ``tokenizer_config.json`` live
    inside the model snapshot folder, and :func:`epago.model.store.snapshot_digest`
    hashes every file in that folder — so the template is covered by the pinned
    model digest, and two validators materializing the same digest render
    byte-identical prompts. The rendering itself is a pure function of
    (template, prompt): no wall clock, no RNG, no environment lookup. (Templates
    that call ``strftime_now`` would break that; none is used here because the
    harness passes no date variables and the reference template takes none.)

    Tokenizing here rather than handing the engine a rendered *string* also
    avoids a double-BOS: a template that emits its own BOS plus an engine that
    prepends one during tokenization would shift every prompt by a token.
    """
    if tokenizer is None:
        return None
    render = getattr(tokenizer, "apply_chat_template", None)
    if render is None or not getattr(tokenizer, "chat_template", None):
        return None
    # A plain string travels as ONE user message (the v3 contract); the v4
    # native harness passes a full messages list and it is rendered verbatim.
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    try:
        token_ids = _flatten_ids(
            render(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    except Exception as exc:  # noqa: BLE001 - a broken template must not abort a duel
        # Loud, because a silent fallback would score this checkpoint raw (and
        # therefore far below its real ability) with nothing in the record.
        logger.warning("chat template rendering failed (%s); using the raw prompt", exc)
        return None
    return {"prompt_token_ids": token_ids}


def _flatten_ids(rendered) -> list[int]:
    """Normalize what ``apply_chat_template(tokenize=True)`` returns to ids.

    Transformers has returned three shapes across versions for a single
    conversation — a flat ``list[int]``, a ``BatchEncoding``/dict keyed by
    ``input_ids`` (the 5.x default), and a batch-of-one nesting — and a
    validator does not get to pin one transformers version across every GPU
    box in the network. Anything else raises, which the caller turns into the
    raw-prompt fallback rather than a wrong prompt.
    """
    ids = rendered
    if hasattr(ids, "keys"):  # BatchEncoding / dict
        ids = ids["input_ids"]
    ids = list(ids)
    if ids and not isinstance(ids[0], int):  # batch-of-one nesting
        ids = list(ids[0])
    if not ids or not all(isinstance(i, int) for i in ids):
        raise TypeError(f"chat template produced non-token output: {type(rendered)!r}")
    return ids


def _apply_stops(text: str, stop: list[str]) -> str:
    """Truncate at the earliest stop sequence, keeping the stop sequence."""
    cut = len(text)
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx + len(s))
    return text[:cut]


class VllmBackend:
    """vLLM engine wrapper over a local snapshot directory.

    Greedy decoding (temperature 0.0) with the pinned rollout seed; sampling
    randomness must never enter a scored path. Engines are cached per resolved
    model directory so the king stays warm across duels — constructing a second
    backend for the same directory reuses the loaded engine; ``close`` evicts it.
    """

    _engines: ClassVar[dict[str, object]] = {}

    def __init__(self, model_dir: Path) -> None:
        try:
            import vllm
        except ImportError as exc:
            raise RuntimeError(
                "VllmBackend requires vllm; install it on GPU validators "
                "(pip install vllm) or use kind='scripted' for tests"
            ) from exc
        self._vllm = vllm
        self._key = str(Path(model_dir).resolve())
        engine = self._engines.get(self._key)
        if engine is None:
            import os

            kwargs = {}
            mem_util = os.environ.get("EPAGO_VLLM_GPU_MEM_UTIL")
            if mem_util:
                # 24 GB-class cards: cap each engine's share explicitly.
                kwargs["gpu_memory_utilization"] = float(mem_util)
            # Shard one model across several cards. Unset (the default) keeps
            # the shipped one-card-per-engine behaviour exactly.
            #
            # Why this is needed: the genesis king is pinned to the bf16 base
            # (~57 GB of safetensors), and the size gate admits any challenger
            # up to 1.05x the king's BYTES — so a legitimate duel can involve a
            # model far larger than one card. Without this the validator simply
            # cannot load its own genesis king on 32 GB-class hardware, and the
            # failure looks like an OOM rather than a missing capability.
            tp = os.environ.get("EPAGO_VLLM_TP")
            if tp:
                kwargs["tensor_parallel_size"] = int(tp)
            # Lowest-noise scoring: CUDA graphs and batch-variant kernels make
            # the same weights produce slightly different logits depending on
            # batch shape, and a multi-turn agent amplifies one flipped token
            # into a whole wrong trajectory. enforce_eager disables the graphs;
            # pairing it with concurrency=1 (batch of one) takes batch shape out
            # of the numerics.
            #
            # It lowers the noise, it does not remove it. Measured on the
            # reference stack (Qwen3-MoE 4-bit, vLLM 0.27, RTX 5090): with this
            # flag set, batch of one, greedy, seeded, and prefix caching
            # disabled, the same prompt still decodes differently on a repeat
            # call — the fused MoE kernels reduce in a nondeterministic order,
            # below anything this flag controls. Nothing downstream assumes
            # bit-reproducibility: the calibration duel measures whatever
            # disagreement remains and the adaptive floor prices it.
            deterministic = os.environ.get("EPAGO_VLLM_DETERMINISTIC", "").lower() in (
                "1", "true", "yes",
            )
            engine = vllm.LLM(
                model=self._key,
                seed=constants.ROLLOUT_SEED,
                max_model_len=constants.ROLLOUT_CONTEXT_TOKENS,
                enforce_eager=deterministic,
                **kwargs,
            )
            self._engines[self._key] = engine
        self._engine = engine
        # The tokenizer travels with the weights, so the chat template is part
        # of the digest-pinned snapshot (see :func:`chat_prompt`). Fetched once
        # per backend; ``None`` means "render nothing", never "crash a duel".
        try:
            self._tokenizer = engine.get_tokenizer()
        except Exception as exc:  # noqa: BLE001 - degrade to raw prompts, loudly
            logger.warning("could not read tokenizer for %s (%s)", self._key, exc)
            self._tokenizer = None

    def _engine_prompt(self, prompt: str):
        """The harness prompt as this checkpoint expects to receive it.

        Templated checkpoints get pre-tokenized chat-formatted ids; a base
        checkpoint with no template gets the raw string, unchanged. Used by
        BOTH :meth:`generate` and :meth:`generate_many` so a batched duel and a
        single probe rollout put identical bytes in front of the model.
        """
        rendered = chat_prompt(self._tokenizer, prompt)
        if rendered is not None:
            return rendered
        return prompt if isinstance(prompt, str) else flatten_messages(prompt)

    def _params(self, max_tokens: int, stop: list[str]):
        return self._vllm.SamplingParams(
            temperature=constants.ROLLOUT_TEMPERATURE,
            seed=constants.ROLLOUT_SEED,
            repetition_penalty=constants.ROLLOUT_REPETITION_PENALTY,
            max_tokens=max_tokens,
            stop=list(stop),
            include_stop_str_in_output=True,
        )

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        outputs = self._engine.generate(
            [self._engine_prompt(prompt)], self._params(max_tokens, stop), use_tqdm=False
        )
        return outputs[0].outputs[0].text

    def generate_many(self, prompts: list[str], max_tokens: int, stop: list[str]) -> list[str]:
        """One engine call for a whole batch step — this is where continuous
        batching turns an hours-long duel into minutes. Outputs come back in
        prompt order; each request is greedy with the pinned seed."""
        outputs = self._engine.generate(
            [self._engine_prompt(p) for p in prompts],
            self._params(max_tokens, stop),
            use_tqdm=False,
        )
        return [o.outputs[0].text for o in outputs]

    def close(self) -> None:
        self._engines.pop(self._key, None)
        self._engine = None
        self._tokenizer = None
        try:  # best-effort: give the next engine the freed VRAM immediately
            import gc

            gc.collect()
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - freeing is opportunistic
            pass

class ScriptedBackend:
    """Deterministic test double used by all tests and smoke/soak scripts.

    Constructed with either a pure ``policy(prompt) -> str`` function or a
    canned ``answers`` map from the task question to its answer text (the
    question is recovered from the harness's ``Question:`` line). Stop
    truncation matches the real engine so harness behavior is identical.
    """

    _QUESTION_RE = re.compile(r"^Question: (.*)$", re.MULTILINE)

    def __init__(
        self,
        policy: Callable[[str], str] | None = None,
        *,
        answers: Mapping[str, str] | None = None,
    ) -> None:
        if (policy is None) == (answers is None):
            raise ValueError("provide exactly one of policy or answers")
        if policy is not None:
            self._policy = policy
        else:
            canned = dict(answers or {})

            def _answer_policy(prompt: str) -> str:
                m = self._QUESTION_RE.search(prompt)
                question = m.group(1) if m else ""
                return f"<answer>{canned.get(question, 'unknown')}</answer>"

            self._policy = _answer_policy

    def generate(self, prompt: "str | list[dict]", max_tokens: int, stop: list[str]) -> str:
        text = prompt if isinstance(prompt, str) else flatten_messages(prompt)
        return _apply_stops(self._policy(text), stop)

    def close(self) -> None:
        return None


def backend_factory(model_dir: Path, kind: str = "vllm") -> ModelBackend:
    """Construct a backend for a local model snapshot directory."""
    if kind == "vllm":
        return VllmBackend(model_dir)
    if kind == "scripted":
        return ScriptedBackend(answers={})
    raise ValueError(f"unknown backend kind: {kind!r}")

"""Minimal example fine-tune skeleton for Epago model miners.

The Epago protocol is output-only: intake and the duel judge the checkpoint
you submit, never the recipe that produced it. SFT, RL, distillation, model
merging, weight surgery — every recipe is equally legal as long as the
resulting checkpoint passes intake (config lock, safetensors-only hygiene,
size cap, not a byte-copy of the king) and beats the king in the duel. This
file is therefore an honest *skeleton*, not a competitive recipe: it shows
the load -> (your training here) -> save-safetensors loop and nothing more.

Typical loop::

    python -m epago.miner.train_example ./challenger ./challenger-v2
    python neurons/miner.py preflight ./challenger-v2 ./king \\
        --repo you/EPAGO-DR-30B-yourname --hotkey <your-hotkey>

Heavy dependencies are import-guarded: this module imports cleanly without
torch/transformers and fails with an actionable message only when run.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_INSTALL_HINT = (
    "this example needs torch and transformers. Install the eval extra:\n"
    "    pip install 'epago[eval]'\n"
    "or the pieces directly:\n"
    "    pip install torch transformers\n"
    "A GPU with bf16 support is strongly recommended for a 4B model."
)


def _import_ml():
    """Late import of the ML stack with an actionable install hint."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def finetune(challenger_dir: Path, out_dir: Path) -> Path:
    """Load the prepared challenger, train (your code), write safetensors back.

    ``challenger_dir`` is the folder produced by ``neurons/miner.py prepare``
    — a byte-copy of the current king. ``out_dir`` receives the trained
    checkpoint in the canonical safetensors layout that intake requires.
    """
    torch, AutoModelForCausalLM, AutoTokenizer = _import_ml()

    challenger_dir = Path(challenger_dir)
    out_dir = Path(out_dir)
    model = AutoModelForCausalLM.from_pretrained(
        challenger_dir,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(challenger_dir)

    # ------------------------------------------------------------------
    # YOUR RECIPE PLUGS IN HERE. Two common starting points:
    #
    # SFT: build deep-research trajectories (search -> browse -> answer)
    #   against a local copy of the pinned corpus snapshot, tokenize them
    #   with `tokenizer`, and run a standard causal-LM training loop over
    #   `model` (e.g. torch optim + cross-entropy on assistant turns).
    #
    # RL: run rollouts through the same tool loop the validators use
    #   (epago.environment.services.ToolSession over SqliteCorpus), reward
    #   corpus-verified answers, and update with your policy-gradient
    #   method of choice.
    #
    # Constraints that DO matter (checked at intake, preflight them first):
    #   * config.json structural keys must stay locked to the king's,
    #   * safetensors only — no .py/.bin/.pt/.pkl files in the repo,
    #   * total safetensors bytes <= MAX_CHALLENGER_SIZE_RATIO x king,
    #   * the checkpoint must not be a byte-copy of the king.
    # ------------------------------------------------------------------

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    # Carry over auxiliary text assets (e.g. chat template) that
    # save_pretrained may not rewrite. Never copy code files.
    for name in ("generation_config.json", "chat_template.jinja"):
        src = challenger_dir / name
        if src.exists() and not (out_dir / name).exists():
            shutil.copy2(src, out_dir / name)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("challenger_dir", type=Path, help="prepared challenger folder (king copy)")
    parser.add_argument("out_dir", type=Path, help="where to write the trained checkpoint")
    args = parser.parse_args(argv)
    try:
        out = finetune(args.challenger_dir, args.out_dir)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1
    print(f"checkpoint written to {out}")
    print("next: run `python neurons/miner.py preflight` before submitting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

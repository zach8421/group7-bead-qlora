"""C3 — Validate the prompt-completion tokenization boundary against the
real Llama-3.1 tokenizer.

Why this exists
---------------
The smoke path runs against `hf-internal-testing/tiny-random-LlamaForCausalLM`
with a custom `<|role|>...<|end|>` chat template — it does NOT exercise the
Llama-3 chat template (`<|begin_of_text|>`, `<|start_header_id|>`,
`<|eot_id|>`). So the first time the production code hits the real tokenizer
is the first real GPU run.

The eval likelihood scorer in `eval_adapter.py:score_completion` depends on a
specific invariant:

    tokenize(prompt) is a strict prefix of tokenize(prompt + completion)

If a BPE merge crosses the prompt/completion boundary — e.g. the trailing
`\\n\\n` of the chat template's `add_generation_prompt=True` rendering merges
with the first char of `"biased"` — the prefix invariant is violated. The
scorer would silently index into the wrong logits, biasing predictions.

This script asserts the invariant against the real tokenizer for both labels
and for several representative BEAD-style statements. It also reports the
token counts for each completion (so the "biased" vs "non-biased" length
asymmetry called out in audit item I1 is visible).

Run
---
    # Locally or on Tillicum; needs HF_TOKEN for the gated Llama tokenizer.
    python scripts/check_chat_template.py

    # Override the model if you want to point at a non-gated mirror locally:
    python scripts/check_chat_template.py --model some-org/some-mirror

Exit 0 on all assertions passing; exit 1 on any failure.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path


REPO_SCRIPTS_DIR = Path(__file__).resolve().parent


def extract_constant(path: Path, name: str):
    """Pull a module-level assignment value out of a Python source file
    without importing the file (avoids loading torch/transformers/etc.)."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def build_prompt(tokenizer, system_instruction: str, statement: str) -> str:
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Statement: {statement}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_full_chat(tokenizer, system_instruction: str, statement: str, label: str) -> str:
    """Render messages including the assistant label — this is what the
    SFTTrainer sees at training time (modulo the trailer)."""
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Statement: {statement}"},
        {"role": "assistant", "content": label},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="Tokenizer to load. Defaults to the gated production model.")
    ap.add_argument("--show-tokens", action="store_true",
                    help="Print the decoded token strings at the prompt/completion boundary for inspection.")
    args = ap.parse_args()

    # 1. Pull SYSTEM_INSTRUCTION and LABEL_STRINGS straight from the production
    #    scripts so we validate the exact constants in use.
    train_path = REPO_SCRIPTS_DIR / "train_qlora.py"
    eval_path = REPO_SCRIPTS_DIR / "eval_adapter.py"
    sys_inst_train = extract_constant(train_path, "SYSTEM_INSTRUCTION")
    sys_inst_eval = extract_constant(eval_path, "SYSTEM_INSTRUCTION")
    label_strings = extract_constant(eval_path, "LABEL_STRINGS")

    if sys_inst_train != sys_inst_eval:
        print(
            f"[check] FAIL: SYSTEM_INSTRUCTION differs between train and eval scripts.\n"
            f"  train_qlora.py:    {sys_inst_train!r}\n"
            f"  eval_adapter.py:   {sys_inst_eval!r}",
            file=sys.stderr,
        )
        return 1
    print(f"[check] SYSTEM_INSTRUCTION matches between train and eval ({len(sys_inst_train)} chars)")
    print(f"[check] LABEL_STRINGS: {label_strings}")

    # 2. Load the real tokenizer. No model weights, no GPU.
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"[check] FAIL: cannot import transformers: {e}", file=sys.stderr)
        return 1

    if (
        args.model.startswith("meta-llama/")
        and not os.environ.get("HF_TOKEN")
        and not os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ):
        print(
            f"[check] WARNING: HF_TOKEN not set and --model={args.model} is gated. "
            "If the tokenizer is not cached, the download will 401.",
            file=sys.stderr,
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    except Exception as e:
        print(f"[check] FAIL: could not load tokenizer for {args.model}: {e}", file=sys.stderr)
        return 1

    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if not has_chat_template:
        print(
            f"[check] FAIL: tokenizer for {args.model} has no chat_template. "
            "This script must run against a tokenizer that ships a chat template "
            "(Llama-3.1-Instruct does).",
            file=sys.stderr,
        )
        return 1
    print(f"[check] Loaded tokenizer: {args.model}  vocab={tokenizer.vocab_size}  chat_template=present")

    # 3. Sample BEAD-style statements to test against. Mix lengths and
    #    punctuation so we catch boundary quirks.
    samples = [
        "Those people always cause problems.",
        "The library will close at 6pm on Sunday.",
        "I think people forget that the primary benefit in incorporating is so you can write off expenses.",
        'listening to "New Young Pony Club - Ice Cream"',
    ]

    failures: list[str] = []
    boundary_reports: list[str] = []

    for stmt in samples:
        prompt = build_prompt(tokenizer, sys_inst_train, stmt)
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        n_prompt = len(prompt_ids)

        for label in label_strings:
            # (a) Replicates eval_adapter.py:score_completion's string concat.
            full_text = prompt + label
            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

            if len(full_ids) < n_prompt:
                failures.append(
                    f"  stmt={stmt!r} label={label!r}: full ids shorter than prompt ids "
                    f"({len(full_ids)} < {n_prompt})"
                )
                continue
            if full_ids[:n_prompt] != prompt_ids:
                # Show the first mismatching position.
                mis = next(
                    (i for i in range(n_prompt) if full_ids[i] != prompt_ids[i]),
                    n_prompt,
                )
                failures.append(
                    f"  stmt={stmt!r} label={label!r}: prompt is NOT a prefix of prompt+completion.\n"
                    f"    first mismatch at position {mis}: prompt_id={prompt_ids[mis]}, full_id={full_ids[mis]}\n"
                    f"    prompt last 3 chars: {prompt[-10:]!r}; completion starts: {label[:10]!r}"
                )
                continue

            n_completion = len(full_ids) - n_prompt
            completion_ids = full_ids[n_prompt:]
            boundary_reports.append(
                f"  stmt[:40]={stmt[:40]!r:<44} label={label!r:<14} "
                f"prompt_tokens={n_prompt:>4d}  completion_tokens={n_completion}  "
                f"completion_ids={completion_ids}"
            )

            # (b) Sanity: the same completion text rendered inside a chat-template
            #     assistant message should tokenize the completion tokens identically
            #     as a substring (modulo the trailing <|eot_id|>). If this drifts,
            #     training and eval are seeing different completion tokens.
            full_chat = build_full_chat(tokenizer, sys_inst_train, stmt, label)
            full_chat_ids = tokenizer(full_chat, add_special_tokens=False).input_ids
            # The chat-template version should start with the same prompt prefix.
            if full_chat_ids[:n_prompt] != prompt_ids:
                failures.append(
                    f"  stmt={stmt!r} label={label!r}: chat-template render with assistant "
                    f"turn does not share the prompt prefix. Training and eval would see "
                    f"different prompt tokens."
                )
                continue
            # And it should contain the completion_ids as a prefix of the remainder.
            chat_after_prompt = full_chat_ids[n_prompt:]
            if chat_after_prompt[: len(completion_ids)] != completion_ids:
                failures.append(
                    f"  stmt={stmt!r} label={label!r}: chat-template completion tokens "
                    f"differ from eval-time string concat.\n"
                    f"    eval concat: {completion_ids}\n"
                    f"    chat tpl:    {chat_after_prompt[: len(completion_ids) + 2]}"
                )

    print("[check] Boundary token report:")
    for line in boundary_reports:
        print(line)

    # 4. Length asymmetry visibility — audit item I1.
    biased_tokens: dict[str, int] = {}
    for label in label_strings:
        # Render with a neutral sample; the per-label length depends only on the
        # completion's tokenization since prompts are identical across labels.
        prompt = build_prompt(tokenizer, sys_inst_train, samples[0])
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(prompt + label, add_special_tokens=False).input_ids
        biased_tokens[label] = len(full_ids) - len(prompt_ids)

    print(f"[check] Completion token counts (length asymmetry check, audit item I1):")
    for label, n in biased_tokens.items():
        print(f"    {label!r:<14} = {n} token(s)")
    if len(set(biased_tokens.values())) > 1:
        print(
            "[check] NOTE: label completions have different token counts; the raw "
            "sum-of-log-probs scorer in eval_adapter.py will favor the shorter one. "
            "This is expected per the audit; flagging here for visibility."
        )

    if args.show_tokens:
        # Dump the boundary as decoded strings for the first sample / first label.
        prompt = build_prompt(tokenizer, sys_inst_train, samples[0])
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        for label in label_strings:
            full_ids = tokenizer(prompt + label, add_special_tokens=False).input_ids
            print(f"[check] Decoded boundary tokens for label={label!r}:")
            tail_prompt = prompt_ids[-3:]
            completion = full_ids[len(prompt_ids):]
            for tid in tail_prompt:
                print(f"    prompt-tail    id={tid:<6d}  decoded={tokenizer.decode([tid])!r}")
            for tid in completion:
                print(f"    completion     id={tid:<6d}  decoded={tokenizer.decode([tid])!r}")

    if failures:
        print(f"[check] FAILED: {len(failures)} invariant violation(s):", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        print(
            "[check] Eval likelihood scoring is unsafe with this tokenizer/template "
            "combination. Investigate before launching the QLoRA sweep.",
            file=sys.stderr,
        )
        return 1

    print("[check] OK — all prefix and chat-template invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

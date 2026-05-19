"""Evaluate a trained QLoRA adapter on the held-out BEAD test split.

Approach
--------
For each test example we score the two completions ("biased" and
"non-biased") under the same instruction prompt, conditional on the input,
and pick the higher-likelihood label. This is fully deterministic, requires
no sampling temperature tuning, and matches how we trained the model
(prompt-completion SFT on those exact strings).

Scoring is batched: every test row produces two padded sequences
(prompt + "biased", prompt + "non-biased"), and `--eval-batch-size` of
those sequences are forwarded together. After the forward, each row's
logits are sliced down to just the completion positions before
log-softmax is applied, so peak memory is bounded by the model's logits
tensor instead of a `(B, T-1, V)` fp32 intermediate. With right-padding
+ attention mask, the per-token log-probs at real (non-pad) completion
positions are numerically equivalent to a batch-size-1 forward up to
bf16 reduction order. In practice the argmax over the two labels is
stable; if you need bit-identical predictions to a previous batch-1 run
you can set `--eval-batch-size 1`.

Outputs
-------
* outputs/.../predictions.jsonl — one row per test example with text, gold,
  predicted label, and per-class summed log-likelihood.
* outputs/.../eval_metrics.json  — accuracy, precision/recall/F1 for the
  positive class plus macro F1.

Smoke test
----------
  python scripts/eval_adapter.py \\
      --adapter-path outputs/tillicum_1k_calibration/adapter_smoke \\
      --test-jsonl data/processed_mock/test_held_out.jsonl \\
      --max-test-rows 8 --smoke-test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)


SYSTEM_INSTRUCTION = (
    "You are a careful annotator. Read the following statement and decide whether "
    "it is biased or non-biased. Respond with exactly one word: either 'biased' or 'non-biased'."
)
LABEL_STRINGS = ["non-biased", "biased"]  # index = label_int


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(tokenizer, statement: str) -> str:
    """Render the user prompt up to (but not including) the assistant's answer."""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Statement: {statement}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def prepare_scoring_items(tokenizer, rows: list[dict]) -> list[dict]:
    """Tokenize each row once into per-(row, label) scoring items.

    Returns one item per (row_idx, label) with `input_ids` (list[int]) and
    `prompt_len` (int, number of prompt tokens — the same across both labels
    for a given row). Tokenizing the prompt once per row instead of once per
    (row, label) drops a redundant pass; the full prompt+label string is
    still tokenized per item so BPE boundary effects between the prompt's
    final token and the label's leading token are preserved exactly as in
    the batch-size-1 path.
    """
    items: list[dict] = []
    for row_idx, row in enumerate(rows):
        prompt = build_prompt(tokenizer, row["text"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)
        for label_idx, label in enumerate(LABEL_STRINGS):
            full_ids = tokenizer(prompt + label, add_special_tokens=False).input_ids
            items.append({
                "row_idx": row_idx,
                "label_idx": label_idx,
                "input_ids": full_ids,
                "prompt_len": prompt_len,
            })
    return items


def score_items_batched(model, tokenizer, items: list[dict], device, batch_size: int) -> torch.Tensor:
    """Forward every (row, label) item through the model in batches; return a
    1-D tensor of summed completion log-probs (on CPU), aligned to `items`.

    Right-pads sequences in each batch with `tokenizer.pad_token_id`, builds
    the attention mask, runs one forward per batch under `inference_mode`,
    and for each batch element slices the model output down to just the
    completion-position rows before computing log-probs. Avoiding the full
    `(B, T-1, V)` log_softmax materialization keeps peak memory bounded by
    the model's own logits tensor (`(B, T, V)` in bf16) — without this,
    bs=64 at max_seq_length=512 OOMs trying to allocate a ~17 GB fp32
    intermediate on top of the bf16 cast.
    """
    pad_id = tokenizer.pad_token_id
    n = len(items)
    out_scores = torch.empty(n, dtype=torch.float32, device=device)

    for start in range(0, n, batch_size):
        chunk = items[start: start + batch_size]
        B = len(chunk)
        max_len = max(len(it["input_ids"]) for it in chunk)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for j, it in enumerate(chunk):
            ids = it["input_ids"]
            L = len(ids)
            input_ids[j, :L] = torch.tensor(ids, dtype=torch.long)
            attention_mask[j, :L] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, T, V)

        # Slice each row down to completion positions before computing log-probs.
        # Position t in logits predicts token t+1. The completion tokens live at
        # input_ids[prompt_len : seq_len], so we read logits at positions
        # [prompt_len-1 : seq_len-1] and target tokens from input_ids at
        # [prompt_len : seq_len]. n_comp per row is typically 1–3 tokens
        # ("biased" or "non-biased"), so each slice is a tiny (n_comp, V)
        # tensor — fp32 cast + log-softmax math is essentially free.
        for j, it in enumerate(chunk):
            prompt_len = it["prompt_len"]
            seq_len = len(it["input_ids"])
            comp_logits = logits[j, prompt_len - 1: seq_len - 1, :].float()    # (n_comp, V)
            comp_targets = input_ids[j, prompt_len: seq_len]                    # (n_comp,)
            comp_log_z = torch.logsumexp(comp_logits, dim=-1)                   # (n_comp,)
            comp_target_logits = comp_logits.gather(
                1, comp_targets.unsqueeze(-1)
            ).squeeze(-1)                                                       # (n_comp,)
            out_scores[start + j] = (comp_target_logits - comp_log_z).sum()

        # Progress: roughly each "row" is 2 items, so divide by 2 for a row count.
        rows_done = (start + B + 1) // 2
        if rows_done % 200 == 0 or start + B == n:
            print(f"[eval] ~{rows_done} rows scored ({start + B}/{n} items)")

    return out_scores.detach().to("cpu")


def load_for_eval(adapter_path: Path, base_model_name: str, smoke_test: bool):
    """Load base model + adapter, with or without 4-bit."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = pick_device()
    t0 = time.perf_counter()

    if smoke_test or not torch.cuda.is_available():
        # CPU/MPS path: full-precision tiny base model. Used in smoke tests.
        print(f"[eval] Loading base model {base_model_name} (no quant) on {device}")
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model.to(device)
    else:
        from transformers import BitsAndBytesConfig
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        print(f"[eval] Loading base model {base_model_name} in 4-bit")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=compute_dtype,
        )

    # Tokenizer can come from the adapter dir (we saved it during training).
    tok_source = adapter_path if (adapter_path / "tokenizer_config.json").exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        # Smoke test fallback (matches train_qlora.py's smoke template).
        tokenizer.chat_template = (
            "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n<|end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
        )

    print(f"[eval] Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    print(f"[eval] Model + adapter ready in {time.perf_counter() - t0:.1f}s")
    return model, tokenizer, device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--test-jsonl", default="data/frozen/beads/sizes/full/test.jsonl")
    ap.add_argument("--eval-dataset", default=None,
                    help="Name of the dataset the test_jsonl was drawn from (e.g. "
                         "beads/babe/cajcodes/wnc). Stamped into eval_metrics.json. "
                         "If omitted, inferred from the test_jsonl path "
                         "(.../frozen/<dataset>/...).")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--smoke-base-model", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    ap.add_argument("--max-test-rows", type=int, default=0,
                    help="If >0, evaluate only the first N rows. Useful for quick checks.")
    ap.add_argument("--eval-batch-size", type=int, default=16,
                    help="Sequences per forward pass. Each test row produces 2 sequences "
                         "(one per label), so the actual minibatch is this value. Default 16.")
    ap.add_argument("--output-dir", default=None,
                    help="Defaults to the adapter's parent directory")
    ap.add_argument("--run-name", default=None,
                    help="Optional short label; stamped into eval_metrics.json so the manifest can pick it up.")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        sys.exit(f"[eval] Adapter not found at {adapter_path}")
    test_path = Path(args.test_jsonl)
    if not test_path.exists():
        sys.exit(f"[eval] Test JSONL not found at {test_path}")

    eval_dataset = args.eval_dataset
    if not eval_dataset:
        parts = test_path.resolve().parts
        if "frozen" in parts:
            i = parts.index("frozen")
            if i + 1 < len(parts):
                eval_dataset = parts[i + 1]
        if not eval_dataset:
            eval_dataset = "unknown"
    print(f"[eval] eval_dataset = {eval_dataset}")

    rows = load_jsonl(test_path)
    if args.max_test_rows > 0:
        rows = rows[: args.max_test_rows]
    print(f"[eval] Evaluating on {len(rows)} test rows  (eval_batch_size={args.eval_batch_size})")

    base_model_name = args.smoke_base_model if args.smoke_test else args.base_model
    model, tokenizer, device = load_for_eval(adapter_path, base_model_name, args.smoke_test)

    output_dir = Path(args.output_dir) if args.output_dir else adapter_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / ("predictions_smoke.jsonl" if args.smoke_test else "predictions.jsonl")
    metrics_path = output_dir / ("eval_metrics_smoke.json" if args.smoke_test else "eval_metrics.json")

    t0 = time.perf_counter()
    print(f"[eval] Tokenizing {len(rows)} rows × {len(LABEL_STRINGS)} labels ...")
    items = prepare_scoring_items(tokenizer, rows)
    print(f"[eval] Tokenization done in {time.perf_counter() - t0:.1f}s; running {len(items)} forwards "
          f"across {(len(items) + args.eval_batch_size - 1) // args.eval_batch_size} batches")

    t_fwd = time.perf_counter()
    scores_flat = score_items_batched(model, tokenizer, items, device, args.eval_batch_size)
    fwd_sec = time.perf_counter() - t_fwd
    print(f"[eval] Forward passes done in {fwd_sec:.1f}s "
          f"({len(items) / max(fwd_sec, 1e-9):.1f} seq/s, "
          f"{len(rows) / max(fwd_sec, 1e-9):.1f} rows/s)")

    # Reshape: items are in (row, label) order, two per row.
    scores_per_row = scores_flat.view(len(rows), len(LABEL_STRINGS)).tolist()
    label_to_idx = {lab: i for i, lab in enumerate(LABEL_STRINGS)}
    non_biased_idx = label_to_idx["non-biased"]
    biased_idx = label_to_idx["biased"]

    gold, pred = [], []
    with preds_path.open("w") as fout:
        for i, r in enumerate(rows):
            s_nb = scores_per_row[i][non_biased_idx]
            s_b = scores_per_row[i][biased_idx]
            pred_int = 1 if s_b >= s_nb else 0
            pred_label = LABEL_STRINGS[pred_int]
            gold_int = int(r["label_int"])
            gold.append(gold_int)
            pred.append(pred_int)
            fout.write(json.dumps({
                "text": r["text"],
                "gold_int": gold_int,
                "gold_str": r["label_str"],
                "pred_int": pred_int,
                "pred_str": pred_label,
                "score_non_biased": s_nb,
                "score_biased": s_b,
            }) + "\n")
    eval_sec = time.perf_counter() - t0

    metrics = {
        "run_name": args.run_name,
        "eval_dataset": eval_dataset,
        "n_examples": len(rows),
        "accuracy": float(accuracy_score(gold, pred)),
        "precision_pos": float(precision_score(gold, pred, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(gold, pred, pos_label=1, zero_division=0)),
        "f1_pos": float(f1_score(gold, pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            gold, pred, target_names=LABEL_STRINGS, zero_division=0, digits=4
        ),
        "eval_seconds": round(eval_sec, 3),
        "eval_forward_seconds": round(fwd_sec, 3),
        "eval_batch_size": args.eval_batch_size,
        "smoke_test": args.smoke_test,
        "adapter_path": str(adapter_path),
        "test_jsonl": str(test_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] Wrote {metrics_path}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()

"""1k-example QLoRA calibration training on Llama-3.1-8B-Instruct.

What this does
--------------
* Loads meta-llama/Llama-3.1-8B-Instruct in 4-bit (bitsandbytes nf4).
* Wraps it with LoRA adapters via PEFT — *no* classification head.
* Treats binary bias classification as prompt-completion SFT: input = an
  instruction prompt around the BEAD sentence, completion = "biased" /
  "non-biased". Loss is next-token cross-entropy on the completion only.
* Logs wall-clock, peak CUDA memory, batch-size accounting, and writes a
  metrics JSON + adapter checkpoint to ./outputs/tillicum_1k_calibration/.

Smoke test (no GPU, no full Llama)
----------------------------------
  python scripts/train_qlora_1k.py --smoke-test --train-jsonl data/processed_mock/train_1k.jsonl

Smoke test uses hf-internal-testing/tiny-random-LlamaForCausalLM (a few MB,
ungated) and disables 4-bit. It validates the data formatting + training loop
end-to-end without touching the real Llama weights.

Tillicum production run
-----------------------
  python scripts/train_qlora_1k.py \
      --train-jsonl data/processed/train_1k.jsonl \
      --output-dir outputs/tillicum_1k_calibration

Notes
-----
* HF_TOKEN must be set in the environment for gated meta-llama/Llama-3.1-8B-Instruct.
* All paths are relative to the project root (current working directory).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


SYSTEM_INSTRUCTION = (
    "You are a careful annotator. Read the following statement and decide whether "
    "it is biased or non-biased. Respond with exactly one word: either 'biased' or 'non-biased'."
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_chat_examples(rows: list[dict]) -> list[dict]:
    """Convert raw {text,label_int,label_str} rows to messages format for SFTTrainer."""
    examples = []
    for r in rows:
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": f"Statement: {r['text']}"},
                {"role": "assistant", "content": r["label_str"]},
            ]
        })
    return examples


def build_smoke_model_and_tokenizer(model_name: str):
    """Download a tiny Llama-shaped model + tokenizer for smoke testing.

    hf-internal-testing/tiny-random-LlamaForCausalLM is ~5 MB and ungated.
    Once cached, smoke tests run offline.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Tiny random model has no chat template; install a minimal one so
    # tokenizer.apply_chat_template works the same way as in production.
    if not getattr(tok, "chat_template", None):
        tok.chat_template = (
            "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n<|end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
        )
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tok


def build_qlora_model_and_tokenizer(model_name: str, compute_dtype: torch.dtype):
    """Load Llama-3.1-8B-Instruct in 4-bit and prep for k-bit LoRA training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model, tok


def attach_lora(model, smoke: bool):
    from peft import LoraConfig, get_peft_model
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    if smoke:
        # Tiny test model may not have all of these; PEFT will pick the ones that exist.
        lora_cfg.target_modules = "all-linear"
    model = get_peft_model(model, lora_cfg)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-jsonl", default="data/processed/train_1k.jsonl")
    ap.add_argument("--output-dir", default="outputs/tillicum_1k_calibration")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--smoke-model-name", default="hf-internal-testing/tiny-random-LlamaForCausalLM",
                    help="Tiny Llama-arch model used when --smoke-test is set")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--num-epochs", type=float, default=1.0)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=0,
                    help="0 = save only at end. Set >0 to save intermediate checkpoints.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true",
                    help="Use tiny Llama-shaped model, no quant, a couple of steps. CPU/MPS friendly.")
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="If >0, cap the training set to this many rows (additional safety knob)")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    device = pick_device()
    print(f"[train] Device: {device}")
    print(f"[train] Smoke test: {args.smoke_test}")

    # Defer heavy imports until after we've parsed args, so --help is fast and
    # smoke test failures are surfaced clearly.
    from datasets import Dataset
    from transformers import TrainingArguments
    try:
        from trl import SFTTrainer, SFTConfig
        HAS_SFTCONFIG = True
    except ImportError:
        from trl import SFTTrainer
        SFTConfig = None
        HAS_SFTCONFIG = False

    train_path = Path(args.train_jsonl)
    if not train_path.exists():
        sys.exit(f"[train] {train_path} not found. Run prepare_bead_splits.py first.")
    rows = load_jsonl(train_path)
    if args.max_train_rows > 0:
        rows = rows[: args.max_train_rows]
    print(f"[train] Loaded {len(rows)} training rows from {train_path}")
    examples = build_chat_examples(rows)
    train_ds = Dataset.from_list(examples)

    if args.smoke_test:
        compute_dtype = torch.float32
        model_name_used = args.smoke_model_name
        print(f"[train] Loading smoke model: {model_name_used}")
        model, tokenizer = build_smoke_model_and_tokenizer(model_name_used)
        model = attach_lora(model, smoke=True)
        if device.type != "cuda":
            model.to(device)
    else:
        if not torch.cuda.is_available():
            sys.exit(
                "[train] No CUDA device available. Real QLoRA training requires CUDA + bitsandbytes. "
                "Use --smoke-test for local CPU validation."
            )
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_name_used = args.model_name
        print(f"[train] Loading {model_name_used} in 4-bit (compute_dtype={compute_dtype})")
        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            print("[train] WARNING: HF_TOKEN not set. Gated Llama download will fail.", file=sys.stderr)
        model, tokenizer = build_qlora_model_and_tokenizer(model_name_used, compute_dtype)
        model = attach_lora(model, smoke=False)

    n_trainable, n_total = 0, 0
    for p in model.parameters():
        n_total += p.numel()
        if p.requires_grad:
            n_trainable += p.numel()
    print(f"[train] Trainable params: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / max(1, n_total):.4f}%)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common_args = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs if not args.smoke_test else 1,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="no" if args.save_steps == 0 else "steps",
        save_steps=args.save_steps if args.save_steps > 0 else 1,
        report_to="none",
        seed=args.seed,
        bf16=(not args.smoke_test) and torch.cuda.is_bf16_supported(),
        fp16=(not args.smoke_test) and torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=not args.smoke_test,
        max_steps=4 if args.smoke_test else -1,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    # SFTTrainer signature changed across TRL versions. trl>=0.10 expects an
    # SFTConfig that bundles training-args + SFT-specific knobs. Older TRL
    # accepts kwargs directly on the trainer.
    if HAS_SFTCONFIG:
        # TRL renamed max_seq_length -> max_length somewhere in the 1.x line.
        # Pass whichever the installed SFTConfig accepts.
        import inspect
        sft_sig = set(inspect.signature(SFTConfig.__init__).parameters.keys())
        sft_extra = {}
        if "max_length" in sft_sig:
            sft_extra["max_length"] = args.max_seq_length
        elif "max_seq_length" in sft_sig:
            sft_extra["max_seq_length"] = args.max_seq_length
        if "packing" in sft_sig:
            sft_extra["packing"] = False
        if "completion_only_loss" in sft_sig:
            sft_extra["completion_only_loss"] = True
        sft_cfg = SFTConfig(**common_args, **sft_extra)
        # Also handle the tokenizer-vs-processing_class rename in SFTTrainer.
        import inspect as _inspect
        sft_trainer_sig = set(_inspect.signature(SFTTrainer.__init__).parameters.keys())
        trainer_kwargs = dict(model=model, args=sft_cfg, train_dataset=train_ds)
        if "processing_class" in sft_trainer_sig:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer
    else:
        targs = TrainingArguments(**common_args)
        trainer_kwargs = dict(
            model=model,
            args=targs,
            train_dataset=train_ds,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            packing=False,
        )
    trainer = SFTTrainer(**trainer_kwargs)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    effective_bs = args.per_device_batch_size * args.grad_accum
    print(f"[train] examples={len(rows)}  per_device_bs={args.per_device_batch_size}  "
          f"grad_accum={args.grad_accum}  effective_bs={effective_bs}  "
          f"max_seq_length={args.max_seq_length}  epochs={args.num_epochs}")

    t0 = time.perf_counter()
    train_result = trainer.train()
    wall_clock_sec = time.perf_counter() - t0

    peak_mem_gb = None
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    if args.smoke_test:
        adapter_path = output_dir / "adapter_smoke"
    else:
        adapter_path = output_dir / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    metrics = {
        "smoke_test": args.smoke_test,
        "model_name": model_name_used,
        "device": str(device),
        "compute_dtype": str(compute_dtype),
        "examples": len(rows),
        "per_device_batch_size": args.per_device_batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": effective_bs,
        "max_seq_length": args.max_seq_length,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "wall_clock_sec": round(wall_clock_sec, 3),
        "wall_clock_min": round(wall_clock_sec / 60.0, 3),
        "peak_cuda_memory_gb": round(peak_mem_gb, 3) if peak_mem_gb is not None else None,
        "train_loss_final": float(train_result.training_loss) if train_result.training_loss is not None else None,
        "global_steps": int(train_result.global_step),
        "throughput_examples_per_sec": round(len(rows) * args.num_epochs / max(wall_clock_sec, 1e-9), 3),
        "adapter_path": str(adapter_path),
        "seed": args.seed,
    }
    metrics_path = output_dir / ("calibration_metrics_smoke.json" if args.smoke_test else "calibration_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[train] Wrote {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

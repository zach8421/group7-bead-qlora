"""Reusable QLoRA fine-tune for the BEAD sweep (Llama-3.1-8B-Instruct).

What this does
--------------
* Loads meta-llama/Llama-3.1-8B-Instruct in 4-bit (bitsandbytes nf4).
* Attaches LoRA adapters via PEFT — no classification head.
* Prompt-completion SFT: input = instruction prompt around the BEAD sentence,
  completion = "biased" / "non-biased". Loss is next-token CE on the completion
  only (completion_only_loss=True).
* Writes adapter + `train_metrics.json` + `run_meta.json` to --output-dir.

`run_meta.json` snapshots everything that influences reproducibility: argparse
values, key library versions, hostname / slurm job id, start/end timestamps,
git head if available, input JSONL sha256, model name, output paths. The
sweep manifest (scripts/update_manifest.py) joins on this file.

Smoke test (no GPU, no full Llama)
----------------------------------
  python scripts/train_qlora.py --smoke-test \\
    --train-jsonl data/frozen/train_100.jsonl \\
    --output-dir outputs/_smoke_qlora --run-name _smoke

Tillicum sweep example
----------------------
  python scripts/train_qlora.py \\
    --train-jsonl data/frozen/train_5k.jsonl \\
    --num-epochs 3 --run-name qlora_5k \\
    --output-dir outputs/qlora_5k
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
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


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def best_effort_git_head(cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, timeout=5
        )
        return out.decode().strip()
    except Exception:
        return None


def lib_versions() -> dict:
    """Best-effort snapshot of the libraries that govern training behavior."""
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "peft", "trl", "bitsandbytes", "datasets", "accelerate"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[mod] = None
    if torch.cuda.is_available():
        try:
            out["cuda"] = torch.version.cuda
            out["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return out


def build_chat_examples(rows: list[dict]) -> list[dict]:
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not getattr(tok, "chat_template", None):
        tok.chat_template = (
            "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n<|end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
        )
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tok


def build_qlora_model_and_tokenizer(model_name: str, compute_dtype: torch.dtype):
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

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if smoke:
        lora_cfg.target_modules = "all-linear"
    return get_peft_model(model, lora_cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--output-dir", required=True,
                    help="Directory the adapter + metrics will be written to.")
    ap.add_argument("--run-name", required=True,
                    help="Short label for this run (e.g. qlora_5k). Written into run_meta.")
    ap.add_argument("--splits-manifest", default="data/frozen/splits_manifest.json",
                    help="Path to splits_manifest.json so the run records which frozen splits it used.")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--smoke-model-name", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--num-epochs", type=float, default=3.0,
                    help="Default 3 for the v2 sweep. Override per run as needed.")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=0,
                    help="0 = save only at end. Set >0 for intermediate checkpoints.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="If >0, cap training rows (safety knob).")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    device = pick_device()
    print(f"[train] Device: {device}  smoke: {args.smoke_test}  run: {args.run_name}")

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
        sys.exit(f"[train] {train_path} not found. Run freeze_splits.py first.")
    train_sha = sha256_of_file(train_path)
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
                "[train] No CUDA device. Real QLoRA training requires CUDA + bitsandbytes. "
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
    print(
        f"[train] Trainable params: {n_trainable:,} / {n_total:,} "
        f"({100 * n_trainable / max(1, n_total):.4f}%)"
    )

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

    if HAS_SFTCONFIG:
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
        sft_trainer_sig = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
        trainer_kwargs = dict(model=model, args=sft_cfg, train_dataset=train_ds)
        if "processing_class" in sft_trainer_sig:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer
    else:
        targs = TrainingArguments(**common_args)
        trainer_kwargs = dict(
            model=model, args=targs, train_dataset=train_ds, tokenizer=tokenizer,
            max_seq_length=args.max_seq_length, packing=False,
        )
    trainer = SFTTrainer(**trainer_kwargs)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    effective_bs = args.per_device_batch_size * args.grad_accum
    print(
        f"[train] examples={len(rows)}  per_device_bs={args.per_device_batch_size}  "
        f"grad_accum={args.grad_accum}  effective_bs={effective_bs}  "
        f"max_seq_length={args.max_seq_length}  epochs={args.num_epochs}"
    )

    started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    t0 = time.perf_counter()
    train_result = trainer.train()
    wall_clock_sec = time.perf_counter() - t0
    finished_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

    peak_mem_gb = None
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    adapter_path = output_dir / ("adapter_smoke" if args.smoke_test else "adapter")
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    train_metrics = {
        "run_name": args.run_name,
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
    metrics_path = output_dir / ("train_metrics_smoke.json" if args.smoke_test else "train_metrics.json")
    metrics_path.write_text(json.dumps(train_metrics, indent=2))

    splits_manifest_sha = None
    splits_manifest_path = Path(args.splits_manifest)
    if splits_manifest_path.exists():
        splits_manifest_sha = sha256_of_file(splits_manifest_path)

    run_meta = {
        "run_name": args.run_name,
        "smoke_test": args.smoke_test,
        "args": vars(args),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_clock_sec": round(wall_clock_sec, 3),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "libs": lib_versions(),
        "git_head": best_effort_git_head(Path.cwd()),
        "inputs": {
            "train_jsonl": str(train_path),
            "train_jsonl_sha256": train_sha,
            "splits_manifest": str(splits_manifest_path) if splits_manifest_path.exists() else None,
            "splits_manifest_sha256": splits_manifest_sha,
        },
        "outputs": {
            "adapter_path": str(adapter_path),
            "train_metrics": str(metrics_path),
        },
        "model": {
            "model_name": model_name_used,
            "trainable_params": n_trainable,
            "total_params": n_total,
        },
    }
    run_meta_path = output_dir / ("run_meta_smoke.json" if args.smoke_test else "run_meta.json")
    run_meta_path.write_text(json.dumps(run_meta, indent=2))

    print(f"[train] Wrote {metrics_path}")
    print(f"[train] Wrote {run_meta_path}")
    print(json.dumps(train_metrics, indent=2))


if __name__ == "__main__":
    main()

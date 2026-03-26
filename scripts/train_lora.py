from __future__ import annotations

import argparse
import json
from pathlib import Path

from ticket_agent.dataset import sft_example_to_training_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an experimental LoRA adapter for the ticket agent.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the JSONL SFT dataset.")
    parser.add_argument("--model-id", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--no-4bit", action="store_true", help="Disable QLoRA and load the base model in standard precision.")
    return parser


def load_training_dataset(dataset_path: Path):
    from datasets import Dataset

    records: list[dict[str, str]] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                example = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_number} of {dataset_path}: {exc}") from exc
            records.append({"text": sft_example_to_training_text(example)})

    if not records:
        raise SystemExit(f"No training examples found in {dataset_path}")

    return Dataset.from_list(records)


def main() -> int:
    args = build_parser().parse_args()

    try:
        import torch
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - runtime guidance
        raise SystemExit(
            "Missing training dependencies. Install them with `pip install -e .[train]` "
            "and install a CUDA-compatible PyTorch build before rerunning."
        ) from exc

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    if not torch.cuda.is_available() and not args.no_4bit:
        raise SystemExit(
            "No CUDA GPU detected. Local 8B QLoRA training is not practical on CPU. "
            "Use a CUDA-enabled PyTorch install or rerun with `--no-4bit` only if you intentionally want CPU training."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=False)
    except OSError as exc:
        message = str(exc).lower()
        if "gated repo" in message or "access to model" in message:
            raise SystemExit(
                "The selected model is gated on Hugging Face. Fastest fix: rerun with "
                "`--model-id microsoft/Phi-3-mini-4k-instruct`. If you must stay on Llama, "
                "request access to the repo in your browser and then run `.venv\\Scripts\\hf.exe auth login`."
            ) from exc
        raise
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch_dtype,
            device_map="auto",
            use_cache=False,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            attn_implementation="eager",
            trust_remote_code=False,
        )
    except OSError as exc:
        message = str(exc).lower()
        if "gated repo" in message or "access to model" in message:
            raise SystemExit(
                "The selected model is gated on Hugging Face. Fastest fix: rerun with "
                "`--model-id microsoft/Phi-3-mini-4k-instruct`. If you must stay on Llama, "
                "request access to the repo in your browser and then run `.venv\\Scripts\\hf.exe auth login`."
            ) from exc
        raise
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    dataset = load_training_dataset(args.dataset)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
        packing=args.packing,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    (args.output_dir / "training_config.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "model_id": args.model_id,
                "use_4bit": not args.no_4bit,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "max_seq_length": args.max_seq_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "packing": args.packing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved LoRA adapter to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

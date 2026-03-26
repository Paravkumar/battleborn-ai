from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge an experimental LoRA adapter into a base model folder.")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def normalize_tied_weight_keys(model) -> None:
    for module in model.modules():
        tied = getattr(module, "_tied_weights_keys", None)
        if isinstance(tied, (list, tuple, set)):
            module._tied_weights_keys = {str(key): str(key) for key in tied}


def main() -> int:
    args = build_parser().parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime guidance
        raise SystemExit(
            "Missing merge dependencies. Install them with `.venv\\Scripts\\python.exe -m pip install \".[train]\"` before rerunning."
        ) from exc

    if not args.adapter_dir.exists():
        raise SystemExit(f"Adapter directory not found: {args.adapter_dir}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    merged = PeftModel.from_pretrained(base_model, args.adapter_dir).merge_and_unload()
    normalize_tied_weight_keys(merged)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

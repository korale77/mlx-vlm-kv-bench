"""
TurboQuant / TriAttention KV-cache benchmark for mlx-vlm on Apple silicon.

Compares KV-cache configurations on the same (long text + image) prompt at
multiple prompt-length tiers:

    BL     - no KV quantization, no pruning (baseline)
    UNI    - uniform 4-bit                  (mlx_lm built-in)
    TBQ    - TurboQuant 3.5-bit             (kv-cache quantization)
    TA512  - TriAttention budget=512        (kv-cache token pruning, PR #985)
    TA2048 - TriAttention budget=2048       (kv-cache token pruning, PR #985)
    TA4096 - TriAttention budget=4096       (kv-cache token pruning, PR #985)
    TA8192 - TriAttention budget=8192       (kv-cache token pruning, PR #985)

TurboQuant and TriAttention are orthogonal axes: TBQ compresses bits-per-KV,
TA prunes tokens-per-KV. They can be combined (not exercised here).

For each (tier x config) we capture prompt_tps, generation_tps, peak_memory
and a snippet of the generated text. Results are printed as a Markdown table
and optionally written to a file.

Run:
    .venv/bin/python bench_turboquant.py

    .venv/bin/python bench_turboquant.py \\
        --model mlx-community/Qwen3.5-9B-8bit \\
        --tiers 8000 24000 \\
        --max-tokens 128 \\
        --output results.md

    # TriAttention (PR #985) — requires one-time calibration first:
    .venv/bin/python -m mlx_vlm.triattention_calibrate \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --output gemma4_26b_calib.safetensors
    .venv/bin/python bench_turboquant.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --triattention-calib gemma4_26b_calib.safetensors \\
        --tiers 8000 24000 48000 60000 \\
        --configs BL TBQ TA512 TA2048 \\
        --output results_gemma4_triattention.md
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.models import cache as cache_mod
from mlx_vlm.prompt_utils import apply_chat_template

DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-4bit"
DEFAULT_IMAGE = "cats.jpg"  # any image; VLM requires one but content doesn't matter
DEFAULT_SEED = Path(__file__).parent / "long_prompt.txt"

QUESTION = (
    "\n\n---\nBased only on the Northwind Station Engineering Handbook above, "
    "answer in one or two sentences: What is the primary power source, what is "
    "its nominal output, and how long can the zinc-bromide flow batteries "
    "sustain hotel load at 40 percent duty cycle?"
)

CONFIGS: list[tuple[str, dict]] = [
    ("BL", {}),
    ("UNI", {"kv_bits": 4, "kv_quant_scheme": "uniform"}),
    ("TBQ", {"kv_bits": 3.5, "kv_quant_scheme": "turboquant"}),
    # TA configs require --triattention-calib; calib path is injected in main().
    ("TA512", {"triattention_budget": 512}),
    ("TA2048", {"triattention_budget": 2048}),
    ("TA4096", {"triattention_budget": 4096}),
    ("TA8192", {"triattention_budget": 8192}),
    ("TA16384", {"triattention_budget": 16384}),
    ("TA32768", {"triattention_budget": 32768}),
]


@dataclass
class RunResult:
    tier: int
    config: str
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    kv_bytes_mb: float  # isolated KV-cache bytes (sum of .nbytes across layers)
    wall_seconds: float
    text: str


def sum_cache_nbytes(prompt_cache) -> int:
    """Sum .nbytes across every entry in a prompt_cache list.

    Each entry is a cache object (KVCache, TurboQuantKVCache, ArraysCache,
    CacheList, ...) — all subclass _BaseCache and implement the .nbytes
    property. CacheList and ArraysCache already recurse internally, so a
    single top-level sum is correct.
    """
    total = 0
    for entry in prompt_cache:
        total += int(entry.nbytes)
    return total


def reset_peak_memory() -> None:
    """Reset peak-memory tracking across mlx versions that differ on the name."""
    reset = getattr(mx, "reset_peak_memory", None)
    if reset is None:
        reset = getattr(mx.metal, "reset_peak_memory", None)
    if reset is not None:
        reset()


def tile_seed(seed_text: str, tokenizer, target_tokens: int) -> str:
    """Repeat seed_text until it reaches at least target_tokens when tokenized."""
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    seed_len = max(1, len(seed_ids))
    copies = max(1, (target_tokens + seed_len - 1) // seed_len)
    return (seed_text + "\n\n") * copies


def build_prompt(
    processor,
    model_config,
    seed_text: str,
    target_tokens: int,
    image_count: int,
) -> tuple[str, int]:
    """Tile seed to target tokens, append the question, apply chat template."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    body = tile_seed(seed_text, tokenizer, target_tokens)
    raw = body + QUESTION
    prompt = apply_chat_template(
        processor,
        model_config,
        raw,
        num_images=image_count,
    )
    # Count tokens in the fully templated prompt (includes <|vision_start|> etc)
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    return prompt, len(encoded)


def run_one(
    model,
    processor,
    prompt: str,
    image: str,
    config_kwargs: dict,
    max_tokens: int,
    quantized_kv_start: int,
) -> tuple[object, float, int]:
    reset_peak_memory()
    # Create an externally-visible prompt cache so we can read .nbytes afterwards.
    # Must pass model.language_model (not the full VLM wrapper) to match what
    # generate_step does internally at mlx_vlm/generate.py:468-471.
    prompt_cache = cache_mod.make_prompt_cache(model.language_model)
    t0 = time.perf_counter()
    result = generate(
        model,
        processor,
        prompt,
        image=[image],
        max_tokens=max_tokens,
        quantized_kv_start=quantized_kv_start,
        temperature=0.0,
        verbose=False,
        prompt_cache=prompt_cache,
        **config_kwargs,
    )
    elapsed = time.perf_counter() - t0
    kv_bytes = sum_cache_nbytes(prompt_cache)
    return result, elapsed, kv_bytes


def fmt_row(r: RunResult) -> str:
    snippet = r.text.strip().replace("|", "\\|").replace("\n", " ")
    return (
        f"| {r.tier:>6d} | {r.config:<6} | {r.prompt_tokens:>7d} | "
        f"{r.prompt_tps:>10.2f} | {r.generation_tps:>10.2f} | "
        f"{r.peak_memory_gb:>8.2f} | {r.kv_bytes_mb:>9.1f} | "
        f"{r.wall_seconds:>6.1f} | {snippet} |"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--seed-file", default=str(DEFAULT_SEED))
    p.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=[8000, 24000],
        help="Target prompt-token counts (before chat template) to benchmark",
    )
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument(
        "--quantized-kv-start",
        type=int,
        default=0,
        help="Force KV quantization from token 0 (default) to expose TBQ savings",
    )
    p.add_argument("--output", default="results.md")
    p.add_argument(
        "--configs",
        nargs="+",
        default=["BL", "UNI", "TBQ"],
        help="Subset of configs to run "
        "(BL, UNI, TBQ, TBQ4, TA512, TA2048, TA4096, TA8192, TA16384, TA32768)",
    )
    p.add_argument(
        "--triattention-calib",
        default=None,
        help="Path to TriAttention calibration .safetensors "
        "(required for TA* configs)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    seed_path = Path(args.seed_file)
    if not seed_path.exists():
        print(f"seed file not found: {seed_path}", file=sys.stderr)
        return 1
    seed_text = seed_path.read_text()

    configs = [(n, k) for n, k in CONFIGS if n in args.configs]
    if not configs:
        print(f"no configs selected from {args.configs}", file=sys.stderr)
        return 1

    # Inject the runtime TriAttention calibration path into any config that
    # uses triattention_budget (TA*, TBQ+TA*, etc).  Key-based detection so
    # future composition configs also pick up the calib path automatically.
    needs_calib = any("triattention_budget" in k for _, k in configs)
    if needs_calib and not args.triattention_calib:
        print(
            "Configs with triattention_budget require --triattention-calib <path>. "
            "Run `python -m mlx_vlm.triattention_calibrate --model ... --output ...` first.",
            file=sys.stderr,
        )
        return 1
    resolved_configs: list[tuple[str, dict]] = []
    for name, kwargs in configs:
        if "triattention_budget" in kwargs:
            merged = {**kwargs, "triattention_calib": args.triattention_calib}
            resolved_configs.append((name, merged))
        else:
            resolved_configs.append((name, kwargs))
    configs = resolved_configs

    print(f"[load] {args.model}", flush=True)
    t_load = time.perf_counter()
    model, processor = load(args.model)
    print(f"[load] done in {time.perf_counter() - t_load:.1f}s", flush=True)

    rows: list[RunResult] = []
    for tier in args.tiers:
        prompt, templated_tokens = build_prompt(
            processor, model.config, seed_text, tier, image_count=1
        )
        print(
            f"\n[tier {tier}] templated prompt has ~{templated_tokens} tokens "
            f"(target was {tier})",
            flush=True,
        )
        for name, kwargs in configs:
            print(f"  [{name}] running...", flush=True)
            try:
                result, wall, kv_bytes = run_one(
                    model,
                    processor,
                    prompt,
                    args.image,
                    kwargs,
                    args.max_tokens,
                    args.quantized_kv_start,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name}] FAILED: {exc}", flush=True)
                continue
            rows.append(
                RunResult(
                    tier=tier,
                    config=name,
                    prompt_tokens=getattr(result, "prompt_tokens", templated_tokens),
                    generation_tokens=getattr(result, "generation_tokens", 0),
                    prompt_tps=getattr(result, "prompt_tps", 0.0),
                    generation_tps=getattr(result, "generation_tps", 0.0),
                    peak_memory_gb=getattr(result, "peak_memory", 0.0),
                    kv_bytes_mb=kv_bytes / (1024 * 1024),
                    wall_seconds=wall,
                    text=getattr(result, "text", str(result)),
                )
            )
            print(
                f"  [{name}] prompt_tps={rows[-1].prompt_tps:.1f} "
                f"gen_tps={rows[-1].generation_tps:.1f} "
                f"peak={rows[-1].peak_memory_gb:.2f}GB "
                f"kv={rows[-1].kv_bytes_mb:.1f}MB "
                f"wall={wall:.1f}s",
                flush=True,
            )

    # Render
    header = (
        "| Tier   | Cfg    | Prompt  | Prefill TPS | Decode TPS | Peak GB  | KV MB     | Wall s | Output |\n"
        "|--------|--------|---------|-------------|------------|----------|-----------|--------|--------|"
    )
    table_lines = [header]
    for r in rows:
        table_lines.append(fmt_row(r))

    table = "\n".join(table_lines)
    print("\n" + table + "\n")

    out_path = Path(args.output)
    calib_line = (
        f"- TriAttention calib: `{args.triattention_calib}`\n"
        if args.triattention_calib
        else ""
    )
    out_path.write_text(
        f"# KV-Cache Benchmark — {args.model}\n\n"
        f"- Image: `{args.image}`\n"
        f"- Seed: `{args.seed_file}`\n"
        f"- `--quantized-kv-start`: `{args.quantized_kv_start}`\n"
        f"- `--max-tokens`: `{args.max_tokens}`\n"
        f"{calib_line}"
        f"\n{table}\n"
    )
    print(f"[write] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

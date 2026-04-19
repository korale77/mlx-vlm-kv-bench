"""
Needle-In-A-Haystack (NIAH) benchmark for TriAttention / TurboQuant
KV-cache optimizations on MLX-VLM.

Inserts a known needle phrase at a controlled depth within tiled filler
text, then asks the model to retrieve it.  Sweeps across context lengths,
needle positions, and KV-cache configurations to produce NIAH-style
pass/fail heatmaps.

Complements bench_northwind.py (Northwind Station QA with fixed needle
position).  This script uses a distinct haystack and needle to keep the
two result sets independent.

Run:
    .venv/bin/python bench_niah.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --image cats.jpg \\
        --haystack niah_haystack.txt \\
        --triattention-calib gemma4_26b_calib.safetensors \\
        --tiers 8000 24000 60000 100000 \\
        --positions 0.10 0.25 0.50 0.75 0.90 \\
        --configs BL TBQ TA512 TA2048 TA8192 TA16384 TA32768 \\
        --max-tokens 64 \\
        --output local/niah_results/gemma4_26b.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.models import cache as cache_mod
from mlx_vlm.prompt_utils import apply_chat_template

DEFAULT_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
DEFAULT_IMAGE = "cats.jpg"
DEFAULT_HAYSTACK = Path(__file__).parent / "niah_haystack.txt"

NEEDLE = (
    "The special magic number is: NIAH-7392-ECHO. "
    "Remember this number: NIAH-7392-ECHO."
)
NEEDLE_KEY = "NIAH-7392-ECHO"

QUESTION = (
    "\n\n---\n"
    "What is the special magic number mentioned somewhere in the text above? "
    "Reply with just the number."
)

DEFAULT_POSITIONS = [0.10, 0.25, 0.50, 0.75, 0.90]

CONFIGS: list[tuple[str, dict]] = [
    ("BL", {}),
    ("TBQ", {"kv_bits": 3.5, "kv_quant_scheme": "turboquant"}),
    ("TA512", {"triattention_budget": 512}),
    ("TA2048", {"triattention_budget": 2048}),
    ("TA4096", {"triattention_budget": 4096}),
    ("TA8192", {"triattention_budget": 8192}),
    ("TA16384", {"triattention_budget": 16384}),
    ("TA32768", {"triattention_budget": 32768}),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class NIAHResult:
    tier: int
    position: float
    config: str
    prompt_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    kv_bytes_mb: float
    wall_seconds: float
    text: str
    passed: bool


# ---------------------------------------------------------------------------
# Helpers (same as bench_northwind.py)
# ---------------------------------------------------------------------------

def sum_cache_nbytes(prompt_cache) -> int:
    total = 0
    for entry in prompt_cache:
        total += int(entry.nbytes)
    return total


def reset_peak_memory() -> None:
    reset = getattr(mx, "reset_peak_memory", None)
    if reset is None:
        reset = getattr(mx.metal, "reset_peak_memory", None)
    if reset is not None:
        reset()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_niah_prompt(
    processor,
    model_config,
    haystack_text: str,
    target_tokens: int,
    position: float,
    image_count: int,
) -> tuple[str, int]:
    """Tile haystack to *target_tokens*, insert needle at *position* %, append question."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Tile haystack to roughly target_tokens (reserve room for needle + question)
    seed_ids = tokenizer.encode(haystack_text, add_special_tokens=False)
    seed_len = max(1, len(seed_ids))
    body_target = max(1, target_tokens - 100)
    copies = max(1, (body_target + seed_len - 1) // seed_len)
    tiled = (haystack_text.strip() + "\n\n") * copies

    # Split into paragraphs for clean insertion
    paragraphs = [p for p in tiled.split("\n\n") if p.strip()]

    # Insert needle at the target depth
    insert_idx = max(1, min(int(len(paragraphs) * position), len(paragraphs) - 1))
    paragraphs.insert(insert_idx, NEEDLE)

    body = "\n\n".join(paragraphs)
    raw = body + QUESTION

    prompt = apply_chat_template(
        processor, model_config, raw, num_images=image_count,
    )
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    return prompt, len(encoded)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_output(text: str) -> bool:
    return NEEDLE_KEY.lower() in text.lower()


# ---------------------------------------------------------------------------
# Run one inference
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def render_heatmaps(
    results: list[NIAHResult],
    positions: list[float],
    tiers: list[int],
) -> str:
    """Per-config NIAH heatmap tables."""
    configs: list[str] = []
    seen: set[str] = set()
    for r in results:
        if r.config not in seen:
            configs.append(r.config)
            seen.add(r.config)

    lines: list[str] = []
    for cfg in configs:
        cfg_results = [r for r in results if r.config == cfg]
        lines.append(f"\n### {cfg}\n")

        pos_headers = " | ".join(f"{p:.0%}" for p in positions)
        lines.append(f"| Context | {pos_headers} |")
        lines.append("|--------:" + "|".join(":------:" for _ in positions) + "|")

        for tier in tiers:
            cells: list[str] = []
            for pos in positions:
                match = [
                    r for r in cfg_results
                    if r.tier == tier and abs(r.position - pos) < 0.01
                ]
                if match:
                    cells.append("PASS" if match[0].passed else "FAIL")
                else:
                    cells.append("--")
            tier_label = f"{tier // 1000}k"
            lines.append(f"| {tier_label:>7} | " + " | ".join(f"{c:^6}" for c in cells) + " |")

    return "\n".join(lines)


def render_detail_table(results: list[NIAHResult]) -> str:
    """Full metrics table for every run."""
    lines: list[str] = []
    lines.append(
        "| Tier | Pos | Config | Tokens | Prefill TPS | Decode TPS "
        "| Peak GB | KV MB | Wall s | Result | Output |"
    )
    lines.append(
        "|-----:|----:|--------|-------:|------------:|-----------:"
        "|--------:|------:|-------:|:------:|--------|"
    )
    for r in results:
        snippet = r.text.strip().replace("|", "\\|").replace("\n", " ")[:120]
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| {r.tier:>5d} | {r.position:.0%} | {r.config:<7} "
            f"| {r.prompt_tokens:>6d} | {r.prompt_tps:>10.1f} "
            f"| {r.generation_tps:>9.1f} | {r.peak_memory_gb:>6.2f} "
            f"| {r.kv_bytes_mb:>5.1f} | {r.wall_seconds:>5.1f} "
            f"| {status:>4} | {snippet} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIAH benchmark for TriAttention / TurboQuant KV-cache opts"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--haystack", default=str(DEFAULT_HAYSTACK))
    p.add_argument(
        "--tiers", type=int, nargs="+", default=[8000, 24000, 60000, 100000],
        help="Target prompt-token counts",
    )
    p.add_argument(
        "--positions", type=float, nargs="+", default=DEFAULT_POSITIONS,
        help="Needle insertion depths as fractions (0.0-1.0)",
    )
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--quantized-kv-start", type=int, default=0)
    p.add_argument("--output", default="local/niah_results/results.md")
    p.add_argument(
        "--configs", nargs="+", default=["BL", "TBQ"],
        help="Subset of configs to run "
        "(BL, TBQ, TA512, TA2048, TA4096, TA8192, TA16384, TA32768)",
    )
    p.add_argument("--triattention-calib", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    haystack_path = Path(args.haystack)
    if not haystack_path.exists():
        print(f"haystack file not found: {haystack_path}", file=sys.stderr)
        return 1
    haystack_text = haystack_path.read_text()

    # Ensure output directory exists before doing anything expensive
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path.with_suffix(".jsonl")
    # Clear incremental log from any previous run
    jsonl_path.write_text("")

    # Resolve configs
    configs = [(n, k) for n, k in CONFIGS if n in args.configs]
    if not configs:
        print(f"no configs selected from {args.configs}", file=sys.stderr)
        return 1

    needs_calib = any("triattention_budget" in k for _, k in configs)
    if needs_calib and not args.triattention_calib:
        print(
            "TA configs require --triattention-calib <path>. "
            "Run `python -m mlx_vlm.triattention_calibrate` first.",
            file=sys.stderr,
        )
        return 1

    resolved_configs: list[tuple[str, dict]] = []
    for name, kwargs in configs:
        if "triattention_budget" in kwargs:
            resolved_configs.append(
                (name, {**kwargs, "triattention_calib": args.triattention_calib})
            )
        else:
            resolved_configs.append((name, kwargs))
    configs = resolved_configs

    # Load model once
    print(f"[load] {args.model}", flush=True)
    t_load = time.perf_counter()
    model, processor = load(args.model)
    print(f"[load] done in {time.perf_counter() - t_load:.1f}s", flush=True)

    positions = sorted(args.positions)
    tiers = sorted(args.tiers)
    total_runs = len(tiers) * len(positions) * len(configs)

    print(
        f"[plan] {len(tiers)} tiers x {len(positions)} positions "
        f"x {len(configs)} configs = {total_runs} runs",
        flush=True,
    )

    results: list[NIAHResult] = []
    run_idx = 0

    for tier in tiers:
        for pos in positions:
            prompt, templated_tokens = build_niah_prompt(
                processor, model.config, haystack_text, tier, pos, image_count=1,
            )
            print(
                f"\n[tier={tier} pos={pos:.0%}] ~{templated_tokens} tokens",
                flush=True,
            )

            for name, kwargs in configs:
                run_idx += 1
                print(f"  [{name}] run {run_idx}/{total_runs}...", flush=True)
                try:
                    result, wall, kv_bytes = run_one(
                        model, processor, prompt, args.image,
                        kwargs, args.max_tokens, args.quantized_kv_start,
                    )
                    text = getattr(result, "text", str(result))
                    passed = score_output(text)
                    r = NIAHResult(
                        tier=tier,
                        position=pos,
                        config=name,
                        prompt_tokens=getattr(
                            result, "prompt_tokens", templated_tokens
                        ),
                        prompt_tps=getattr(result, "prompt_tps", 0.0),
                        generation_tps=getattr(result, "generation_tps", 0.0),
                        peak_memory_gb=getattr(result, "peak_memory", 0.0),
                        kv_bytes_mb=kv_bytes / (1024 * 1024),
                        wall_seconds=wall,
                        text=text,
                        passed=passed,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{name}] ERROR: {exc}", flush=True)
                    r = NIAHResult(
                        tier=tier, position=pos, config=name,
                        prompt_tokens=templated_tokens,
                        prompt_tps=0.0, generation_tps=0.0,
                        peak_memory_gb=0.0, kv_bytes_mb=0.0,
                        wall_seconds=0.0, text=f"ERROR: {exc}",
                        passed=False,
                    )

                results.append(r)
                status = "PASS" if r.passed else "FAIL"
                print(
                    f"  [{name}] {status} | prefill={r.prompt_tps:.1f} "
                    f"decode={r.generation_tps:.1f} "
                    f"peak={r.peak_memory_gb:.2f}GB "
                    f"kv={r.kv_bytes_mb:.1f}MB "
                    f"wall={r.wall_seconds:.1f}s",
                    flush=True,
                )

                # Write incrementally so partial results survive crashes
                with open(jsonl_path, "a") as f:
                    d = asdict(r)
                    d["text"] = d["text"][:200]
                    f.write(json.dumps(d) + "\n")

    # ----- Final output -----
    heatmaps = render_heatmaps(results, positions, tiers)
    detail = render_detail_table(results)

    total = len(results)
    passed = sum(1 for r in results if r.passed)

    calib_line = (
        f"- TriAttention calib: `{args.triattention_calib}`\n"
        if args.triattention_calib
        else ""
    )

    md = (
        f"# NIAH Benchmark — {args.model}\n\n"
        f"- Haystack: `{args.haystack}`\n"
        f"- Image: `{args.image}`\n"
        f"- Needle: `{NEEDLE_KEY}`\n"
        f"- `--quantized-kv-start`: `{args.quantized_kv_start}`\n"
        f"- `--max-tokens`: `{args.max_tokens}`\n"
        f"{calib_line}"
        f"- Total: **{passed}/{total} passed**\n\n"
        f"## Heatmaps\n{heatmaps}\n\n"
        f"## Detail\n\n{detail}\n"
    )

    out_path.write_text(md)
    print(f"\n[write] {out_path}")
    print(f"[write] {jsonl_path} ({total} rows)")

    # Print summary to stdout
    print(f"\n{'=' * 60}")
    print(f"NIAH Results: {passed}/{total} passed")
    print(f"{'=' * 60}")
    print(heatmaps)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

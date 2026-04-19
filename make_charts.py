"""Chart generator for the mlx-vlm KV optimization thread.

Reads the tier-x-config markdown tables in `local/m4max_results/` and
writes PNGs to `charts/`. Run:

    .venv/bin/python make_charts.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "local" / "m4max_results"
OUT_DIR = HERE / "charts"
OUT_DIR.mkdir(exist_ok=True)

COLORS = {
    "BL": "#c0392b",
    "TBQ": "#8e44ad",
    "TBQ4": "#9b59b6",
    "TA512": "#2980b9",
    "TA2048": "#16a085",
    "TA4096": "#27ae60",
    "TA8192": "#d35400",
}
LABELS = {
    "BL": "Baseline (no KV opt)",
    "TBQ": "TurboQuant 3.5-bit",
    "TBQ4": "TurboQuant 4-bit",
    "TA512": "TriAttention budget=512",
    "TA2048": "TriAttention budget=2048",
    "TA4096": "TriAttention budget=4096",
    "TA8192": "TriAttention budget=8192",
}


@dataclass(frozen=True)
class Row:
    tier: int
    cfg: str
    prompt: int
    prefill_tps: float
    decode_tps: float
    peak_gb: float
    kv_mb: float
    wall_s: float
    output: str


def parse_results_md(path: Path) -> list[Row]:
    """Parse a bench harness markdown table.

    Handles both the current 9-column layout (includes `Wall s`) and the
    older 8-column layout from the M1 session (`Wall s` absent).
    """
    rows: list[Row] = []
    header: list[str] | None = None
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None and cells[0].lower() == "tier":
            header = [c.lower() for c in cells]
            continue
        if not cells or set(cells[0]) <= {"-"} or not cells[0][0].isdigit():
            continue
        if header is None:
            continue
        by_name = dict(zip(header, cells))
        rows.append(
            Row(
                tier=int(by_name["tier"]),
                cfg=by_name["cfg"],
                prompt=int(by_name["prompt"]),
                prefill_tps=float(by_name["prefill tps"]),
                decode_tps=float(by_name["decode tps"]),
                peak_gb=float(by_name["peak gb"]),
                kv_mb=float(by_name["kv mb"]),
                wall_s=float(by_name.get("wall s", "0") or 0),
                output=by_name.get("output", ""),
            )
        )
    return rows


def load_all() -> dict[str, list[Row]]:
    files = {
        "26b_stretch_bl": RESULTS_DIR / "results_gemma4_26b_stretch_bl.md",
        "26b_stretch_ta512": RESULTS_DIR / "results_gemma4_26b_stretch_ta512.md",
        "26b_large_budgets": RESULTS_DIR / "results_gemma4_26b_large_budgets.md",
        "26b_domain_calib": RESULTS_DIR / "results_gemma4_26b_domain_calib.md",
        "31b": RESULTS_DIR / "results_gemma4_31b.md",
        "qwen35_35b": RESULTS_DIR / "results_qwen35_35b_a3b.md",
        # 26B-A4B TBQ-3.5 and TA-2048 cells were measured in the earlier
        # session and not re-run later. Retrieval outputs are deterministic
        # in (model, prompt, MLX version, KV-opt config) — overlapping cells
        # produced byte-identical outputs across re-runs — so they're loaded
        # into Chart 3's 26B grid without a source distinction.
        "26b_extra": HERE / "local" / "results_gemma4_triattention.md",
    }
    optional = {
        # Long-context Gemma-4-31B TA stretch (Block 9, Measurement A).
        "31b_stretch": RESULTS_DIR / "results_gemma4_31b_stretch.md",
        # Block 10: Qwen35 TBQ retrieval at max_tokens=384.
        "qwen35_35b_retrieval": RESULTS_DIR / "results_qwen35_35b_retrieval.md",
        # Block 11: Qwen3-VL-8B dense TBQ-4 retry.
        "qwen3vl_8b_tbq4": RESULTS_DIR / "results_qwen3vl_8b_tbq4.md",
        # Round 1 (Block 12): 31B with TA-16384 / TA-32768 at 100k / 128k.
        "31b_bigbudget": RESULTS_DIR / "results_gemma4_31b_bigbudget.md",
        # Round 2 BL fill-ins (Block 13): unlock Chart 4 savings computations.
        "qwen35_35b_stretch": RESULTS_DIR / "results_qwen35_35b_stretch.md",
        "26b_bl_200k": RESULTS_DIR / "results_gemma4_26b_bl_200k.md",
        "31b_bl_100k": RESULTS_DIR / "results_gemma4_31b_bl_100k.md",
        # Round 3 (Block 14): 31B TA-16384/32768 at 200k.
        "31b_bigbudget_200k": RESULTS_DIR / "results_gemma4_31b_bigbudget_200k.md",
        # P1: 26B-A4B TA-16384/32768 at 60k + 100k — ratio rule verification.
        "26b_bigbudget": RESULTS_DIR / "results_gemma4_26b_bigbudget.md",
        # P3: 31B TBQ at 100k — fills Chart 4 TBQ-31B line.
        "31b_tbq_100k": RESULTS_DIR / "results_gemma4_31b_tbq_100k.md",
        # P2: long-output decode test at 60k with max_tokens=2048.
        "26b_longoutput": RESULTS_DIR / "results_gemma4_26b_longoutput.md",
        # Option-1 follow-up: 31B BL+TBQ at 128k — extends Chart 4 lines.
        "31b_128k": RESULTS_DIR / "results_gemma4_31b_128k.md",
        # Pareto sweep: 26B-A4B, BL + TA-{512,8192,16384,32768} at 8k/60k/128k,
        # max_tokens=2048 — feeds Chart 5 (the design-space frontier).
        "26b_pareto": RESULTS_DIR / "results_gemma4_26b_pareto.md",
        # Pareto extension: same configs, extra tiers (24k, 48k, 100k, 200k)
        # for denser trajectories on Chart 5.
        "26b_pareto_extra": RESULTS_DIR / "results_gemma4_26b_pareto_extra.md",
        # Chart 3 gap-fills: big budgets at short context (max_tokens=64).
        "31b_chart3_fill": RESULTS_DIR / "results_gemma4_31b_chart3_fill.md",
        "26b_chart3_fill": RESULTS_DIR / "results_gemma4_26b_chart3_fill.md",
        "26b_chart3_fill2": RESULTS_DIR / "results_gemma4_26b_chart3_fill2.md",
        # TBQ stretch on 26B-A4B (8k-200k) — feeds Chart 1 third line.
        "26b_tbq_stretch": RESULTS_DIR / "results_gemma4_26b_tbq_stretch.md",
        "26b_tbq_200k": RESULTS_DIR / "results_gemma4_26b_tbq_200k.md",
        # TA-16384 rerun at 60k+100k on 26B — overwrites noisy Block 15 cells.
        "26b_ta16384_rerun": RESULTS_DIR / "results_gemma4_26b_ta16384_rerun2.md",
    }
    data = {k: parse_results_md(v) for k, v in files.items()}
    for k, v in optional.items():
        data[k] = parse_results_md(v) if v.exists() else []
    return data


def _style_axes(ax, *, title, subtitle=None, xlabel=None, ylabel=None):
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=13, pad=12)
        ax.title.set_fontweight("bold")
    else:
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _footer(fig, text):
    fig.text(
        0.01,
        0.01,
        text,
        fontsize=8,
        color="#888",
        ha="left",
    )


def chart1_flatline(data):
    """Chart 1 — KV cache flatline for Gemma4-26B-A4B, 8k → 200k."""
    # Merge the 8k-128k stretch with the 200k fill-in so BL extends all the way.
    bl = sorted(
        [*data["26b_stretch_bl"], *data["26b_bl_200k"]],
        key=lambda r: r.tier,
    )
    ta = sorted(data["26b_stretch_ta512"], key=lambda r: r.tier)
    tbq = sorted(
        [*data.get("26b_tbq_stretch", []), *data.get("26b_tbq_200k", [])],
        key=lambda r: r.tier,
    )

    fig, (ax_kv, ax_peak) = plt.subplots(
        1, 2, figsize=(14, 7.2), gridspec_kw={"wspace": 0.3}
    )

    # Left panel: KV MB
    ax_kv.plot(
        [r.tier / 1000 for r in bl],
        [r.kv_mb for r in bl],
        marker="o",
        markersize=7,
        color=COLORS["BL"],
        linewidth=2.5,
        label="Baseline",
    )
    if tbq:
        ax_kv.plot(
            [r.tier / 1000 for r in tbq],
            [r.kv_mb for r in tbq],
            marker="D",
            markersize=6,
            color=COLORS["TBQ"],
            linewidth=2.5,
            label="TurboQuant 3.5-bit",
        )
    ax_kv.plot(
        [r.tier / 1000 for r in ta],
        [r.kv_mb for r in ta],
        marker="s",
        markersize=7,
        color=COLORS["TA512"],
        linewidth=2.5,
        label="TriAttention (budget=512)",
    )
    # savings-zone shading between BL and TA
    ax_kv.fill_between(
        [r.tier / 1000 for r in bl],
        [r.kv_mb for r in bl],
        [
            next((t.kv_mb for t in ta if t.tier == r.tier), r.kv_mb)
            for r in bl
        ],
        color=COLORS["TA512"],
        alpha=0.06,
    )
    ax_kv.annotate(
        "210 MB @ 200k context",
        xy=(200, 210.7),
        xytext=(110, 1000),
        fontsize=11,
        fontweight="bold",
        color=COLORS["TA512"],
        arrowprops=dict(
            arrowstyle="->", color=COLORS["TA512"], lw=1.4, connectionstyle="arc3,rad=-0.25"
        ),
    )
    _style_axes(
        ax_kv,
        title="KV cache size vs. context length",
        subtitle="TA caps tokens (flat), TBQ compresses bits (proportional)",
        xlabel="Context length (k tokens)",
        ylabel="KV cache size (MB)",
    )
    ax_kv.legend(loc="upper left", frameon=False, fontsize=10)
    ax_kv.set_xlim(0, 210)
    ax_kv.set_ylim(0, 4500)

    # Right panel: Peak GB
    ax_peak.plot(
        [r.tier / 1000 for r in bl],
        [r.peak_gb for r in bl],
        marker="o",
        markersize=7,
        color=COLORS["BL"],
        linewidth=2.5,
        label="Baseline",
    )
    if tbq:
        ax_peak.plot(
            [r.tier / 1000 for r in tbq],
            [r.peak_gb for r in tbq],
            marker="D",
            markersize=6,
            color=COLORS["TBQ"],
            linewidth=2.5,
            label="TurboQuant 3.5-bit",
        )
    ax_peak.plot(
        [r.tier / 1000 for r in ta],
        [r.peak_gb for r in ta],
        marker="s",
        markersize=7,
        color=COLORS["TA512"],
        linewidth=2.5,
        label="TriAttention (budget=512)",
    )
    ax_peak.fill_between(
        [r.tier / 1000 for r in bl],
        [r.peak_gb for r in bl],
        [
            next((t.peak_gb for t in ta if t.tier == r.tier), r.peak_gb)
            for r in bl
        ],
        color=COLORS["TA512"],
        alpha=0.06,
    )
    ax_peak.annotate(
        "23.7 GB @ 200k",
        xy=(200, 23.72),
        xytext=(120, 16),
        fontsize=11,
        fontweight="bold",
        color=COLORS["TA512"],
        arrowprops=dict(
            arrowstyle="->", color=COLORS["TA512"], lw=1.4, connectionstyle="arc3,rad=-0.25"
        ),
    )
    _style_axes(
        ax_peak,
        title="Peak memory during generation",
        subtitle="Model weights + KV + activations",
        xlabel="Context length (k tokens)",
        ylabel="Peak memory (GB)",
    )
    ax_peak.legend(loc="upper left", frameon=False, fontsize=10)
    ax_peak.set_xlim(0, 210)
    ax_peak.set_ylim(14, 40)

    fig.suptitle(
        "Gemma-4-26B-A4B  ·  Mac Studio M4 Max  ·  TriAttention PR #985",
        fontsize=12,
        color="#555",
        y=0.995,
    )
    _footer(
        fig,
        "source: snippets/mlx-vlm-turboquant/m4max_results/ · mlx-vlm 0.31.1 · 2026-04-10",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT_DIR / "northwind_flatline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'northwind_flatline.png'}")


def chart2_decode_tps(data):
    """Chart 2 — Decode tok/s vs context, BL vs TA budgets on Gemma4-26B-A4B."""
    bl = sorted(
        [*data["26b_stretch_bl"], *data["26b_bl_200k"]],
        key=lambda r: r.tier,
    )
    ta512 = sorted(data["26b_stretch_ta512"], key=lambda r: r.tier)

    # TA-8192 from all available sources (decode TPS is consistent across max_tokens)
    ta8192_rows = [
        r for r in [
            *data["26b_large_budgets"],
            *data["26b_chart3_fill2"],
            *data.get("26b_pareto", []),
            *data.get("26b_pareto_extra", []),
        ]
        if r.cfg == "TA8192"
    ]
    # Deduplicate by tier (keep last)
    ta8192_by_tier: dict[int, Row] = {}
    for r in ta8192_rows:
        ta8192_by_tier[r.tier] = r
    ta8192 = sorted(ta8192_by_tier.values(), key=lambda r: r.tier)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(
        [r.tier / 1000 for r in bl],
        [r.decode_tps for r in bl],
        marker="o",
        markersize=8,
        color=COLORS["BL"],
        linewidth=2.8,
        label="Baseline",
    )
    if ta8192:
        ax.plot(
            [r.tier / 1000 for r in ta8192],
            [r.decode_tps for r in ta8192],
            marker="D",
            markersize=7,
            color=COLORS["TA8192"],
            linewidth=2.5,
            label="TriAttention (budget=8192, retrieves to 60k)",
        )
    ax.plot(
        [r.tier / 1000 for r in ta512],
        [r.decode_tps for r in ta512],
        marker="s",
        markersize=8,
        color=COLORS["TA512"],
        linewidth=2.8,
        label="TriAttention (budget=512, retrieval fails)",
    )

    _style_axes(
        ax,
        title="Decode speed survives context growth",
        subtitle="Gemma-4-26B-A4B · Mac Studio M4 Max",
        xlabel="Context length (k tokens)",
        ylabel="Decode speed (tokens / second)",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    ax.set_xlim(0, 215)
    ax.set_ylim(30, 120)

    fig.suptitle(
        "Gemma-4-26B-A4B  ·  Mac Studio M4 Max  ·  TriAttention PR #985",
        fontsize=12,
        color="#555",
        y=0.995,
    )
    _footer(
        fig,
        "source: snippets/mlx-vlm-turboquant/m4max_results/ · mlx-vlm 0.31.1 · 2026-04-10",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT_DIR / "northwind_decode_tps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'northwind_decode_tps.png'}")


# Retrieval outcome scoring (0=fail, 1=partial, 2=correct).
#
# Complication: the old harness truncates outputs to ~60 chars so we often
# only see the start of the answer ("... thorium-fluori..."). The newer
# full-text mode reveals explicit refusals like "does not state how long".
# The scorer handles both:
#   2 = mentions thorium-fluor[ide] AND does not also refuse a later fact
#   1 = mentions reactor/thorium but is weaker (missing fluoride qualifier
#       or refuses a downstream fact like the battery duration)
#   0 = refuses AND has no reactor mention at all
def _score_output(output: str) -> int:
    low = output.lower()
    has_fluoride = "thorium-fluor" in low or "thorium fluor" in low
    has_reactor_weak = (
        has_fluoride
        or "thorium-based" in low
        or "thorium based" in low
        or "reactor" in low
    )
    refuses = (
        "does not contain" in low
        or "does not mention" in low
        or "does not state" in low
    )
    # Tightened: if the output claims a duty cycle but gets the number
    # wrong (e.g. "48-hour duty cycle" instead of "40 percent duty cycle"),
    # that's a factual error — downgrade from full to partial.
    has_correct_duty = "40 percent" in low or "40%" in low or "40 per cent" in low
    has_wrong_duty = "duty cycle" in low and not has_correct_duty

    if has_fluoride and not refuses and not has_wrong_duty:
        return 2  # full answer, or truncated-but-starting-right
    if has_fluoride and not refuses and has_wrong_duty:
        return 1  # reactor correct but duty-cycle fact wrong
    if has_fluoride and refuses:
        return 1  # first fact correct, refused a later one
    if has_reactor_weak and not refuses:
        return 1  # reactor named but missing the fluoride qualifier
    if has_reactor_weak and refuses:
        return 1  # named reactor AND refused something — still partial
    return 0  # "does not ..." with no reactor mention, or nothing at all


def chart3_heatmap(data):
    """Chart 3 — retrieval reliability heatmap: model × TA budget × context.

    The 31B grid runs to 200k because Measurement A stretched TA-512 and
    TA-8192 out there. The 26B-A4B grid stops at 60k (no long-context
    runs for budgets other than TA-512 that we could compare).
    """
    tiers_31b = [8000, 24000, 48000, 60000, 100000, 128000, 200000]
    tiers_26b = [8000, 24000, 48000, 60000, 100000]
    configs = ["BL", "TBQ", "TA512", "TA2048", "TA4096", "TA8192", "TA16384", "TA32768"]
    cfg_labels = ["BL", "TBQ-3.5", "TA-512", "TA-2048", "TA-4096", "TA-8192", "TA-16384", "TA-32768"]
    score_color = {0: "#c0392b", 1: "#f39c12", 2: "#27ae60"}
    score_symbol = {0: "✗", 1: "~", 2: "✓"}

    def grid_for(rows_sources: list[list[Row]], tiers: list[int]):
        rows = [r for src in rows_sources for r in src]
        grid = [[None] * len(tiers) for _ in configs]
        for r in rows:
            if r.cfg not in configs or r.tier not in tiers:
                continue
            ci = configs.index(r.cfg)
            ti = tiers.index(r.tier)
            grid[ci][ti] = _score_output(r.output)
        return grid

    grid_31b = grid_for(
        [
            data["31b"],
            data["31b_stretch"],
            data["31b_bigbudget"],
            data["31b_bigbudget_200k"],
            data["31b_bl_100k"],
            data["31b_tbq_100k"],
            data["31b_128k"],
            data["31b_chart3_fill"],
        ],
        tiers_31b,
    )
    grid_26b = grid_for(
        [
            data["26b_stretch_bl"],
            data["26b_stretch_ta512"],
            data["26b_large_budgets"],
            data["26b_extra"],
            data["26b_bl_200k"],
            data["26b_bigbudget"],
            data["26b_chart3_fill"],
            data["26b_chart3_fill2"],
            data["26b_ta16384_rerun"],  # must be last to overwrite Block 15 cells
        ],
        tiers_26b,
    )

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(10, 12), gridspec_kw={"hspace": 0.4, "height_ratios": [8, 6]}
    )

    def plot_grid(ax, grid, tiers, title, subtitle=None):
        for ci, cfg in enumerate(configs):
            for ti, tier in enumerate(tiers):
                score = grid[ci][ti]
                if score is None:
                    color = "#ecf0f1"
                    symbol = "—"
                    fg = "#95a5a6"
                else:
                    color = score_color[score]
                    symbol = score_symbol[score]
                    fg = "white"
                ax.add_patch(
                    plt.Rectangle(
                        (ti, len(configs) - 1 - ci),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=2,
                    )
                )
                ax.text(
                    ti + 0.5,
                    len(configs) - 1 - ci + 0.5,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=22,
                    fontweight="bold",
                    color=fg,
                )
        ax.set_xlim(0, len(tiers))
        ax.set_ylim(0, len(configs))
        ax.set_xticks([t + 0.5 for t in range(len(tiers))])
        ax.set_xticklabels([f"{t // 1000}k" for t in tiers], fontsize=11)
        ax.set_yticks([len(configs) - 1 - i + 0.5 for i in range(len(configs))])
        ax.set_yticklabels(cfg_labels, fontsize=11)
        if subtitle:
            ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=12, pad=10)
            ax.title.set_fontweight("bold")
        else:
            ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(axis="both", which="both", length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    plot_grid(
        ax_a,
        grid_31b,
        tiers_31b,
        "Gemma-4-31B-it-4bit",
        subtitle="retrieval follows the budget ÷ context ratio (~13 %)",
    )
    plot_grid(
        ax_b,
        grid_26b,
        tiers_26b,
        "Gemma-4-26B-A4B",
        subtitle="ratio rule is noisier on smaller models",
    )

    # shared legend
    legend_entries = [
        plt.Rectangle((0, 0), 1, 1, facecolor=score_color[2], edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor=score_color[1], edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor=score_color[0], edgecolor="white"),
    ]
    fig.legend(
        legend_entries,
        ["✓ full answer", "~ partial", "✗ failed to find answer"],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.03),
    )

    fig.suptitle(
        "KV optimization vs. retrieval accuracy",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.965,
        "NIAH-style QA task · BL and TBQ as baselines, TA budgets sized to context",
        fontsize=10,
        color="#666",
        ha="center",
        style="italic",
    )
    _footer(
        fig,
        "source: m4max_results/ Blocks 1, 2, 3, 9, 12, 14 · mlx-vlm 0.31.1 · TriAttention PR #985 · 2026-04-10",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(OUT_DIR / "northwind_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'northwind_heatmap.png'}")


def chart4_savings_scaling(data):
    """Chart 4 — KV savings % vs context length for multiple (model, config) pairs."""
    # For each series, compute savings% at each tier vs BL on the same model.
    def savings_series(bl_rows_list: list[list[Row]], opt_rows_list: list[list[Row]], cfg: str) -> list[tuple[int, float]]:
        bl_rows = [r for src in bl_rows_list for r in src]
        opt_rows = [r for src in opt_rows_list for r in src]
        bl_by_tier = {r.tier: r.kv_mb for r in bl_rows if r.cfg == "BL"}
        out = []
        for r in opt_rows:
            if r.cfg != cfg or r.tier not in bl_by_tier:
                continue
            out.append((r.tier, 100.0 * (1.0 - r.kv_mb / bl_by_tier[r.tier])))
        out.sort()
        return out

    # 26B-A4B: combine stretch BL + BL@200k fill-in for the full 8k-200k range.
    bl_26b = [data["26b_stretch_bl"], data["26b_bl_200k"]]
    # 31B: Block 1 (8k/24k/48k/60k) + BL@100k fill-in + 128k BL/TBQ fill-in.
    bl_31b = [data["31b"], data["31b_bl_100k"], data["31b_128k"]]
    # 31B TA rows: Block 1 + Measurement A (100k/128k/200k) + Round 1/3 big-budget runs.
    ta_31b = [data["31b"], data["31b_stretch"], data["31b_bigbudget"], data["31b_bigbudget_200k"]]
    # Qwen35-35B: Block 6 (8k/24k/60k) + Round 2 stretch (100k/128k).
    qwen35 = [data["qwen35_35b"], data["qwen35_35b_stretch"]]

    series = []
    # TA-512 on 26B-A4B (stretch) — now goes to 200k
    ta_26b = savings_series(bl_26b, [data["26b_stretch_ta512"]], "TA512")
    series.append(("TA-512 · Gemma-4-26B-A4B", ta_26b, COLORS["TA512"], "s"))
    # TA-8192 on 31B — now goes to 100k (BL fill-in landed)
    ta8192_31b = savings_series(bl_31b, ta_31b, "TA8192")
    series.append(("TA-8192 · Gemma-4-31B", ta8192_31b, COLORS["TA8192"], "D"))
    # TA-16384 on 31B — new! Only 100k is computable (no BL at 128k+)
    ta16384_31b = savings_series(bl_31b, ta_31b, "TA16384")
    if ta16384_31b:
        series.append(("TA-16384 · Gemma-4-31B", ta16384_31b, "#e67e22", "v"))
    # TBQ on 31B — 8k/24k/48k/60k from Block 1 + 100k from P3 + 128k from Option-1
    tbq_31b = savings_series(
        bl_31b, [data["31b"], data["31b_tbq_100k"], data["31b_128k"]], "TBQ"
    )
    series.append(("TBQ-3.5 · Gemma-4-31B", tbq_31b, COLORS["TBQ"], "o"))
    # TBQ on Qwen3.5-35B-A3B — now goes to 128k
    tbq_qwen = savings_series(qwen35, qwen35, "TBQ")
    series.append(("TBQ-3.5 · Qwen3.5-35B-A3B MoE", tbq_qwen, "#2c3e50", "^"))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for label, points, color, marker in series:
        xs = [t / 1000 for t, _ in points]
        ys = [s for _, s in points]
        ax.plot(xs, ys, marker=marker, markersize=8, color=color, linewidth=2.5, label=label)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylim(20, 100)
    ax.set_xlim(0, 210)

    _style_axes(
        ax,
        title="KV cache savings — flat vs. growing",
        subtitle="TriAttention savings stay flat (the KV is a hard cap).\nTurboQuant savings grow with context because the quantization rate is fixed.",
        xlabel="Context length (k tokens)",
        ylabel="KV savings vs. Baseline",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    _footer(
        fig,
        "source: m4max_results/ Blocks 1, 2, 6, 12, 13, 14 · mlx-vlm 0.31.1 · 2026-04-10",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "northwind_savings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'northwind_savings.png'}")


# -- Shared helpers for the Pareto charts --

_V2_SCORE_COLOR = {0: "#c0392b", 1: "#f39c12", 2: "#27ae60"}
_V2_SCORE_LABEL = {0: "fails retrieval", 1: "partial", 2: "full retrieval"}

_V2_CFG_MARKER = {
    "BL": "o",
    "TA512": "s",
    "TA8192": "D",
    "TA16384": "v",
    "TA32768": "^",
}
_V2_CFG_LABEL = {
    "BL": "Baseline",
    "TA512": "TA-512",
    "TA8192": "TA-8192",
    "TA16384": "TA-16384",
    "TA32768": "TA-32768",
}
_V2_CFG_LINESTYLE = {
    "BL": (0, (4, 2)),       # dashed
    "TA512": "solid",
    "TA8192": (0, (1, 1)),    # dotted
    "TA16384": (0, (3, 1, 1, 1)),  # dash-dot
    "TA32768": (0, (5, 1)),   # long dash
}
_V2_EDGE = "#333"
_V2_LINE = "#555"


def _v2_plot_pareto(ax, rows, *, annotate_envelope=False):
    """Shared scatter + trajectory plotter for v2 Pareto charts.

    - Trajectory lines: dark gray, distinguished by linestyle
    - Marker edges: dark gray (shape = config)
    - Marker fill: retrieval score color (green/orange/red)
    """
    by_cfg: dict[str, list[Row]] = {}
    for r in rows:
        by_cfg.setdefault(r.cfg, []).append(r)

    # Trajectory lines
    for cfg, cfg_rows in by_cfg.items():
        cfg_rows = sorted(cfg_rows, key=lambda r: r.tier)
        ax.plot(
            [r.peak_gb for r in cfg_rows],
            [r.decode_tps for r in cfg_rows],
            color=_V2_LINE,
            linewidth=1.8,
            linestyle=_V2_CFG_LINESTYLE.get(cfg, "solid"),
            alpha=0.6,
            zorder=1,
        )

    # Endpoint tier labels
    endpoint_ids = set()
    for cfg, cfg_rows in by_cfg.items():
        sorted_rows = sorted(cfg_rows, key=lambda r: r.tier)
        endpoint_ids.add((cfg, sorted_rows[0].tier))
        endpoint_ids.add((cfg, sorted_rows[-1].tier))

    # Scatter: fill = retrieval score, edge + shape = config
    for r in rows:
        score = _score_output(r.output)
        ax.scatter(
            r.peak_gb,
            r.decode_tps,
            s=200,
            marker=_V2_CFG_MARKER.get(r.cfg, "o"),
            color=_V2_SCORE_COLOR[score],
            edgecolor=_V2_EDGE,
            linewidth=1.8,
            zorder=3,
        )
        if (r.cfg, r.tier) in endpoint_ids:
            ax.annotate(
                f"{r.tier // 1000}k",
                xy=(r.peak_gb, r.decode_tps),
                xytext=(10, 4),
                textcoords="offset points",
                ha="left",
                fontsize=9,
                color="#333",
                zorder=4,
            )

    if annotate_envelope:
        # Shade the TA operating envelope (all TA configs cluster 17-24 GB)
        ta_rows = [r for r in rows if r.cfg != "BL"]
        if ta_rows:
            x_min = min(r.peak_gb for r in ta_rows) - 0.4
            x_max = max(r.peak_gb for r in ta_rows) + 0.6
            y_min = min(r.decode_tps for r in ta_rows) - 4
            y_max = max(r.decode_tps for r in ta_rows) + 4
            rect = plt.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=1.5, edgecolor="#2980b9", facecolor="#2980b9",
                alpha=0.06, linestyle="--", zorder=0,
            )
            ax.add_patch(rect)
            ax.text(
                x_min + 0.2, y_max + 1,
                "TA operating envelope\n(memory stays bounded)",
                fontsize=9, color="#2980b9", fontstyle="italic",
                va="bottom",
            )

        # Annotate BL trajectory
        bl_rows = sorted(by_cfg.get("BL", []), key=lambda r: r.tier)
        if len(bl_rows) >= 2:
            last = bl_rows[-1]
            ax.annotate(
                "Baseline: memory + speed\nboth degrade with context",
                xy=(last.peak_gb, last.decode_tps),
                xytext=(-60, 30),
                textcoords="offset points",
                fontsize=9,
                color="#555",
                fontstyle="italic",
                arrowprops=dict(
                    arrowstyle="->", color="#999", lw=1.2,
                    connectionstyle="arc3,rad=0.2",
                ),
            )


def _v2_legends(ax):
    """Add shape + retrieval legends for v2 Pareto charts."""
    cfg_handles = [
        plt.Line2D(
            [0], [0],
            marker=_V2_CFG_MARKER[c], color="w",
            markerfacecolor="#ccc", markeredgecolor=_V2_EDGE,
            markeredgewidth=2.0, markersize=12, linestyle="none",
            label=_V2_CFG_LABEL[c],
        )
        for c in ("BL", "TA512", "TA8192", "TA16384", "TA32768")
    ]
    color_handles = [
        plt.Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=_V2_SCORE_COLOR[s], markeredgecolor=_V2_EDGE,
            markersize=12, linestyle="none",
            label=_V2_SCORE_LABEL[s],
        )
        for s in (2, 1, 0)
    ]
    leg1 = ax.legend(
        handles=cfg_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#ddd",
        fontsize=10,
        title="Config (shape)",
        title_fontsize=10,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=color_handles,
        loc="center right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#ddd",
        fontsize=10,
        title="Retrieval (fill color)",
        title_fontsize=10,
    )


def chart5_pareto(data):
    """Chart 5 — single-panel Pareto frontier, annotated with TA envelope."""
    rows = [*data.get("26b_pareto", []), *data.get("26b_pareto_extra", [])]
    if not rows:
        print("chart5_pareto: no data — skipping")
        return

    fig, ax = plt.subplots(figsize=(13, 8))
    _v2_plot_pareto(ax, rows, annotate_envelope=True)
    _v2_legends(ax)

    _style_axes(
        ax,
        title="Pick your spot in the speed × memory × accuracy space",
        subtitle="Gemma-4-26B-A4B · Mac Studio M4 Max · 2 048 tokens generated per run",
        xlabel="Peak memory during generation (GB)",
        ylabel="Decode speed (tokens / second)",
    )

    xs = [r.peak_gb for r in rows]
    ys = [r.decode_tps for r in rows]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 2.0)
    ax.set_ylim(min(ys) - 10, max(ys) + 18)

    _footer(
        fig,
        "source: m4max_results/results_gemma4_26b_pareto.md · mlx-vlm 0.31.1 · TriAttention PR #985 · 2026-04-15",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "northwind_pareto.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'northwind_pareto.png'}")


def chart5_pareto_dual(data):
    """Chart 5 dual — zoomed TA envelope + full range with Baseline."""
    rows = [*data.get("26b_pareto", []), *data.get("26b_pareto_extra", [])]
    if not rows:
        print("chart5_pareto_dual: no data — skipping")
        return

    ta_rows = [r for r in rows if r.cfg != "BL"]
    ta_x_min = min(r.peak_gb for r in ta_rows) - 0.6
    ta_x_max = max(r.peak_gb for r in ta_rows) + 1.5

    fig, (ax_zoom, ax_full) = plt.subplots(
        1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [3, 2], "wspace": 0.22}
    )

    # Left panel: zoomed into the TA envelope
    _v2_plot_pareto(ax_zoom, [r for r in rows if r.peak_gb <= ta_x_max + 1], annotate_envelope=False)
    _style_axes(
        ax_zoom,
        title="TA configs: speed/accuracy tradeoff at bounded memory",
        subtitle="Zoomed to the TriAttention operating region",
        xlabel="Peak memory during generation (GB)",
        ylabel="Decode speed (tokens / second)",
    )
    ta_ys = [r.decode_tps for r in ta_rows]
    ax_zoom.set_xlim(ta_x_min, 26)
    ax_zoom.set_ylim(min(ta_ys) - 8, max(ta_ys) + 12)

    # Legends on the right side — extending xlim to 26 gives enough
    # breathing room since no data points land past ~24 GB.
    _v2_legends(ax_zoom)

    # Right panel: full range showing BL trajectory for context
    _v2_plot_pareto(ax_full, rows, annotate_envelope=True)
    _style_axes(
        ax_full,
        title="Full picture (with Baseline)",
        subtitle="BL sprawls right as context grows; TA stays compact",
        xlabel="Peak memory during generation (GB)",
        ylabel="Decode speed (tokens / second)",
    )
    all_xs = [r.peak_gb for r in rows]
    all_ys = [r.decode_tps for r in rows]
    ax_full.set_xlim(min(all_xs) - 0.8, max(all_xs) + 2.0)
    ax_full.set_ylim(min(all_ys) - 10, max(all_ys) + 18)

    fig.suptitle(
        "Gemma-4-26B-A4B  ·  Mac Studio M4 Max  ·  2 048 tokens generated per run",
        fontsize=12,
        color="#555",
        y=0.995,
    )
    _footer(
        fig,
        "source: m4max_results/results_gemma4_26b_pareto.md · mlx-vlm 0.31.1 · TriAttention PR #985 · 2026-04-15",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(OUT_DIR / "northwind_pareto_dual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'northwind_pareto_dual.png'}")


# ---------------------------------------------------------------------------
# NIAH heatmaps — built from niah_results/*.jsonl
# ---------------------------------------------------------------------------

NIAH_DIR = HERE / "local" / "niah_results"


def _load_niah_jsonl(*filenames: str) -> list[dict]:
    """Load and merge one or more NIAH JSONL result files."""
    import json

    rows: list[dict] = []
    for name in filenames:
        path = NIAH_DIR / name
        if not path.exists():
            print(f"  warning: {path} not found, skipping")
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _niah_pass_rate(rows: list[dict], config: str, tier: int) -> float | None:
    """Return pass rate (0.0-1.0) for a config at a tier, or None if untested."""
    subset = [r for r in rows if r["config"] == config and r["tier"] == tier]
    if not subset:
        return None
    return sum(1 for r in subset if r["passed"]) / len(subset)


def niah_heatmap(rows_26b: list[dict], rows_31b: list[dict]):
    """NIAH heatmap — pass rate per (config, tier) aggregated over needle positions.

    Two panels: 31B (top, 8k-200k) and 26B (bottom, 8k-200k).
    Cell color: green (5/5), yellow (partial), red (0/5), gray (not tested).
    Cell text: "N/5" pass count.
    """
    tiers = [8000, 24000, 48000, 60000, 100000, 128000, 200000]
    configs_31b = ["BL", "TBQ", "TA32768", "TA16384", "TA8192", "TA4096", "TA2048", "TA512"]
    configs_26b = ["BL", "TBQ", "TA32768", "TA16384", "TA8192", "TA4096", "TA2048", "TA512"]
    cfg_labels_31b = ["BL", "TBQ-3.5", "TA-32768", "TA-16384", "TA-8192", "TA-4096", "TA-2048", "TA-512"]
    cfg_labels_26b = ["BL", "TBQ-3.5", "TA-32768", "TA-16384", "TA-8192", "TA-4096", "TA-2048", "TA-512"]

    # Inferred cells: configs that consistently pass/fail at shorter context.
    inferred_zero = {
        "31b": {
            ("TA4096", 128000), ("TA4096", 200000),
            ("TA2048", 128000), ("TA2048", 200000),
            ("TA512", 128000), ("TA512", 200000),
        },
        "26b": {
            ("TA512", 128000), ("TA512", 200000),
            ("TA2048", 128000), ("TA2048", 200000),
            ("TA4096", 128000), ("TA4096", 200000),
        },
    }
    inferred_pass = {
        "31b": {("BL", 200000), ("TBQ", 200000)},
        "26b": set(),
    }

    def pass_count(rows, config, tier):
        subset = [r for r in rows if r["config"] == config and r["tier"] == tier]
        if not subset:
            return None, None
        passed = sum(1 for r in subset if r["passed"])
        return passed, len(subset)

    def color_for_rate(passed, total):
        if total == 0 or passed is None:
            return "#ecf0f1"
        rate = passed / total
        if rate >= 0.95:
            return "#27ae60"
        elif rate >= 0.6:
            return "#f1c40f"
        elif rate >= 0.2:
            return "#e67e22"
        else:
            return "#c0392b"

    fig, (ax_31b, ax_26b) = plt.subplots(
        2, 1, figsize=(12, 13), gridspec_kw={"hspace": 0.35, "height_ratios": [7, 8]}
    )

    def plot_niah_grid(ax, rows, configs, cfg_labels, tiers, title, subtitle, model_key):
        for ci, cfg in enumerate(configs):
            for ti, tier in enumerate(tiers):
                if (cfg, tier) in inferred_pass[model_key]:
                    color = "#27ae60"
                    label = "5/5*"
                    fg = "#222"
                elif (cfg, tier) in inferred_zero[model_key]:
                    color = "#c0392b"
                    label = "0/5*"
                    fg = "white"
                else:
                    passed, total = pass_count(rows, cfg, tier)
                    if passed is None:
                        color = "#ecf0f1"
                        label = "—"
                        fg = "#95a5a6"
                    else:
                        color = color_for_rate(passed, total)
                        label = f"{passed}/{total}"
                        fg = "white" if passed / total < 0.6 else "#222"

                ax.add_patch(
                    plt.Rectangle(
                        (ti, len(configs) - 1 - ci), 1, 1,
                        facecolor=color, edgecolor="white", linewidth=2,
                    )
                )
                ax.text(
                    ti + 0.5, len(configs) - 1 - ci + 0.5, label,
                    ha="center", va="center", fontsize=13, fontweight="bold", color=fg,
                )

        ax.set_xlim(0, len(tiers))
        ax.set_ylim(0, len(configs))
        ax.set_xticks([t + 0.5 for t in range(len(tiers))])
        ax.set_xticklabels([f"{t // 1000}k" for t in tiers], fontsize=11)
        ax.set_yticks([len(configs) - 1 - i + 0.5 for i in range(len(configs))])
        ax.set_yticklabels(cfg_labels, fontsize=11)
        ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=12, pad=10, fontweight="bold")
        ax.tick_params(axis="both", which="both", length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    plot_niah_grid(
        ax_31b, rows_31b, configs_31b, cfg_labels_31b, tiers,
        "Gemma-4-31B-it-4bit",
        "10 full-attention layers · robust to 128k at TA-16384+",
        "31b",
    )
    plot_niah_grid(
        ax_26b, rows_26b, configs_26b, cfg_labels_26b, tiers,
        "Gemma-4-26B-A4B-it-4bit",
        "5 full-attention layers · degrades earlier",
        "26b",
    )

    # Legend
    legend_entries = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#27ae60", edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#f1c40f", edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#e67e22", edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#c0392b", edgecolor="white"),
    ]
    fig.legend(
        legend_entries,
        ["5/5 pass", "3-4/5", "1-2/5", "0/5"],
        loc="lower center", ncol=4, frameon=False, fontsize=11,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.suptitle(
        "Needle-In-A-Haystack retrieval · 5 positions × N/5 pass rate",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.965,
        "Single-needle retrieval (NIAH-7392-ECHO) at 10%/25%/50%/75%/90% depth  ·  * = inferred from shorter-context results",
        fontsize=9, color="#666", ha="center", style="italic",
    )
    _footer(fig, "source: niah_results/*.jsonl · mlx-vlm 0.31.1 · TriAttention PR #985 · 2026-04-19")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(OUT_DIR / "niah_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'niah_heatmap.png'}")


def niah_position_heatmaps(rows_26b: list[dict], rows_31b: list[dict]):
    """NIAH position heatmap — combined side-by-side for both models.

    Two columns (31B left, 26B right), one row per TA config.
    Each cell: x = context tier, y = needle depth (10%-90%), color = pass/fail.
    """
    tiers = [8000, 24000, 48000, 60000, 100000, 128000, 200000]
    positions = [0.10, 0.25, 0.50, 0.75, 0.90]
    pos_labels = ["10%", "25%", "50%", "75%", "90%"]
    pass_color = {True: "#27ae60", False: "#c0392b"}
    pass_symbol = {True: "P", False: "F"}
    configs = ["TA32768", "TA16384", "TA8192", "TA4096", "TA2048"]
    cfg_labels = ["TA-32768", "TA-16384", "TA-8192", "TA-4096", "TA-2048"]
    n_configs = len(configs)

    fig, axes = plt.subplots(
        n_configs, 2, figsize=(16, 2.2 * n_configs + 2.0),
        gridspec_kw={"hspace": 0.55, "wspace": 0.15},
    )

    # Inferred all-fail cells (config fails at shorter context → fails here too).
    pos_inferred_fail = {
        "31b": {
            ("TA4096", 128000), ("TA4096", 200000),
            ("TA2048", 128000), ("TA2048", 200000),
        },
        "26b": {
            ("TA4096", 128000), ("TA4096", 200000),
            ("TA2048", 128000), ("TA2048", 200000),
        },
    }
    model_keys = ["31b", "26b"]

    model_data = [
        (rows_31b, "Gemma-4-31B"),
        (rows_26b, "Gemma-4-26B-A4B"),
    ]

    for col, (rows, model_name) in enumerate(model_data):
        mkey = model_keys[col]
        for row, (cfg, cfg_label) in enumerate(zip(configs, cfg_labels)):
            ax = axes[row][col]
            for ti, tier in enumerate(tiers):
                for pi, pos in enumerate(positions):
                    if (cfg, tier) in pos_inferred_fail[mkey]:
                        color = pass_color[False]
                        symbol = "F*"
                        fg = "white"
                    else:
                        match = [
                            r for r in rows
                            if r["config"] == cfg and r["tier"] == tier
                            and abs(r["position"] - pos) < 0.01
                        ]
                        if match:
                            passed = match[0]["passed"]
                            color = pass_color[passed]
                            symbol = pass_symbol[passed]
                            fg = "white"
                        else:
                            color = "#ecf0f1"
                            symbol = "—"
                            fg = "#95a5a6"

                    ax.add_patch(
                        plt.Rectangle(
                            (ti, len(positions) - 1 - pi), 1, 1,
                            facecolor=color, edgecolor="white", linewidth=1.5,
                        )
                    )
                    ax.text(
                        ti + 0.5, len(positions) - 1 - pi + 0.5, symbol,
                        ha="center", va="center", fontsize=10, fontweight="bold", color=fg,
                    )

            ax.set_xlim(0, len(tiers))
            ax.set_ylim(0, len(positions))
            ax.set_xticks([t + 0.5 for t in range(len(tiers))])
            ax.set_xticklabels([f"{t // 1000}k" for t in tiers], fontsize=8)
            ax.set_yticks([len(positions) - 1 - i + 0.5 for i in range(len(positions))])
            ax.set_yticklabels(pos_labels if col == 0 else [], fontsize=9)
            # Title: config label on left column, model name on top row
            if row == 0:
                ax.set_title(model_name, fontsize=12, fontweight="bold", pad=8)
            # Config label on the left edge
            if col == 0:
                ax.set_ylabel(cfg_label, fontsize=11, fontweight="bold", rotation=0,
                              labelpad=60, va="center")
            ax.tick_params(axis="both", which="both", length=0)
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)

    fig.suptitle(
        "NIAH needle-depth sensitivity by model",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.968,
        "Per-position pass/fail · needle inserted at 10%/25%/50%/75%/90% depth in haystack",
        fontsize=10, color="#666", ha="center", style="italic",
    )
    legend_entries = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#27ae60", edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#c0392b", edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#ecf0f1", edgecolor="#ccc"),
    ]
    fig.legend(
        legend_entries, ["Pass", "Fail", "Not tested"],
        loc="lower center", ncol=3, frameon=False, fontsize=10,
        bbox_to_anchor=(0.5, 0.005),
    )
    _footer(fig, "source: niah_results/*.jsonl · mlx-vlm 0.31.1 · 2026-04-19")
    fig.tight_layout(rect=[0.08, 0.04, 1, 0.96])
    fig.savefig(OUT_DIR / "niah_positions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'niah_positions.png'}")


def main():
    data = load_all()
    chart1_flatline(data)
    chart2_decode_tps(data)
    chart3_heatmap(data)
    chart4_savings_scaling(data)
    chart5_pareto(data)
    chart5_pareto_dual(data)

    # NIAH charts — loaded from JSONL, not from m4max_results markdown
    rows_26b = _load_niah_jsonl(
        "gemma4_26b.jsonl", "gemma4_26b_extended.jsonl", "gemma4_26b_ta4096.jsonl",
        "gemma4_26b_48k.jsonl",
    )
    rows_31b = _load_niah_jsonl(
        "gemma4_31b.jsonl", "gemma4_31b_gap.jsonl", "gemma4_31b_gap2.jsonl",
        "gemma4_31b_extended.jsonl", "gemma4_31b_200k.jsonl",
        "gemma4_31b_ta4096.jsonl",
    )
    if rows_26b or rows_31b:
        niah_heatmap(rows_26b, rows_31b)
        niah_position_heatmaps(rows_26b, rows_31b)


if __name__ == "__main__":
    main()

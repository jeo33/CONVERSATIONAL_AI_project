#!/usr/bin/env python3
"""
Parse logs/*.out files and plot ROUGE-L vs budget% with recent% as line colour.
Generates one set of plots per dataset.

File naming convention (new):
  h2o_{dataset}_{mode}_{strategy}_{jobid}.out
  dataset examples: cnn_dailymail, gov_report, qmsum, vcsum
  mode examples: full, h2o_b3_r1, random_b2, local_b3
  strategy: per_head | layer_shared
"""

import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.lines import Line2D

# ── Config ────────────────────────────────────────────────────────────────────
LOGS_DIR   = Path(__file__).parent.parent / "logs"
OUTPUT_DIR = Path(__file__).parent.parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CACHE_SIZE = 512

KNOWN_DATASETS = ["cnn_dailymail", "gov_report", "qmsum", "vcsum"]

# ── Parse .out files ──────────────────────────────────────────────────────────
ROUGE_RE = re.compile(r"ROUGE-L:\s+([\d.]+)\s+\(min=")
LAT_RE   = re.compile(r"Latency:\s+([\d.]+)s \| mem_peak:\s+([\d.]+)MB")
# Matches: h2o_{dataset}_{mode}_{strategy}_{jobid}.out
MODE_RE  = re.compile(
    r"^h2o_(" + "|".join(KNOWN_DATASETS) + r")_(.+)_(per_head|layer_shared)_\d+\.out$"
)

all_records = []
for f in sorted(LOGS_DIR.glob("*.out")):
    m = MODE_RE.match(f.name)
    if not m:
        continue
    dataset, mode_raw, strategy = m.group(1), m.group(2), m.group(3)

    text = f.read_text(errors="replace")

    rm = ROUGE_RE.search(text)
    if not rm:
        print(f"  [SKIP] no final ROUGE-L in {f.name}")
        continue
    rouge = float(rm.group(1))

    lat_matches  = LAT_RE.findall(text)
    avg_latency  = sum(float(l) for l, _ in lat_matches) / len(lat_matches) if lat_matches else None
    avg_mem_peak = sum(float(mp) for _, mp in lat_matches) / len(lat_matches) if lat_matches else None

    if mode_raw == "full":
        budget_pct = 100
        recent_pct = 100
        method = "full"
    elif mode_raw.startswith("random_b"):
        bm = re.match(r"random_b(\d+)", mode_raw)
        budget_pct = int(bm.group(1)) * 10 if bm else None
        recent_pct = None
        method = "random"
    elif mode_raw.startswith("local_b"):
        bm = re.match(r"local_b(\d+)", mode_raw)
        budget_pct = int(bm.group(1)) * 10 if bm else None
        recent_pct = budget_pct
        method = "local"
    elif mode_raw.startswith("h2o_b"):
        bm = re.match(r"h2o_b(\d+)_r(\d+)", mode_raw)
        budget_pct = int(bm.group(1)) * 10 if bm else None
        recent_pct = int(bm.group(2)) * 10 if bm else None
        method = "h2o"
    else:
        print(f"  [SKIP] unrecognised mode: {mode_raw}")
        continue

    all_records.append({
        "dataset":     dataset,
        "mode":        mode_raw,
        "method":      method,
        "strategy":    strategy,
        "budget_pct":  budget_pct,
        "recent_pct":  recent_pct,
        "rouge_l":     rouge,
        "avg_latency":  avg_latency,
        "avg_mem_peak": avg_mem_peak,
        "file":        f.name,
    })
    print(f"  [{dataset}] {mode_raw:30s} {strategy:15s}  budget={budget_pct}%  rouge={rouge:.4f}")

# Group by dataset
records_by_dataset = {}
for r in all_records:
    records_by_dataset.setdefault(r["dataset"], []).append(r)

print(f"\nParsed {len(all_records)} records across {len(records_by_dataset)} datasets: "
      f"{list(records_by_dataset.keys())}\n")

if not all_records:
    print("No records found — check LOGS_DIR path:", LOGS_DIR)
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────
STRATEGY_LS = {"per_head": "-",  "layer_shared": "--"}
STRATEGY_MK = {"per_head": "o",  "layer_shared": "s"}
METHOD_COLOURS = {
    "Full Cache":             "black",
    "Random":                 "steelblue",
    "Local (sliding window)": "orange",
    "H2O best":               "crimson",
}
LOCAL_COLOUR  = "darkorange"
RANDOM_COLOUR = "steelblue"

def to_total_budget(budget_pct):
    return max(16, int(CACHE_SIZE * budget_pct / 100))

def to_recent_budget(recent_pct):
    return max(8, int(CACHE_SIZE * recent_pct / 100))

def set_xaxis(ax, budgets):
    ticks = sorted(set(budgets), reverse=True)
    ax.set_xlim(max(ticks) + 2, min(ticks) - 2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks], fontsize=8)


# ── Per-dataset plot function ─────────────────────────────────────────────────
def plot_dataset(dataset: str, records: list):
    out_dir = OUTPUT_DIR / dataset
    out_dir.mkdir(exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}  ({len(records)} records)")
    print(f"  Output : {out_dir}")
    print(f"{'='*60}")

    full_rouge    = next((r["rouge_l"]      for r in records if r["method"] == "full"), None)
    full_latency  = next((r["avg_latency"]  for r in records if r["method"] == "full"), None)
    full_mem_peak = next((r["avg_mem_peak"] for r in records if r["method"] == "full"), None)
    print(f"  Baseline ROUGE-L: {full_rouge}  latency: {full_latency}s  mem: {full_mem_peak}MB\n")

    h2o_records = [r for r in records if r["method"] == "h2o"]
    recent_vals = sorted(set(r["recent_pct"] for r in h2o_records))
    cmap        = plt.colormaps["tab10"].resampled(max(len(recent_vals), 1))
    h2o_colour  = {rv: cmap(i) for i, rv in enumerate(recent_vals)}

    # ── Plot 1: H2O + Local — ROUGE-L vs budget ───────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(f"[{dataset}]  H2O + Local KV Cache — ROUGE-L  (cache_size={CACHE_SIZE})",
                 fontsize=13, fontweight="bold")

    all_budgets = []
    if full_rouge:
        ax.axhline(full_rouge, color="black", linestyle=":", linewidth=1.4, label="Full cache")

    for rv in recent_vals:
        recent_tok = to_recent_budget(rv)
        for strategy in ["per_head", "layer_shared"]:
            ls = STRATEGY_LS[strategy]
            mk = STRATEGY_MK[strategy]
            grp = sorted([r for r in h2o_records
                          if r["recent_pct"] == rv and r["strategy"] == strategy],
                         key=lambda r: r["budget_pct"])
            if not grp:
                continue
            xs = [r["budget_pct"] for r in grp]
            ys = [r["rouge_l"]    for r in grp]
            all_budgets.extend(xs)
            lbl = f"H2O recent={rv}% ({recent_tok} tok)" if strategy == "per_head" else None
            ax.plot(xs, ys, marker=mk, linestyle=ls,
                    color=h2o_colour[rv], linewidth=1.6, markersize=5, label=lbl)

    rand = sorted([r for r in records if r["method"] == "random"], key=lambda r: r["budget_pct"])
    if rand:
        xs = [r["budget_pct"] for r in rand]
        ys = [r["rouge_l"]    for r in rand]
        all_budgets.extend(xs)
        ax.plot(xs, ys, marker="^", linestyle="-",
                color=RANDOM_COLOUR, linewidth=2.2, markersize=6, label="Random")

    for strategy in ["per_head", "layer_shared"]:
        ls  = STRATEGY_LS[strategy]
        mk  = STRATEGY_MK[strategy]
        local = sorted([r for r in records if r["method"] == "local"
                        and r["strategy"] == strategy],
                       key=lambda r: r["budget_pct"])
        if not local:
            continue
        xs = [r["budget_pct"] for r in local]
        ys = [r["rouge_l"]    for r in local]
        all_budgets.extend(xs)
        ax.plot(xs, ys, marker=mk, linestyle=ls, color=LOCAL_COLOUR, linewidth=2.2, markersize=6,
                label=f"Local ({strategy.replace('_',' ')})")

    if all_budgets:
        ticks = sorted(set(all_budgets))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t}%\n({to_total_budget(t)} tok)" for t in ticks], fontsize=8)
        ax.set_xlim(max(ticks) + 2, min(ticks) - 2)
    ax.set_xlabel("TOTAL_BUDGET  [budget_ratio %  (= max(16, 512 x ratio) tokens)]")
    ax.set_ylabel("ROUGE-L")

    style_handles = [
        Line2D([0], [0], color="gray", linestyle="-",  marker="o", label="solid = per head"),
        Line2D([0], [0], color="gray", linestyle="--", marker="s", label="dashed = layer shared"),
    ]
    dh, dl = ax.get_legend_handles_labels()
    ax.legend(handles=style_handles + dh,
              labels=["solid = per head", "dashed = layer shared"] + dl,
              fontsize=7, ncol=1, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    out = out_dir / "h2o_rouge_vs_budget.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 1b: Coverage (% of full ROUGE-L) ─────────────────────────────────
    if full_rouge:
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.suptitle(f"[{dataset}]  H2O + Local — Coverage (% of Full-Cache ROUGE-L)  "
                     f"(cache_size={CACHE_SIZE})",
                     fontsize=13, fontweight="bold")

        all_budgets = []
        ax.axhline(100, color="black", linestyle=":", linewidth=1.4, label="Full cache (100%)")

        for rv in recent_vals:
            recent_tok = to_recent_budget(rv)
            for strategy in ["per_head", "layer_shared"]:
                ls = STRATEGY_LS[strategy]
                mk = STRATEGY_MK[strategy]
                grp = sorted([r for r in h2o_records
                              if r["recent_pct"] == rv and r["strategy"] == strategy],
                             key=lambda r: r["budget_pct"])
                if not grp:
                    continue
                xs = [r["budget_pct"]                  for r in grp]
                ys = [r["rouge_l"] / full_rouge * 100  for r in grp]
                all_budgets.extend(xs)
                lbl = f"H2O recent={rv}% ({recent_tok} tok)" if strategy == "per_head" else None
                ax.plot(xs, ys, marker=mk, linestyle=ls,
                        color=h2o_colour[rv], linewidth=1.6, markersize=5, label=lbl)

        rand = sorted([r for r in records if r["method"] == "random"], key=lambda r: r["budget_pct"])
        if rand:
            xs = [r["budget_pct"]                 for r in rand]
            ys = [r["rouge_l"] / full_rouge * 100 for r in rand]
            all_budgets.extend(xs)
            ax.plot(xs, ys, marker="^", linestyle="-",
                    color=RANDOM_COLOUR, linewidth=2.2, markersize=6, label="Random")

        for strategy in ["per_head", "layer_shared"]:
            ls  = STRATEGY_LS[strategy]
            mk  = STRATEGY_MK[strategy]
            local = sorted([r for r in records if r["method"] == "local"
                            and r["strategy"] == strategy],
                           key=lambda r: r["budget_pct"])
            if not local:
                continue
            xs = [r["budget_pct"]                  for r in local]
            ys = [r["rouge_l"] / full_rouge * 100  for r in local]
            all_budgets.extend(xs)
            ax.plot(xs, ys, marker=mk, linestyle=ls, color=LOCAL_COLOUR, linewidth=2.2, markersize=6,
                    label=f"Local ({strategy.replace('_',' ')})")

        if all_budgets:
            ticks = sorted(set(all_budgets))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t}%\n({to_total_budget(t)} tok)" for t in ticks], fontsize=8)
            ax.set_xlim(max(ticks) + 2, min(ticks) - 2)
        ax.set_xlabel("TOTAL_BUDGET  [budget_ratio %  (= max(16, 512 x ratio) tokens)]")
        ax.set_ylabel("% of Full-Cache ROUGE-L")

        style_handles = [
            Line2D([0], [0], color="gray", linestyle="-",  marker="o", label="solid = per head"),
            Line2D([0], [0], color="gray", linestyle="--", marker="s", label="dashed = layer shared"),
        ]
        dh, dl = ax.get_legend_handles_labels()
        ax.legend(handles=style_handles + dh,
                  labels=["solid = per head", "dashed = layer shared"] + dl,
                  fontsize=7, ncol=1, loc="upper left",
                  bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        out = out_dir / "h2o_coverage_vs_budget.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")

    # ── Plot 2: Method comparison ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"[{dataset}]  Method Comparison  "
                 f"(solid=per_head  dashed=layer_shared)",
                 fontsize=13, fontweight="bold")

    for ax_idx, (ylabel, transform) in enumerate([
        ("ROUGE-L",         lambda y: y),
        ("% of Full Cache", lambda y: y / full_rouge * 100 if full_rouge else y),
    ]):
        ax = axes[ax_idx]
        all_budgets = []

        if full_rouge:
            ref_y = transform(full_rouge)
            ax.axhline(ref_y, color="black", linestyle=":", linewidth=1.4, label="Full cache")

        rand = sorted([r for r in records if r["method"] == "random"], key=lambda r: r["budget_pct"])
        if rand:
            xs = [r["budget_pct"]         for r in rand]
            ys = [transform(r["rouge_l"]) for r in rand]
            ax.plot(xs, ys, marker="o", linestyle="-", color=METHOD_COLOURS["Random"],
                    linewidth=2, label="Random")
            all_budgets.extend(xs)

        for strategy in ["per_head", "layer_shared"]:
            ls  = STRATEGY_LS[strategy]
            mk  = STRATEGY_MK[strategy]
            strat_label = strategy.replace("_", " ")

            local = sorted([r for r in records if r["method"] == "local"
                            and r["strategy"] == strategy],
                           key=lambda r: r["budget_pct"])
            if local:
                xs = [r["budget_pct"]         for r in local]
                ys = [transform(r["rouge_l"]) for r in local]
                lbl = f"Local ({strat_label})" if ax_idx == 0 else None
                ax.plot(xs, ys, marker=mk, linestyle=ls,
                        color=METHOD_COLOURS["Local (sliding window)"],
                        linewidth=2, label=lbl)
                all_budgets.extend(xs)

            h2o_strat = [r for r in h2o_records if r["strategy"] == strategy]
            best_h2o  = {}
            for r in h2o_strat:
                b = r["budget_pct"]
                if b not in best_h2o or r["rouge_l"] > best_h2o[b]["rouge_l"]:
                    best_h2o[b] = r
            if best_h2o:
                xs = sorted(best_h2o.keys())
                ys = [transform(best_h2o[b]["rouge_l"]) for b in xs]
                lbl = f"H2O best ({strat_label})" if ax_idx == 0 else None
                ax.plot(xs, ys, marker=mk, linestyle=ls,
                        color=METHOD_COLOURS["H2O best"],
                        linewidth=2, label=lbl)
                all_budgets.extend(xs)

        if all_budgets:
            set_xaxis(ax, all_budgets)
        ax.set_xlabel("Budget % ->  (100% = full cache)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if ax_idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "method_comparison_rouge_vs_budget.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 2b: Coverage ─────────────────────────────────────────────────────
    if full_rouge:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"[{dataset}]  Coverage: % of Full-Cache ROUGE-L Retained by Method",
                     fontsize=13, fontweight="bold")

        all_budgets = []
        rand = sorted([r for r in records if r["method"] == "random"], key=lambda r: r["budget_pct"])
        if rand:
            xs = [r["budget_pct"]                 for r in rand]
            ys = [r["rouge_l"] / full_rouge * 100 for r in rand]
            ax.plot(xs, ys, marker="^", linestyle="-", color=METHOD_COLOURS["Random"],
                    linewidth=2, markersize=6, label="Random")
            all_budgets.extend(xs)

        for strategy in ["per_head", "layer_shared"]:
            ls  = STRATEGY_LS[strategy]
            mk  = STRATEGY_MK[strategy]
            strat_label = strategy.replace("_", " ")

            local = sorted([r for r in records if r["method"] == "local"
                            and r["strategy"] == strategy],
                           key=lambda r: r["budget_pct"])
            if local:
                xs = [r["budget_pct"]                  for r in local]
                ys = [r["rouge_l"] / full_rouge * 100  for r in local]
                ax.plot(xs, ys, marker=mk, linestyle=ls,
                        color=METHOD_COLOURS["Local (sliding window)"],
                        linewidth=2, markersize=6, label=f"Local ({strat_label})")
                all_budgets.extend(xs)

            h2o_strat = [r for r in h2o_records if r["strategy"] == strategy]
            best_h2o  = {}
            for r in h2o_strat:
                b = r["budget_pct"]
                if b not in best_h2o or r["rouge_l"] > best_h2o[b]["rouge_l"]:
                    best_h2o[b] = r
            if best_h2o:
                xs = sorted(best_h2o.keys())
                ys = [best_h2o[b]["rouge_l"] / full_rouge * 100 for b in xs]
                ax.plot(xs, ys, marker=mk, linestyle=ls,
                        color=METHOD_COLOURS["H2O best"],
                        linewidth=2, markersize=6, label=f"H2O best ({strat_label})")
                all_budgets.extend(xs)

        ax.axhline(100, color="black", linestyle=":", linewidth=1.4, label="Full cache (100%)")
        if all_budgets:
            set_xaxis(ax, all_budgets)
        ax.set_xlabel("Budget %")
        ax.set_ylabel("% of Full-Cache ROUGE-L")
        ax.set_title("solid = per head   |   dashed = layer shared", fontsize=9, color="gray")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = out_dir / "coverage_vs_budget.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")

    # ── Plot 3: Latency & Memory ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle(f"[{dataset}]  H2O + Local KV Cache — Latency & Memory  "
                 f"(cache_size={CACHE_SIZE})",
                 fontsize=13, fontweight="bold")

    for ax_idx, (ylabel, key, ref_val) in enumerate([
        ("Avg Latency (s)",   "avg_latency",  full_latency),
        ("Avg mem_peak (MB)", "avg_mem_peak", full_mem_peak),
    ]):
        ax = axes[ax_idx]
        all_budgets = []

        for rv in recent_vals:
            recent_tok = to_recent_budget(rv)
            for strategy in ["per_head", "layer_shared"]:
                ls = STRATEGY_LS[strategy]
                mk = STRATEGY_MK[strategy]
                grp = sorted([r for r in h2o_records
                              if r["recent_pct"] == rv and r["strategy"] == strategy
                              and r[key] is not None],
                             key=lambda r: r["budget_pct"])
                if not grp:
                    continue
                xs = [r["budget_pct"] for r in grp]
                ys = [r[key]          for r in grp]
                all_budgets.extend(xs)
                lbl = (f"H2O recent={rv}% ({recent_tok} tok)"
                       if strategy == "per_head" and ax_idx == 0 else None)
                ax.plot(xs, ys, marker=mk, linestyle=ls,
                        color=h2o_colour.get(rv, "gray"), linewidth=1.6, markersize=5, label=lbl)

        for strategy in ["per_head", "layer_shared"]:
            ls  = STRATEGY_LS[strategy]
            mk  = STRATEGY_MK[strategy]
            local = sorted([r for r in records if r["method"] == "local"
                            and r["strategy"] == strategy and r[key] is not None],
                           key=lambda r: r["budget_pct"])
            if not local:
                continue
            xs = [r["budget_pct"] for r in local]
            ys = [r[key]          for r in local]
            all_budgets.extend(xs)
            lbl = f"Local ({strategy.replace('_',' ')})" if ax_idx == 0 else None
            ax.plot(xs, ys, marker=mk, linestyle=ls,
                    color=LOCAL_COLOUR, linewidth=2.2, markersize=6, label=lbl)

        if all_budgets:
            ticks = sorted(set(all_budgets))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t}%\n({to_total_budget(t)} tok)" for t in ticks], fontsize=8)
            ax.set_xlim(max(ticks) + 2, min(ticks) - 2)
        all_vals = [r[key] for r in records if r[key] is not None and r["method"] != "full"]
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            pad = (hi - lo) * 0.3 or 0.05
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("TOTAL_BUDGET  [budget_ratio %]")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)

    style_handles = [
        Line2D([0], [0], color="gray", linestyle="-",  marker="o", label="solid = per head"),
        Line2D([0], [0], color="gray", linestyle="--", marker="s", label="dashed = layer shared"),
    ]
    dh, dl = axes[0].get_legend_handles_labels()
    axes[1].legend(handles=style_handles + dh,
                   labels=["solid = per head", "dashed = layer shared"] + dl,
                   fontsize=7, ncol=1, loc="upper left",
                   bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    out = out_dir / "latency_memory_vs_budget.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 4a: Horizontal bar — all configs ranked ───────────────────────────
    sorted_recs = sorted(records, key=lambda r: r["rouge_l"], reverse=True)
    labels  = [f"{r['mode'][:12]}\n{r['strategy'][:3]}" for r in sorted_recs]
    vals    = [r["rouge_l"] for r in sorted_recs]
    colours = ["steelblue"  if r["method"] == "h2o"    else
               "darkorange" if r["method"] == "local"  else
               "crimson"    if r["method"] == "full"   else "gray"
               for r in sorted_recs]
    ypos = range(len(vals))
    fig, ax = plt.subplots(figsize=(10, max(6, len(vals) * 0.18)))
    ax.barh(list(ypos), vals, color=colours, height=0.7)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("ROUGE-L Score")
    ax.set_title(f"[{dataset}]  ROUGE-L Comparison Across All Methods",
                 fontsize=11, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=4)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = out_dir / "rouge_all_configs_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 4b: Avg ROUGE-L by method group ──────────────────────────────────
    groups = {
        "baseline (full)": [r["rouge_l"] for r in records if r["method"] == "full"],
        "h2o":             [r["rouge_l"] for r in records if r["method"] == "h2o"],
        "local":           [r["rouge_l"] for r in records if r["method"] == "local"],
        "random":          [r["rouge_l"] for r in records if r["method"] == "random"],
    }
    grp_colours = {"baseline (full)": "teal", "h2o": "yellowgreen",
                   "local": "silver", "random": "steelblue"}
    grp_labels  = list(groups.keys())
    grp_means   = [np.mean(v) if v else 0 for v in groups.values()]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(grp_labels, grp_means, color=[grp_colours[g] for g in grp_labels])
    for bar, val in zip(bars, grp_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Average ROUGE-L Score")
    ax.set_title(f"[{dataset}]  Average ROUGE-L by Method Type",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = out_dir / "rouge_avg_by_method.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 4c: Boxplot by strategy ──────────────────────────────────────────
    strat_data = {s: [r["rouge_l"] for r in records if r.get("strategy") == s]
                  for s in ["layer_shared", "per_head"]}
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([strat_data["layer_shared"], strat_data["per_head"]],
               labels=["layer_shared", "per_head"],
               patch_artist=True,
               boxprops=dict(facecolor="salmon", alpha=0.6),
               medianprops=dict(color="darkred", linewidth=2))
    ax.set_ylabel("ROUGE-L Score")
    ax.set_title(f"[{dataset}]  ROUGE-L Distribution by Strategy",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = out_dir / "rouge_boxplot_by_strategy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 4d: Scatter — H2O budget vs ROUGE-L ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, colour in [("layer_shared", "steelblue"), ("per_head", "darkorange")]:
        grp = [r for r in h2o_records if r["strategy"] == strategy]
        xs  = [f"b{r['budget_pct']//10}" for r in grp]
        ys  = [r["rouge_l"] for r in grp]
        ax.scatter(xs, ys, color=colour, label=strategy, s=50, alpha=0.8)
    ax.set_xlabel("Budget Block (b)")
    ax.set_ylabel("ROUGE-L Score")
    ax.set_title(f"[{dataset}]  H2O: Budget vs ROUGE-L Score",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "rouge_h2o_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 5: Recent-ratio lines ─────────────────────────────────────────────
    budget_levels = sorted(set(r["budget_pct"] for r in h2o_records))
    cmap_b   = plt.colormaps["viridis"].resampled(max(len(budget_levels), 1))
    b_colour = {b: cmap_b(i) for i, b in enumerate(budget_levels)}

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.suptitle(f"[{dataset}]  H2O (All Strategies): ROUGE-L vs Recent Ratio\n"
                 f"(Solid=per_head, Dash=layer_shared)",
                 fontsize=12, fontweight="bold")

    if full_rouge:
        ax.axhline(full_rouge, color="red", linestyle="--", linewidth=1.8,
                   label=f"Baseline: {full_rouge:.4f}")

    for b in budget_levels:
        blabel = f"b{b//10}"
        for strategy, ls in [("per_head", "-"), ("layer_shared", "--")]:
            pts = sorted([r for r in h2o_records
                          if r["budget_pct"] == b and r["strategy"] == strategy
                          and r["recent_pct"] <= b],
                         key=lambda r: r["recent_pct"])
            if not pts:
                continue
            xs  = [f"{r['recent_pct']}%" for r in pts]
            ys  = [r["rouge_l"]          for r in pts]
            mk  = "o" if strategy == "per_head" else "s"
            lbl = blabel if strategy == "per_head" else None
            ax.plot(xs, ys, marker=mk, linestyle=ls, color=b_colour[b],
                    linewidth=1.8, markersize=6, label=lbl)

    ax.set_xlabel("Recent Ratio (%)")
    ax.set_ylabel("ROUGE-L Score")
    ax.grid(True, alpha=0.3)

    strat_legend = [
        Line2D([0], [0], color="gray", ls="-",  marker="o", label="per_head"),
        Line2D([0], [0], color="gray", ls="--", marker="s", label="layer_shared"),
        Line2D([0], [0], color="red",  ls="--",
               label=f"Baseline: {full_rouge:.4f}" if full_rouge else "Baseline"),
    ]
    budget_handles = [Line2D([0], [0], color=b_colour[b], marker="o", ls="-",
                             label=f"b{b//10}") for b in budget_levels]
    leg1 = ax.legend(handles=budget_handles, title="Budget Level",
                     fontsize=8, loc="upper left", bbox_to_anchor=(0, 1))
    ax.add_artist(leg1)
    ax.legend(handles=strat_legend, title="Strategy / Reference",
              fontsize=8, loc="upper right")

    plt.tight_layout()
    out = out_dir / "rouge_recent_ratio_lines.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # ── Plot 6: Heatmap — budget x recent ─────────────────────────────────────
    for strategy in ["per_head", "layer_shared"]:
        b_vals = sorted(set(r["budget_pct"] for r in h2o_records))
        r_vals = sorted(set(r["recent_pct"] for r in h2o_records))
        mat    = np.full((len(b_vals), len(r_vals)), np.nan)
        lookup = {(r["budget_pct"], r["recent_pct"]): r["rouge_l"]
                  for r in h2o_records if r["strategy"] == strategy}
        for bi, b in enumerate(b_vals):
            for ri, rv in enumerate(r_vals):
                if rv <= b:
                    mat[bi, ri] = lookup.get((b, rv), np.nan)

        cmap_hm = mcolors.LinearSegmentedColormap.from_list(
            "rg", ["#d73027", "#fee08b", "#1a9850"])
        vmin_v = np.nanmin(mat) if not np.all(np.isnan(mat)) else 0
        vmax_v = np.nanmax(mat) if not np.all(np.isnan(mat)) else 1

        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(mat, cmap=cmap_hm, vmin=vmin_v, vmax=vmax_v, aspect="auto")
        plt.colorbar(im, ax=ax, label="ROUGE-L Score")
        ax.set_xticks(range(len(r_vals)))
        ax.set_xticklabels([f"r{rv//10}" for rv in r_vals])
        ax.set_yticks(range(len(b_vals)))
        ax.set_yticklabels([f"b{b//10}" for b in b_vals])
        ax.set_xlabel("Recent Ratio (r)")
        ax.set_ylabel("Budget Ratio (b)")
        ax.set_title(f"[{dataset}]  H2O {strategy.replace('_',' ').title()}: "
                     f"ROUGE-L Score Heatmap",
                     fontsize=11, fontweight="bold")
        for bi in range(len(b_vals)):
            for ri in range(len(r_vals)):
                v = mat[bi, ri]
                if not np.isnan(v):
                    colour = "white" if v < (vmin_v + (vmax_v - vmin_v) * 0.4) else "black"
                    ax.text(ri, bi, f"{v:.3f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=colour)
        plt.tight_layout()
        out = out_dir / f"rouge_budget_heatmap_{strategy}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")


# ── Cross-dataset comparison plot ─────────────────────────────────────────────
def plot_cross_dataset(records_by_dataset: dict):
    """One subplot per dataset — H2O best vs Local vs Random vs Full baseline."""
    datasets = sorted(records_by_dataset.keys())
    n = len(datasets)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Cross-Dataset: H2O Best vs Local vs Random vs Full  (ROUGE-L)",
                 fontsize=13, fontweight="bold")

    for ax, dataset in zip(axes, datasets):
        records    = records_by_dataset[dataset]
        h2o_recs   = [r for r in records if r["method"] == "h2o"]
        full_rouge = next((r["rouge_l"] for r in records if r["method"] == "full"), None)

        if full_rouge:
            ax.axhline(full_rouge, color="black", linestyle=":", linewidth=1.4, label="Full")

        rand = sorted([r for r in records if r["method"] == "random"],
                      key=lambda r: r["budget_pct"])
        if rand:
            ax.plot([r["budget_pct"] for r in rand],
                    [r["rouge_l"]    for r in rand],
                    marker="^", linestyle="-", color=RANDOM_COLOUR,
                    linewidth=1.8, markersize=5, label="Random")

        for strategy in ["per_head", "layer_shared"]:
            ls  = STRATEGY_LS[strategy]
            mk  = STRATEGY_MK[strategy]
            local = sorted([r for r in records if r["method"] == "local"
                            and r["strategy"] == strategy],
                           key=lambda r: r["budget_pct"])
            if local:
                ax.plot([r["budget_pct"] for r in local],
                        [r["rouge_l"]    for r in local],
                        marker=mk, linestyle=ls, color=LOCAL_COLOUR,
                        linewidth=1.8, markersize=5,
                        label=f"Local {strategy[:3]}")

            best_h2o = {}
            for r in [r for r in h2o_recs if r["strategy"] == strategy]:
                b = r["budget_pct"]
                if b not in best_h2o or r["rouge_l"] > best_h2o[b]["rouge_l"]:
                    best_h2o[b] = r
            if best_h2o:
                xs = sorted(best_h2o.keys())
                ax.plot(xs, [best_h2o[b]["rouge_l"] for b in xs],
                        marker=mk, linestyle=ls,
                        color=METHOD_COLOURS["H2O best"],
                        linewidth=1.8, markersize=5,
                        label=f"H2O {strategy[:3]}")

        all_b = [r["budget_pct"] for r in records if r["budget_pct"] is not None]
        if all_b:
            ticks = sorted(set(all_b))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t}%" for t in ticks], fontsize=7, rotation=45)
            ax.set_xlim(max(ticks) + 2, min(ticks) - 2)
        ax.set_title(dataset.replace("_", "\n"), fontsize=10, fontweight="bold")
        ax.set_xlabel("Budget %")
        ax.set_ylabel("ROUGE-L")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "cross_dataset_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved cross-dataset plot: {out}")


# ── Run ───────────────────────────────────────────────────────────────────────
for dataset, records in sorted(records_by_dataset.items()):
    plot_dataset(dataset, records)

plot_cross_dataset(records_by_dataset)

print(f"\nAll plots saved under: {OUTPUT_DIR}")

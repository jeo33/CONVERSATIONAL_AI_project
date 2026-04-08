#!/usr/bin/env python3
"""
compare_predictions.py

Reads all .out files in LOGS/ (same directory used by plot_rouge.py) and
extracts the quick-test prediction (sample 0) printed at the start of each job.

Outputs:
  plots/prediction_comparison_sample0.png   — bar chart of ROUGE-L per config
  plots/prediction_comparison_sample0.txt   — text report with full predictions

Usage:
  python script/compare_predictions.py
  python script/compare_predictions.py --logs-dir /path/to/LOGS
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_LOGS_DIR = Path(__file__).parent.parent / "LOGS"
OUTPUT_DIR       = Path(__file__).parent.parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Compare quick-test predictions across KV configs")
parser.add_argument("--logs-dir", type=str, default=None,
                    help=f"Directory of .out files (default: {DEFAULT_LOGS_DIR})")
args = parser.parse_args()

logs_dir = Path(args.logs_dir) if args.logs_dir else DEFAULT_LOGS_DIR
if not logs_dir.exists():
    print(f"ERROR: logs_dir not found: {logs_dir}")
    sys.exit(1)

# ── Regexes ───────────────────────────────────────────────────────────────────
FILE_RE    = re.compile(r"^h2o_(.+)_(per_head|layer_shared)_\d+\.out$")
REF_RE     = re.compile(r"Reference:\s+(.+)")
PRED_RE    = re.compile(r"Prediction raw: '(.+)'")
ROUGE_RE   = re.compile(r"ROUGE-L:\s+([\d.]+)\s+✓")

# ── Parse .out files ──────────────────────────────────────────────────────────
entries = []
reference_text = None

for f in sorted(logs_dir.glob("*.out")):
    m = FILE_RE.match(f.name)
    if not m:
        continue
    mode_raw, strategy = m.group(1), m.group(2)

    text = f.read_text(errors="replace")

    ref_m  = REF_RE.search(text)
    pred_m = PRED_RE.search(text)
    rl_m   = ROUGE_RE.search(text)

    if not (ref_m and pred_m and rl_m):
        print(f"  [SKIP] incomplete output in {f.name}")
        continue

    reference   = ref_m.group(1).strip()
    prediction  = pred_m.group(1).strip()
    rouge_l     = float(rl_m.group(1))

    if reference_text is None:
        reference_text = reference

    # Parse mode label
    if mode_raw == "full":
        method = "full"
    elif mode_raw.startswith("random"):
        method = "random"
    elif mode_raw.startswith("local"):
        method = "local"
    else:
        method = "h2o"

    entries.append({
        "label":      f"{mode_raw}\n{strategy.replace('_', ' ')}",
        "mode":       mode_raw,
        "method":     method,
        "strategy":   strategy,
        "rouge_l":    rouge_l,
        "prediction": prediction,
        "reference":  reference,
    })
    print(f"  {mode_raw:30s} {strategy:14s}  ROUGE-L={rouge_l:.4f}")

if not entries:
    print("No entries found — check logs_dir.")
    sys.exit(1)

print(f"\nLoaded {len(entries)} configs\n")

# ── Sort ──────────────────────────────────────────────────────────────────────
METHOD_RANK = {"full": 0, "random": 1, "local": 2, "h2o": 3}

def sort_key(e):
    return (METHOD_RANK.get(e["method"], 9), e["strategy"], e["mode"])

entries.sort(key=sort_key)

# ── Bar chart ─────────────────────────────────────────────────────────────────
METHOD_COLOUR = {
    "full":   "black",
    "random": "steelblue",
    "local":  "darkorange",
    "h2o":    "crimson",
}
HATCH = {"per_head": "", "layer_shared": "//"}

n = len(entries)
fig, ax = plt.subplots(figsize=(max(14, n * 0.55), 6))

xs     = np.arange(n)
rouges = [e["rouge_l"] for e in entries]
colours = [METHOD_COLOUR[e["method"]] for e in entries]
hatches = [HATCH[e["strategy"]] for e in entries]

bars = ax.bar(xs, rouges, color=colours, edgecolor="white", linewidth=0.5)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# Value labels on top
for bar, val in zip(bars, rouges):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

ax.set_xticks(xs)
ax.set_xticklabels([e["label"] for e in entries], fontsize=6.5, rotation=45, ha="right")
ax.set_ylabel("ROUGE-L")
ax.set_title(
    f"ROUGE-L per Config — Quick-test sample (validation[0])\n"
    f"Reference: \"{reference_text[:120]}...\"",
    fontsize=9
)
ax.set_ylim(0, max(rouges) * 1.18)
ax.grid(axis="y", alpha=0.3)

legend_handles = [
    mpatches.Patch(facecolor=METHOD_COLOUR["full"],   label="Full cache"),
    mpatches.Patch(facecolor=METHOD_COLOUR["random"], label="Random"),
    mpatches.Patch(facecolor=METHOD_COLOUR["local"],  label="Local"),
    mpatches.Patch(facecolor=METHOD_COLOUR["h2o"],    label="H2O"),
    mpatches.Patch(facecolor="white", edgecolor="gray", label="solid = per head"),
    mpatches.Patch(facecolor="white", edgecolor="gray", hatch="//", label="hatch = layer shared"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

plt.tight_layout()
chart_path = OUTPUT_DIR / "prediction_comparison_sample0.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {chart_path}")

# ── Text report ───────────────────────────────────────────────────────────────
txt_path = OUTPUT_DIR / "prediction_comparison_sample0.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("SAMPLE COMPARISON  —  validation[0]  (quick-test sample from each job)\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"REFERENCE:\n  {reference_text}\n\n")
    f.write("-" * 80 + "\n")
    for e in entries:
        f.write(f"\nCONFIG:  {e['mode']}  |  strategy={e['strategy']}\n")
        f.write(f"ROUGE-L: {e['rouge_l']:.4f}\n")
        f.write(f"PREDICTION:\n  {e['prediction']}\n")
        f.write("-" * 80 + "\n")

print(f"Saved: {txt_path}")
print(f"\nDone. {len(entries)} configs compared.")


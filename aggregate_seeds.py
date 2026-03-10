#!/usr/bin/env python3
"""
Aggregate results across seeds and produce paper-quality figures.

Usage:
    python aggregate_seeds.py --results_dir results --n_seeds 5

Expects:  results/seed_0/all_results.json
          results/seed_1/all_results.json
          ...
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="results")
parser.add_argument("--n_seeds", type=int, default=5)
parser.add_argument("--outdir", type=str, default="figures")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ── load all seeds ─────────────────────────────────────────────

seeds_data = []
for s in range(args.n_seeds):
    path = os.path.join(args.results_dir, f"seed_{s}", "all_results.json")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        continue
    with open(path) as f:
        seeds_data.append(json.load(f))

n_seeds = len(seeds_data)
print(f"Loaded {n_seeds} seeds")
assert n_seeds >= 2, "Need at least 2 seeds for error bars"


# ── helper: extract metric across seeds ────────────────────────

def get_metric(key_path, seeds=seeds_data):
    """Extract a metric from all seeds given a dot-separated key path."""
    vals = []
    for sd in seeds:
        obj = sd
        for k in key_path:
            obj = obj[k]
        vals.append(obj)
    return np.array(vals)

def mean_sem(arr):
    return np.mean(arr), np.std(arr) / np.sqrt(len(arr))


# ════════════════════════════════════════════════════════════════
# FIGURE 1: Main perturbation results (6-panel)
# ════════════════════════════════════════════════════════════════

COLORS = {
    "Intact": "#2ca02c", "SOM_suppressed": "#d62728",
    "VIP_suppressed": "#1f77b4", "Linear_SOM": "#ff7f0e",
    "PV_weakened": "#9467bd", "SOM_VIP_off": "#8c564b",
    "Baseline": "#7f7f7f",
}

LABELS = {
    "Intact": "Intact", "SOM_suppressed": "SOM suppressed",
    "VIP_suppressed": "VIP suppressed", "Linear_SOM": "Linear SOM",
    "PV_weakened": "PV weakened", "SOM_VIP_off": "SOM+VIP off",
    "Baseline": "Baseline",
}

main_conds = ["Intact", "SOM_suppressed", "VIP_suppressed", "PV_weakened"]
all_conds = ["Intact", "SOM_suppressed", "VIP_suppressed", "Linear_SOM",
             "PV_weakened", "SOM_VIP_off", "Baseline"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Pure-SSN Model: Interneuron Perturbation Results\n"
             f"(mean ± SEM, n={n_seeds} seeds)",
             fontsize=14, fontweight="bold")

# A) Psychometric curves
ax = axes[0, 0]
for c in main_conds + ["Baseline"]:
    psych_all = np.array([sd["perturbation"][c]["psych"] for sd in seeds_data])
    m = psych_all.mean(axis=0)
    se = psych_all.std(axis=0) / np.sqrt(n_seeds)
    ax.plot(range(8), m, "o-", color=COLORS[c], label=LABELS[c], lw=2)
    ax.fill_between(range(8), m - se, m + se, color=COLORS[c], alpha=0.15)
ax.axvline(3.5, ls="--", color="gray", alpha=0.4)
ax.set_xlabel("Cue step k"); ax.set_ylabel("P(PA | cue_k)")
ax.set_title("A) Categorical sharpness"); ax.legend(fontsize=7)

# B) Context length curve
ax = axes[0, 1]
for c in main_conds + ["Baseline"]:
    accL_all = np.array([sd["perturbation"][c]["accL"] for sd in seeds_data])
    Ls = seeds_data[0]["perturbation"][c]["Ls"]
    m = accL_all.mean(axis=0)
    se = accL_all.std(axis=0) / np.sqrt(n_seeds)
    ax.plot(Ls, m, "s-", color=COLORS[c], label=LABELS[c], lw=2)
    ax.fill_between(Ls, m - se, m + se, color=COLORS[c], alpha=0.15)
ax.set_xlabel("Truncation length L"); ax.set_ylabel("Accuracy (B2)")
ax.set_title("B) Temporal integration"); ax.legend(fontsize=7)

# C) Intrusion curve
ax = axes[0, 2]
for c in main_conds + ["Baseline"]:
    acclag_all = np.array([sd["perturbation"][c]["acclag"] for sd in seeds_data])
    lags = seeds_data[0]["perturbation"][c]["lags"]
    m = acclag_all.mean(axis=0)
    se = acclag_all.std(axis=0) / np.sqrt(n_seeds)
    ax.plot(lags, m, "^-", color=COLORS[c], label=LABELS[c], lw=2)
    ax.fill_between(lags, m - se, m + se, color=COLORS[c], alpha=0.15)
ax.set_xlabel("Distractor lag"); ax.set_ylabel("Accuracy (B2)")
ax.set_title("C) Context intrusion"); ax.legend(fontsize=7)

# D) Dissociation scatter
ax = axes[1, 0]
for c in all_conds:
    benefits = np.array([sd["perturbation"][c]["benefit"] for sd in seeds_data])
    intrusions = np.array([sd["perturbation"][c]["intrusion_idx"] for sd in seeds_data])
    bm, bse = mean_sem(benefits)
    im, ise = mean_sem(intrusions)
    ax.errorbar(bm, im, xerr=bse, yerr=ise, fmt="o", color=COLORS[c],
                markersize=8, capsize=3, zorder=5)
    ax.annotate(LABELS[c], (bm, im), fontsize=6, xytext=(5, 5),
                textcoords="offset points")
ax.set_xlabel("Context benefit (B2 − B1)")
ax.set_ylabel("Intrusion susceptibility")
ax.set_title("D) Double dissociation"); ax.grid(alpha=0.2)
ax.axhline(0, ls=":", color="gray", alpha=0.3)
ax.axvline(0, ls=":", color="gray", alpha=0.3)

# E) SOM nonlinearity sweep
ax = axes[1, 1]
n_vals = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
ben_all = np.array([[sd["som_sweep"][str(nv)]["benefit"] for nv in n_vals]
                     for sd in seeds_data])
intr_all = np.array([[sd["som_sweep"][str(nv)]["intrusion"] for nv in n_vals]
                      for sd in seeds_data])
bm, bse = ben_all.mean(0), ben_all.std(0) / np.sqrt(n_seeds)
im, ise = intr_all.mean(0), intr_all.std(0) / np.sqrt(n_seeds)
ax.errorbar(n_vals, bm, yerr=bse, fmt="o-", color="#d62728",
            label="Context benefit", lw=2, capsize=3)
ax.errorbar(n_vals, im, yerr=ise, fmt="s--", color="#1f77b4",
            label="Intrusion suscept.", lw=2, capsize=3)
ax.set_xlabel("SOM exponent n_S"); ax.set_ylabel("Metric")
ax.set_title("E) SOM nonlinearity sweep"); ax.legend(fontsize=8)

# F) Bar chart: B1 vs B2
ax = axes[1, 2]
bar_conds = ["Intact", "SOM_suppressed", "VIP_suppressed", "PV_weakened", "Baseline"]
xp = np.arange(len(bar_conds))
b1_all = np.array([[sd["perturbation"][c]["b1"] for c in bar_conds] for sd in seeds_data])
b2_all = np.array([[sd["perturbation"][c]["b2"] for c in bar_conds] for sd in seeds_data])
w = 0.35
ax.bar(xp - w/2, b1_all.mean(0), w, yerr=b1_all.std(0)/np.sqrt(n_seeds),
       label="B1 (local)", color="#aec7e8", capsize=3)
ax.bar(xp + w/2, b2_all.mean(0), w, yerr=b2_all.std(0)/np.sqrt(n_seeds),
       label="B2 (context)", color="#ffbb78", capsize=3)
ax.set_xticks(xp)
ax.set_xticklabels([LABELS[c] for c in bar_conds], rotation=30, ha="right", fontsize=7)
ax.set_ylabel("Accuracy"); ax.set_ylim(0.0, 1.0)
ax.set_title("F) B1 vs B2 by condition"); ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(args.outdir, "fig1_main_results.png"),
            dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(args.outdir, "fig1_main_results.pdf"),
            bbox_inches="tight")
plt.show()
print("Saved fig1_main_results")


# ════════════════════════════════════════════════════════════════
# FIGURE 2: SOM trace decodability
# ════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

trace_conds = ["Intact", "SOM_suppressed", "VIP_suppressed", "Linear_SOM"]
for c in trace_conds:
    decode_all = []
    for sd in seeds_data:
        decode_all.append(sd["som_trace"][c]["decodability"])
    # pad to same length
    max_len = max(len(d) for d in decode_all)
    padded = np.full((len(decode_all), max_len), np.nan)
    for i, d in enumerate(decode_all):
        padded[i, :len(d)] = d
    m = np.nanmean(padded, axis=0)
    se = np.nanstd(padded, axis=0) / np.sqrt(np.sum(~np.isnan(padded), axis=0))
    pos = np.arange(max_len)
    ax.plot(pos, m, "o-", color=COLORS[c], label=LABELS[c], lw=2, markersize=4)
    ax.fill_between(pos, m - se, m + se, color=COLORS[c], alpha=0.15)

ax.axhline(0.5, ls="--", color="gray", alpha=0.4, label="Chance")
ax.set_xlabel("Sequence position")
ax.set_ylabel("MODE decodability from SOM trace")
ax.set_title(f"MODE Information in SOM Carry-Over State\n"
             f"(mean ± SEM, n={n_seeds} seeds)")
ax.legend(fontsize=9); ax.set_ylim(0.4, 1.05); ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "fig2_som_trace.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(args.outdir, "fig2_som_trace.pdf"), bbox_inches="tight")
plt.show()
print("Saved fig2_som_trace")


# ════════════════════════════════════════════════════════════════
# FIGURE 3: Per-task evaluation
# ════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

pt_conds = ["Intact", "SOM_suppressed", "VIP_suppressed",
            "Linear_SOM", "PV_weakened", "Baseline"]
tasks = ["A", "B1", "B2"]
task_colors = {"A": "#66c2a5", "B1": "#aec7e8", "B2": "#ffbb78"}

xp = np.arange(len(pt_conds))
w = 0.25
for ti, task in enumerate(tasks):
    vals = np.array([[sd["per_task"][c][task] for c in pt_conds] for sd in seeds_data])
    m, se = vals.mean(0), vals.std(0) / np.sqrt(n_seeds)
    ax.bar(xp + (ti - 1) * w, m, w, yerr=se,
           label=f"Task {task}", color=task_colors[task], capsize=3)

ax.set_xticks(xp)
ax.set_xticklabels([LABELS[c] for c in pt_conds], rotation=25, ha="right")
ax.set_ylabel("Accuracy"); ax.set_ylim(0.4, 1.0)
ax.set_title(f"Per-Task Accuracy by Condition (n={n_seeds} seeds)")
ax.legend(); ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "fig3_per_task.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(args.outdir, "fig3_per_task.pdf"), bbox_inches="tight")
plt.show()
print("Saved fig3_per_task")


# ════════════════════════════════════════════════════════════════
# FIGURE 4: Sequence length analysis
# ════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

sl_conds = ["Intact", "SOM_suppressed", "VIP_suppressed", "Linear_SOM", "Baseline"]
bin_keys = ["short_8_12", "medium_13_17", "long_18_22", "vlong_23p"]
bin_labels = ["Short\n(8-12)", "Medium\n(13-17)", "Long\n(18-22)", "V.Long\n(23+)"]

for c in sl_conds:
    accs_per_seed = []
    for sd in seeds_data:
        row = []
        for bk in bin_keys:
            v = sd["seq_length"][c][bk]["acc"]
            row.append(v if v is not None else np.nan)
        accs_per_seed.append(row)
    arr = np.array(accs_per_seed)
    m = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
    valid = ~np.isnan(m)
    ax.errorbar(np.arange(len(bin_keys))[valid], m[valid], yerr=se[valid],
                fmt="o-", color=COLORS[c], label=LABELS[c], lw=2,
                markersize=8, capsize=3)

ax.set_xticks(range(len(bin_labels)))
ax.set_xticklabels(bin_labels)
ax.set_xlabel("Sequence Length"); ax.set_ylabel("B2 Accuracy")
ax.set_title(f"B2 Accuracy vs Sequence Length (n={n_seeds} seeds)")
ax.legend(); ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "fig4_seq_length.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(args.outdir, "fig4_seq_length.pdf"), bbox_inches="tight")
plt.show()
print("Saved fig4_seq_length")


# ════════════════════════════════════════════════════════════════
# FIGURE 5: Graded gain sweeps (dose-response)
# ════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Graded Perturbation Dose-Response (n={n_seeds} seeds)",
             fontsize=13, fontweight="bold")

# SOM gain
ax = axes[0]
k_vals = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04]
for metric, label, col, ls in [("benefit", "Context benefit", "#d62728", "-"),
                                 ("intrusion", "Intrusion suscept.", "#1f77b4", "--")]:
    vals = np.array([[sd["som_gain_sweep"][str(kv)][metric] for kv in k_vals]
                      for sd in seeds_data])
    m, se = vals.mean(0), vals.std(0) / np.sqrt(n_seeds)
    ax.errorbar(k_vals, m, yerr=se, fmt="o" + ls, color=col,
                label=label, lw=2, capsize=3)
ax.set_xlabel("k_SOM (gain)"); ax.set_ylabel("Metric")
ax.set_title("SOM gain sweep"); ax.legend()

# VIP gain
ax = axes[1]
for metric, label, col, ls in [("benefit", "Context benefit", "#d62728", "-"),
                                 ("intrusion", "Intrusion suscept.", "#1f77b4", "--")]:
    vals = np.array([[sd["vip_gain_sweep"][str(kv)][metric] for kv in k_vals]
                      for sd in seeds_data])
    m, se = vals.mean(0), vals.std(0) / np.sqrt(n_seeds)
    ax.errorbar(k_vals, m, yerr=se, fmt="o" + ls, color=col,
                label=label, lw=2, capsize=3)
ax.set_xlabel("k_VIP (gain)"); ax.set_ylabel("Metric")
ax.set_title("VIP gain sweep"); ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(args.outdir, "fig5_gain_sweeps.png"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(args.outdir, "fig5_gain_sweeps.pdf"), bbox_inches="tight")
plt.show()
print("Saved fig5_gain_sweeps")


# ════════════════════════════════════════════════════════════════
# SUMMARY TABLE (for paper)
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("SUMMARY TABLE FOR PAPER")
print("=" * 100)
print(f"\n{'Condition':<20s} {'B1':>12s} {'B2':>12s} {'Benefit':>14s} "
      f"{'Intrusion':>14s}")
print("-" * 75)

for c in all_conds:
    b1s = np.array([sd["perturbation"][c]["b1"] for sd in seeds_data])
    b2s = np.array([sd["perturbation"][c]["b2"] for sd in seeds_data])
    bens = np.array([sd["perturbation"][c]["benefit"] for sd in seeds_data])
    intrs = np.array([sd["perturbation"][c]["intrusion_idx"] for sd in seeds_data])

    print(f"{LABELS[c]:<20s} "
          f"{b1s.mean():.3f}±{b1s.std()/np.sqrt(n_seeds):.3f} "
          f"{b2s.mean():.3f}±{b2s.std()/np.sqrt(n_seeds):.3f} "
          f"{bens.mean():+.3f}±{bens.std()/np.sqrt(n_seeds):.3f} "
          f"{intrs.mean():+.3f}±{intrs.std()/np.sqrt(n_seeds):.3f}")

print("\nDone. All figures saved to", args.outdir)

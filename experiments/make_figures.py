"""Generate all paper figures from the results/*.csv and results/*.json
artifacts produced by run_main_experiment.py, run_barren_plateau.py, and
run_depth_ablation.py. All figures are built directly from measured data
-- nothing here is illustrative or fabricated.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = "results"
FIGDIR = os.path.join("docs", "figures")
os.makedirs(FIGDIR, exist_ok=True)

COLORS = {
    "VQC": "#5B8DEF",
    "MatchedMLP": "#E4572E",
    "FullFeatureMLP": "#2E933C",
}


def fig_convergence():
    df = pd.read_csv(os.path.join(RESULTS, "main_results.csv"))
    fig, axes = plt.subplots(3, 1, figsize=(3.3, 4.2), sharex=True)
    for ax, nq in zip(axes, [4, 6, 8]):
        for model in ["VQC", "MatchedMLP"]:
            sub = df[(df.model == model) & (df.qubits == nq)]
            g = sub.groupby("round")["test_auc"].agg(["mean", "std"]).reset_index()
            ax.plot(g["round"], g["mean"], label=model, color=COLORS[model])
            ax.fill_between(g["round"], g["mean"] - g["std"], g["mean"] + g["std"],
                             color=COLORS[model], alpha=0.2)
        full = df[df.model == "FullFeatureMLP"]
        g = full.groupby("round")["test_auc"].agg(["mean", "std"]).reset_index()
        ax.plot(g["round"], g["mean"], "--", label="FullFeatureMLP (ref.)",
                 color=COLORS["FullFeatureMLP"])
        ax.set_title(f"Q = {nq} qubits", fontsize=9)
        ax.set_ylabel("Test ROC-AUC", fontsize=8)
        ax.set_ylim(0.45, 0.95)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Federated round", fontsize=8)
    axes[0].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_convergence.png"), dpi=220)
    plt.close(fig)


def fig_comm_accuracy_tradeoff():
    summary = pd.read_csv(os.path.join(RESULTS, "main_summary.csv"))
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for model in summary["model"].unique():
        sub = summary[summary.model == model].sort_values("n_params")
        ax.errorbar(
            sub["n_params"], sub["auc_mean"], yerr=sub["auc_std"],
            marker="o", capsize=3, label=model, markersize=4, linewidth=1.2,
            color=COLORS.get(model, "#888"),
        )
        for _, row in sub.iterrows():
            label = f"Q={int(row.qubits)}" if not pd.isna(row.qubits) else "raw-11"
            ax.annotate(label, (row.n_params, row.auc_mean),
                        textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax.set_xlabel("Params transmitted / client / round", fontsize=8)
    ax.set_ylabel("Final test ROC-AUC", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_comm_accuracy_tradeoff.png"), dpi=220)
    plt.close(fig)


def fig_barren_plateau():
    with open(os.path.join(RESULTS, "barren_plateau.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for L in sorted(df.n_layers.unique()):
        sub = df[df.n_layers == L].sort_values("n_qubits")
        ax.semilogy(sub.n_qubits, sub.grad_variance, marker="o", label=f"L={L}",
                     markersize=4, linewidth=1.2)
    ax.set_xlabel("Number of qubits", fontsize=8)
    ax.set_ylabel(r"Grad. variance Var$[\partial_\theta C]$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="Depth", fontsize=6.5, title_fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_barren_plateau.png"), dpi=220)
    plt.close(fig)


def fig_depth_tradeoff():
    summary = pd.read_csv(os.path.join(RESULTS, "depth_ablation_summary.csv"))
    fig, ax1 = plt.subplots(figsize=(3.3, 2.7))
    ax2 = ax1.twinx()
    ax1.errorbar(summary.depth, summary.auc_mean, yerr=summary.auc_std,
                 marker="o", color=COLORS["VQC"], label="Test ROC-AUC", markersize=4, linewidth=1.2)
    ax2.plot(summary.depth, summary.n_params, marker="s", color="#888",
              linestyle="--", label="Parameters", markersize=4, linewidth=1.2)
    ax1.set_xlabel("Circuit depth (8 qubits)", fontsize=8)
    ax1.set_ylabel("Test ROC-AUC", color=COLORS["VQC"], fontsize=8)
    ax2.set_ylabel("Trainable parameters", color="#888", fontsize=8)
    ax1.set_xticks(summary.depth)
    ax1.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_depth_tradeoff.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig_convergence()
    fig_comm_accuracy_tradeoff()
    if os.path.exists(os.path.join(RESULTS, "barren_plateau.json")):
        fig_barren_plateau()
    if os.path.exists(os.path.join(RESULTS, "depth_ablation_summary.csv")):
        fig_depth_tradeoff()
    print("Figures written to", FIGDIR)

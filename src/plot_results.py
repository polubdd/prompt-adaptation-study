"""Графики (точность по методам, качество vs стоимость, кривая k-shot) и markdown-таблица."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


METHOD_ORDER = [
    "zero_shot", "few_shot", "cot", "few_shot_cot",
    "knn_few_shot_cot", "self_consistency",
]
METHOD_LABELS = {
    "zero_shot":        "Zero-shot",
    "few_shot":         "Few-shot (k=3)",
    "cot":              "Zero-shot CoT",
    "few_shot_cot":     "Few-shot CoT",
    "knn_few_shot_cot": "kNN Few-shot CoT",
    "self_consistency": "Self-consistency",
}
PALETTE = {
    "zero_shot": "#90A4AE", "few_shot": "#42A5F5", "cot": "#66BB6A",
    "few_shot_cot": "#26A69A", "knn_few_shot_cot": "#AB47BC",
    "self_consistency": "#EF5350",
}

RESULTS = Path("./results")
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def load_data():
    with open(RESULTS / "results.json") as f:
        return pd.DataFrame(json.load(f))


def plot_accuracy_by_method(df):
    tasks = sorted(df["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), squeeze=False)
    fig.suptitle("Training-free adaptation — accuracy by method",
                 fontsize=14, fontweight="bold")

    for ax, task in zip(axes[0], tasks):
        sub = df[df["task"] == task].set_index("method").reindex(METHOD_ORDER).dropna()
        labels = [METHOD_LABELS[m] for m in sub.index]
        colors = [PALETTE[m] for m in sub.index]
        bars = ax.bar(labels, sub["accuracy"], color=colors, edgecolor="white")
        for b, v in zip(bars, sub["accuracy"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                    ha="center", fontsize=9)
        ax.set_title(task.upper(), fontsize=12)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, max(sub["accuracy"]) * 1.2)
        ax.tick_params(axis="x", rotation=30)
        sns.despine(ax=ax)

    plt.tight_layout()
    p = FIGS / "accuracy_by_method.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
    plt.close(fig)


def plot_accuracy_vs_cost(df):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    gsm = df[df["task"] == "gsm8k"]
    for _, row in gsm.iterrows():
        c = PALETTE.get(row["method"], "gray")
        ax.scatter(row["avg_completion_tokens"], row["accuracy"], s=220, color=c,
                   edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(METHOD_LABELS.get(row["method"], row["method"]),
                    (row["avg_completion_tokens"], row["accuracy"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Avg completion tokens (cost ↑)", fontsize=11)
    ax.set_ylabel("GSM8K accuracy (%)", fontsize=11)
    ax.set_title("Quality vs. cost of reasoning — GSM8K", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    p = FIGS / "accuracy_vs_cost.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
    plt.close(fig)


def plot_kshot_curve(kshot_path="./results/kshot.json"):
    p_in = Path(kshot_path)
    if not p_in.exists():
        return
    with open(p_in) as f:
        data = pd.DataFrame(json.load(f))
    fig, ax = plt.subplots(figsize=(7, 5))
    for task in sorted(data["task"].unique()):
        sub = data[data["task"] == task].sort_values("k")
        ax.plot(sub["k"], sub["accuracy"], "o-", linewidth=2, label=task.upper())
    ax.set_xlabel("Number of in-context examples (k)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs. number of few-shot examples", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    p = FIGS / "kshot_curve.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved {p}")
    plt.close(fig)


def generate_markdown_table(df) -> str:
    tasks = sorted(df["task"].unique())
    header = "| Method | " + " | ".join(t.upper() + " acc (%)" for t in tasks) + " | Avg tokens (GSM8K) |"
    sep = "|--------|" + "|".join(["----------"] * len(tasks)) + "|---------------------|"
    rows = []
    gsm_tokens = df[df["task"] == "gsm8k"].set_index("method")["avg_completion_tokens"].to_dict()
    for m in METHOD_ORDER:
        cells = [METHOD_LABELS[m]]
        for t in tasks:
            v = df[(df["method"] == m) & (df["task"] == t)]["accuracy"]
            cells.append(f"{v.values[0]:.1f}" if len(v) else "—")
        cells.append(f"{gsm_tokens.get(m, 0):.0f}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


if __name__ == "__main__":
    df = load_data()
    print("\n" + generate_markdown_table(df) + "\n")
    plot_accuracy_by_method(df)
    plot_accuracy_vs_cost(df)
    plot_kshot_curve()
    print("Figures saved to results/figures/")

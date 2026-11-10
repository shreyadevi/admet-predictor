"""Turn benchmark results into a Markdown report and figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import get_endpoint

# Categorical slots 1 and 2 of the reference palette (blue, orange); text in neutral inks.
SPLIT_COLORS = {"random": "#2a78d6", "scaffold": "#eb6834"}
INK = "#3d3d3a"
MUTED = "#8a8980"
GRID = "#e6e5e0"


def _fmt(mean, std, digits=3) -> str:
    if mean is None or pd.isna(mean):
        return "n/a"
    if std is None or pd.isna(std) or std == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def benchmark_markdown(summary: pd.DataFrame, n_seeds: int) -> str:
    lines = ["# Benchmark results", "",
             f"Test-set metrics, mean ± standard deviation over {n_seeds} seed(s). "
             "Random split: 80/10/10 by molecule. Scaffold split: 80/10/10 by Bemis-Murcko scaffold, "
             "so test molecules come from chemical series unseen in training.", ""]
    for endpoint_name, df in summary.groupby("endpoint", sort=False):
        endpoint = get_endpoint(endpoint_name)
        lines += [f"## {endpoint_name} ({endpoint.admet})", "", endpoint.description, ""]
        if endpoint.is_regression:
            header = f"| Model | Split | RMSE ({endpoint.units}) ↓ | MAE ↓ | R² ↑ | Spearman ↑ |"
            cols = ["rmse", "mae", "r2", "spearman"]
        else:
            header = "| Model | Split | ROC-AUC ↑ | PR-AUC ↑ | Balanced accuracy ↑ |"
            cols = ["roc_auc", "pr_auc", "balanced_accuracy"]
        lines += [header, "|" + "---|" * (len(cols) + 2)]
        for _, row in df.sort_values(["split", "config"]).iterrows():
            cells = [_fmt(row[f"{c}_mean"], row[f"{c}_std"]) for c in cols]
            lines.append(f"| {row['model']} | {row['split']} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def plot_benchmark(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """One horizontal bar chart per endpoint: headline metric per model, random vs scaffold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for endpoint_name, df in summary.groupby("endpoint", sort=False):
        endpoint = get_endpoint(endpoint_name)
        metric, label = ("rmse", f"Test RMSE ({endpoint.units}), lower is better") if endpoint.is_regression \
            else ("roc_auc", "Mean test ROC-AUC across 12 assays, higher is better")
        models = list(dict.fromkeys(df.sort_values("config")["model"]))
        y = np.arange(len(models))
        height = 0.36
        fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(models) + 1.4), dpi=150)
        for k, split in enumerate(["random", "scaffold"]):
            sub = df[df["split"] == split].set_index("model").reindex(models)
            means = sub[f"{metric}_mean"].to_numpy(dtype=float)
            stds = np.nan_to_num(sub[f"{metric}_std"].to_numpy(dtype=float))
            pos = y + (k - 0.5) * height
            ax.barh(pos, means, height=height - 0.04, xerr=stds, color=SPLIT_COLORS[split],
                    error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2}, label=f"{split} split")
            for p, m in zip(pos, means):
                if not np.isnan(m):
                    ax.text(m, p, f" {m:.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, models)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        if not endpoint.is_regression:
            ax.axvline(0.5, color=MUTED, lw=1, ls="--")
            ax.set_xlim(0.4, 1.0)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"{endpoint_name}: random vs scaffold split", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        _style(ax)
        fig.tight_layout()
        path = out_dir / f"benchmark_{endpoint_name}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_parity(y_true, y_pred, title: str, units: str, path: Path) -> Path:
    """Predicted vs measured, with the identity line and ±1 log unit band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    lo = float(min(y_true.min(), y_pred.min())) - 0.5
    hi = float(max(y_true.max(), y_pred.max())) + 0.5
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    within = float(np.mean(np.abs(y_true - y_pred) <= 1.0))

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
    ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color=GRID, alpha=0.7, lw=0, label="±1 log unit")
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
    ax.scatter(y_true, y_pred, s=22, color=SPLIT_COLORS["scaffold"], edgecolor="white", linewidth=0.6,
               label="test molecules")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Measured ({units})")
    ax.set_ylabel(f"Predicted ({units})")
    ax.set_title(title, loc="left", fontsize=11)
    ax.text(0.03, 0.97, f"RMSE {rmse:.2f}\n{within:.0%} within ±1 log unit\nn = {len(y_true)}",
            transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

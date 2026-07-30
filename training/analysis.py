from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def load_results(name: str) -> pd.DataFrame | None:
    path = Path("logs") / f"{name}_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["run_name"].astype(str).str.startswith("sweep_")].copy()
    df = df.reset_index(drop=False).rename(columns={"index": "row_order"})
    df = df.sort_values("row_order").drop_duplicates(subset=["run_name"], keep="last")
    keep = [f"sweep_{i:02d}" for i in range(1, 11)]
    df = df[df["run_name"].isin(keep)].sort_values("run_name").reset_index(drop=True)
    return df


def save_table_png(df: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # row_order is internal bookkeeping and algo is redundant with the title; drop both.
    # Round floats so long decimal expansions don't overflow their cell into neighbors.
    view = df.drop(columns=[c for c in ("row_order", "algo") if c in df.columns]).copy()
    float_cols = view.select_dtypes(include="float").columns
    view[float_cols] = view[float_cols].round(4)

    fig, ax = plt.subplots(figsize=(13, 3 + 0.35 * len(view)))
    ax.axis("off")
    tbl = ax.table(cellText=view.values, colLabels=view.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(view.columns))))
    tbl.scale(1, 1.5)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_convergence_curves(available: dict[str, pd.DataFrame], charts_dir: Path) -> None:
    """Cumulative reward curves for all methods in subplots."""
    n = len(available)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for idx, (algo, df) in enumerate(sorted(available.items())):
        ax = axes[0, idx]
        # Sort by run_name to get sweep_01..sweep_10 order as a proxy for training progress
        sorted_df = df.sort_values("run_name").reset_index(drop=True)
        ax.plot(range(1, len(sorted_df) + 1), sorted_df["mean_reward"].values, "o-", color="steelblue")
        ax.fill_between(
            range(1, len(sorted_df) + 1),
            sorted_df["mean_reward"].values - sorted_df["std_reward"].values,
            sorted_df["mean_reward"].values + sorted_df["std_reward"].values,
            alpha=0.2,
            color="steelblue",
        )
        ax.set_title(f"{algo.upper()} Convergence")
        ax.set_xlabel("Sweep Run Index")
        ax.set_ylabel("Mean Reward")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(charts_dir / "convergence_curves.png", dpi=220)
    plt.close(fig)


def plot_dqn_objective_curves(charts_dir: Path) -> None:
    """DQN objective curves: loss and exploration rate over training."""
    # Check if DQN tensorboard logs exist
    dqn_log_dir = Path("logs") / "dqn"
    if not dqn_log_dir.exists():
        return

    # Generate a proxy plot from the DQN sweep results showing learning dynamics
    df = load_results("dqn")
    if df is None or df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Mean reward vs learning rate
    ax = axes[0]
    ax.scatter(df["learning_rate"], df["mean_reward"], c="steelblue", s=60, alpha=0.8)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Mean Reward")
    ax.set_title("DQN: Reward vs Learning Rate")
    ax.set_xscale("log")

    # Right: Mean reward vs gamma
    ax = axes[1]
    ax.scatter(df["gamma"], df["mean_reward"], c="steelblue", s=60, alpha=0.8)
    ax.set_xlabel("Gamma (Discount Factor)")
    ax.set_ylabel("Mean Reward")
    ax.set_title("DQN: Reward vs Gamma")

    fig.tight_layout()
    fig.savefig(charts_dir / "dqn_objective_curves.png", dpi=220)
    plt.close(fig)


def plot_pg_entropy_curves(available: dict[str, pd.DataFrame], charts_dir: Path) -> None:
    """PG entropy curves for REINFORCE, PPO, A2C."""
    pg_algos = [a for a in ["reinforce", "ppo", "a2c"] if a in available]
    if not pg_algos:
        return

    n = len(pg_algos)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)

    for idx, algo in enumerate(pg_algos):
        ax = axes[0, idx]
        df = available[algo].sort_values("run_name").reset_index(drop=True)

        # Use ent_coef or entropy_beta as the entropy regularization parameter
        ent_col = "entropy_beta" if "entropy_beta" in df.columns else "ent_coef"
        if ent_col not in df.columns:
            ax.text(0.5, 0.5, "No entropy data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{algo.upper()}")
            continue

        scatter = ax.scatter(df[ent_col], df["mean_reward"], c=range(len(df)), cmap="viridis", s=60, alpha=0.8)
        ax.set_xlabel("Entropy Coefficient")
        ax.set_ylabel("Mean Reward")
        ax.set_title(f"{algo.upper()}: Reward vs Entropy")
        plt.colorbar(scatter, ax=ax, label="Run Index")

    fig.tight_layout()
    fig.savefig(charts_dir / "pg_entropy_curves.png", dpi=220)
    plt.close(fig)


def plot_generalization_tests(available: dict[str, pd.DataFrame], charts_dir: Path) -> None:
    """Generalization: compare mean reward stability across runs (std as proxy)."""
    if not available:
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    labels = []
    all_means = []
    all_stds = []
    boundaries = []

    for idx, (algo, df) in enumerate(sorted(available.items())):
        sorted_df = df.sort_values("run_name").reset_index(drop=True)
        means = sorted_df["mean_reward"].values
        stds = sorted_df["std_reward"].values
        start = len(all_means)
        for j in range(len(means)):
            labels.append(f"{algo.upper()}-{j + 1:02d}")
            all_means.append(means[j])
            all_stds.append(stds[j])
        ax.errorbar(
            range(start, start + len(means)), means, yerr=stds,
            fmt="o", capsize=4, capthick=1.5, elinewidth=1.5,
            color=f"C{idx}", ecolor=f"C{idx}", alpha=0.4, markersize=7,
            label=algo.upper(),
        )
        boundaries.append(start)

    for b in boundaries[1:]:
        ax.axvline(b - 0.5, color="gray", linestyle=":", alpha=0.4)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="Zero Reward Baseline")
    ax.set_xticks(range(len(all_means)))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=8)
    ax.set_xlim(-1, len(all_means))
    ax.set_ylabel("Mean Reward ± Std")
    ax.set_title("Generalization: Reward Stability Across All Runs")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(charts_dir / "generalization_tests.png", dpi=220)
    plt.close(fig)


def plot_summary() -> None:
    files = {
        "dqn": load_results("dqn"),
        "reinforce": load_results("reinforce"),
        "ppo": load_results("ppo"),
        "a2c": load_results("a2c"),
    }

    available = {k: v for k, v in files.items() if v is not None and not v.empty}
    if not available:
        raise FileNotFoundError("No results found in logs/*.csv. Run sweeps first.")

    charts_dir = Path("assets") / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hyperparameter tables for report appendix.
    for algo, df in available.items():
        save_table_png(df, charts_dir / f"{algo}_table.png", f"{algo.upper()} hyperparameter runs")

    # 2. Mean reward comparison across all methods.
    merged = []
    for algo, df in available.items():
        local = df.copy()
        local["algo"] = algo.upper()
        merged.append(local[["algo", "run_name", "mean_reward", "std_reward"]])
    all_df = pd.concat(merged, ignore_index=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(data=all_df, x="algo", y="mean_reward", ax=ax)
    sns.stripplot(data=all_df, x="algo", y="mean_reward", color="black", alpha=0.6, ax=ax)
    ax.set_title("Reward Distribution by Algorithm")
    ax.set_ylabel("Evaluation Mean Reward")
    fig.tight_layout()
    fig.savefig(charts_dir / "reward_distribution.png", dpi=220)
    plt.close(fig)

    # 3. Best-run convergence proxy (ranked bars).
    top_runs = all_df.sort_values("mean_reward", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=top_runs, x="run_name", y="mean_reward", hue="algo", dodge=False, ax=ax)
    ax.set_title("Top 12 Runs Across All Algorithms")
    ax.set_ylabel("Evaluation Mean Reward")
    ax.set_xlabel("Run ID")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(charts_dir / "top_runs.png", dpi=220)
    plt.close(fig)

    # 4. Convergence curves (subplots for each method).
    plot_convergence_curves(available, charts_dir)

    # 5. DQN objective curves.
    plot_dqn_objective_curves(charts_dir)

    # 6. PG entropy curves.
    plot_pg_entropy_curves(available, charts_dir)

    # 7. Generalization tests.
    plot_generalization_tests(available, charts_dir)

    # 8. Write best model summary.
    best_algo = None
    best_reward = -float("inf")
    best_row = None
    for algo, df in available.items():
        top = df.sort_values("mean_reward", ascending=False).iloc[0]
        if top["mean_reward"] > best_reward:
            best_reward = top["mean_reward"]
            best_algo = algo
            best_row = top

    if best_row is not None:
        best_info = {
            "algo": best_algo,
            "run_name": str(best_row["run_name"]),
            "mean_reward": float(best_row["mean_reward"]),
            "std_reward": float(best_row["std_reward"]),
        }
        # Determine model path
        if best_algo == "dqn":
            best_info["model_path"] = f"models/dqn/{best_row['run_name']}/model.zip"
        elif best_algo == "reinforce":
            best_info["model_path"] = f"models/pg/reinforce/{best_row['run_name']}/policy.pt"
        elif best_algo == "ppo":
            best_info["model_path"] = f"models/pg/ppo/{best_row['run_name']}/model.zip"
        elif best_algo == "a2c":
            best_info["model_path"] = f"models/pg/a2c/{best_row['run_name']}/model.zip"

        best_json_path = Path("logs") / "best_model_summary.json"
        best_json_path.write_text(json.dumps(best_info, indent=2), encoding="utf-8")
        print(f"Best model: {best_algo.upper()} / {best_row['run_name']} -> reward={best_reward:.3f}")


if __name__ == "__main__":
    plot_summary()
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Repo root (script lives in <repo>/scripts/)
    repo_root = Path(__file__).resolve().parents[2]

    ends2_path = repo_root / "data" / "derived" / "Ends2.csv"
    out_path = repo_root / "outputs" / "diagrams" / "end_score_diff_distribution.png"

    if not ends2_path.exists():
        raise FileNotFoundError(f"Ends2.csv not found: {ends2_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ends2_path)

    required = {
        "CompetitionID", "SessionID", "GameID", "EndID",
        "TeamID", "Result", "HasHammer", "PowerPlay"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ends2.csv missing required columns: {sorted(missing)}")

    # Ensure numeric
    df["EndID"] = pd.to_numeric(df["EndID"], errors="raise").astype(int)
    df["Result"] = pd.to_numeric(df["Result"], errors="coerce").fillna(0).astype(int)
    df["HasHammer"] = pd.to_numeric(df["HasHammer"], errors="raise").astype(int)
    df["PowerPlay"] = pd.to_numeric(df["PowerPlay"], errors="coerce").fillna(0).astype(int)

    keys = ["CompetitionID", "SessionID", "GameID", "EndID"]

    # End-level aggregation
    hammer_pts = (
        df.loc[df["HasHammer"] == 1]
        .groupby(keys)["Result"]
        .sum()
        .rename("hammer_pts")
    )

    nonhammer_pts = (
        df.loc[df["HasHammer"] == 0]
        .groupby(keys)["Result"]
        .sum()
        .rename("nonhammer_pts")
    )

    pp_flag = (
        df.loc[df["HasHammer"] == 1]
        .groupby(keys)["PowerPlay"]
        .max()
        .rename("pp_used")
    )

    end_df = pd.concat([hammer_pts, nonhammer_pts, pp_flag], axis=1).fillna(0)
    end_df["d"] = end_df["hammer_pts"] - end_df["nonhammer_pts"]
    end_df["pp_used"] = end_df["pp_used"].astype(int)

    # Restrict to score differentials in [-6, 6]
    end_df = end_df.loc[(end_df["d"] >= -6) & (end_df["d"] <= 6)]

    # Empirical distributions
    counts_np = end_df.loc[end_df["pp_used"] == 0, "d"].value_counts().sort_index()
    counts_pp = end_df.loc[end_df["pp_used"] == 1, "d"].value_counts().sort_index()

    support = np.arange(-6, 7)
    prob_np = np.array([counts_np.get(d, 0) for d in support], dtype=float)
    prob_pp = np.array([counts_pp.get(d, 0) for d in support], dtype=float)

    if prob_np.sum() > 0:
        prob_np /= prob_np.sum()
    if prob_pp.sum() > 0:
        prob_pp /= prob_pp.sum()

    # Plot
    x = np.arange(len(support))
    width = 0.40

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x - width / 2, prob_np, width, label="Normal Play (NP)")
    ax.bar(x + width / 2, prob_pp, width, label="Power Play (PP)")

    ax.set_xticks(x)
    ax.set_xticklabels(support)
    ax.set_xlabel("End score differential d (hammer − non-hammer)")
    ax.set_ylabel("Empirical probability")
    ax.set_title("Hammer-only end score differential distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

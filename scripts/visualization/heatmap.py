from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_t_by_s_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    first_col = df.columns[0]
    if first_col != "t":
        df = df.rename(columns={first_col: "t"})

    df["t"] = pd.to_numeric(df["t"], errors="raise").astype(int)
    df = df.set_index("t")

    df.columns = [int(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.loc[df.index >= 1, [s for s in df.columns if -6 <= s <= 6]]

    df = df.sort_index(ascending=False)
    df = df[sorted(df.columns)]
    return df


def plot_heatmap(df: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    im = ax.imshow(df.values, aspect="auto", origin="upper", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)

    ax.set_xlabel("Score differential s (you − opponent)")
    ax.set_ylabel("Ends remaining t")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parents[2]

    base_in = repo_root / "outputs" / "posterior_mc"
    base_out = repo_root / "outputs" / "diagrams"

    targets = [
        (
            base_in / "you_hammer_bothPP" / "you_hammer_bothPP_post_mean_winprob_t_by_s.csv",
            base_out / "heatmap_you_hammer_bothPP_mean_winprob.png",
            "Posterior mean win probability\n(you have hammer; both Power Plays available)",
        ),
        (
            base_in / "you_hammer_youPP_only" / "you_hammer_youPP_only_post_mean_winprob_t_by_s.csv",
            base_out / "heatmap_you_hammer_youPP_only_mean_winprob.png",
            "Posterior mean win probability\n(you have hammer; only your Power Play available)",
        ),
    ]

    for csv_path, out_path, title in targets:
        df = read_t_by_s_csv(csv_path)
        plot_heatmap(df, out_path, title)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

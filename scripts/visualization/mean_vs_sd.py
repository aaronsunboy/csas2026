from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_t_by_s_csv(path: Path) -> pd.DataFrame:
    """
    Reads a t-by-s CSV:
      - first column is t (ends remaining)
      - remaining columns are score differentials s (e.g., -10..10)
    Returns DataFrame indexed by t with integer columns s.
    """
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


def plot_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    im = ax.imshow(df.values.astype(float), aspect="auto", origin="upper", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(np.arange(len(df.index)))
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

    base_in = repo_root / "outputs" / "posterior_mc" / "you_hammer_bothPP"
    base_out = repo_root / "outputs" / "diagrams"

    mean_csv = base_in / "you_hammer_bothPP_post_mean_winprob_t_by_s.csv"
    sd_csv = base_in / "you_hammer_bothPP_post_sd_winprob_t_by_s.csv"

    if not mean_csv.exists():
        raise FileNotFoundError(f"Missing mean CSV: {mean_csv}")
    if not sd_csv.exists():
        raise FileNotFoundError(f"Missing SD CSV: {sd_csv}")

    mean_df = read_t_by_s_csv(mean_csv)
    sd_df = read_t_by_s_csv(sd_csv)

    plot_heatmap(
        mean_df,
        base_out / "heatmap_you_hammer_bothPP_post_mean_winprob_s-6to6_t1plus.png",
        title="Posterior mean win probability\n(you have hammer; both Power Plays available)",
        vmin=0.0,
        vmax=1.0,
    )

    plot_heatmap(
        sd_df,
        base_out / "heatmap_you_hammer_bothPP_post_sd_winprob_s-6to6_t1plus.png",
        title="Posterior SD of win probability\n(you have hammer; both Power Plays available)",
        vmin=None,
        vmax=None,
    )

    print("Wrote:")
    print(f"  {base_out / 'heatmap_you_hammer_bothPP_post_mean_winprob_s-6to6_t1plus.png'}")
    print(f"  {base_out / 'heatmap_you_hammer_bothPP_post_sd_winprob_s-6to6_t1plus.png'}")


if __name__ == "__main__":
    main()

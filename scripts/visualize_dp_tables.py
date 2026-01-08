# visualize_dp_tables.py
#
# Reads dp_win_probabilities.csv (from solve_pp_dp.py) and prints / saves
# table-style summaries with rows = t (ends remaining) and columns = s (score lead).
#
# Assumptions:
# - dp_win_probabilities.csv has columns:
#   t, s, h, p_you, p_opp, WinProb, BestAction
#
# Output:
# - Prints tables for selected regimes.
# - Saves each table as a CSV in the current directory.

import pandas as pd

IN_PATH = "/Users/aaronlin/Documents/CSAS_2026/data/dp_win_probabilities.csv"
T_MAX = 8
S_MIN, S_MAX = -10, 10

# Regimes to visualize (edit as desired)
REGIMES = [
    # (name, h, p_you, p_opp)
    ("you_hammer_bothPP", 1, 1, 1),
    ("you_hammer_youPP_only", 1, 1, 0),
    ("you_hammer_oppPP_only", 1, 0, 1),
    ("you_hammer_noPP", 1, 0, 0),
    ("opp_hammer_bothPP", 0, 1, 1),
]

def make_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    piv = df.pivot(index="t", columns="s", values=value_col)
    # Ensure full t and s grid exists (fill missing with NA)
    piv = piv.reindex(index=list(range(T_MAX, -1, -1)), columns=list(range(S_MIN, S_MAX + 1)))
    return piv

def print_table(title: str, table: pd.DataFrame):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    # Pretty formatting for display
    print(table.to_string())

def main():
    df = pd.read_csv(IN_PATH)

    # Ensure correct dtypes
    for c in ["t", "s", "h", "p_you", "p_opp"]:
        df[c] = df[c].astype(int)

    # WinProb should be numeric (already rounded in the exporter; keep as float for formatting)
    df["WinProb"] = df["WinProb"].astype(float)

    # Restrict to the requested score window
    df = df[(df["s"] >= S_MIN) & (df["s"] <= S_MAX) & (df["t"] >= 0) & (df["t"] <= T_MAX)].copy()

    for name, h, p_you, p_opp in REGIMES:
        sub = df[(df["h"] == h) & (df["p_you"] == p_you) & (df["p_opp"] == p_opp)].copy()

        if sub.empty:
            print(f"\n[WARN] No rows found for regime {name} (h={h}, p_you={p_you}, p_opp={p_opp}). Skipping.")
            continue

        # 1) Win probability table
        win_piv = make_pivot(sub, "WinProb")

        # Format to 3 decimals (string table for console readability)
        win_fmt = win_piv.applymap(lambda x: "" if pd.isna(x) else f"{x:.3f}")

        # 2) Best action table
        act_piv = make_pivot(sub, "BestAction")

        # 3) Combined cell: "0.742 PP" (or "0.742 NP")
        comb = sub.copy()
        comb["WinProbStr"] = comb["WinProb"].map(lambda x: f"{x:.3f}")
        comb["Cell"] = comb["WinProbStr"] + " " + comb["BestAction"].astype(str)
        cell_piv = make_pivot(comb, "Cell")

        # Print to console
        print_table(
            f"{name}: WinProb (rows=t {T_MAX}..0, cols=s {S_MIN}..{S_MAX}) | h={h}, p_you={p_you}, p_opp={p_opp}",
            win_fmt
        )
        print_table(
            f"{name}: BestAction (PP/NP) | h={h}, p_you={p_you}, p_opp={p_opp}",
            act_piv.fillna("")
        )
        print_table(
            f"{name}: Combined (WinProb Action) | h={h}, p_you={p_you}, p_opp={p_opp}",
            cell_piv.fillna("")
        )

        # Save CSVs
        win_piv.to_csv(f"{name}_winprob_t_by_s.csv", index=True)
        act_piv.to_csv(f"{name}_bestaction_t_by_s.csv", index=True)
        cell_piv.to_csv(f"{name}_combined_t_by_s.csv", index=True)

        print(f"\nSaved: {name}_winprob_t_by_s.csv, {name}_bestaction_t_by_s.csv, {name}_combined_t_by_s.csv")

if __name__ == "__main__":
    main()

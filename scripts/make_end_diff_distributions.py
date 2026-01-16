import pandas as pd

ENDS2_PATH = "data/derived/Ends2.csv"
OUT_PATH = "data/derived/end_diff_distributions.csv"

KEYS = ["CompetitionID", "SessionID", "GameID", "EndID"]

def main():
    df = pd.read_csv(ENDS2_PATH)

    # End-level PP flag
    df["IsPowerPlay"] = df["PowerPlay"].notna()

    # Compute opponent score USING BOTH TEAMS, then filter to hammer rows
    df["OpponentScore"] = df.groupby(KEYS)["Result"].transform("sum") - df["Result"]
    df["ScoreDiff"] = df["Result"] - df["OpponentScore"]  # team perspective

    # Restrict to hammer team rows only
    df_h = df[df["HasHammer"] == 1].copy()

    # Action label (decision available only on hammer rows)
    df_h["Action"] = df_h["IsPowerPlay"].map({False: "NP", True: "PP"})

    # Tally counts of d = ScoreDiff under each action
    counts = (
        df_h.groupby(["Action", "ScoreDiff"])
            .size()
            .rename("Count")
            .reset_index()
            .rename(columns={"ScoreDiff": "d"})
            .sort_values(["Action", "d"])
    )

    # Convert to probabilities within each action
    counts["Prob"] = counts["Count"] / counts.groupby("Action")["Count"].transform("sum")

    # Save
    counts.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
    print(counts)

if __name__ == "__main__":
    main()
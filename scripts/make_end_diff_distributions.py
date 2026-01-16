import pandas as pd

ENDS2_PATH = "data/derived/Ends2.csv"
OUT_PATH = "data/derived/end_diff_distributions.csv"

KEYS = ["CompetitionID", "SessionID", "GameID", "EndID"]

def main():
    df = pd.read_csv(ENDS2_PATH)

    df["IsPowerPlay"] = df["PowerPlay"].notna()

    df["OpponentScore"] = df.groupby(KEYS)["Result"].transform("sum") - df["Result"]
    df["ScoreDiff"] = df["Result"] - df["OpponentScore"]  # team perspective

    df_h = df[df["HasHammer"] == 1].copy()

    df_h["Action"] = df_h["IsPowerPlay"].map({False: "NP", True: "PP"})

    counts = (
        df_h.groupby(["Action", "ScoreDiff"])
            .size()
            .rename("Count")
            .reset_index()
            .rename(columns={"ScoreDiff": "d"})
            .sort_values(["Action", "d"])
    )

    counts["Prob"] = counts["Count"] / counts.groupby("Action")["Count"].transform("sum")

    counts.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
    print(counts)

if __name__ == "__main__":
    main()
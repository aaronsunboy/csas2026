import pandas as pd
import numpy as np
import re

GAMES_PATH = "Games.csv"
ENDS_PATH  = "Ends.csv"
OUT_PATH   = "Ends2.csv"

# ----------------------------
# Helpers
# ----------------------------
def canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())

def require(df: pd.DataFrame, candidates):
    cmap = {canon(c): c for c in df.columns}
    for cand in candidates:
        k = canon(cand)
        if k in cmap:
            return cmap[k]
    raise KeyError(f"Could not find any of {candidates} in columns: {list(df.columns)}")

def guess_team_cols(games: pd.DataFrame):
    """
    Identify which columns in Games.csv correspond to NOC1 and NOC2 team IDs.
    Common possibilities: TeamID1/TeamID2, Team1ID/Team2ID, NOC1/NOC2, Competitor1ID/Competitor2ID.
    Returns (col_for_noc1, col_for_noc2).
    """
    # Most common patterns
    pairs = [
        ("TeamID1", "TeamID2"),
        ("TeamId1", "TeamId2"),
        ("Team1ID", "Team2ID"),
        ("Team1Id", "Team2Id"),
        ("Competitor1ID", "Competitor2ID"),
        ("CompetitorId1", "CompetitorId2"),
        ("NOC1", "NOC2"),
        ("Noc1", "Noc2"),
    ]
    cmap = {canon(c): c for c in games.columns}
    for a, b in pairs:
        if canon(a) in cmap and canon(b) in cmap:
            return cmap[canon(a)], cmap[canon(b)]

    # Fallback: try to find any two columns that look like "team...1" and "team...2"
    cols = list(games.columns)
    ccanon = [canon(c) for c in cols]

    def find(pattern):
        for c, cc in zip(cols, ccanon):
            if pattern in cc:
                return c
        return None

    a = find("teamid1") or find("team1id") or find("competitorid1") or find("noc1")
    b = find("teamid2") or find("team2id") or find("competitorid2") or find("noc2")
    if a and b:
        return a, b

    raise KeyError(
        "Could not infer NOC1/NOC2 team identifier columns in Games.csv. "
        "Please inspect Games.csv columns and map them manually."
    )

# ----------------------------
# Load
# ----------------------------
games = pd.read_csv(GAMES_PATH)
ends  = pd.read_csv(ENDS_PATH)

# Standard keys (adjust here only if your CSV uses different names)
KEYS = ["CompetitionID", "SessionID", "GameID"]

# Validate keys exist
for k in KEYS:
    if k not in games.columns or k not in ends.columns:
        raise KeyError(f"Missing key column {k} in Games or Ends. Games cols={list(games.columns)}")

# Required columns
LSFE_COL = require(games, ["LSFE"])
TEAM_COL = require(ends,  ["TeamID"])
END_COL  = require(ends,  ["EndID"])
RES_COL  = require(ends,  ["Result"])

NOC1_COL, NOC2_COL = guess_team_cols(games)

# Keep only relevant columns in games
games_small = games[KEYS + [NOC1_COL, NOC2_COL, LSFE_COL]].copy()

# Clean Result: drop weird value 9 if present (based on prior observation)
ends2 = ends.copy()
ends2.loc[ends2[RES_COL] == 9, RES_COL] = np.nan
ends2 = ends2.dropna(subset=[RES_COL])
ends2[RES_COL] = ends2[RES_COL].astype(int)

# ----------------------------
# Build end-level wide table: points for NOC1 and NOC2
# ----------------------------
# Attach NOC1/NOC2 IDs + LSFE to each end-team row
ends_m = ends2.merge(games_small, on=KEYS, how="left")

# Sanity: ensure mapping exists
if ends_m[[NOC1_COL, NOC2_COL, LSFE_COL]].isna().any().any():
    bad = ends_m[ends_m[[NOC1_COL, NOC2_COL, LSFE_COL]].isna().any(axis=1)].head(10)
    raise ValueError(
        "Some Ends rows did not match Games rows (missing NOC1/NOC2/LSFE after merge). "
        "Check that CompetitionID/SessionID/GameID keys align.\n"
        f"Example unmatched rows:\n{bad}"
    )

# For each end, compute NOC1 points and NOC2 points
END_KEYS = KEYS + [END_COL]

# Create columns for points by role
ends_m["IsNOC1"] = ends_m[TEAM_COL] == ends_m[NOC1_COL]
ends_m["IsNOC2"] = ends_m[TEAM_COL] == ends_m[NOC2_COL]

# Aggregate to one row per end (per game)
end_wide = (
    ends_m.groupby(END_KEYS, as_index=False)
          .agg(
              NOC1_ID=(NOC1_COL, "first"),
              NOC2_ID=(NOC2_COL, "first"),
              LSFE=(LSFE_COL, "first"),
              NOC1_Points=(RES_COL, lambda s: int(s[ends_m.loc[s.index, "IsNOC1"]].sum()) if any(ends_m.loc[s.index, "IsNOC1"]) else 0),
              NOC2_Points=(RES_COL, lambda s: int(s[ends_m.loc[s.index, "IsNOC2"]].sum()) if any(ends_m.loc[s.index, "IsNOC2"]) else 0),
          )
)

# Note: The lambda above sums the single matching row; it’s written defensively.

# ----------------------------
# Compute hammer per end
# ----------------------------
# Sort by end number within each game
end_wide = end_wide.sort_values(END_KEYS).copy()

def compute_hammer_for_game(df_game: pd.DataFrame) -> pd.DataFrame:
    """
    df_game contains one game's ends, sorted by EndID.
    Returns same df with HammerTeamID per end.
    """
    df_game = df_game.sort_values(END_COL).copy()

    # End 1 hammer from LSFE: 1 => NOC1, 0 => NOC2
    hammer = df_game["NOC1_ID"].iloc[0] if int(df_game["LSFE"].iloc[0]) == 1 else df_game["NOC2_ID"].iloc[0]
    hammer_list = []

    prev_scoring_team = None

    for i, row in df_game.iterrows():
        # Current end hammer
        hammer_list.append(hammer)

        # Determine scoring team for this end (for updating next end)
        p1 = int(row["NOC1_Points"])
        p2 = int(row["NOC2_Points"])

        if p1 == 0 and p2 == 0:
            scoring_team = None  # blank
        elif p1 > p2:
            scoring_team = row["NOC1_ID"]
        else:
            scoring_team = row["NOC2_ID"]

        # Update hammer for next end
        if scoring_team is None:
            # blank end: hammer stays
            pass
        else:
            # scoring team loses hammer
            if scoring_team == row["NOC1_ID"]:
                hammer = row["NOC2_ID"]
            else:
                hammer = row["NOC1_ID"]

    df_game["HammerTeamID"] = hammer_list
    return df_game

end_wide = (
    end_wide.groupby(KEYS, group_keys=False)
            .apply(compute_hammer_for_game)
)

# ----------------------------
# Attach hammer back to the original Ends rows
# ----------------------------
hammer_map = end_wide[END_KEYS + ["HammerTeamID"]]

ends_out = ends2.merge(hammer_map, on=END_KEYS, how="left")
ends_out["HasHammer"] = (ends_out[TEAM_COL] == ends_out["HammerTeamID"]).astype(int)

# ----------------------------
# Write Ends2.csv
# ----------------------------
ends_out.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH} with HasHammer column.")

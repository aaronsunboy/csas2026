from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DIST_PATH = "data/derived/end_diff_distributions.csv"

T = 8
S_MIN, S_MAX = -10, 10

M = 50000

ALPHA0 = 0.5

TIE_VALUE = 0.5

REGIMES = [
    ("you_hammer_bothPP", 1, 1, 1),
    ("you_hammer_youPP_only", 1, 1, 0),
    ("you_hammer_oppPP_only", 1, 0, 1),
    ("you_hammer_noPP", 1, 0, 0),
    ("opp_hammer_bothPP", 0, 1, 1),
]

@dataclass(frozen=True)
class State:
    t: int
    s: int
    h: int
    p_you: int
    p_opp: int


def terminal_win_prob(s: int, tie_value: float = 0.5) -> float:
    if s > 0:
        return 1.0
    if s < 0:
        return 0.0
    return tie_value


def next_hammer(h: int, d_hammer_minus_non: int) -> int:
    """
    d is defined as (hammer team points) - (non-hammer team points) for that end.
    """
    if h == 1:
        return 0 if d_hammer_minus_non > 0 else 1
    else:
        return 1 if d_hammer_minus_non > 0 else 0


def allowed_actions(p_available: int) -> List[str]:
    return ["NP", "PP"] if p_available == 1 else ["NP"]


def solve_dp(T: int, dist: Dict[str, List[Tuple[int, float]]], tie_value: float) -> Tuple[Dict[State, float], Dict[State, str]]:
    """
    Finite-horizon zero-sum DP with minimax structure:
      - when h=1: you choose action to maximize your win probability
      - when h=0: opponent chooses action to minimize your win probability
    """
    max_abs_d = max(abs(d) for a in dist for d, _ in dist[a])
    Smax = max_abs_d * T 

    V: Dict[State, float] = {}
    Pi: Dict[State, str] = {}

    for s in range(-Smax, Smax + 1):
        for h in (0, 1):
            for p_you in (0, 1):
                for p_opp in (0, 1):
                    st = State(0, s, h, p_you, p_opp)
                    V[st] = terminal_win_prob(s, tie_value)
                    Pi[st] = "TERM"

    for t in range(1, T + 1):
        for s in range(-Smax, Smax + 1):
            for h in (0, 1):
                for p_you in (0, 1):
                    for p_opp in (0, 1):
                        st = State(t, s, h, p_you, p_opp)

                        if h == 1:
                            acts = allowed_actions(p_you)
                            best_val = -1.0
                            best_act = "NP"
                            for a in acts:
                                p_you_next = 0 if a == "PP" else p_you
                                val = 0.0
                                for d, prob in dist[a]:
                                    s2 = s + d
                                    if s2 < -Smax:
                                        s2 = -Smax
                                    elif s2 > Smax:
                                        s2 = Smax
                                    h2 = next_hammer(1, d)
                                    st2 = State(t - 1, s2, h2, p_you_next, p_opp)
                                    val += prob * V[st2]
                                if val > best_val:
                                    best_val = val
                                    best_act = a
                            V[st] = best_val
                            Pi[st] = best_act

                        else:
                            acts = allowed_actions(p_opp)
                            worst_val = 2.0
                            worst_act = "NP"
                            for a in acts:
                                p_opp_next = 0 if a == "PP" else p_opp
                                val = 0.0
                                for d, prob in dist[a]:
                                    s2 = s - d
                                    if s2 < -Smax:
                                        s2 = -Smax
                                    elif s2 > Smax:
                                        s2 = Smax
                                    h2 = next_hammer(0, d)
                                    st2 = State(t - 1, s2, h2, p_you, p_opp_next)
                                    val += prob * V[st2]
                                if val < worst_val:
                                    worst_val = val
                                    worst_act = a
                            V[st] = worst_val
                            Pi[st] = worst_act

    return V, Pi

def read_counts(path: str) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Returns:
      d_support: sorted list of all d in union of NP and PP supports
      counts_np: counts aligned to d_support
      counts_pp: counts aligned to d_support
    Requires columns: Action, d, Count
    """
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError(f"No rows read from {path}")

    for required in ["Action", "d", "Count"]:
        if required not in rows[0]:
            raise ValueError(f"{path} must contain column '{required}'. Found columns: {list(rows[0].keys())}")

    d_all = sorted({int(r["d"]) for r in rows})
    idx = {d: i for i, d in enumerate(d_all)}

    counts_np = np.zeros(len(d_all), dtype=float)
    counts_pp = np.zeros(len(d_all), dtype=float)

    for r in rows:
        a = r["Action"].strip()
        d = int(r["d"])
        c = float(r["Count"])
        if a == "NP":
            counts_np[idx[d]] += c
        elif a == "PP":
            counts_pp[idx[d]] += c
        else:
            raise ValueError(f"Unknown Action={a} in {path}")

    if counts_np.sum() <= 0 or counts_pp.sum() <= 0:
        raise ValueError("Counts for NP and/or PP are zero. Check your distribution file creation step.")

    return d_all, counts_np, counts_pp


def sample_dist(rng: np.random.Generator, d_support: List[int], counts_np: np.ndarray, counts_pp: np.ndarray, alpha0: float) -> Dict[str, List[Tuple[int, float]]]:
    """
    Samples NP and PP probability vectors from Dirichlet(alpha0 + counts),
    returns dist dict with list of (d, prob) for NP and PP.
    """
    alpha_np = counts_np + alpha0
    alpha_pp = counts_pp + alpha0

    p_np = rng.dirichlet(alpha_np)
    p_pp = rng.dirichlet(alpha_pp)

    dist = {
        "NP": [(d_support[i], float(p_np[i])) for i in range(len(d_support))],
        "PP": [(d_support[i], float(p_pp[i])) for i in range(len(d_support))],
    }
    return dist

def init_aggregators() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns dict regime_name -> dict of accumulators:
      sumV, sumV2, countPP, n
    Each table is shape (T+1, S_range) where rows are t=0..T and cols are s=S_MIN..S_MAX.
    """
    s_range = S_MAX - S_MIN + 1
    aggs: Dict[str, Dict[str, np.ndarray]] = {}
    for name, _, _, _ in REGIMES:
        aggs[name] = {
            "sumV": np.zeros((T + 1, s_range), dtype=float),
            "sumV2": np.zeros((T + 1, s_range), dtype=float),
            "countPP": np.zeros((T + 1, s_range), dtype=float),
            "n": np.zeros((T + 1, s_range), dtype=float),
        }
    return aggs


def update_aggregators(aggs: Dict[str, Dict[str, np.ndarray]], V: Dict[State, float], Pi: Dict[State, str]):
    s_vals = list(range(S_MIN, S_MAX + 1))
    for (name, h, p_you, p_opp) in REGIMES:
        A = aggs[name]
        for t in range(0, T + 1):
            for j, s in enumerate(s_vals):
                st = State(t=t, s=s, h=h, p_you=p_you, p_opp=p_opp)
                v = V.get(st)
                a = Pi.get(st)
                if v is None or a is None:
                    continue
                A["sumV"][t, j] += v
                A["sumV2"][t, j] += v * v
                A["countPP"][t, j] += 1.0 if a == "PP" else 0.0
                A["n"][t, j] += 1.0


def tables_to_csv(name: str, table: np.ndarray, out_path: str):
    """
    Writes a t-by-s table to CSV with rows t=T..0 and columns s=S_MIN..S_MAX.
    """
    df = pd.DataFrame(
        table,
        index=list(range(0, T + 1)),
        columns=list(range(S_MIN, S_MAX + 1)),
    )
    df = df.reindex(index=list(range(T, -1, -1)))
    df.index.name = "t"
    df.to_csv(out_path)
    print(f"Wrote {out_path}")


def main():
    rng = np.random.default_rng(12345)

    d_support, counts_np, counts_pp = read_counts(DIST_PATH)

    aggs = init_aggregators()

    for m in range(1, M + 1):
        dist = sample_dist(rng, d_support, counts_np, counts_pp, ALPHA0)
        V, Pi = solve_dp(T=T, dist=dist, tie_value=TIE_VALUE)
        update_aggregators(aggs, V, Pi)

        if m % 500 == 0:
            print(f"Completed {m}/{M} posterior draws")

    for (name, _, _, _) in REGIMES:
        A = aggs[name]
        n = A["n"]
        n_safe = np.where(n == 0, 1.0, n)

        mean = A["sumV"] / n_safe
        var = A["sumV2"] / n_safe - mean * mean
        var = np.maximum(var, 0.0)
        sd = np.sqrt(var)
        p_pp_opt = A["countPP"] / n_safe

        mean_r = np.round(mean, 3)
        sd_r = np.round(sd, 3)
        ppp_r = np.round(p_pp_opt, 3)

        tables_to_csv(name, mean_r, f"{name}_post_mean_winprob_t_by_s.csv")
        tables_to_csv(name, sd_r, f"{name}_post_sd_winprob_t_by_s.csv")
        tables_to_csv(name, ppp_r, f"{name}_post_prob_PP_optimal_t_by_s.csv")

    key = "you_hammer_bothPP"
    j0 = 0 - S_MIN
    t0 = T
    mean0 = aggs[key]["sumV"][t0, j0] / aggs[key]["n"][t0, j0]
    ppp0 = aggs[key]["countPP"][t0, j0] / aggs[key]["n"][t0, j0]
    print("\nKey initial state (t=8, s=0, you hammer, both PP available):")
    print(f"Posterior mean win prob: {mean0:.3f}")
    print(f"Posterior Pr(PP optimal): {ppp0:.3f}")

if __name__ == "__main__":
    main()

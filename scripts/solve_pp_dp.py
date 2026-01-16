from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd

DIST_PATH = "data/test/end_diff_distributions.csv"

# ----------------------------
# Model / game assumptions
# ----------------------------
# d = (hammer points) - (non-hammer points)
# If YOU have hammer, your score lead s updates as s' = s + d.
# If OPP has hammer, your score lead updates as s' = s - d.
#
# Hammer retention rule (role-based):
# - If hammer team scores (d > 0), hammer SWITCHES next end.
# - If blank or steal against hammer (d <= 0), hammer STAYS with same team.

@dataclass(frozen=True)
class State:
    t: int          # ends remaining
    s: int          # your lead (your score - opp score)
    h: int          # 1 if you have hammer now, 0 if opponent has hammer
    p_you: int      # 1 if your PP available, else 0
    p_opp: int      # 1 if opponent PP available, else 0

def read_distributions(path: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Returns dict: action -> list of (d, prob), where prob sums to 1 per action.
    """
    dist: Dict[str, List[Tuple[int, float]]] = {"NP": [], "PP": []}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            action = row["Action"].strip()
            d = int(row["d"])
            p = float(row["Prob"])
            if action not in dist:
                raise ValueError(f"Unknown Action={action} in {path}")
            dist[action].append((d, p))

    # Basic sanity
    for a, lst in dist.items():
        total = sum(p for _, p in lst)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Probabilities for {a} sum to {total}, not 1. Check {path}.")
    return dist

def terminal_win_prob(s: int, tie_value: float = 0.5) -> float:
    """
    Terminal win probability at t=0. For ties, uses tie_value.
    Replace tie_value if you have a specific tie-break rule.
    """
    if s > 0:
        return 1.0
    if s < 0:
        return 0.0
    return tie_value

def next_hammer(h: int, d_hammer_minus_non: int) -> int:
    """
    Returns h' (1 if you have hammer next, 0 otherwise),
    given current h and d defined as (hammer - non-hammer).
    """
    if h == 1:
        # You are hammer team now
        if d_hammer_minus_non > 0:
            return 0  # you scored -> opponent gets hammer
        return 1      # blank or steal -> you keep hammer
    else:
        # Opponent is hammer team now
        if d_hammer_minus_non > 0:
            return 1  # opponent scored -> you get hammer
        return 0      # blank or opponent stolen against -> opponent keeps hammer

def allowed_actions(h: int, p_available: int) -> List[str]:
    """
    If hammer team has PP available, they can choose NP or PP.
    Otherwise only NP.
    """
    if h == 1:
        # You to act when you have hammer
        return ["NP", "PP"] if p_available == 1 else ["NP"]
    else:
        # Opponent acts when they have hammer; we will use their p_opp separately
        return ["NP", "PP"] if p_available == 1 else ["NP"]

def solve_dp(
    T: int,
    dist: Dict[str, List[Tuple[int, float]]],
    tie_value: float = 0.5
) -> Tuple[Dict[State, float], Dict[State, str]]:
    """
    Solves the finite-horizon zero-sum game by backward induction on discrete score s.
    Returns:
      V[state] = value (win prob)
      Pi[state] = optimal action when it's that side's turn (your action if h=1, opponent action if h=0 stored for completeness)
    """
    # Bound score range to keep state space finite:
    # max_abs_d * T is a safe bound.
    max_abs_d = max(abs(d) for a in dist for d, _ in dist[a])
    Smax = max_abs_d * T

    V: Dict[State, float] = {}
    Pi: Dict[State, str] = {}

    # Initialize terminal layer t=0
    for s in range(-Smax, Smax + 1):
        for h in (0, 1):
            for p_you in (0, 1):
                for p_opp in (0, 1):
                    st = State(t=0, s=s, h=h, p_you=p_you, p_opp=p_opp)
                    V[st] = terminal_win_prob(s, tie_value=tie_value)
                    Pi[st] = "TERM"

    # Backward induction
    for t in range(1, T + 1):
        for s in range(-Smax, Smax + 1):
            for h in (0, 1):
                for p_you in (0, 1):
                    for p_opp in (0, 1):
                        st = State(t=t, s=s, h=h, p_you=p_you, p_opp=p_opp)

                        if h == 1:
                            # You have hammer: choose action to MAX your win prob
                            acts = allowed_actions(h=1, p_available=p_you)
                            best_val = -1.0
                            best_act = "NP"
                            for a in acts:
                                p_you_next = 0 if a == "PP" else p_you
                                val = 0.0
                                for d, prob in dist[a]:
                                    s2 = s + d
                                    if s2 < -Smax or s2 > Smax:
                                        # Outside bounds shouldn't happen if Smax chosen as above, but guard anyway.
                                        s2 = max(-Smax, min(Smax, s2))
                                    h2 = next_hammer(h=1, d_hammer_minus_non=d)
                                    st2 = State(t=t-1, s=s2, h=h2, p_you=p_you_next, p_opp=p_opp)
                                    val += prob * V[st2]
                                if val > best_val:
                                    best_val = val
                                    best_act = a
                            V[st] = best_val
                            Pi[st] = best_act

                        else:
                            # Opponent has hammer: opponent chooses action to MIN your win prob
                            acts = allowed_actions(h=0, p_available=p_opp)
                            worst_val = 2.0
                            worst_act = "NP"
                            for a in acts:
                                p_opp_next = 0 if a == "PP" else p_opp
                                val = 0.0
                                for d, prob in dist[a]:
                                    # Opponent hammer => your lead decreases by d
                                    s2 = s - d
                                    if s2 < -Smax or s2 > Smax:
                                        s2 = max(-Smax, min(Smax, s2))
                                    h2 = next_hammer(h=0, d_hammer_minus_non=d)
                                    st2 = State(t=t-1, s=s2, h=h2, p_you=p_you, p_opp=p_opp_next)
                                    val += prob * V[st2]
                                if val < worst_val:
                                    worst_val = val
                                    worst_act = a
                            V[st] = worst_val
                            Pi[st] = worst_act

    return V, Pi

def print_policy_slice(Pi: Dict[State, str], T: int, s_min: int, s_max: int):
    """
    Prints a simple policy slice when you have hammer and both PP available.
    """
    print("\nPolicy slice: when YOU have hammer and both PP available (p_you=1, p_opp=1)")
    for t in range(T, 0, -1):
        row = []
        for s in range(s_min, s_max + 1):
            st = State(t=t, s=s, h=1, p_you=1, p_opp=1)
            row.append(Pi.get(st, "?"))
        print(f"t={t:2d}:", " ".join(f"{a:>2s}" for a in row), f"  (s={s_min}..{s_max})")

def write_value_table(V, Pi, path="data/derived/dp_win_probabilities.csv"):
    rows = []
    for st, val in V.items():
        rows.append({
            "t": st.t,
            "s": st.s,
            "h": st.h,
            "p_you": st.p_you,
            "p_opp": st.p_opp,
            "WinProb": round(val, 3),
            "BestAction": Pi.get(st, "")
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["t", "s", "h", "p_you", "p_opp"])
    df.to_csv(path, index=False)
    print(f"Wrote {path}")

def main():
    dist = read_distributions(DIST_PATH)

    # Choose match length (Mixed Doubles often 8 ends; adjust as needed)
    T = 8

    V, Pi = solve_dp(T=T, dist=dist, tie_value=0.5)

    # Example: print a policy slice around close games
    print_policy_slice(Pi, T=T, s_min=-3, s_max=3)

    # Example: evaluate initial state: tied game, you have hammer, both PP available
    init = State(t=T, s=0, h=1, p_you=1, p_opp=1)
    print("\nExample initial state:", init)
    print("Optimal action:", Pi[init])
    print("Win probability:", V[init])
    
    write_value_table(V, Pi, path="data/derived/dp_win_probabilities.csv")


if __name__ == "__main__":
    main()
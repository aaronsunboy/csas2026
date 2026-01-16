from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd

DIST_PATH = "data/test/end_diff_distributions.csv"

@dataclass(frozen=True)
class State:
    t: int         
    s: int       
    h: int         
    p_you: int     
    p_opp: int    

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
        if d_hammer_minus_non > 0:
            return 0  
        return 1      
    else:
        if d_hammer_minus_non > 0:
            return 1 
        return 0    

def allowed_actions(h: int, p_available: int) -> List[str]:
    """
    If hammer team has PP available, they can choose NP or PP.
    Otherwise only NP.
    """
    if h == 1:
        return ["NP", "PP"] if p_available == 1 else ["NP"]
    else:
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
    max_abs_d = max(abs(d) for a in dist for d, _ in dist[a])
    Smax = max_abs_d * T

    V: Dict[State, float] = {}
    Pi: Dict[State, str] = {}

    for s in range(-Smax, Smax + 1):
        for h in (0, 1):
            for p_you in (0, 1):
                for p_opp in (0, 1):
                    st = State(t=0, s=s, h=h, p_you=p_you, p_opp=p_opp)
                    V[st] = terminal_win_prob(s, tie_value=tie_value)
                    Pi[st] = "TERM"

    for t in range(1, T + 1):
        for s in range(-Smax, Smax + 1):
            for h in (0, 1):
                for p_you in (0, 1):
                    for p_opp in (0, 1):
                        st = State(t=t, s=s, h=h, p_you=p_you, p_opp=p_opp)

                        if h == 1:
                            acts = allowed_actions(h=1, p_available=p_you)
                            best_val = -1.0
                            best_act = "NP"
                            for a in acts:
                                p_you_next = 0 if a == "PP" else p_you
                                val = 0.0
                                for d, prob in dist[a]:
                                    s2 = s + d
                                    if s2 < -Smax or s2 > Smax:
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
                            acts = allowed_actions(h=0, p_available=p_opp)
                            worst_val = 2.0
                            worst_act = "NP"
                            for a in acts:
                                p_opp_next = 0 if a == "PP" else p_opp
                                val = 0.0
                                for d, prob in dist[a]:
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

    T = 8

    V, Pi = solve_dp(T=T, dist=dist, tie_value=0.5)

    print_policy_slice(Pi, T=T, s_min=-3, s_max=3)

    init = State(t=T, s=0, h=1, p_you=1, p_opp=1)
    print("\nExample initial state:", init)
    print("Optimal action:", Pi[init])
    print("Win probability:", V[init])
    
    write_value_table(V, Pi, path="data/derived/dp_win_probabilities.csv")

if __name__ == "__main__":
    main()
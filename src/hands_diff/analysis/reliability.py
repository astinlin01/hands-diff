"""
Percentiles vs reference players, and reliability: how much of an average over N games is
a stable player signal rather than game-to-game luck.

    reference_percentiles     a profile's percentile vs reference players' averages
    single_game_reliability   ICC(1): how much one game says about a player
    profile_reliability       Spearman-Brown: reliability of an N-game average

Used to validate the performance areas (tools/diagnostics.py) and by the gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


# ---------- Player profiles and percentiles ----------

def reference_percentiles(profile: pd.Series, reference: pd.DataFrame) -> pd.Series:
    """Percentile of each trait vs the reference players' profiles (ties count half)."""
    result = {}
    for trait, value in profile.items():
        ref = reference[trait].dropna() if trait in reference.columns else pd.Series(dtype=float)
        result[trait] = float(percentileofscore(ref, value, kind="mean")) if len(ref) and pd.notna(value) else np.nan
    return pd.Series(result)


# ---------- Reliability ----------

def single_game_reliability(scores: pd.DataFrame, puuids: pd.Series) -> pd.Series:
    """ICC(1) per trait: the share of game-to-game variance that's between players.

    One-way ANOVA with players as groups (unbalanced form):
        ICC = (MSB - MSW) / (MSB + (n0 - 1) * MSW)
    where n0 is the effective games-per-player. Read it as "how much one game tells you
    about a player's typical value". Only players with 2+ games for the trait count
    (a single game has no within-player spread). Clamped at 0 (sampling noise can push
    the estimate slightly negative when there's no real player effect).
    """
    result = {}
    for trait in scores.columns:
        frame = pd.DataFrame({"x": scores[trait].to_numpy(), "p": puuids.to_numpy()}).dropna()
        sizes = frame.groupby("p")["x"].transform("size")
        frame = frame[sizes >= 2]
        groups = frame.groupby("p")["x"]
        n_i = groups.size().to_numpy(dtype=float)
        k, n = len(n_i), n_i.sum()
        if k < 2:
            result[trait] = np.nan
            continue
        means = groups.mean()
        grand = frame["x"].mean()
        msb = float((n_i * (means.to_numpy() - grand) ** 2).sum() / (k - 1))
        msw = float(((frame["x"] - frame["p"].map(means)) ** 2).sum() / (n - k))
        n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
        denominator = msb + (n0 - 1) * msw
        result[trait] = max(0.0, (msb - msw) / denominator) if denominator > 0 else np.nan
    return pd.Series(result)


def profile_reliability(single_game: float, games: int) -> float:
    """Spearman-Brown: reliability of an average over `games` games."""
    if games <= 0 or not np.isfinite(single_game):
        return 0.0
    return games * single_game / (1 + (games - 1) * single_game)

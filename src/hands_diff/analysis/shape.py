"""
Performance shape: five areas of an ADC's (or any laner's) output, in two views.

Areas (each = mean of standardized features, sign-flipped where lower is better):
  Laning      lead over the lane opponent in the first 14 minutes, plates
  Farming     CS, gold income, share of the team's gold
  Damage      damage output, share of the team's damage, damage per gold
  Survival    deaths (fewer = higher)
  Objectives  building and objective damage, turret participation

Views:
  "games"  what your games look like: end-of-game features adjusted for the teammates'
           lead and game length only (so a stomp by your teammates doesn't inflate your
           numbers); early (<=14:00) features as measured. Champion choices are included.
  "hands"  how you compare with players on the same champions: additionally adjusted for
           the champion played (rare champions pooled by class). Early features are
           adjusted for champion only -- teammates' lead and game length are measured at
           the end of the game, after the 14:00 snapshot.

Both views are standardized against the whole role population (all tiers), so 0 = a
typical game in the role; percentiles against a chosen tier are taken on player averages.
The matchup (enemy laners) is deliberately not adjusted for: over 20+ games it changes
player profiles by less than 2% (r 0.98-0.99 vs champion-only; see the 2026-11 analysis),
because players don't choose their opponents.

Why these areas: Fighting (kill participation, kills) was dropped -- it tracked Damage at
0.86 and was the least reliable area (~0.45 at 20 games).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from hands_diff.analysis.adjustment import MIN_CHAMPION_GAMES, apply_outcome_adjuster, fit_outcome_adjuster
from hands_diff.analysis.adjustment import (
    apply_early_adjuster,
    fit_early_adjuster,
    fit_standardizer,
    standardize,
)
from hands_diff.features.timeline import TIMELINE_FEATURE_COLUMNS

# area -> [(feature, sign)], in display order.
AREAS: dict[str, list[tuple[str, int]]] = {
    "Laning": [
        ("cs_diff_at_10", 1), ("gold_diff_at_14", 1), ("xp_diff_at_14", 1),
        ("max_cs_advantage_lane_opp", 1), ("turret_plates_taken", 1),
    ],
    "Farming": [
        ("cs_per_min", 1), ("gold_earned_per_min", 1), ("team_gold_share", 1), ("lane_minions_first_10", 1),
    ],
    "Damage": [("damage_per_minute", 1), ("team_damage_pct", 1), ("damage_per_gold", 1)],
    "Survival": [("deaths_per_min", -1), ("early_deaths", -1)],
    "Objectives": [
        ("damage_to_buildings_per_min", 1), ("turret_participation", 1), ("damage_to_objectives_per_min", 1),
    ],
}
AREA_FEATURES = [feature for spec in AREAS.values() for feature, _ in spec]
VIEWS = ("games", "hands")

# The raw numbers shown under each area (as played, vs the tier's median player):
# (feature, label, decimals, signed, percent). Lower is better wherever the area's sign is -1.
AREA_STATS: dict[str, list[tuple[str, str, int, bool, bool]]] = {
    "Laning": [("cs_diff_at_10", "CS diff @10", 1, True, False), ("gold_diff_at_14", "Gold diff @14", 0, True, False)],
    "Farming": [("cs_per_min", "CS/min", 2, False, False), ("team_gold_share", "Gold share", 1, False, True)],
    "Damage": [("damage_per_minute", "Damage/min", 0, False, False), ("damage_per_gold", "Damage per gold", 2, False, False)],
    "Survival": [("deaths_per_game", "Deaths/game", 1, False, False), ("early_deaths", "Deaths before 14:00", 1, False, False)],
    "Objectives": [("damage_to_buildings_per_min", "Building dmg/min", 0, False, False),
                   ("turret_participation", "Turret participation", 0, False, True)],
}
STAT_FEATURES = [s[0] for stats in AREA_STATS.values() for s in stats]


def add_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns the areas need. damage_per_gold = damage to champions per gold
    earned ("what you do with what you get"); undefined (NaN) with no gold.
    deaths_per_game is for display only (players count deaths per game, not per minute)."""
    df = df.copy()
    gold = df["gold_earned_per_min"].astype(float)
    df["damage_per_gold"] = (df["damage_to_champions_per_min"].astype(float) / gold).where(gold > 0)
    if "deaths_per_min" in df.columns and "gameLength" in df.columns:
        df["deaths_per_game"] = df["deaths_per_min"].astype(float) * df["gameLength"].astype(float) / 60.0
    return df


def _split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    early = [c for c in AREA_FEATURES if c in TIMELINE_FEATURE_COLUMNS and c in df.columns]
    late = [c for c in AREA_FEATURES if c not in TIMELINE_FEATURE_COLUMNS and c in df.columns]
    return late, early


def _adjust(rows: pd.DataFrame, view_model: dict) -> pd.DataFrame:
    late = apply_outcome_adjuster(rows, view_model["outcome"])
    early_columns = view_model["early_columns"]
    if not early_columns:
        return late
    if view_model["early"] is None:
        early = rows[early_columns].astype(float)
    else:
        early = apply_early_adjuster(rows, view_model["early"])
    return pd.concat([late, early], axis=1)


def fit_shape_model(
    role_df: pd.DataFrame,
    champion_classes: dict[str, str],
    min_champion_games: int = MIN_CHAMPION_GAMES,
) -> dict:
    """Fit both views' adjusters and standardizers on one role's games (all tiers)."""
    late, early = _split_columns(role_df)
    model = {}
    for view in VIEWS:
        with_champion = view == "hands"
        view_model = {
            "outcome": fit_outcome_adjuster(
                role_df,
                champion_classes=champion_classes if with_champion else None,
                min_champion_games=min_champion_games,
                feature_columns=late,
            ),
            "early": fit_early_adjuster(role_df, early, champion_classes, min_champion_games) if (with_champion and early) else None,
            "early_columns": early,
        }
        view_model["standardizer"] = fit_standardizer(_adjust(role_df, view_model))
        model[view] = view_model
    return model


def combine_areas(z: pd.DataFrame) -> pd.DataFrame:
    """Per game: each area = mean of its (sign-applied) standardized features that are
    present, skipping missing ones. An area with nothing available is NaN."""
    out = {}
    for area, spec in AREAS.items():
        parts = [z[feature] * sign for feature, sign in spec if feature in z.columns]
        out[area] = pd.concat(parts, axis=1).mean(axis=1, skipna=True) if parts else pd.Series(np.nan, index=z.index)
    return pd.DataFrame(out, index=z.index)


def game_area_scores(rows: pd.DataFrame, model: dict, view: str) -> pd.DataFrame:
    """Area scores per game in one view ("games" or "hands"), 0 = typical game in the role."""
    view_model = model[view]
    adjusted = _adjust(rows, view_model)
    return combine_areas(standardize(adjusted, view_model["standardizer"]))


def player_area_averages(scores: pd.DataFrame, puuids: pd.Series, min_games: int) -> pd.DataFrame:
    """Mean area score per player, players with >= min_games games only."""
    keys = np.asarray(puuids)
    counts = pd.Series(keys).value_counts()
    keep = np.isin(keys, counts[counts >= min_games].index)
    return scores[keep].groupby(keys[keep]).mean()


def area_percentile(value: float, reference: pd.Series) -> float:
    """Percentile of value among reference player averages (ties count half)."""
    reference = reference.dropna()
    if pd.isna(value) or reference.empty:
        return float("nan")
    return float(percentileofscore(reference, value, kind="mean"))


def champion_effect(
    games: pd.DataFrame,
    hands: pd.DataFrame,
    champions: pd.Series,
    reference_sd: pd.Series,
) -> dict[str, dict]:
    """How much one player's champions move each area, from their own games.

    effect_sd: mean per-game (games - hands) score, in units of the reference players'
      spread (sd of player averages). Positive = the champions lift the area. SD units
      rather than percentile points because percentiles squeeze together near 0 and 100.
    driver: the champion contributing most in the direction of the total effect;
      driver_share: its contribution / the total (can exceed 1 when other champions
      pull the opposite way).
    """
    diff = games.reset_index(drop=True) - hands.reset_index(drop=True)
    names = np.asarray(champions)
    out = {}
    for area in diff.columns:
        column = diff[area]
        total = float(column.sum())
        sd = float(reference_sd.get(area, np.nan))
        effect = float(column.mean() / sd) if sd and np.isfinite(sd) else float("nan")
        driver, share = None, float("nan")
        if total != 0 and np.isfinite(total):
            by_champion = column.groupby(names).sum() * np.sign(total)
            driver = str(by_champion.idxmax())
            share = float(by_champion.max() / abs(total))
        out[area] = {"effect_sd": effect, "driver": driver, "driver_share": share}
    return out

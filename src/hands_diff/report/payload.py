"""
Building blocks for the dashboard export (export_dashboard_data.py): everything the static
page needs to recompute profiles, pool stats, percentiles and gaps in the browser
(report/web/profile_math.js).

The page never sees other players' games, only *reference curves*: for each stat, the
51 percentile cutoffs (0th, 2nd, ..., 100th) of the reference players' averages, plus
their mean, median and count. Reference players = players with 10+ games in the role and
tier, as everywhere else; a tier with fewer than the minimum number of such players
gets no curves, only its player count, so the page can say "insufficient data (n=X)".
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from hands_diff.analysis.pool import effective_champions, reference_class_mix
from hands_diff.analysis.shape import (
    AREA_STATS,
    AREAS,
    STAT_FEATURES,
    VIEWS,
    fit_shape_model,
    game_area_scores,
)

BREAKPOINT_COUNT = 51  # percentile grid: 0, 2, 4, ..., 100
MIN_PLAYER_GAMES = 10
MIN_REFERENCE_PLAYERS = 20


def json_number(value, digits: int = 4):
    """JSON-safe number: NaN/None -> None, numpy scalars -> Python, floats rounded."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    return None if math.isnan(value) else round(value, digits)


def _curve(series: pd.Series) -> dict:
    quantiles = np.linspace(0, 100, BREAKPOINT_COUNT)
    return {
        "ref_avg": json_number(series.mean()),
        "std": json_number(series.std(ddof=0) or 1.0),
        "n": int(len(series)),
        "median": json_number(series.median()),
        "breakpoints": [json_number(v) for v in np.percentile(series, quantiles)],
    }


def _eligible(rows: pd.DataFrame, min_games: int) -> pd.Series:
    """Boolean mask: rows of players with >= min_games rows here."""
    return rows.groupby("puuid")["puuid"].transform("size") >= min_games


def player_average_curves(
    rows: pd.DataFrame,
    values: pd.DataFrame,
    min_games: int = MIN_PLAYER_GAMES,
    min_players: int = MIN_REFERENCE_PLAYERS,
) -> dict:
    """{stat: curve} over the averages of players with >= min_games rows. A player's
    average skips their missing values (e.g. games without a timeline). {} if fewer
    than min_players players qualify."""
    keep = _eligible(rows, min_games)
    averages = values[keep.to_numpy()].groupby(rows.loc[keep, "puuid"].to_numpy()).mean()
    if len(averages) < min_players:
        return {}
    curves = {}
    for column in averages.columns:
        series = averages[column].dropna()
        if len(series) >= min_players:
            curves[column] = _curve(series)
    return curves


def reference_by_tier(
    rows: pd.DataFrame,
    values: pd.DataFrame,
    regions: list[str],
    tiers: list[str],
    min_games: int = MIN_PLAYER_GAMES,
    min_players: int = MIN_REFERENCE_PLAYERS,
) -> dict:
    """{region: {tier: curves}} for tiers with enough players, plus
    "_player_counts": {region: {tier: qualifying players}} for every tier with any."""
    reference: dict = {"_player_counts": {}}
    for region in regions:
        for tier in tiers:
            mask = ((rows["region"] == region) & (rows["tier"] == tier)).to_numpy()
            if not mask.any():
                continue
            subset = rows[mask]
            players = int((subset.groupby("puuid").size() >= min_games).sum())
            reference["_player_counts"].setdefault(region, {})[tier] = players
            curves = player_average_curves(subset, values[mask], min_games, min_players)
            if curves:
                reference.setdefault(region, {})[tier] = curves
    return reference


def pool_reference(rows: pd.DataFrame, min_games: int = MIN_PLAYER_GAMES, min_players: int = MIN_REFERENCE_PLAYERS) -> dict:
    """The tier's champion-pool baseline: average class mix (every player weighs the
    same) and the spread of effective champion counts. rows: one role/tier's games with
    championName and champion_class (support subclasses already applied)."""
    eligible = rows[_eligible(rows, min_games)]
    players = eligible["puuid"].nunique()
    if players < min_players:
        return {"n": int(players)}
    mixes = pd.crosstab(eligible["puuid"], eligible["champion_class"], normalize="index")
    effective = eligible.groupby("puuid")["championName"].apply(lambda s: effective_champions(s.value_counts()))
    return {
        "n": int(players),
        "class_mix": {cls: json_number(share) for cls, share in reference_class_mix(mixes).items()},
        "effective_champions": _curve(effective.dropna()),
    }


# ---------- Scatter points ----------

def player_points(
    rows: pd.DataFrame,
    scores: pd.DataFrame,
    traits: list[str],
    regions: list[str],
    tiers: list[str],
    min_games: int = MIN_PLAYER_GAMES,
    min_players: int = MIN_REFERENCE_PLAYERS,
) -> dict:
    """{region: {tier: [[trait averages in `traits` order], ...]}}: one point per reference
    player with >= min_games games, for the style map. Anonymous on purpose (no puuid or
    name). 3 decimals is far finer than a dot on screen; null = no data for that trait.
    Tiers with fewer than min_players players are left out (the page says so)."""
    points: dict = {}
    for region in regions:
        for tier in tiers:
            mask = ((rows["region"] == region) & (rows["tier"] == tier)).to_numpy()
            if not mask.any():
                continue
            subset = rows[mask]
            keep = _eligible(subset, min_games).to_numpy()
            if not keep.any():
                continue
            averages = scores.loc[mask, traits][keep].groupby(subset.loc[keep, "puuid"].to_numpy()).mean()
            if len(averages) < min_players:
                continue
            points.setdefault(region, {})[tier] = [
                [json_number(v, 3) for v in row] for row in averages.to_numpy()
            ]
    return points


# ---------- Performance shape ----------

SHAPE_POINT_AREAS = ["Damage", "Survival"]  # the risk-for-damage quadrant


def shape_payload(
    role_df: pd.DataFrame,
    champion_classes: dict[str, str],
    regions: list[str],
    tiers: list[str],
) -> tuple[dict, dict]:
    """(role-level shape payload, {row index: {"games": {area: score}, "hands": {...}}}).

    role_df needs add_shape_features() applied. Both views are fitted on the role's whole
    season population; the page averages your filtered games and ranks them against the
    reference curves of the tier you pick.
    """
    model = fit_shape_model(role_df, champion_classes)
    stat_columns = [c for c in STAT_FEATURES if c in role_df.columns]
    scores = {view: game_area_scores(role_df, model, view) for view in VIEWS}
    payload = {
        "areas": list(AREAS),
        "reference": {view: reference_by_tier(role_df, scores[view], regions, tiers) for view in VIEWS},
        # Quadrant dots: every reference player's (Damage, Survival) averages, hands view.
        "points_areas": SHAPE_POINT_AREAS,
        "points": player_points(role_df, scores["hands"], SHAPE_POINT_AREAS, regions, tiers),
        # Stat chips: raw values as played, compared with the tier's median player.
        "stats": {area: [list(s) for s in stats] for area, stats in AREA_STATS.items()},
        "stat_reference": reference_by_tier(role_df, role_df[stat_columns].astype(float), regions, tiers),
    }
    per_game = {
        i: {
            **{view: {area: json_number(scores[view].at[i, area]) for area in AREAS} for view in VIEWS},
            "stats": {c: json_number(role_df.at[i, c]) for c in stat_columns},
        }
        for i in role_df.index
    }
    return payload, per_game


# ---------- Player card splash art ----------

def splash_champions(games: pd.DataFrame, min_games: int) -> set[str]:
    """Champions that get splash art on the player card: each account's most-played
    champion per role (roles with >= min_games games, where the page shows a profile), plus
    each account's overall most-played one for the all-roles view. Ties -> alphabetical."""
    def top(series: pd.Series) -> str:
        counts = series.value_counts()
        return sorted(counts.index, key=lambda name: (-counts[name], name))[0]

    champions = set()
    for _, account_games in games.groupby("account"):
        champions.add(top(account_games["championName"]))
        for _, role_games in account_games.groupby("teamPosition"):
            if len(role_games) >= min_games:
                champions.add(top(role_games["championName"]))
    return champions


def crop_splash(image_bytes: bytes, width: int = 1200, height: int = 480, quality: int = 72) -> bytes:
    """Crop a Data Dragon splash (1215x717) to the card's wide shape and re-encode small.

    Keeps the full width and takes the band starting 3% down: splash art puts the
    champion's head near the top (12% cut Kai'Sa's off), and the card overlays text on
    the left, so the lower edge matters least."""
    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    source_w, source_h = image.size
    target_ratio = width / height
    if source_w / source_h > target_ratio:  # too wide: trim the sides evenly
        crop_w = round(source_h * target_ratio)
        left = (source_w - crop_w) // 2
        box = (left, 0, left + crop_w, source_h)
    else:  # too tall (the usual case): keep a band from 12% down
        crop_h = round(source_w / target_ratio)
        top = min(round(source_h * 0.03), source_h - crop_h)
        box = (0, top, source_w, top + crop_h)
    out = BytesIO()
    image.crop(box).resize((width, height), Image.LANCZOS).save(out, "JPEG", quality=quality, optimize=True, progressive=True)
    return out.getvalue()

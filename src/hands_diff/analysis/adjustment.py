"""
Game-context adjustment, shared by the performance shape and the performance gaps.

Almost every per-minute stat (damage, gold, CS, deaths, ...) moves with *how the game
went* and with game length, not just with how the player played. Per role, each feature is
regressed on game context -- the teammates' kill and gold lead (your 4 teammates vs the 4
enemies other than your lane opponent), a cubic spline of game length, and optionally the
champion played -- and the residual is put back on the feature's own scale at a "typical
game" (even teammates, median length, the role's average champion mix).

Your own lead stays in on purpose: team lead = teammates' lead + your own lane lead, and
only the teammates' part is outside your control.

Also here: the early-game (<=14:00) adjustment, which is champion-only because the
teammates' lead and game length are measured after the 14:00 snapshot, and the
standardizer that puts every feature on the role population's z scale.

Raw feature columns are untouched in the feature table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

from hands_diff.config.champions import champion_class
from hands_diff.config.features import FEATURE_COLUMN_NAMES, add_team_context_features, lane_opponent_values

# Game state is split into what the player did and what their teammates did:
#     team lead = teammates' lead + own lane lead
# where "teammates' lead" compares your 4 teammates against the 4 enemies other than your
# lane opponent. The adjustment controls only for the teammates' part (outside your
# control); your own lane lead -- and any win your style causes -- stays in as style.
TEAM_STATE_COLUMNS = ("team_kill_diff_per_min", "team_gold_diff_per_min")
LANE_STATE_COLUMNS = ("lane_kill_diff_per_min", "lane_gold_diff_per_min")
TEAMMATE_STATE_COLUMNS = ("teammates_kill_diff_per_min", "teammates_gold_diff_per_min")
GAME_STATE_COLUMNS = TEAM_STATE_COLUMNS + LANE_STATE_COLUMNS + TEAMMATE_STATE_COLUMNS
STATE_COLUMNS = TEAMMATE_STATE_COLUMNS
# Game length enters the adjustment as a cubic spline (see fit_outcome_adjuster).
LENGTH_SPLINE_KNOTS = 6
# Champions with fewer games than this in a role share one adjustment with the rest of
# their class (see champion_groups) -- too few games to estimate their own typical stats.
MIN_CHAMPION_GAMES = 30

# Exact or near-exact duplicates of another feature (Riot's precomputed per-minute value
# vs. our own division of the same total), and kill-count features that mostly restate
# "got kills". Left out of the default adjustment set so a fit doesn't double-weight them;
# they stay in the feature table for percentile comparisons.
DUPLICATE_FEATURES = frozenset(
    {
        "gold_per_minute",  # == gold_earned_per_min
        "damage_to_champions_per_min",  # == damage_per_minute
        "vision_score_per_min_raw",  # == vision_score_per_minute
        "cs_total_per_min",  # == cs_per_min + neutral_cs_per_min
    }
)

REDUNDANT_KILL_FEATURES = frozenset(
    {"kda", "killing_sprees", "largest_killing_spree", "largest_multikill", "multikills"}
)

# Default feature set for fit_outcome_adjuster (callers normally pass their own).
DEFAULT_FEATURE_COLUMNS = [c for c in FEATURE_COLUMN_NAMES if c not in DUPLICATE_FEATURES | REDUNDANT_KILL_FEATURES]


def add_game_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach team / own-lane / teammates' kill and gold differentials per minute, plus
    the team-context features (team_gold_share, lane_gold_diff_per_min). No-op if present.

    Must be called on the full feature table, not a subset of one player's rows.

    Derived from the feature table itself: within a match, the 5 rows sharing a `win`
    value are one team. Rows whose opposing team or lane opponent is missing from the
    table (shouldn't happen after build_features' remake filter, but possible for partial
    data or lane swaps) get 0 for that part, i.e. "even".
    """
    df = add_team_context_features(df)
    if all(column in df.columns for column in GAME_STATE_COLUMNS):
        return df

    team = (
        df.groupby(["matchId", "win"])[["kills_per_min", "gold_earned_per_min"]]
        .sum()
        .reset_index()
    )
    opponent = team.copy()
    opponent["win"] = ~opponent["win"].astype(bool)
    merged = team.merge(opponent, on=["matchId", "win"], suffixes=("_team", "_opp"))
    merged["team_kill_diff_per_min"] = merged["kills_per_min_team"] - merged["kills_per_min_opp"]
    merged["team_gold_diff_per_min"] = merged["gold_earned_per_min_team"] - merged["gold_earned_per_min_opp"]

    out = df.drop(columns=[c for c in GAME_STATE_COLUMNS if c in df.columns and c not in LANE_STATE_COLUMNS]).merge(
        merged[["matchId", "win", *TEAM_STATE_COLUMNS]],
        on=["matchId", "win"],
        how="left",
    )
    for column in TEAM_STATE_COLUMNS:
        out[column] = out[column].fillna(0.0)

    # lane_gold_diff_per_min is also a style feature (computed in add_team_context_features);
    # it's reused here as-is. Missing lane opponent -> 0 for the decomposition only.
    out["lane_kill_diff_per_min"] = out["kills_per_min"] - lane_opponent_values(out, "kills_per_min")
    for stat in ("kill", "gold"):
        own = out[f"lane_{stat}_diff_per_min"].fillna(0.0)
        out[f"teammates_{stat}_diff_per_min"] = out[f"team_{stat}_diff_per_min"] - own
    out["lane_kill_diff_per_min"] = out["lane_kill_diff_per_min"].fillna(0.0)
    return out


def _game_minutes(df: pd.DataFrame) -> np.ndarray:
    return (df["gameLength"].astype(float).to_numpy() / 60.0).reshape(-1, 1)


def _covariate_matrix(
    state: np.ndarray, minutes: np.ndarray, length_spline: SplineTransformer
) -> np.ndarray:
    return np.hstack([state, length_spline.transform(minutes)])


def _row_covariates(
    df: pd.DataFrame, length_spline: SplineTransformer, state_columns=STATE_COLUMNS
) -> np.ndarray:
    return _covariate_matrix(df[list(state_columns)].astype(float).to_numpy(), _game_minutes(df), length_spline)


def _typical_state(state_columns) -> np.ndarray:
    """The "typical game" state: even teams (0 lead); win=0.5 for legacy artifacts."""
    return np.array([[0.5 if column == "win" else 0.0 for column in state_columns]])


def champion_groups(champions: pd.Series, classes: dict[str, str], min_games: int) -> pd.Series:
    """Champion name if it has >= min_games rows, else its class pool ("class:Mage")."""
    counts = champions.value_counts()
    common = set(counts[counts >= min_games].index)
    return champions.map(lambda name: name if name in common else f"class:{champion_class(name, classes)}")


def _champion_onehot(champions: pd.Series, adjuster: dict) -> np.ndarray:
    """One-hot champion-group matrix for rows, using the groups seen at fit time.

    A champion not seen at fit time falls back to its class pool; if the class pool wasn't
    seen either, the row gets the population's champion mix, i.e. no champion adjustment.
    """
    levels: list[str] = adjuster["champion_group_levels"]
    shares: np.ndarray = adjuster["champion_group_shares"]
    index = {level: i for i, level in enumerate(levels)}
    onehot = np.tile(shares, (len(champions), 1))
    for row, name in enumerate(champions.to_numpy()):
        group = name if name in index else f"class:{champion_class(name, adjuster['champion_classes'])}"
        if group in index:
            onehot[row] = 0.0
            onehot[row, index[group]] = 1.0
    return onehot


def fit_outcome_adjuster(
    role_df: pd.DataFrame,
    champion_classes: dict[str, str] | None = None,
    min_champion_games: int = MIN_CHAMPION_GAMES,
    feature_columns: list[str] | None = None,
) -> dict:
    """Fit the per-role feature-on-context regression. role_df must have game-state columns.

    Covariates: the teammates' kill/gold differential per minute (your 4 teammates vs the
    4 enemies other than your lane opponent -- deliberately *not* the win flag or the full
    team lead, so a style that wins games isn't subtracted out), and a smooth (cubic
    spline) function of game length. Game length matters on its own: champion level, killing
    sprees, turret plates, max lane leads etc. are per-game totals/maxima that grow with
    length -- non-linearly (champion level follows the XP curve and caps at 18) -- and
    even per-minute rates like gold/min drift with it. Without this, the shortest games
    looked uniformly "low everything".

    With champion_classes, the champion played is a covariate too (one indicator per
    champion, rare champions pooled by class), so residuals mean "how this game differs
    from a typical game *on the same champion*" -- otherwise playstyle mostly reduces to
    which champion was picked (Malphite always "tanks more" than Jinx).

    feature_columns defaults to DEFAULT_FEATURE_COLUMNS; the shape and gaps pass their own.
    """
    columns = list(feature_columns) if feature_columns is not None else list(DEFAULT_FEATURE_COLUMNS)
    medians = role_df[columns].astype(float).median()
    features = role_df[columns].astype(float).fillna(medians)
    minutes = _game_minutes(role_df)
    # Knots at length quantiles so the curve has resolution where games actually are;
    # constant extrapolation keeps unusually short/long games from blowing up.
    length_spline = SplineTransformer(
        n_knots=LENGTH_SPLINE_KNOTS, degree=3, knots="quantile", extrapolation="constant"
    ).fit(minutes)
    covariates = _row_covariates(role_df, length_spline, STATE_COLUMNS)
    covariate_names = list(STATE_COLUMNS) + ["game_minutes_spline"]

    champion_terms: dict = {}
    if champion_classes is not None:
        groups = champion_groups(role_df["championName"], champion_classes, min_champion_games)
        levels = sorted(groups.unique())
        onehot = (groups.to_numpy()[:, None] == np.array(levels)[None, :]).astype(float)
        champion_terms = {
            "champion_group_levels": levels,
            # The population's champion mix -- the "typical game" champion (see apply).
            "champion_group_shares": onehot.mean(axis=0),
            "champion_classes": dict(champion_classes),
            "champion_min_games": min_champion_games,
        }
        covariates = np.hstack([covariates, onehot])
        covariate_names.append("champion_group")

    regression = LinearRegression().fit(covariates, features)
    return {
        "regression": regression,
        "length_spline": length_spline,
        "median_game_minutes": float(np.median(minutes)),
        "feature_medians": medians,
        "feature_columns": columns,
        "state_columns": list(STATE_COLUMNS),
        "covariates": covariate_names,
        **champion_terms,
    }


def apply_outcome_adjuster(rows: pd.DataFrame, adjuster: dict) -> pd.DataFrame:
    """Residualize rows' features against game context, keeping each feature's scale.

    Residual + the prediction for a "typical game" (even teammates, median game length for
    the role, and -- if champion-adjusted -- the role's average champion mix) keeps
    adjusted values in readable units, e.g. "champion level this game would have had in
    an even, median-length game".
    """
    columns = adjuster["feature_columns"]
    state_columns = adjuster["state_columns"]
    features = rows[columns].fillna(adjuster["feature_medians"])
    regression: LinearRegression = adjuster["regression"]
    length_spline: SplineTransformer = adjuster["length_spline"]
    covariates = _row_covariates(rows, length_spline, state_columns)
    typical = _covariate_matrix(
        _typical_state(state_columns),
        np.array([[adjuster["median_game_minutes"]]]),
        length_spline,
    )
    if "champion_group_levels" in adjuster:  # fitted without champion_classes: no champion terms
        covariates = np.hstack([covariates, _champion_onehot(rows["championName"], adjuster)])
        typical = np.hstack([typical, adjuster["champion_group_shares"].reshape(1, -1)])
    predicted = regression.predict(covariates)
    adjusted = features.to_numpy() - predicted + regression.predict(typical)
    return pd.DataFrame(adjusted, columns=columns, index=rows.index)


# ---------- Early (timeline) feature adjustment ----------
#
# Early features are measured at/before 14:00, so the end-of-game context the main
# adjuster uses (teammates' lead, game length) is *after* them in time -- adjusting for
# it would subtract out effects the early game caused. They're adjusted for the champion
# played only. With champion as the only covariate, the regression reduces to group
# means: adjusted = value - mean(value | champion group) + mean(value), i.e. "how this
# game compares to a typical game on the same champion, in the role's average units".

def fit_early_adjuster(
    role_df: pd.DataFrame,
    early_columns: list[str],
    champion_classes: dict[str, str],
    min_champion_games: int = MIN_CHAMPION_GAMES,
) -> dict:
    """Per-champion-group means of each early column (NaN rows = no timeline, skipped).

    Groups are decided on all rows (as in the main adjuster). A group with no timeline
    rows for a column simply gets no adjustment for it.
    """
    groups = champion_groups(role_df["championName"], champion_classes, min_champion_games)
    group_means = {}
    overall = {}
    for column in early_columns:
        values = role_df[column].astype(float)
        group_means[column] = values.groupby(groups).mean().dropna().to_dict()
        overall[column] = float(values.mean())
    return {
        "early_columns": list(early_columns),
        "group_means": group_means,
        "overall_means": overall,
        "groups": sorted(set(groups)),
        "champion_classes": dict(champion_classes),
    }


def _row_groups(champions: pd.Series, adjuster: dict) -> pd.Series:
    """Champion's own group if it had one at fit time, else its class pool."""
    known = set(adjuster["groups"])
    classes = adjuster["champion_classes"]
    return champions.map(lambda name: name if name in known else f"class:{champion_class(name, classes)}")


def apply_early_adjuster(rows: pd.DataFrame, adjuster: dict) -> pd.DataFrame:
    """Champion-adjusted early columns. NaN (no timeline / undefined) stays NaN."""
    groups = _row_groups(rows["championName"], adjuster)
    out = {}
    for column in adjuster["early_columns"]:
        overall = adjuster["overall_means"][column]
        # Unseen group -> its mean is the overall mean, i.e. no adjustment.
        group_mean = groups.map(adjuster["group_means"][column]).astype(float).fillna(overall)
        out[column] = rows[column].astype(float) - group_mean + overall
    return pd.DataFrame(out, index=rows.index)


# ---------- Standardizing ----------

def fit_standardizer(df: pd.DataFrame) -> dict:
    """Population mean and std (ddof=0) per column, ignoring NaN."""
    std = df.std(ddof=0)
    # A constant column carries no style information: scale 1 maps it to all zeros
    # instead of dividing by zero.
    return {"mean": df.mean(), "std": std.where(std > 0, 1.0)}


def standardize(df: pd.DataFrame, standardizer: dict) -> pd.DataFrame:
    columns = [c for c in df.columns if c in standardizer["mean"].index]
    return (df[columns] - standardizer["mean"][columns]) / standardizer["std"][columns]

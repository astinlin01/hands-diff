"""
Method diagnostics: do the two shape views measure what the README says they do?

For each role (season-matched), on the five performance areas (performance_shape.AREAS),
"your games" view vs "your hands" view (the second also adjusted for the champion played):

- champion share of variance  eta^2 = SS_between_champions / SS_total. "How much of this
                              area is just which champion was played." Should drop in the
                              hands view.
- |correlation with win|      "How much of this area is just whether the game went well."
                              Not expected to reach zero: both views remove only the
                              teammates' share of the outcome and keep your own.
- split-half reliability      each player's games alternate into two halves; correlate the
                              half means across players (10+ games). "Is there a stable
                              per-player signal?" Raw correlation, not Spearman-Brown corrected.

Each measure is averaged over areas per role, and reported per area (median over roles).

Run from repo root (after build_features.py and build_timeline_features.py):
    python src/method_diagnostics.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from hands_diff.config.champions import load_champion_classes
from hands_diff.config.features import VALID_ROLES, current_season
from hands_diff.features.timeline import merge_timeline_features
from hands_diff.analysis.shape import AREAS, add_shape_features, fit_shape_model, game_area_scores
from hands_diff.analysis.adjustment import add_game_state_columns

MIN_PLAYER_GAMES = 10
MEASURES = ("champion_variance_share", "abs_win_correlation", "split_half_reliability")


def champion_variance_share(values: pd.DataFrame, champions: pd.Series) -> pd.Series:
    """Per feature: share of variance between champions (eta squared). Missing values dropped."""
    out = {}
    for column in values.columns:
        mask = values[column].notna().to_numpy()
        v = values[column].to_numpy(dtype=float)[mask]
        groups = champions.to_numpy()[mask]
        ss_total = ((v - v.mean()) ** 2).sum()
        group_means = pd.Series(v).groupby(groups).transform("mean").to_numpy()
        ss_between = ((group_means - v.mean()) ** 2).sum()
        out[column] = ss_between / ss_total if ss_total > 0 else np.nan
    return pd.Series(out)


def win_correlation(values: pd.DataFrame, win: pd.Series) -> pd.Series:
    """Per feature: |Pearson correlation| with the win flag. Missing values dropped."""
    w = pd.Series(win.to_numpy(dtype=float), index=values.index)
    return values.apply(lambda col: abs(col.corr(w)))


def split_half_reliability(values: pd.DataFrame, puuids: pd.Series, min_games: int = MIN_PLAYER_GAMES) -> pd.Series:
    """Per feature: correlation across players of their two half-means (games alternate in
    row order, so sort rows by time first). Players with fewer than min_games valid games
    for that feature are excluded."""
    out = {}
    players = pd.Series(puuids.to_numpy(), index=values.index)
    for column in values.columns:
        df = pd.DataFrame({"v": values[column], "p": players}).dropna()
        counts = df["p"].map(df["p"].value_counts())
        df = df[counts >= min_games]
        if df["p"].nunique() < 3:
            out[column] = np.nan
            continue
        half = df.groupby("p").cumcount() % 2
        means = df.groupby([df["p"], half])["v"].mean().unstack()
        out[column] = means[0].corr(means[1])
    return pd.Series(out)


def summarize_role(values: pd.DataFrame, rows: pd.DataFrame, min_games: int = MIN_PLAYER_GAMES) -> dict:
    """Each measure averaged over features (ignoring features where it's undefined; NaN if
    it's undefined for all of them, e.g. fewer than 3 players with enough games)."""
    per_feature = {
        "champion_variance_share": champion_variance_share(values, rows["championName"]),
        "abs_win_correlation": win_correlation(values, rows["win"]),
        "split_half_reliability": split_half_reliability(values, rows["puuid"], min_games),
    }
    return {name: float(s.mean()) if s.notna().any() else np.nan for name, s in per_feature.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Games vs hands view diagnostics per role.")
    parser.add_argument("--features-path", default="data/processed/participant_features.parquet")
    parser.add_argument("--timeline-features-path", default="data/processed/timeline_features.parquet")
    parser.add_argument("--report-path", default="data/processed/method_diagnostics.md")
    parser.add_argument("--min-games", type=int, default=MIN_PLAYER_GAMES)
    return parser.parse_args(argv)


def _range(results: dict, version: str, measure: str) -> str:
    vals = [results[role][version][measure] for role in results]
    return f"{min(vals):.2f}–{max(vals):.2f}"


VIEWS = ("games", "hands")


def run(args: argparse.Namespace) -> None:
    df = add_game_state_columns(pd.read_parquet(args.features_path))
    df = add_shape_features(merge_timeline_features(df, args.timeline_features_path))
    season = current_season(df)
    if season is not None:
        df = df[df["season"] == season]
    classes = load_champion_classes()

    results: dict[str, dict] = {}
    per_area: dict[str, dict] = {}
    for role in VALID_ROLES:
        role_df = df[df["teamPosition"] == role].sort_values("gameCreation").reset_index(drop=True)
        if role_df.empty:
            continue
        model = fit_shape_model(role_df, classes)
        scores = {view: game_area_scores(role_df, model, view) for view in VIEWS}
        results[role] = {view: summarize_role(scores[view], role_df, args.min_games) for view in VIEWS}
        results[role]["n_games"] = len(role_df)
        results[role]["n_players"] = int((role_df["puuid"].value_counts() >= args.min_games).sum())
        for view in VIEWS:
            for area in AREAS:
                per_area.setdefault(area, {}).setdefault(view, []).append(
                    summarize_role(scores[view][[area]], role_df, args.min_games)
                )
        r = results[role]
        print(f"[{role}] n={r['n_games']}, players with {args.min_games}+ games={r['n_players']}  "
              + "  ".join(f"{m}: {r['games'][m]:.3f} -> {r['hands'][m]:.3f}" for m in MEASURES))

    header = ("| {} | champion share games | hands | abs(corr win) games | hands | split-half games | hands |")
    lines = [
        f"# Method diagnostics (season {season})",
        "",
        f"The {len(AREAS)} performance areas ({', '.join(AREAS)}), each measure averaged over areas. "
        "Games view = teammates' lead + game length removed; hands view = also the champion (performance_shape).",
        "",
        "| role | games | players (10+) " + header.format("x")[5:],
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role, r in results.items():
        cells = [f"{r[v][m]:.3f}" for m in MEASURES for v in VIEWS]
        lines.append(f"| {role} | {r['n_games']} | {r['n_players']} | " + " | ".join(cells) + " |")
    lines += ["", "Per area (median over roles):", "", header.format("area"), "|---|---:|---:|---:|---:|---:|---:|"]
    for area, views in per_area.items():
        cells = [f"{np.nanmedian([d[m] for d in views[v]]):.3f}" for m in MEASURES for v in VIEWS]
        lines.append(f"| {area} | " + " | ".join(cells) + " |")
    lines += ["", "Range across roles:", ""]
    for m in MEASURES:
        lines.append(f"- {m}: games {_range(results, 'games', m)}, hands {_range(results, 'hands', m)}")
    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))
    print(f"Report written to {args.report_path}")


if __name__ == "__main__":
    run(parse_args())

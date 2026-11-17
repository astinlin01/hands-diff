"""
Phase 5 exporter tests (dashboard_payload.py): what the static page receives.

- Reference curves come from player averages (players with 10+ games in the role), per
  region and tier, and are dropped when a tier has too few players (the page then shows
  "insufficient data (n=X)"). Each curve carries its median for the gap noise check.
- Style: per-game trait scores for the role's traits, plus single-game reliability.
- Champion pool reference: the tier's average class mix and the spread of effective
  champion counts.
- Everything is JSON-safe: NaN becomes null.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES_DIR

from hands_diff.features.build import build_feature_rows
from hands_diff.report.payload import json_number, player_average_curves, pool_reference, reference_by_tier
from hands_diff.config.features import add_derived_features
from hands_diff.analysis.adjustment import add_game_state_columns


def rows_for(players: dict[str, int], region="NA1", tier="CHALLENGER") -> pd.DataFrame:
    """players: {puuid: games}. One row per game."""
    puuids = [p for p, n in players.items() for _ in range(n)]
    return pd.DataFrame({"puuid": puuids, "region": region, "tier": tier})


# --- Curves -------------------------------------------------------------------------------------

def test_curves_use_player_averages_with_min_games():
    rows = rows_for({"a": 10, "b": 10, "c": 3})
    values = pd.DataFrame({"x": [1.0] * 10 + [3.0] * 10 + [100.0] * 3})
    curves = player_average_curves(rows, values, min_games=10, min_players=2)
    assert curves["x"]["n"] == 2                      # c (3 games) left out
    assert curves["x"]["ref_avg"] == pytest.approx(2.0)
    assert curves["x"]["median"] == pytest.approx(2.0)
    assert curves["x"]["breakpoints"][0] == pytest.approx(1.0)
    assert curves["x"]["breakpoints"][-1] == pytest.approx(3.0)
    assert len(curves["x"]["breakpoints"]) == 51


def test_curves_skip_missing_values_per_player():
    # A player's average ignores games where the stat is missing (e.g. no timeline).
    rows = rows_for({"a": 10, "b": 10})
    values = pd.DataFrame({"x": [2.0] * 5 + [np.nan] * 5 + [4.0] * 10})
    curves = player_average_curves(rows, values, min_games=10, min_players=2)
    assert curves["x"]["ref_avg"] == pytest.approx(3.0)


def test_too_few_players_gives_no_curves():
    rows = rows_for({"a": 10})
    values = pd.DataFrame({"x": [1.0] * 10})
    assert player_average_curves(rows, values, min_games=10, min_players=2) == {}


def test_reference_by_tier_keys_and_insufficient_tiers():
    rows = pd.concat([
        rows_for({"a": 10, "b": 10}, tier="CHALLENGER"),
        rows_for({"c": 10}, tier="DIAMOND"),             # 1 player: below the minimum
    ], ignore_index=True)
    values = pd.DataFrame({"x": np.arange(len(rows), dtype=float)})
    reference = reference_by_tier(rows, values, ["NA1"], ["CHALLENGER", "DIAMOND"], min_games=10, min_players=2)
    assert set(reference["NA1"]) == {"CHALLENGER"}
    # The page needs n for "insufficient data (n=X)", so tiers that fall short report their size.
    assert reference["_player_counts"]["NA1"]["DIAMOND"] == 1


# --- Champion pool reference ----------------------------------------------------------------------

def test_pool_reference():
    rows = rows_for({"a": 10, "b": 10})
    rows["championName"] = ["Malphite"] * 10 + ["Jinx"] * 5 + ["Ashe"] * 5
    rows["champion_class"] = ["Tank"] * 10 + ["Marksman"] * 10
    pool = pool_reference(rows, min_games=10, min_players=2)
    assert pool["class_mix"] == pytest.approx({"Tank": 0.5, "Marksman": 0.5})
    # a: 1 champion; b: 5/5 -> unbiased HHI 40/90 -> 2.25, capped at 2 champions played.
    assert pool["effective_champions"]["breakpoints"][0] == pytest.approx(1.0)
    assert pool["effective_champions"]["breakpoints"][-1] == pytest.approx(2.0)
    assert pool["n"] == 2


@pytest.fixture(scope="module")
def fixture_df():
    path = os.path.join(FIXTURES_DIR, "matches.ndjson")
    rows = build_feature_rows(path, info_by_puuid={}, min_game_length_seconds=300)
    return add_game_state_columns(add_derived_features(pd.DataFrame(rows)))


# --- JSON safety ------------------------------------------------------------------------------------------

def test_json_number():
    assert json_number(float("nan")) is None
    assert json_number(None) is None
    assert json_number(np.float64(1.234567)) == 1.2346
    assert json.dumps(json_number(np.int64(3))) == "3"


# --- Style map: one point per reference player ------------------------------------------------------
# The map plots every qualifying reference player (10+ games) as their average trait scores,
# in the role's trait order; the page turns them into percentiles with the tier's curves.

from hands_diff.report.payload import crop_splash, player_points, splash_champions


def test_player_points_are_player_averages():
    rows = rows_for({"a": 10, "b": 12, "few": 3})
    # a: Lane Bully all 1.0, Duelist 0.0/2.0 alternating -> 1.0; b: 2.0 and 4.0; few: excluded
    scores = pd.DataFrame({
        "Lane Bully": [1.0] * 10 + [2.0] * 12 + [9.0] * 3,
        "Duelist": [0.0, 2.0] * 5 + [4.0] * 12 + [9.0] * 3,
    })
    points = player_points(rows, scores, ["Lane Bully", "Duelist"], ["NA1"], ["CHALLENGER"], min_players=2)
    assert sorted(points["NA1"]["CHALLENGER"]) == [[1.0, 1.0], [2.0, 4.0]]


def test_player_points_missing_is_null_and_small_tiers_dropped():
    rows = rows_for({"a": 10, "b": 10})
    scores = pd.DataFrame({"Lane Bully": [np.nan] * 10 + [0.12345] * 10})
    points = player_points(rows, scores, ["Lane Bully"], ["NA1"], ["CHALLENGER"], min_players=2)
    assert sorted(points["NA1"]["CHALLENGER"], key=str) == [[0.123], [None]]  # 3 decimals
    assert player_points(rows, scores, ["Lane Bully"], ["NA1"], ["CHALLENGER"], min_players=3) == {}


# --- Player card splash art -----------------------------------------------------------------------

def test_splash_champions_top_pick_per_account_and_role():
    games = pd.DataFrame({
        "account": ["main"] * 9 + ["alt"] * 4,
        "teamPosition": ["BOTTOM"] * 6 + ["TOP"] * 3 + ["MIDDLE"] * 4,
        "championName": ["Kaisa"] * 4 + ["Ezreal"] * 2 + ["Gnar"] * 3 + ["Ahri"] * 4,
    })
    # main BOTTOM -> Kaisa (6 games >= 5). main TOP has 3 games and alt MIDDLE has 4: too few
    # for a role card, but each account's overall top pick is kept for the all-roles view.
    assert splash_champions(games, min_games=5) == {"Kaisa", "Ahri"}


def test_crop_splash_shape_and_size():
    from io import BytesIO

    from PIL import Image
    source = BytesIO()
    Image.effect_noise((1215, 717), 64).convert("RGB").save(source, "JPEG", quality=95)  # splash size
    out = crop_splash(source.getvalue(), width=1200, height=480)
    image = Image.open(BytesIO(out))
    assert image.format == "JPEG" and image.size == (1200, 480)
    assert len(out) < len(source.getvalue())


# --- Enchanter only where it can happen ----------------------------------------------------------
# Heals/shields on teammates are zero in 86-94% of TOP/MIDDLE/BOTTOM games (82-93% of games are
# on champions that essentially never do it; checked 2026-11-13, season 16). There the
# champion-adjusted score mostly reflects which champions you play, so the trait is dropped,
# as it already was for JUNGLE.

# --- Performance shape payload ---------------------------------------------------------------------
# Per game: area scores in both views (the page averages them over the filtered games).
# Per role: reference curves per view and tier, and anonymous Damage/Survival points per tier.

from hands_diff.report.payload import shape_payload
from hands_diff.analysis.shape import AREAS, add_shape_features


def test_shape_payload_structure(fixture_df):
    role_df = add_shape_features(fixture_df[fixture_df["teamPosition"] == "BOTTOM"].reset_index(drop=True))
    payload, per_game = shape_payload(role_df, champion_classes={}, regions=["NA1"], tiers=["CHALLENGER"])
    assert payload["areas"] == list(AREAS)
    assert set(payload["reference"]) == {"games", "hands"}
    assert "_player_counts" in payload["reference"]["hands"]
    assert payload["points_areas"] == ["Damage", "Survival"]
    # One entry per game in the role, both views, every area (None where undefined).
    assert len(per_game) == len(role_df)
    first = per_game[role_df.index[0]]
    assert set(first) == {"games", "hands", "stats"}
    assert set(first["hands"]) == set(AREAS)
    assert all(v is None or isinstance(v, float) for v in first["games"].values())


def test_shape_payload_without_timelines_still_scores_late_areas(fixture_df):
    # The fixture has no timeline columns: Farming/Damage/Objectives come from end-of-game
    # features; Laning and Survival keep their end-of-game features (lead, plates, deaths).
    role_df = add_shape_features(fixture_df[fixture_df["teamPosition"] == "BOTTOM"].reset_index(drop=True))
    _, per_game = shape_payload(role_df, champion_classes={}, regions=["NA1"], tiers=["CHALLENGER"])
    assert all(game["hands"]["Damage"] is not None for game in per_game.values())
    assert all(game["hands"]["Survival"] is not None for game in per_game.values())


def test_shape_payload_carries_stat_chips(fixture_df):
    from hands_diff.analysis.shape import AREA_STATS
    role_df = add_shape_features(fixture_df[fixture_df["teamPosition"] == "BOTTOM"].reset_index(drop=True))
    payload, per_game = shape_payload(role_df, champion_classes={}, regions=["NA1"], tiers=["CHALLENGER"])
    features = [s[0] for stats in AREA_STATS.values() for s in stats]
    assert payload["stats"] == {area: [list(s) for s in stats] for area, stats in AREA_STATS.items()}
    assert "_player_counts" in payload["stat_reference"]
    first = per_game[role_df.index[0]]
    # Timeline stats (CS diff @10, early deaths, ...) only when the table has them; the fixture doesn't.
    assert set(first["stats"]) == {f for f in features if f in role_df.columns}
    assert "deaths_per_game" in first["stats"]

"""
Performance shape (performance_shape.py): five areas, two views.

  "Your games" view: features adjusted for the teammates' lead and game length only
                     (early <=14:00 features unadjusted) -- what your games look like,
                     champion choices included.
  "Your hands" view: also adjusted for the champion played (early features: champion
                     only) -- how you compare with players on the same champions.

Area score per game = mean of the area's standardized features (signs flipped where lower
is better, e.g. deaths), skipping missing values. A player's area value = mean over games.

Synthetic data where the right answer is known by construction:
- "Kaisa" farms +2 CS/min more than "Jinx" for everyone (a champion effect), and the
  player "picker" only plays Kai'Sa, with otherwise average play;
- the player "hands" deals +250 damage/min more than others on the same champion
  (a player effect), on a 50/50 champion mix.
"""

import math

import numpy as np
import pandas as pd
import pytest

from performance_shape import (
    AREAS,
    add_shape_features,
    area_percentile,
    champion_effect,
    combine_areas,
    fit_shape_model,
    game_area_scores,
    player_area_averages,
)

CLASSES = {"kaisa": "Marksman", "jinx": "Marksman"}
BASE = {
    # feature: (mean, sd)
    "cs_diff_at_10": (0.0, 8.0), "gold_diff_at_14": (0.0, 600.0), "xp_diff_at_14": (0.0, 500.0),
    "max_cs_advantage_lane_opp": (10.0, 6.0), "turret_plates_taken": (1.5, 1.0),
    "cs_per_min": (7.5, 0.6), "gold_earned_per_min": (420.0, 40.0), "team_gold_share": (0.23, 0.02),
    "lane_minions_first_10": (60.0, 6.0),
    "damage_per_minute": (800.0, 120.0), "team_damage_pct": (0.25, 0.04), "damage_to_champions_per_min": (750.0, 110.0),
    "deaths_per_min": (0.18, 0.05), "early_deaths": (1.5, 1.0),
    "damage_to_buildings_per_min": (250.0, 80.0), "turret_participation": (0.4, 0.15), "damage_to_objectives_per_min": (500.0, 150.0),
}
FARM_BONUS_KAISA = 2.0      # CS/min, champion effect (~3.3 sd)
DAMAGE_BONUS_HANDS = 250.0  # damage/min, player effect (~2 sd)


def synthetic_role_df(seed: int = 0, games: int = 2400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(games):
        if i % 12 == 0:
            player, champion = "picker", "Kaisa"
        elif i % 12 == 1:
            player, champion = "hands", ("Kaisa" if i % 24 == 1 else "Jinx")
        else:
            player, champion = f"p{i % 40}", ("Kaisa" if rng.random() < 0.5 else "Jinx")
        win = bool(rng.integers(0, 2))
        row = {
            "puuid": player, "championName": champion, "win": win,
            "gameLength": float(rng.uniform(22, 36) * 60),
            "teammates_kill_diff_per_min": float(rng.normal(0.2 if win else -0.2, 0.1)),
            "teammates_gold_diff_per_min": float(rng.normal(80 if win else -80, 30)),
        }
        for feature, (mean, sd) in BASE.items():
            row[feature] = float(rng.normal(mean, sd))
        if champion == "Kaisa":
            row["cs_per_min"] += FARM_BONUS_KAISA
        if player == "hands":
            row["damage_per_minute"] += DAMAGE_BONUS_HANDS
        rows.append(row)
    return add_shape_features(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def scored():
    df = synthetic_role_df()
    model = fit_shape_model(df, CLASSES, min_champion_games=30)
    games = game_area_scores(df, model, "games")
    hands = game_area_scores(df, model, "hands")
    return df, model, games, hands


# --- Area definitions --------------------------------------------------------------

def test_five_areas_in_display_order():
    assert list(AREAS) == ["Laning", "Farming", "Damage", "Survival", "Objectives"]


def test_no_feature_counts_twice():
    features = [f for spec in AREAS.values() for f, _ in spec]
    assert len(features) == len(set(features))


def test_deaths_count_against_survival():
    assert dict(AREAS["Survival"]) == {"deaths_per_min": -1, "early_deaths": -1}
    assert all(sign == 1 for area, spec in AREAS.items() if area != "Survival" for _, sign in spec)


# --- Derived feature -----------------------------------------------------------------

def test_damage_per_gold():
    df = pd.DataFrame({"damage_to_champions_per_min": [1000.0, 300.0, np.nan], "gold_earned_per_min": [500.0, 0.0, 400.0]})
    out = add_shape_features(df)
    assert out["damage_per_gold"].iloc[0] == pytest.approx(2.0)  # 1000 / 500
    assert math.isnan(out["damage_per_gold"].iloc[1])            # no gold: undefined, not inf
    assert math.isnan(out["damage_per_gold"].iloc[2])


# --- Combining standardized features into areas --------------------------------------

def test_combine_areas_signs_and_missing():
    z = pd.DataFrame({
        "deaths_per_min": [1.0, 1.0], "early_deaths": [0.5, np.nan],
        "damage_per_minute": [2.0, 0.0], "team_damage_pct": [1.0, 0.0], "damage_per_gold": [0.0, np.nan],
    })
    areas = combine_areas(z)
    # Survival = -(1.0 + 0.5) / 2; second game has no early_deaths -> -(1.0) / 1
    assert areas.loc[0, "Survival"] == pytest.approx(-0.75)
    assert areas.loc[1, "Survival"] == pytest.approx(-1.0)
    # Damage = (2 + 1 + 0) / 3; second game skips the missing feature: (0 + 0) / 2
    assert areas.loc[0, "Damage"] == pytest.approx(1.0)
    assert areas.loc[1, "Damage"] == pytest.approx(0.0)
    # An area with none of its features present is missing, not zero.
    assert areas["Laning"].isna().all()


# --- The two views -------------------------------------------------------------------

def test_champion_driven_farming_shows_in_games_view_only(scored):
    df, _, games, hands = scored
    pick = (df["puuid"] == "picker").to_numpy()
    # As played, Kai'Sa's +2 CS/min lifts the picker's Farming above average. Hand check:
    # with ~half the games on Kai'Sa, the role's CS/min sd = sqrt(0.6^2 + 1^2) ~= 1.17 and
    # its mean ~= 8.5, so the picker (9.5) is ~+0.8 sd on 1 of 4 Farming features -> ~+0.2.
    # On the same champion it's average again.
    assert games.loc[pick, "Farming"].mean() > 0.12
    assert abs(hands.loc[pick, "Farming"].mean()) < 0.05


def test_player_driven_damage_survives_both_views(scored):
    df, _, games, hands = scored
    mine = (df["puuid"] == "hands").to_numpy()
    # +250 damage/min on 1 of 3 Damage features (~2 sd): about +0.65 on the area in both views.
    assert hands.loc[mine, "Damage"].mean() > 0.4
    assert games.loc[mine, "Damage"].mean() == pytest.approx(hands.loc[mine, "Damage"].mean(), abs=0.1)


def test_views_are_standardized_against_the_whole_role(scored):
    _, _, games, hands = scored
    for view in (games, hands):
        assert view.mean().abs().max() < 0.05


# --- Player averages, percentiles, champion effect ------------------------------------

def test_player_area_averages_respects_min_games():
    scores = pd.DataFrame({"Damage": [1.0, 3.0, 5.0]})
    puuids = pd.Series(["a", "a", "b"])
    out = player_area_averages(scores, puuids, min_games=2)
    assert list(out.index) == ["a"]
    assert out.loc["a", "Damage"] == pytest.approx(2.0)


def test_area_percentile_counts_ties_half():
    reference = pd.Series([1.0, 2.0, 3.0, 4.0])
    # 2 below, 1 tie counted half -> 2.5 of 4 = 62.5
    assert area_percentile(3.0, reference) == pytest.approx(62.5)
    assert math.isnan(area_percentile(np.nan, reference))


def test_champion_effect_size_and_driver():
    games = pd.DataFrame({"Farming": [0.8, 0.8, 0.1]})
    hands = pd.DataFrame({"Farming": [0.5, 0.5, 0.2]})
    champions = pd.Series(["Kaisa", "Kaisa", "Jinx"])
    out = champion_effect(games, hands, champions, reference_sd=pd.Series({"Farming": 0.5}))
    # per-game (games - hands) = [0.3, 0.3, -0.1]: mean 1/6, / sd 0.5 = 0.333
    assert out["Farming"]["effect_sd"] == pytest.approx(1 / 3)
    # Kai'Sa pushes +0.6 of the +0.5 total -> share 1.2; Jinx pulls the other way.
    assert out["Farming"]["driver"] == "Kaisa"
    assert out["Farming"]["driver_share"] == pytest.approx(1.2)


def test_champion_effect_on_synthetic_picker(scored):
    df, _, games, hands = scored
    pick = (df["puuid"] == "picker").to_numpy()
    sd = player_area_averages(hands, df["puuid"], min_games=10).std()
    out = champion_effect(games[pick], hands[pick], df.loc[pick, "championName"], reference_sd=sd)
    assert out["Farming"]["effect_sd"] > 0.5
    assert out["Farming"]["driver"] == "Kaisa"
    assert abs(out["Damage"]["effect_sd"]) < 0.2


# --- Stat chips: the raw numbers behind each area -------------------------------------------------

from performance_shape import AREA_STATS


def test_every_area_has_stat_chips_with_formats():
    assert list(AREA_STATS) == list(AREAS)
    for area, stats in AREA_STATS.items():
        assert 1 <= len(stats) <= 3
        for feature, label, decimals, signed, percent in stats:
            assert isinstance(label, str) and decimals in (0, 1, 2) and isinstance(signed, bool) and isinstance(percent, bool)


def test_deaths_per_game():
    df = pd.DataFrame({"damage_to_champions_per_min": [1.0], "gold_earned_per_min": [1.0],
                       "deaths_per_min": [0.2], "gameLength": [1800.0]})
    # 0.2 deaths/min over a 30-minute game = 6 deaths
    assert add_shape_features(df)["deaths_per_game"].iloc[0] == pytest.approx(6.0)

"""
The dashboard recomputes the shape, pool stats and gaps in the browser
(so the game-window and champion filters stay live on a static page). That math lives in
report/web/profile_math.js, a second copy of the Python logic -- these tests run both
copies on the same inputs through Node and require the same answers, so the JS can't
silently drift from the tested Python.

Skipped if Node isn't installed.
"""

import json
import math
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from hands_diff.analysis.pool import class_mix, effective_champions, top_champion
from hands_diff.analysis.gaps import biggest_gaps, own_sample_status

from hands_diff.report.build import WEB_DIR

JS_PATH = os.path.join(WEB_DIR, "profile_math.js")
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")


def js(function: str, *args):
    """Call ProfileMath.<function>(*args) in Node and return the JSON result."""
    script = (
        "const M = require(process.argv[1]);"
        "const args = JSON.parse(process.argv[2]);"
        f"process.stdout.write(JSON.stringify(M.{function}(...args)));"
    )
    out = subprocess.run(
        [NODE, "-e", script, os.path.abspath(JS_PATH), json.dumps(args)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def nan_to_none(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


# --- Area profile: one player's average per area ------------------------------------------------

def test_area_profile_hand_checked():
    games = [{"Damage": 1.0, "Survival": 0.5}, {"Damage": 3.0, "Survival": None}, {"Damage": None, "Survival": 2.0}]
    result = js("areaProfile", games, ["Damage", "Survival"])
    # Damage: (1 + 3) / 2 over 2 games; Survival: (0.5 + 2) / 2 over 2 games
    assert result["profile"] == {"Damage": pytest.approx(2.0), "Survival": pytest.approx(1.25)}
    assert result["counts"] == {"Damage": 2, "Survival": 2}


def test_area_profile_all_missing_is_null():
    result = js("areaProfile", [{"Damage": None}], ["Damage"])
    assert result["profile"]["Damage"] is None
    assert result["counts"]["Damage"] == 0


@pytest.mark.parametrize("games", [0, 4, 5, 9, 10, 50])
def test_own_sample_status_matches_python(games):
    assert js("ownSampleStatus", games) == own_sample_status(games)


# --- Champion pool --------------------------------------------------------------------------

@pytest.mark.parametrize("counts", [
    {"Jinx": 27, "Ashe": 2, "Ezreal": 1},
    {"Jinx": 10, "Ashe": 10, "Ezreal": 10},
    {f"c{i}": 1 for i in range(10)},
    {"Jinx": 1},
])
def test_pool_stats_match_python(counts):
    series = pd.Series(counts)
    expected = effective_champions(series)
    result = js("effectiveChampions", counts)
    if math.isnan(expected):
        assert result is None
    else:
        assert result == pytest.approx(expected)
    name, share = top_champion(series)
    assert js("topChampion", counts) == [name, pytest.approx(share)]


def test_class_mix_matches_python():
    # The page receives each game's class already resolved (support overrides applied in
    # Python), so the JS only has to count.
    labels = ["Tank", "Tank", "Pick", "Enchanter"]
    expected = class_mix(pd.Series(labels), "TOP", {"tank": "Tank", "pick": "Pick", "enchanter": "Enchanter"}, {})
    result = js("classMix", labels)
    assert result == pytest.approx(expected.to_dict())


# --- Percentiles and gaps ----------------------------------------------------------------------

def test_percentile_from_breakpoints_close_to_exact():
    # The page only has 51 percentile cutoffs per stat, not every player's average, so
    # its percentiles are interpolated; they should stay within ~2 points of exact.
    from scipy.stats import percentileofscore

    rng = np.random.default_rng(3)
    reference = rng.normal(500, 80, 400)
    breakpoints = np.percentile(reference, np.linspace(0, 100, 51)).tolist()
    for value in [380.0, 450.0, 500.0, 560.0, 640.0]:
        exact = percentileofscore(reference, value, kind="mean")
        assert js("percentileFromBreakpoints", value, breakpoints) == pytest.approx(exact, abs=2.0)


def test_biggest_gaps_match_python():
    # Same fixture as test_performance's gap tests: reference player averages 0..9.
    gap_features = {
        "gold": ["Farming", 1], "cs": ["Farming", 1], "deaths": ["Survival", -1],
        "vision": ["Vision", 1], "kills": ["Combat", 1],
    }
    reference = pd.DataFrame({c: np.arange(10, dtype=float) for c in gap_features})
    mine = pd.DataFrame({
        "gold": [2.0] * 4, "cs": [1.0] * 4, "deaths": [7.0] * 4,
        "vision": [0.0, 6.0, 0.0, 6.0], "kills": [6.0] * 4,
    })
    expected = biggest_gaps(mine, reference, {k: tuple(v) for k, v in gap_features.items()}, top=3)

    # The page gets each stat's reference as percentile cutoffs (median = the 50th).
    curves = {
        c: {"breakpoints": np.percentile(reference[c], np.linspace(0, 100, 51)).tolist(),
            "median": float(reference[c].median())}
        for c in gap_features
    }
    result = js("biggestGaps", mine.to_dict("records"), curves, gap_features, 3)
    assert [g["feature"] for g in result] == [g["feature"] for g in expected]
    assert [g["theme"] for g in result] == [g["theme"] for g in expected]
    for got, want in zip(result, expected):
        assert got["percentile"] == pytest.approx(want["percentile"], abs=6.0)  # coarse grid on 10 players
        assert got["your_avg"] == pytest.approx(want["your_avg"])
        assert got["n_games"] == want["n_games"]


@pytest.mark.parametrize("feature, you, median", [
    ("gold_earned_per_min", 504.0, 515.0),
    ("deaths_per_min", 0.25, 0.20),
    ("kill_participation", 0.55, 0.62),
    ("kda", 3.1, 3.5),
])
def test_gap_size_matches_python(feature, you, median):
    from hands_diff.analysis.gaps import gap_size
    expected = gap_size(feature, you, median, game_minutes=30.0, percent_features=["kill_participation"])
    result = js("gapSize", feature, you, median, 30.0, ["kill_participation"])
    assert result["amount"] == pytest.approx(expected["amount"])
    assert result["per_game"] == expected["per_game"]


# --- Performance shape --------------------------------------------------------------------------

from hands_diff.analysis.shape import champion_effect


def test_champion_effect_matches_python():
    games = [{"Farming": 0.8, "Damage": 0.5}, {"Farming": 0.8, "Damage": None}, {"Farming": 0.1, "Damage": 0.2}]
    hands = [{"Farming": 0.5, "Damage": 0.5}, {"Farming": 0.5, "Damage": None}, {"Farming": 0.2, "Damage": 0.1}]
    champions = ["Kaisa", "Kaisa", "Jinx"]
    sd = {"Farming": 0.5, "Damage": 0.4}
    expected = champion_effect(
        pd.DataFrame(games, dtype=float), pd.DataFrame(hands, dtype=float), pd.Series(champions), pd.Series(sd)
    )
    got = js("championEffect", games, hands, champions, sd, ["Farming", "Damage"])
    for area in ("Farming", "Damage"):
        assert got[area]["effect_sd"] == pytest.approx(expected[area]["effect_sd"])
        assert got[area]["driver"] == expected[area]["driver"]
        assert got[area]["driver_share"] == pytest.approx(expected[area]["driver_share"])


def test_champion_effect_hand_checked():
    # Farming per-game (games - hands): [0.3, 0.3, -0.1] -> mean 1/6, / sd 0.5 = 1/3;
    # Kai'Sa pushes +0.6 of the +0.5 total -> share 1.2.
    got = js("championEffect", [{"Farming": 0.8}, {"Farming": 0.8}, {"Farming": 0.1}],
             [{"Farming": 0.5}, {"Farming": 0.5}, {"Farming": 0.2}], ["Kaisa", "Kaisa", "Jinx"], {"Farming": 0.5}, ["Farming"])
    assert got["Farming"] == {"effect_sd": pytest.approx(1 / 3), "driver": "Kaisa", "driver_share": pytest.approx(1.2)}


@pytest.mark.parametrize("percentile, top, band, verdict", [
    (95.3, 5, "top 10%", "strength"),
    (89.8, 10, "top 10%", "strength"),     # rounds to top 10%: the label and the band must agree
    (87.4, 13, "top 25%", "strength"),
    (75.0, 25, "top 25%", "strength"),
    (60.0, 40, "middle", "middle"),
    (25.0, 75, "bottom 25%", "weakness"),
    (8.0, 92, "bottom 10%", "weakness"),
    (99.9, 1, "top 10%", "strength"),       # never "top 0%"
])
def test_shape_bands(percentile, top, band, verdict):
    assert js("topPct", percentile) == top
    assert js("shapeBand", percentile) == band
    assert js("shapeVerdict", percentile) == verdict


@pytest.mark.parametrize("damage, survival, label", [
    (90, 78, "Efficient carry"), (20, 80, "Safe, low impact"), (85, 10, "Glass cannon"), (30, 30, "Struggling"),
    (50, 50, "Efficient carry"),  # the median player sits on the corner; ties go up and right
])
def test_quadrant_label(damage, survival, label):
    assert js("quadrantLabel", damage, survival) == label

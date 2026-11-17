"""
End to end: the real export (hands-diff export) on the anonymized fixture games, with their
early-game features merged in, then the page build (hands-diff page).

Unit tests cover each piece; this one catches wiring mistakes between them (a function
moved without its import, a payload key renamed on one side only). No network: icons and
splash art come from the (empty) cache, champion classes are stubbed, and every file lives
in a temporary folder.
"""
import json
import os

import pandas as pd
import pytest
from conftest import FIXTURES_DIR

from hands_diff.analysis.shape import AREAS
from hands_diff.features.build import build_feature_rows
from hands_diff.features.timeline import extract_timeline_features
from hands_diff.report import build as page_build
from hands_diff.report import export
from hands_diff.storage import iter_ndjson

ME = "me#na1"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export")
    matches_path = os.path.join(FIXTURES_DIR, "matches.ndjson")

    features = pd.DataFrame(build_feature_rows(matches_path, info_by_puuid={}, min_game_length_seconds=300))
    features["tier"], features["region"] = "CHALLENGER", "NA1"  # one reference tier for every fixture player
    features.to_parquet(tmp / "features.parquet", index=False)

    matches = {m["metadata"]["matchId"]: m for m in iter_ndjson(matches_path)}
    rows = []
    for timeline in iter_ndjson(os.path.join(FIXTURES_DIR, "timelines.ndjson")):
        match = matches.get(timeline["metadata"]["matchId"])
        if match:
            rows.extend(extract_timeline_features(timeline, match))
    pd.DataFrame(rows).to_parquet(tmp / "timeline.parquet", index=False)

    # "You" = the bottom laner of the first fixture game.
    me = features[features["teamPosition"] == "BOTTOM"].iloc[0]
    with open(tmp / "match_ids.ndjson", "w", encoding="utf-8") as f:
        f.write(json.dumps({"matchId": me["matchId"], "source_riot_id": ME, "source_puuid": me["puuid"]}) + "\n")

    mp = pytest.MonkeyPatch()
    mp.chdir(tmp)  # icon/splash caches resolve under the temp folder: empty, so nothing is downloaded
    mp.setattr(export, "load_champion_classes", lambda: {})
    mp.setattr(export, "default_compare_tiers", lambda: {ME: "CHALLENGER"})
    try:
        export.run(export.parse_args([
            "--features-path", str(tmp / "features.parquet"),
            "--timeline-features-path", str(tmp / "timeline.parquet"),
            "--matches-path", matches_path,
            "--match-ids-path", str(tmp / "match_ids.ndjson"),
            "--pro-features-path", str(tmp / "no_pro.parquet"),
            "--output-path", str(tmp / "data.json"),
            "--riot-id", ME,
            "--cached-icons-only",
        ]))
        page_build.run(page_build.parse_args(["--data-path", str(tmp / "data.json"), "--output-path", str(tmp / "page.html")]))
    finally:
        mp.undo()
    with open(tmp / "data.json", encoding="utf-8") as f:
        data = json.load(f)
    return data, tmp, me


def test_your_games_carry_both_views_and_stat_chips(exported):
    data, _, me = exported
    mine = [g for g in data["games"] if g["account"] == ME]
    assert [g["matchId"] for g in mine] == [me["matchId"]]
    shape = mine[0]["shape"]
    assert set(shape) == {"games", "hands", "stats"}
    assert set(shape["hands"]) == set(AREAS)
    # Early-game stats are present only if the timeline merge ran.
    assert shape["stats"]["cs_diff_at_10"] is not None


def test_role_payload_has_the_shape_and_no_retired_keys(exported):
    data, _, _ = exported
    bottom = data["roles"]["BOTTOM"]
    assert bottom["shape"]["areas"] == list(AREAS)
    for retired in ("traits", "trait_reliability", "archetypes", "style_reference", "trait_points"):
        assert retired not in bottom
    assert "confidence_thresholds" not in data


def test_comparison_tier_comes_from_compare_tiers(exported):
    data, _, _ = exported
    assert data["compare_tiers"] == {ME: "CHALLENGER"}


def test_page_embeds_the_data(exported):
    _, tmp, _ = exported
    html = (tmp / "page.html").read_text(encoding="utf-8")
    assert page_build.DATA_PLACEHOLDER not in html and page_build.MATH_PLACEHOLDER not in html
    assert '"compare_tiers"' in html

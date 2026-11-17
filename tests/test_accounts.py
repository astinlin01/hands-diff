"""Your Riot IDs come from the RIOT_IDS setting (.env), not from hardcoded lists in the scripts."""
import pytest

from accounts import parse_riot_ids, require_riot_ids


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("river3#na1", ["river3#na1"]),
        ("a#na1, b#euw ,c#kr", ["a#na1", "b#euw", "c#kr"]),  # whitespace around commas
        ('"a#na1,b#euw"', ["a#na1", "b#euw"]),  # quoted value, as some .env editors write it
        ("a#na1,,A#NA1", ["a#na1"]),  # blanks dropped; Riot IDs are case-insensitive, so dedupe
        ("Gamer Name#NA1", ["Gamer Name#NA1"]),  # spaces inside a game name are legal
        ("", []),
        (None, []),
    ],
)
def test_parse_riot_ids(raw, expected):
    assert parse_riot_ids(raw) == expected


def test_parse_rejects_missing_tag():
    with pytest.raises(ValueError, match="gameName#tagLine"):
        parse_riot_ids("river3")


def test_require_prefers_cli_ids_over_env(monkeypatch):
    monkeypatch.setenv("RIOT_IDS", "env#na1")
    assert require_riot_ids(["cli#na1"]) == ["cli#na1"]
    assert require_riot_ids([]) == ["env#na1"]


def test_require_explains_how_to_configure_when_empty(monkeypatch):
    monkeypatch.setenv("RIOT_IDS", "")
    with pytest.raises(SystemExit, match="RIOT_IDS"):
        require_riot_ids([])


# --- Which accounts go on the public page ----------------------------------------------------------
# PUBLISH_RIOT_IDS is separate from RIOT_IDS (all your accounts, used for fetching), so a plain
# rebuild can't put an account on the public site that you didn't choose to publish.

from accounts import require_publish_riot_ids


def test_publish_uses_its_own_list(monkeypatch):
    monkeypatch.setenv("RIOT_IDS", "main#na1,alt#na1")
    monkeypatch.setenv("PUBLISH_RIOT_IDS", "main#na1")
    assert require_publish_riot_ids([]) == ["main#na1"]


def test_publish_prefers_cli_ids(monkeypatch):
    monkeypatch.setenv("PUBLISH_RIOT_IDS", "main#na1")
    assert require_publish_riot_ids(["cli#na1"]) == ["cli#na1"]


def test_publish_never_falls_back_to_all_accounts(monkeypatch):
    # Unset publish list + set RIOT_IDS must stop, not quietly publish every account.
    monkeypatch.setenv("RIOT_IDS", "main#na1,alt#na1")
    monkeypatch.setenv("PUBLISH_RIOT_IDS", "")
    with pytest.raises(SystemExit, match="PUBLISH_RIOT_IDS"):
        require_publish_riot_ids([])


# --- Which tier each account is compared against (COMPARE_TIERS) ---------------------------------
# Matchmaking puts you against players of your MMR, so the fair ruler is the tier you actually
# play in -- which can differ from your displayed rank (e.g. Master rank, GM lobbies).

from accounts import compare_tier_for, parse_compare_tiers


def test_parse_compare_tiers():
    raw = "river#ell:GRANDMASTER, River3#NA1:master"
    assert parse_compare_tiers(raw) == {"river#ell": "GRANDMASTER", "river3#na1": "MASTER"}


def test_parse_compare_tiers_accepts_gm_shorthand_and_blank():
    assert parse_compare_tiers("a#na1:GM") == {"a#na1": "GRANDMASTER"}
    assert parse_compare_tiers("") == {}
    assert parse_compare_tiers(None) == {}


@pytest.mark.parametrize("raw, message", [
    ("river#ell", "name#tag:TIER"),          # no tier
    ("river:MASTER", "gameName#tagLine"),    # no tag
    ("river#ell:IRON", "IRON"),              # a tier we have no reference data for
])
def test_parse_compare_tiers_rejects_bad_entries(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_compare_tiers(raw)


def test_compare_tier_lookup_is_case_insensitive_and_optional():
    tiers = parse_compare_tiers("river#ell:GRANDMASTER")
    assert compare_tier_for("River#ELL", tiers) == "GRANDMASTER"
    assert compare_tier_for("other#na1", tiers) is None  # unset: the page's default applies

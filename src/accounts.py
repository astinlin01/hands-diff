"""Which Riot accounts count as "you".

Set them once in .env (gitignored) instead of in code:

    RIOT_IDS=yourname#na1, smurf#na1
    PUBLISH_RIOT_IDS=yourname#na1

RIOT_IDS is every account you own (fetching, comparisons). PUBLISH_RIOT_IDS is the subset
that goes on the public dashboard; the export never falls back to RIOT_IDS, so an account
only reaches the public site if you listed it there.

Every script that needs your accounts calls default_riot_ids() / require_riot_ids(), so the
list lives in one place and your accounts never land in the public repo.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

ENV_VAR = "RIOT_IDS"
PUBLISH_ENV_VAR = "PUBLISH_RIOT_IDS"


def parse_riot_ids(raw: str | None, env_var: str = ENV_VAR) -> list[str]:
    """Split a comma-separated RIOT_IDS value into clean, deduplicated gameName#tagLine strings."""
    if not raw:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for part in raw.strip().strip('"').strip("'").split(","):
        riot_id = part.strip()
        if not riot_id:
            continue
        name, _, tag = riot_id.partition("#")
        if not name.strip() or not tag.strip():
            raise ValueError(f"{env_var}: {riot_id!r} is not in gameName#tagLine form")
        if riot_id.lower() not in seen:  # Riot IDs are case-insensitive
            seen.add(riot_id.lower())
            ids.append(riot_id)
    return ids


COMPARE_ENV_VAR = "COMPARE_TIERS"
# Tiers with reference data (see export_dashboard_data.TIER_RANK); "GM" is accepted as shorthand.
COMPARE_TIER_NAMES = ("CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND")
_TIER_ALIASES = {"GM": "GRANDMASTER"}


def parse_compare_tiers(raw: str | None) -> dict[str, str]:
    """COMPARE_TIERS=name#tag:TIER[, ...] -> {lowercased riot id: TIER}.

    The tier each account is ranked against by default. Matchmaking pairs you with players
    of your MMR, so this should be the tier you actually play in, which can differ from
    the rank shown on your profile (e.g. Master rank with GM-level lobbies).
    """
    if not raw:
        return {}
    tiers: dict[str, str] = {}
    for part in raw.strip().strip('"').strip("'").split(","):
        entry = part.strip()
        if not entry:
            continue
        riot_id, sep, tier = entry.rpartition(":")
        if not sep:
            raise ValueError(f"{COMPARE_ENV_VAR}: {entry!r} is not in name#tag:TIER form")
        parse_riot_ids(riot_id, COMPARE_ENV_VAR)  # validates gameName#tagLine
        tier = _TIER_ALIASES.get(tier.strip().upper(), tier.strip().upper())
        if tier not in COMPARE_TIER_NAMES:
            raise ValueError(f"{COMPARE_ENV_VAR}: tier {tier!r} has no reference data (use one of {', '.join(COMPARE_TIER_NAMES)})")
        tiers[riot_id.strip().lower()] = tier
    return tiers


def default_compare_tiers() -> dict[str, str]:
    load_dotenv()
    return parse_compare_tiers(os.getenv(COMPARE_ENV_VAR))


def compare_tier_for(riot_id: str, tiers: dict[str, str]) -> str | None:
    """The account's comparison tier, or None if unset (the page's default applies)."""
    return tiers.get(riot_id.strip().lower())


def default_riot_ids() -> list[str]:
    """Riot IDs from RIOT_IDS (environment or .env). Empty list if unset."""
    load_dotenv()  # no override: a real environment variable beats .env
    return parse_riot_ids(os.getenv(ENV_VAR))


def require_riot_ids(cli_ids: list[str] | None) -> list[str]:
    """CLI --riot-id values if given, else RIOT_IDS; exit with setup instructions if neither."""
    ids = list(cli_ids or []) or default_riot_ids()
    if not ids:
        raise SystemExit(
            f"No Riot IDs given. Pass --riot-id name#tag, or add {ENV_VAR}=name#tag[,other#tag] to .env."
        )
    return ids


def require_publish_riot_ids(cli_ids: list[str] | None) -> list[str]:
    """Accounts for the public dashboard: CLI --riot-id values if given, else PUBLISH_RIOT_IDS.
    Deliberately no fallback to RIOT_IDS -- exits with instructions instead."""
    load_dotenv()
    ids = list(cli_ids or []) or parse_riot_ids(os.getenv(PUBLISH_ENV_VAR), PUBLISH_ENV_VAR)
    if not ids:
        raise SystemExit(
            f"No accounts to publish. Add {PUBLISH_ENV_VAR}=name#tag[,other#tag] to .env (the accounts "
            f"shown on the public dashboard), or pass --riot-id name#tag. {ENV_VAR} is not used here on purpose."
        )
    return ids

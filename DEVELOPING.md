# Developing hands-diff

Layout, every command, the data files and the full run order. For what the report shows and why the method works this way, see the [README](README.md).

Run everything from the repo root: commands read and write `data/` relative to the current folder.

## Layout

```
src/hands_diff/
  cli.py              the hands-diff command (also: python -m hands_diff)
  storage.py          NDJSON read/append, checkpoints, timestamps
  config/             accounts.py   RIOT_IDS, PUBLISH_RIOT_IDS, COMPARE_TIERS (.env)
                      features.py   curated match-v5 fields -> features, derived features, season tags
                      champions.py  Data Dragon class per champion (+ support subclasses)
  riot/client.py      the Riot API client: routing, auth, rate limiting, retries
  collect/            API fetchers: recent_self, league_pool, match_ids, match_details, timelines
  features/           build.py (feature table), timeline.py + build_timeline.py (early game), pro.py
  analysis/           adjustment.py   teammates' lead / game length / champion adjustment, early-game
                                      adjustment, standardizing
                      shape.py        the five areas, both views, champion effect, stat chips
                      reliability.py  percentiles vs reference players, ICC, Spearman-Brown
                      gaps.py         per-stat percentiles and biggest gaps
                      pool.py         champion pool (effective champions, class mix)
  report/             export.py (JSON payload), payload.py (its building blocks), build.py (the page)
                      web/template.html, web/profile_math.js (the page source and its browser-side math)
  tools/              diagnostics.py, eda.py, feature_summary.py, anonymize_fixtures.py
tests/                mirrors the package; fixtures/ holds anonymized real games
docs/                 the published site only (GitHub Pages serves main /docs)
data/                 gitignored: raw API data and everything derived from it
```

## Commands

`hands-diff` with no arguments lists them; each takes `--help`. Every command is also a module (`python -m hands_diff.collect.match_ids`).

| Command | Module | Does |
|---|---|---|
| `hands-diff refresh` | | `fetch recent` → `build features` → `fetch timelines` → `build timelines` → `export` → `page` |
| `hands-diff fetch recent` | `collect.recent_self` | your accounts' ranked history (`RIOT_IDS`), appended to the shared match files |
| `hands-diff fetch pool` | `collect.league_pool` | a reference player pool (league entries) for one tier and region |
| `hands-diff fetch match-ids` | `collect.match_ids` | deduped match IDs for a pool |
| `hands-diff fetch matches` | `collect.match_details` | full match payloads |
| `hands-diff fetch timelines` | `collect.timelines` | timelines: by default your games plus players with 10+ games in a role; `--all` for the whole season |
| `hands-diff build features` | `features.build` | `data/processed/participant_features.parquet` |
| `hands-diff build timelines` | `features.build_timeline` | `data/processed/timeline_features.parquet` |
| `hands-diff build pro` | `features.pro` | pro features from Oracle's Elixir |
| `hands-diff export` | `report.export` | `data/processed/dashboard_data.json` (accounts in `PUBLISH_RIOT_IDS`) |
| `hands-diff page` | `report.build` | `docs/index.html` |
| `hands-diff diagnostics` | `tools.diagnostics` | `data/processed/method_diagnostics.md`: the README's "Does it hold up?" numbers |
| `hands-diff eda` | `tools.eda` | `data/raw/eda_report.md`: schema and stats of the raw files |
| `hands-diff feature-summary` | `tools.feature_summary` | `data/raw/feature_variable_summary.md`: candidate fields by theme |

## Building the reference population

The first full build takes many hours on a development key (about 1,900 timelines an hour; a key expires every 24h). Every fetcher resumes where it stopped, so a fresh key in `.env` and a rerun continues. Keep the machine from sleeping during long fetches.

1. **Player pools,** one file per tier *and* region. `build features` infers apex tiers from the filename, so don't merge tiers into one file. `fetch pool` tags every row with its `--platform`, so region is never ambiguous.
   ```
   hands-diff fetch pool --tier DIAMOND --division II --num-pages 5
   hands-diff fetch pool --tier MASTER --output-path data/raw/league_entries_master.json
   hands-diff fetch pool --tier GRANDMASTER --output-path data/raw/league_entries_grandmaster.json
   hands-diff fetch pool --tier CHALLENGER --output-path data/raw/league_entries_challenger.json
   hands-diff fetch pool --platform kr --tier CHALLENGER --output-path data/raw/league_entries_kr_challenger.json
   ```
   Top tiers (Challenger, Grandmaster, Master) use the league endpoint (no division or pages). `--positive-winrate-only` keeps players with more wins than losses; `--append` merges into an existing file by puuid.

2. **Match IDs,** once per pool file (one `--league-entries-path` per run; matching `--platform` for non-NA files). Master pools can hit the API's 10,000-player page cap, so cap them with `--max-players`. Your own accounts (`RIOT_IDS`) are always included; add `--skip-default-riot-ids` on repeat runs to skip re-resolving them. Match IDs carry their platform (`KR_...`, `NA1_...`), so every region shares one `match_ids.ndjson`.
   ```
   hands-diff fetch match-ids --resume
   hands-diff fetch match-ids --league-entries-path data/raw/league_entries_master.json --max-players 300 --skip-default-riot-ids --checkpoint-path data/raw/checkpoints/match_ids_master_checkpoint.json
   ```

3. **Match data.** `fetch matches` routes each ID to its regional cluster from the prefix, so one run handles a mixed file. Timelines are stored gzipped, one file per match (`data/raw/timelines/<matchId>.json.gz`, ~35–40 KB each), so new early-game features can be extracted later without refetching.
   ```
   hands-diff fetch matches --resume
   hands-diff fetch timelines --all
   ```

4. **Features.**
   ```
   hands-diff build features
   hands-diff build timelines
   hands-diff build pro            # optional, for the pro comparison
   ```
   `build features` keeps the five solo roles, drops remakes under `--min-game-length-seconds` (default 5 minutes), normalizes per minute, adds derived features (e.g. `cs_total_per_min` = lane minions + jungle camps), tags `patch`, `season` and `gameCreation`, and attaches tier and region per puuid from the pool files.

5. **The page.**
   ```
   hands-diff diagnostics          # optional: re-check the method on the new population
   hands-diff export
   hands-diff page
   ```

After that, `hands-diff refresh` is the everyday update: it pulls only your new games and rebuilds everything derived from them.

## Data files

| Path | What |
|---|---|
| `data/raw/league_entries*.json` | player pools, one per tier and region |
| `data/raw/match_ids.ndjson`, `matches.ndjson` | match IDs (with the account that found them) and full payloads |
| `data/raw/timelines/` | gzipped timelines, one per match |
| `data/raw/ddragon/champion.json` | Data Dragon champion data (classes) |
| `data/raw/champion_icon_cache/`, `champion_splash_cache/` | cached art for the page |
| `data/raw/checkpoints/`, `failures/` | resume points and failed requests |
| `data/processed/participant_features.parquet` | one row per player per game |
| `data/processed/timeline_features.parquet` | early-game (≤14:00) features |
| `data/processed/pro_player_features.parquet` | pro games (Oracle's Elixir) |
| `data/processed/dashboard_data.json` | the page's data |

## The page

`report/export.py` writes everything the page needs, per role:
- your games: headline stats, adjusted stats, both views' area scores, stat chips, champion class;
- reference curves per region and tier: the 0th–100th percentile cutoffs of reference players' averages (10+ games) for raw stats, adjusted stats, area scores and the champion-pool baseline;
- anonymous Damage/Survival points for the risk chart;
- each account's comparison tier.

`report/build.py` merges that JSON and `web/profile_math.js` into `web/template.html` and writes `docs/index.html`. Rebuilding and committing that file is the site update (about 3 MB per rebuild in git history).

The page recomputes everything in the browser when a filter changes, with `web/profile_math.js`: a JavaScript copy of the area-profile, champion-effect, verdict, pool and gap math. `tests/report/test_profile_math.py` runs it through Node against the Python functions on the same inputs, so the two copies can't drift apart.

Any section without enough data says so with the numbers ("insufficient data (n=X)"): under 5 of your games, or under 20 reference players with 10+ games. With 5–9 games, results show with a low-confidence note.

Champion icons and splash art come from Data Dragon at export time, are cached under `data/raw/`, and are embedded as data URIs, so the page works offline. `--cached-icons-only` skips the download.

## Testing

```
pytest
```

- Write or update the tests first, run them, then change the code until they pass; regenerate the page only when the suite is green.
- Expected values in fixtures are checked by hand, with the working in a comment next to them.
- `tests/fixtures/` holds real match payloads and timelines with every identity replaced by a salted hash. To add games, anonymize them first: `python -m hands_diff.tools.anonymize_fixtures <files>`.
- `tests/report/test_export_end_to_end.py` runs the real export and page build on the fixtures. It's the test that catches wiring mistakes between modules.

## Notes

- Riot development keys expire every 24h. Requests then fail with 401: regenerate the key and update `.env`.
- Oracle's Elixir downloads are hosted on Google Drive, whose anonymous quota is sometimes exhausted. Download the file in a logged-in browser instead and pass it as `--input-path`.

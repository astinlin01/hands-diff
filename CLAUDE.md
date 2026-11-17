# hands-diff

League of Legends performance shape: five areas (Laning, Farming, Damage, Survival, Objectives) in two views, "your games" (as played) and "your hands" (on equal champions), ranked against the tier the player actually plays in. Package: `src/hands_diff/`; command: `hands-diff` (see DEVELOPING.md).

## Analysis conventions
- **CS** includes jungle camps: total minions killed + neutral minions killed.
- **Season-match** every comparison: compare a player only against baselines from the same season.
- **Normalize** for game length (per-minute stats) and remove the teammates' lead (not the win flag), so a stomp by your team doesn't inflate your numbers; your own lead stays in.
- **Your hands** also removes the champion played (rare champions pooled by class); early-game (≤14:00) stats are adjusted for champion only. The matchup is not adjusted for: it washes out over a profile.
- **Averages vs averages:** rank a player's average against reference players' averages (10+ games), never against single games.
- **Show comparisons as percentiles**, not basis points or raw differences.
- If a page section doesn't have enough data, show an explicit "insufficient data (n=X)" message. Never render it blank.

## Testing
- Accuracy tests encode the conventions above, using a few real games whose expected values were checked by hand.
- The page's JavaScript math (`report/web/profile_math.js`) is tested against its Python originals through Node.
- Workflow: write or update the tests, run them and show the results, fix the pipeline until they're green, and only then regenerate the page.

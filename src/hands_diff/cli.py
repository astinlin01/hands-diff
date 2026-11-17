"""
The `hands-diff` command (also `python -m hands_diff`). Run from the repo root: every step
reads and writes data/ relative to the current folder.

Each command is a module with parse_args(argv) and run(args); its own --help lists options:

    hands-diff fetch match-ids --help
"""

from __future__ import annotations

import importlib
import sys

# command words -> module. Order = the order they're listed in help.
COMMANDS: dict[tuple[str, ...], str] = {
    ("fetch", "recent"): "hands_diff.collect.recent_self",
    ("fetch", "pool"): "hands_diff.collect.league_pool",
    ("fetch", "match-ids"): "hands_diff.collect.match_ids",
    ("fetch", "matches"): "hands_diff.collect.match_details",
    ("fetch", "timelines"): "hands_diff.collect.timelines",
    ("build", "features"): "hands_diff.features.build",
    ("build", "timelines"): "hands_diff.features.build_timeline",
    ("build", "pro"): "hands_diff.features.pro",
    ("export",): "hands_diff.report.export",
    ("page",): "hands_diff.report.build",
    ("diagnostics",): "hands_diff.tools.diagnostics",
    ("eda",): "hands_diff.tools.eda",
    ("feature-summary",): "hands_diff.tools.feature_summary",
}

DESCRIPTIONS = {
    ("fetch", "recent"): "pull your accounts' new ranked games (RIOT_IDS)",
    ("fetch", "pool"): "fetch a reference player pool (league entries)",
    ("fetch", "match-ids"): "fetch match IDs for the reference pool",
    ("fetch", "matches"): "fetch full match payloads",
    ("fetch", "timelines"): "fetch match timelines (default: the subset around your games)",
    ("build", "features"): "rebuild the per-player feature table",
    ("build", "timelines"): "rebuild the early-game (<=14:00) feature table",
    ("build", "pro"): "build pro features from Oracle's Elixir data",
    ("export",): "export the dashboard data (PUBLISH_RIOT_IDS)",
    ("page",): "build the self-contained page (docs/index.html)",
    ("diagnostics",): "method check: the areas in both views, per role",
    ("eda",): "overview report of the raw collection files",
    ("feature-summary",): "summary of candidate match-v5 fields",
}

# The everyday update: your new games, then everything derived from them, the page last.
REFRESH_STEPS: list[tuple[str, ...]] = [
    ("fetch", "recent"), ("build", "features"), ("fetch", "timelines"),
    ("build", "timelines"), ("export",), ("page",),
]


def usage() -> str:
    lines = ["usage: hands-diff <command> [options]   (run from the repo root)", "", "commands:"]
    lines.append(f"  {'hands-diff refresh':<30} the whole update: " + " -> ".join(" ".join(c) for c in REFRESH_STEPS))
    for command in COMMANDS:
        lines.append(f"  {'hands-diff ' + ' '.join(command):<30} {DESCRIPTIONS[command]}")
    lines += ["", "Each command takes --help for its own options."]
    return "\n".join(lines)


def _run(command: tuple[str, ...], argv: list[str]) -> None:
    module = importlib.import_module(COMMANDS[command])
    module.run(module.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    if argv[0] == "refresh":
        for step in REFRESH_STEPS:
            print(f"\n=== hands-diff {' '.join(step)} ===")
            _run(step, [])
        return 0
    for length in (2, 1):
        command = tuple(argv[:length])
        if command in COMMANDS:
            _run(command, argv[length:])
            return 0
    print(f"Unknown command: {' '.join(argv[:2])}\n")
    print(usage())
    return 2


if __name__ == "__main__":
    sys.exit(main())

"""The `hands-diff` command: one entry point for every pipeline step.

    hands-diff fetch timelines --all      -> hands_diff.collect.timelines, args passed through
    hands-diff refresh                    -> the everyday update, steps in a fixed order

Steps are swapped for recorders here, so nothing calls the Riot API or touches data/.
"""
import importlib

import pytest

from hands_diff import cli


def test_every_command_points_at_a_runnable_module():
    for command, module_name in cli.COMMANDS.items():
        module = importlib.import_module(module_name)
        assert callable(module.parse_args), command
        assert callable(module.run), command


def test_command_args_are_passed_through(monkeypatch):
    seen = {}
    module = importlib.import_module(cli.COMMANDS[("fetch", "timelines")])
    monkeypatch.setattr(module, "run", lambda args: seen.setdefault("args", args))
    assert cli.main(["fetch", "timelines", "--all", "--max-matches", "5"]) == 0
    assert seen["args"].all is True
    assert seen["args"].max_matches == 5


def test_single_word_commands(monkeypatch):
    seen = []
    module = importlib.import_module(cli.COMMANDS[("export",)])
    monkeypatch.setattr(module, "run", lambda args: seen.append(args.cached_icons_only))
    assert cli.main(["export", "--cached-icons-only"]) == 0
    assert seen == [True]


def test_refresh_runs_the_update_in_order(monkeypatch):
    order = []
    for command in cli.REFRESH_STEPS:
        module = importlib.import_module(cli.COMMANDS[command])
        monkeypatch.setattr(module, "run", lambda args, c=command: order.append(c))
    assert cli.main(["refresh"]) == 0
    # Your new games first, then everything derived from them, the page last.
    assert order == [
        ("fetch", "recent"), ("build", "features"), ("fetch", "timelines"),
        ("build", "timelines"), ("export",), ("page",),
    ]


def test_unknown_command_lists_the_commands(capsys):
    assert cli.main(["fetch", "nonsense"]) == 2
    out = capsys.readouterr().out
    assert "fetch match-ids" in out and "refresh" in out


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "hands-diff refresh" in capsys.readouterr().out

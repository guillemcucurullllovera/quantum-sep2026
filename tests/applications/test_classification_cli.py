"""Tests for `applications.classification.cli`. `compare` and `run` are
exercised only with fast classical/lightweight-quantum algorithms and small
default datasets -- the CLI's default dataset sizes are tuned for a real
demo run, not test speed, so tests must not invoke it with every algorithm
(several dispatcher algorithms are deliberately expensive at full scale;
see `test_classification_dispatcher.py` for the per-algorithm smoke tests)."""
from __future__ import annotations

import json

from applications.classification.cli import build_parser, main
from applications.classification.task_spec import Algorithm


def test_build_parser_has_all_subcommands():
    parser = build_parser()
    sub_actions = [a for a in parser._actions if a.dest == "command"]
    assert sub_actions
    choices = set(sub_actions[0].choices.keys())
    assert choices == {"list-algorithms", "run", "compare", "rolling"}


def test_list_algorithms_prints_every_algorithm(capsys):
    rc = main(["list-algorithms"])
    assert rc == 0
    out = capsys.readouterr().out
    for algo in Algorithm:
        assert algo.value in out


def test_run_subcommand_produces_parseable_result(capsys):
    rc = main(
        [
            "run",
            "--task-type", "binary",
            "--algorithm", "classical_logistic_regression",
            "--n-qubits", "4",
            "--test-fraction", "0.3",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["algorithm"] == "classical_logistic_regression"
    assert payload["task_type"] == "binary"
    assert 0.0 <= payload["metrics"]["accuracy"] <= 1.0


def test_rolling_subcommand_produces_parseable_result(capsys):
    rc = main(
        [
            "rolling",
            "--task-type", "binary",
            "--algorithm", "classical_logistic_regression",
            "--n-splits", "2",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["algorithm"] == "classical_logistic_regression"
    assert payload["n_windows"] == 2

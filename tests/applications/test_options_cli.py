from __future__ import annotations

from applications.options.cli import main


def test_cli_price_black_scholes(capsys):
    rc = main(["price", "--payoff", "european_call", "--spot", "100", "--strike", "100", "--method", "black_scholes"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "price:" in out
    assert "black_scholes" in out


def test_cli_compare(capsys):
    rc = main(["compare", "--payoff", "call_spread", "--spot", "100", "--strike", "95",
               "--strike2", "115", "--shots", "1024"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "black_scholes" in out
    assert "qae_mlae" in out


def test_cli_resources(capsys):
    rc = main(["resources", "--payoff", "up_and_out_call", "--spot", "100", "--strike", "100",
               "--barrier", "115", "--target-epsilon", "0.02"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "IQAE expected queries" in out


def test_cli_greeks(capsys):
    rc = main(["greeks", "--payoff", "european_call", "--spot", "100", "--strike", "100",
               "--method", "black_scholes"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "delta" in out


def test_cli_hedge(capsys):
    rc = main(["hedge", "--notional", "1000000", "--spot", "100", "--realized-vol", "0.3",
               "--regime", "trend_down"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "CVaR reduction" in out


def test_cli_invalid_spec_returns_nonzero_and_prints_error(capsys):
    rc = main(["price", "--payoff", "up_and_out_call", "--spot", "100", "--strike", "100",
               "--barrier", "90"])  # up_and_out barrier below spot -- infeasible
    assert rc == 1
    err = capsys.readouterr().err
    assert "ValidationError" in err

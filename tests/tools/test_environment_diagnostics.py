from tools.environment_diagnostics import diagnose


def test_diagnose_reports_no_credential_value(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    report = diagnose()
    assert report["ibm_quantum_token_present"] is False
    # never a literal token value anywhere in the report
    assert "token" not in str(report.get("ibm_quantum_token_value", ""))


def test_diagnose_reports_fake_backend_availability():
    report = diagnose()
    assert report["fake_backends_ok"] is True
    assert report["fake_backends_available"] > 0


def test_cli_json_output(capsys):
    from tools.environment_diagnostics import main

    rc = main(["--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"ibm_quantum_token_present"' in out

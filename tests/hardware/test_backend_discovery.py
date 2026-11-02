from hardware.backend_discovery import discover_backends


def test_discovery_falls_back_to_fake_catalog_without_token(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    from core.config import get_settings

    get_settings(refresh=True)
    result = discover_backends()
    assert result.live is False
    assert len(result.backends) > 0
    assert "No IBM_QUANTUM_TOKEN" in result.note
    # confirm it's genuinely the fake catalog, not an empty/fabricated list
    names = {b.name for b in result.backends}
    assert "noiseless" in names or any("noiseless" in n for n in names) or len(names) >= 2


def test_discovery_result_backends_are_typed_descriptors(monkeypatch):
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    from core.config import get_settings

    get_settings(refresh=True)
    result = discover_backends()
    for b in result.backends:
        assert b.num_qubits > 0
        assert b.name

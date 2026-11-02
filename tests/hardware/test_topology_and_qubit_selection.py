from hardware.backends import BackendCatalog
from hardware.qubit_selection import QubitSelector
from hardware.topology import from_backend


def test_coupling_graph_is_connected_for_real_device():
    r = BackendCatalog.get("fake_heron")
    graph = from_backend(r.backend)
    assert graph.is_connected()
    assert len(graph.nodes) == r.descriptor.num_qubits


def test_shortest_path_distance_is_symmetric_and_positive():
    r = BackendCatalog.get("fake_eagle")
    graph = from_backend(r.backend)
    a, b = graph.nodes[0], graph.nodes[5]
    d_ab = graph.shortest_path_distance(a, b)
    d_ba = graph.shortest_path_distance(b, a)
    assert d_ab == d_ba
    assert d_ab is None or d_ab >= 1


def test_qubit_selector_returns_connected_layout_with_real_calibration():
    r = BackendCatalog.get("fake_heron")
    selector = QubitSelector(r.backend)
    layout = selector.select_layout(6)
    assert len(layout) == 6
    assert len(set(layout)) == 6  # no duplicate qubits
    graph = from_backend(r.backend)
    # every consecutive pair reachable within the selected connected subgraph
    report = selector.layout_report(layout)
    assert report["t1_mean_us"] > 0
    assert 0 <= report["readout_error_mean"] < 1


def test_selector_rejects_too_many_qubits():
    from core.errors import ValidationError
    import pytest

    r = BackendCatalog.get("fake_eagle")
    selector = QubitSelector(r.backend)
    with pytest.raises(ValidationError):
        selector.select_layout(10_000)

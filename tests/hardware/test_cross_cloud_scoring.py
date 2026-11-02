"""hardware.backend_selection cross-cloud ranking -- score/rank
across BOTH IBM fake backends and Braket's local simulator for a given
circuit's qubit requirement, using the normalized BackendCapabilities."""
from __future__ import annotations

from hardware.backend_selection import (
    CrossCloudBackendScore,
    rank_all_clouds_for_circuit,
    score_braket_local_simulator,
)


def test_braket_local_simulator_scores_as_ideal_reference():
    score = score_braket_local_simulator(5)
    assert isinstance(score, CrossCloudBackendScore)
    assert score.backend_name == "braket_local_simulator"
    assert score.cloud == "aws_braket_local"
    assert score.composite_score == float("inf")


def test_cross_cloud_ranking_includes_both_ibm_and_braket_entries():
    ranked = rank_all_clouds_for_circuit(5)
    clouds = {s.cloud for s in ranked}
    assert "ibm_quantum" in clouds
    assert "aws_braket_local" in clouds
    assert all(s.num_qubits >= 5 for s in ranked)


def test_cross_cloud_ranking_excludes_backend_too_small_for_either_cloud():
    ranked = rank_all_clouds_for_circuit(500)  # too big for IBM fake backends AND Braket local sim
    assert ranked == []


def test_cross_cloud_ranking_excludes_braket_when_circuit_too_wide_for_it_but_not_ibm():
    n = 30  # bigger than Braket's ~25-qubit practical ceiling, well within IBM fake backends
    ranked = rank_all_clouds_for_circuit(n)
    clouds = {s.cloud for s in ranked}
    assert "ibm_quantum" in clouds
    assert "aws_braket_local" not in clouds


def test_cross_cloud_ranking_is_sorted_descending():
    ranked = rank_all_clouds_for_circuit(5)
    scores = [s.composite_score for s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_cross_cloud_ranking_each_entry_carries_its_own_capabilities():
    ranked = rank_all_clouds_for_circuit(5)
    for s in ranked:
        assert s.capabilities.name == s.backend_name
        assert s.capabilities.cloud == s.cloud

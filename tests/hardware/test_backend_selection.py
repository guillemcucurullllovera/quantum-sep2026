from hardware.backend_selection import rank_backends_for_circuit


def test_ranking_returns_only_backends_with_enough_qubits():
    ranked = rank_backends_for_circuit(5)
    assert all(s.num_qubits >= 5 for s in ranked)
    assert {"FakeSherbrooke", "FakeTorino"} <= {s.backend_name for s in ranked}


def test_ranking_excludes_backend_too_small():
    ranked = rank_backends_for_circuit(500)  # bigger than any fake backend
    assert ranked == []


def test_ranking_is_sorted_descending_by_score():
    ranked = rank_backends_for_circuit(5)
    scores = [s.composite_score for s in ranked]
    assert scores == sorted(scores, reverse=True)

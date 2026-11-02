import math

import numpy as np
import pytest

from algorithms_extended.qiskit_finance_extensions.uncertainty_models import (
    DiscretizedBivariateLogNormal,
    DiscretizedLogNormal,
)
from algorithms_extended.qiskit_finance_extensions.barrier_payoff import (
    build_A_barrier_payoff,
    build_barrier_payoff_integrand,
    price_from_amplitude as barrier_price_from_amplitude,
    vanilla_reference_price,
)
from algorithms_extended.qiskit_finance_extensions.basket_payoff import (
    build_A_basket_payoff,
    build_basket_call_integrand,
    price_from_amplitude as basket_price_from_amplitude,
)
from algorithms_extended.qiskit_finance_extensions.resource_estimation import (
    canonical_ae_query_count,
    estimate,
    iqae_expected_query_count,
)
from algorithms_extended.qiskit_finance_extensions.scenario_analysis import run_barrier_scenario_sweep


def test_discretized_lognormal_probabilities_sum_to_one():
    dist = DiscretizedLogNormal.from_sigma_window(s0=100, r=0.04, sigma=0.2, t=0.25, num_qubits=4)
    probs = dist.probabilities()
    assert len(probs) == 16
    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(p >= 0 for p in probs)


def test_discretized_lognormal_state_preparation_matches_probabilities():
    from qiskit.quantum_info import Statevector

    dist = DiscretizedLogNormal.from_sigma_window(s0=100, r=0.04, sigma=0.2, t=0.25, num_qubits=3)
    qc = dist.state_preparation()
    sv = Statevector(qc)
    amp_probs = np.abs(sv.data) ** 2
    expected = dist.probabilities()
    assert np.allclose(amp_probs, expected, atol=1e-9)


def test_bivariate_lognormal_joint_probabilities_sum_to_one_and_marginalize_correctly():
    dist = DiscretizedBivariateLogNormal.from_sigma_window(
        s0_a=100, s0_b=100, r=0.04, sigma_a=0.2, sigma_b=0.2, rho=0.0, t=0.25,
        num_qubits_a=2, num_qubits_b=2,
    )
    probs = dist.joint_probabilities()
    assert len(probs) == 16
    assert abs(sum(probs) - 1.0) < 1e-6
    # rho=0 -> marginal over b (summing each row) should match the independent 1D distribution
    marginal_a = DiscretizedLogNormal(
        s0=100, r=0.04, sigma=0.2, t=0.25, num_qubits=2, low=dist.low_a, high=dist.high_a
    ).probabilities()
    joint_marginal_a = [sum(probs[ib * 4 + ia] for ib in range(4)) for ia in range(4)]
    assert np.allclose(joint_marginal_a, marginal_a, atol=1e-6)


def test_up_and_out_barrier_masks_payoff_above_barrier():
    integrand = build_barrier_payoff_integrand(
        "up_and_out", s0=100, strike=95, barrier=110, r=0.04, sigma=0.2, t=0.25, n=4,
        n_mc_paths=5000,
    )
    for s, alive, theta in zip(integrand.grid, integrand.mask_alive, integrand.payoff_angle):
        if s >= 110:
            assert alive is False
            assert theta == 0.0 or abs(theta) < 1e-12


def test_barrier_payoff_never_exceeds_vanilla_price():
    integrand = build_barrier_payoff_integrand(
        "up_and_out", s0=100, strike=95, barrier=115, r=0.04, sigma=0.2, t=0.25, n=4,
        n_mc_paths=5000,
    )
    upper = vanilla_reference_price(100, 95, 0.04, 0.2, 0.25)
    assert integrand.exact_grid_price <= upper + 1e-6


def test_barrier_payoff_terminal_only_matches_grid_estimate():
    """The grid-discretized exact price and a terminal-only Monte Carlo price
    (same barrier definition: checked only at maturity) must agree closely —
    this is the real apples-to-apples parity check for what the A-circuit
    actually prices."""
    integrand = build_barrier_payoff_integrand(
        "down_and_out", s0=100, strike=100, barrier=85, r=0.04, sigma=0.2, t=0.25, n=6,
        n_mc_paths=300_000, seed=7,
    )
    rel_err = abs(integrand.exact_grid_price - integrand.monte_carlo_terminal_only) / max(
        integrand.monte_carlo_terminal_only, 1e-6
    )
    assert rel_err < 0.15  # coarse n=6 grid vs continuous MC; loose but real bound


def test_barrier_payoff_path_monitored_is_cheaper_than_terminal_only():
    """A discretely-monitored barrier knocks out more often than a
    terminal-only check (checking every step is strictly more restrictive),
    so its price must be <= the terminal-only price. This is the honest
    real-world gap the module's docstring documents."""
    integrand = build_barrier_payoff_integrand(
        "up_and_out", s0=100, strike=95, barrier=110, r=0.04, sigma=0.25, t=0.5, n=4,
        n_mc_paths=200_000, n_mc_steps=50, seed=3,
    )
    assert integrand.monte_carlo_path_monitored <= integrand.monte_carlo_terminal_only + 3 * integrand.monte_carlo_path_stderr


def test_A_barrier_payoff_circuit_amplitude_matches_exact_price():
    """Build the actual quantum circuit, compute its exact statevector
    (no shot noise), and confirm the ancilla-|1> probability reproduces the
    classically-computed exact_grid_price via price_from_amplitude — a real
    end-to-end structural correctness check of the quantum construction."""
    from qiskit.quantum_info import Statevector

    integrand = build_barrier_payoff_integrand(
        "up_and_out", s0=100, strike=95, barrier=110, r=0.04, sigma=0.2, t=0.25, n=3,
        n_mc_paths=1000,
    )
    qc = build_A_barrier_payoff(integrand)
    sv = Statevector(qc)
    n = integrand.dist.num_qubits
    anc = n
    a = sum(
        abs(sv.data[i]) ** 2 for i in range(len(sv.data)) if (i >> anc) & 1 == 1
    )
    reconstructed_price = barrier_price_from_amplitude(integrand, a)
    assert abs(reconstructed_price - integrand.exact_grid_price) < 1e-9


def test_basket_call_circuit_amplitude_matches_exact_price():
    from qiskit.quantum_info import Statevector

    integrand = build_basket_call_integrand(
        s0_a=100, s0_b=100, strike=100, weight_a=0.5, weight_b=0.5,
        r=0.04, sigma_a=0.2, sigma_b=0.25, rho=0.3, t=0.25, n_a=2, n_b=2,
        n_mc_paths=2000,
    )
    qc = build_A_basket_payoff(integrand)
    sv = Statevector(qc)
    n = integrand.dist.num_qubits_a + integrand.dist.num_qubits_b
    a = sum(abs(sv.data[i]) ** 2 for i in range(len(sv.data)) if (i >> n) & 1 == 1)
    reconstructed_price = basket_price_from_amplitude(integrand, a)
    assert abs(reconstructed_price - integrand.exact_grid_price) < 1e-9


def test_basket_call_grid_price_agrees_with_monte_carlo():
    integrand = build_basket_call_integrand(
        s0_a=100, s0_b=100, strike=100, weight_a=0.5, weight_b=0.5,
        r=0.04, sigma_a=0.2, sigma_b=0.2, rho=0.5, t=0.25, n_a=3, n_b=3,
        n_mc_paths=500_000, seed=11,
    )
    rel_err = abs(integrand.exact_grid_price - integrand.monte_carlo_price) / max(
        integrand.monte_carlo_price, 1e-6
    )
    assert rel_err < 0.2  # coarse 6-qubit joint grid vs continuous MC


def test_basket_call_rejects_impractically_large_grid():
    with pytest.raises(ValueError, match="impractically large"):
        build_basket_call_integrand(n_a=4, n_b=4)


def test_iqae_needs_fewer_queries_than_canonical_ae():
    canon = canonical_ae_query_count(epsilon=0.01, confidence=0.95)
    iqae = iqae_expected_query_count(epsilon=0.01, confidence=0.95)
    assert iqae < canon


def test_resource_estimate_reports_real_circuit_stats_when_circuit_given():
    integrand = build_barrier_payoff_integrand(
        "up_and_out", s0=100, strike=95, barrier=110, r=0.04, sigma=0.2, t=0.25, n=3,
        n_mc_paths=1000,
    )
    qc = build_A_barrier_payoff(integrand)
    est = estimate(qc, target_epsilon=0.01)
    assert est.circuit_stats is not None
    assert est.circuit_stats.n_qubits == qc.num_qubits
    assert est.circuit_stats.depth > 0


def test_resource_estimate_rejects_invalid_epsilon():
    with pytest.raises(ValueError):
        canonical_ae_query_count(epsilon=1.5)


def test_execution_wiring_runs_on_noiseless_backend():
    """Real end-to-end execution through hardware.ibm_runtime on the noiseless
    simulator path — confirms execution.py's transpile-before-run fix
    (StatePreparation is not a basis gate; run_sampler's noiseless path does
    not transpile internally, unlike its FAKE-backend path)."""
    from algorithms_extended.qiskit_finance_extensions.execution import run_payoff_circuit

    integrand = build_barrier_payoff_integrand(
        "up_and_out", s0=100, strike=95, barrier=110, n=2, n_mc_paths=500,
    )
    qc = build_A_barrier_payoff(integrand)
    result = run_payoff_circuit(qc, backend_name="noiseless", shots=1024, mitigation="none")
    assert sum(result.counts.values()) == 1024
    assert result.backend_name == "aer_simulator_noiseless"


def test_execution_wiring_readout_m3_on_fake_backend():
    """Regression test for the hardware.ibm_runtime.run_sampler FAKE-backend
    qubit-truncation bug (previously xfail strict=True here). The bug: the
    FAKE path ran AerSimulator.from_backend(fake_eagle/fake_heron) against
    the circuit embedded in the FULL 127/133-qubit device with no qubit
    truncation. An all-Clifford circuit (e.g. a bare Bell state) let Aer
    silently auto-select its cheap stabilizer method regardless of qubit
    count, masking the problem in the existing hardware tests. Any circuit
    with a non-Clifford rotation gate -- every real algorithm circuit here,
    this module's payoff circuits included -- forced a full statevector/
    density-matrix fallback sized to the whole backend and exceeded
    max_memory_mb. Fixed directly in hardware/ibm_runtime/primitives.py
    (`_truncate_to_active_qubits`): before simulating, the FAKE path now
    remaps the transpiled circuit down to only the physical qubits it
    actually touches (plus their measurements) instead of the full device
    width, so this 2-qubit payoff circuit (padded out to 127 physical
    qubits by transpile+measure_all) is simulated as the ~2-4 active qubits
    it really is."""
    from algorithms_extended.qiskit_finance_extensions.execution import run_payoff_circuit

    integrand = build_barrier_payoff_integrand(
        "up_and_out", s0=100, strike=95, barrier=110, n=2, n_mc_paths=500,
    )
    qc = build_A_barrier_payoff(integrand)
    result = run_payoff_circuit(qc, backend_name="fake_eagle", shots=1024, mitigation="readout_m3")
    assert result.mitigation == "readout_m3"
    assert result.shots == 1024
    assert sum(result.counts.values()) > 0


def test_barrier_scenario_sweep_produces_monotone_price_in_barrier_distance():
    """A farther-away up-and-out barrier should give a price closer to the
    (unbarriered) vanilla upper bound than a barrier close to the money."""
    rows = run_barrier_scenario_sweep(
        "up_and_out", spots=[100.0], vols=[0.2], barriers=[105.0, 130.0],
        strike=95.0, n=4, n_mc_paths=5000,
    )
    assert len(rows) == 2
    close_barrier, far_barrier = rows[0], rows[1]
    assert close_barrier.exact_grid_price <= far_barrier.exact_grid_price + 1e-9
    assert far_barrier.exact_grid_price <= far_barrier.vanilla_upper_bound + 1e-6

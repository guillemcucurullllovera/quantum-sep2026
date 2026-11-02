"""Independent scalar-oracle comparison for the release's grid acceleration."""
import numpy as np
import pytest

from algorithms_extended.qae_spinqit.mlae import _log_likelihood_grid, _mle_point, _nll, p_k


@pytest.mark.parametrize('ks', [(0,), (0, 1, 2), (0, 1, 2, 3)])
@pytest.mark.parametrize('amplitude', [.01, .35, .77, .99])
def test_vectorized_grid_matches_every_original_scalar_likelihood(ks, amplitude):
    shots=np.full(len(ks),2048)
    successes=np.array([round(p_k(amplitude,k)*n) for k,n in zip(ks,shots)])
    grid,logp,logq=_log_likelihood_grid(ks,401)
    actual=-(successes@logp+(shots-successes)@logq)
    expected=np.array([_nll(float(a),ks,successes,shots) for a in grid])
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=1e-8)
    assert np.argmin(actual)==np.argmin(expected)
    estimate=_mle_point(ks,successes,shots,grid_size=401)
    assert _nll(estimate,ks,successes,shots)<=expected.min()+1e-8
    assert not grid.flags.writeable and not logp.flags.writeable and not logq.flags.writeable

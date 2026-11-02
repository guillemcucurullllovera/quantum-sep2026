import numpy as np

from hardware.mitigation import DDMitigator
from hardware.mitigation.dd import xy4_is_identity, xy4_suppression_factor


def test_xy4_is_identity_up_to_global_phase():
    assert xy4_is_identity() is True


def test_suppression_factor_increases_with_more_pulses():
    low = xy4_suppression_factor(dephasing_rate=0.02, t_idle=1.0, n_pulses=4, n_cycles=1)
    high = xy4_suppression_factor(dephasing_rate=0.02, t_idle=1.0, n_pulses=4, n_cycles=4)
    assert high >= low


def test_dd_mitigator_self_verifies_on_construction():
    mit = DDMitigator()
    summary = mit.summary()
    assert summary["sequence"] == "XY4"
    assert summary["t2_improvement_factor"] > 1

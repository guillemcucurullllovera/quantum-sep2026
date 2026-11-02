# Modified from upstream QOKit's qokit/fur/__init__.py (Apache-2.0, (C) 2023 JP
# Morgan Chase & Co.). Original file selects among C-extension, pure-Python,
# GPU (numba.cuda), and MPI+GPU simulator backends. This environment has no C
# compiler toolchain, no CUDA/numba.cuda GPU, and no MPI runtime available, so
# only the pure-Python FUR backend (fur/python/) is wired here. The C/GPU/MPI
# source trees were not ported (see THIRD_PARTY_NOTICES.md) since they cannot
# be built or executed in this environment and would be untestable dead code.
from .qaoa_simulator_base import QAOAFastSimulatorBase, ParamType, CostsType, TermsType
from .python.qaoa_simulator import (
    QAOAFURXSimulator,
    QAOAFURXZSimulator,
    QAOAFURXYRingSimulator,
    QAOAFURXYCompleteSimulator,
)

SIMULATORS = {
    "x": {"python": QAOAFURXSimulator},
    "xz": {"python": QAOAFURXZSimulator},
    "xyring": {"python": QAOAFURXYRingSimulator},
    "xycomplete": {"python": QAOAFURXYCompleteSimulator},
}


def get_available_simulator_names(type: str = "x") -> list:
    if type not in SIMULATORS:
        raise ValueError(f"Unknown simulator type: {type}")
    return ["python"]


def get_available_simulators(type: str = "x") -> list:
    return [SIMULATORS[type]["python"]]


def choose_simulator(name="auto", **kwargs):
    return SIMULATORS["x"].get(name, QAOAFURXSimulator) if name != "auto" else QAOAFURXSimulator


def choose_simulator_xz(name="auto", **kwargs):
    return QAOAFURXZSimulator


def choose_simulator_xyring(name="auto", **kwargs):
    return QAOAFURXYRingSimulator


def choose_simulator_xycomplete(name="auto", **kwargs):
    return QAOAFURXYCompleteSimulator


__all__ = [
    "QAOAFastSimulatorBase",
    "QAOAFURXSimulator",
    "QAOAFURXZSimulator",
    "QAOAFURXYRingSimulator",
    "QAOAFURXYCompleteSimulator",
    "choose_simulator",
    "choose_simulator_xz",
    "choose_simulator_xyring",
    "choose_simulator_xycomplete",
]

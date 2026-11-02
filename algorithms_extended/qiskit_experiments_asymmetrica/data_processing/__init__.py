# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
===========================================================
Data Processing (:mod:`algorithms_extended.qiskit_experiments_asymmetrica.data_processing`)
===========================================================

.. currentmodule:: algorithms_extended.qiskit_experiments_asymmetrica.data_processing

Data processing is the act of taking the data returned by the backend and
converting it into a format that can be analyzed.
It is implemented as a chain of data processing steps that transform various input data,
e.g. IQ data, into a desired format, e.g. population, which can be analyzed.

These data transformations may consist of multiple steps, such as kerneling and discrimination.
Each step is implemented by a :class:`~algorithms_extended.qiskit_experiments_asymmetrica.data_processing.data_action.DataAction`
also called a `node`.

The data processor implements the :meth:`__call__` method. Once initialized, it
can thus be used as a standard python function:

.. code-block:: python

    processor = DataProcessor(input_key="memory", [Node1(), Node2(), ...])
    out_data = processor(in_data)

The data input to the processor is a sequence of dictionaries each representing the result
of a single circuit. The output of the processor is a numpy array whose shape and data type
depend on the combination of the nodes in the data processor.

Uncertainties that arise from quantum measurements or finite sampling can be taken into account
in the nodes: a standard error can be generated in a node and can be propagated
through the subsequent nodes in the data processor.
Correlation between computed values is also considered.


Classes
=======
.. autosummary::
    :toctree: ../stubs/

    DataProcessor
    DataAction
    TrainableDataAction


Data Processing Nodes
=====================
.. autosummary::
    :toctree: ../stubs/

    Probability
    MarginalizeCounts
    ToImag
    ToReal
    SVD
    DiscriminatorNode
    MemoryToCounts
    AverageData
    BasisExpectationValue
    MinMaxNormalize
    ShotOrder


Discriminators
==============
.. autosummary::
    :toctree: ../stubs/

    BaseDiscriminator
    SkLDA
    SkQDA

Mitigators
==========
.. autosummary::
    :toctree: ../stubs/

    BaseReadoutMitigator
    LocalReadoutMitigator
    CorrelatedReadoutMitigator
"""

from .data_action import DataAction, TrainableDataAction
from .nodes import (
    Probability,
    MarginalizeCounts,
    ToImag,
    ToReal,
    SVD,
    DiscriminatorNode,
    MemoryToCounts,
    AverageData,
    BasisExpectationValue,
    MinMaxNormalize,
    ShotOrder,
)

from .data_processor import DataProcessor
from .discriminator import BaseDiscriminator

# Readout-mitigation classes are deliberately NOT vendored here: Asymmetrica
# already independently re-implemented this technique in
# algorithms_extended/error_mitigation/readout_mitigation.py (see that
# module's THIRD_PARTY_NOTICES.md). These names are re-exported from the installed
# upstream `qiskit-experiments` dependency purely for API compatibility with
# code (in this vendored subset) that references qiskit_experiments.data_processing.*
from qiskit_experiments.data_processing.mitigation.base_readout_mitigator import (
    BaseReadoutMitigator,
)
from qiskit_experiments.data_processing.mitigation.correlated_readout_mitigator import (
    CorrelatedReadoutMitigator,
)
from qiskit_experiments.data_processing.mitigation.local_readout_mitigator import (
    LocalReadoutMitigator,
)
from .sklearn_discriminators import SkLDA, SkQDA

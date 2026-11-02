# This code is part of Qiskit.
#
# (C) Copyright IBM 2021-2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Asymmetrica adaptation of IBM's ``qiskit-ibm-runtime`` (Apache-2.0).

import. Source: the upstream GitHub repo (``qiskit-ibm-runtime``, ``main`` branch).
See ``THIRD_PARTY_NOTICES.md`` in this directory for the full breakdown of
what was imported, what was modified, and the import-path strategy (most
supporting infrastructure -- accounts/api/options/models/utils/etc -- is
reused from the ordinary installed ``qiskit-ibm-runtime`` pip dependency
rather than re-vendored; a small handful of files that postdate the
installed 0.49.0 release were vendored under ``constants.py`` / ``utils/``
because the installed package does not have them).

Vendored here, largely verbatim (original copyright headers and docstrings
preserved):

- ``sampler.py`` / ``estimator.py`` / ``base_primitive.py`` -- the
  SamplerV2/EstimatorV2 primitive classes and their shared base class.
- ``session.py`` / ``batch.py`` -- session and batch execution-mode
  management.
- ``base_runtime_job.py`` / ``runtime_job_v2.py`` -- RuntimeJob
  lifecycle (status polling, cancellation, result retrieval).
- ``fake_provider/`` -- the complete fake-backend provider: 69 IBM device
  models with real T1/T2/gate-error/readout-error calibration snapshots
  (``conf_*.json`` / ``props_*.json``), plus the local (no-credentials)
  execution path (``local_service.py`` / ``local_runtime_job.py``) that
  ``SamplerV2``/``EstimatorV2`` route through when given a fake backend.
- ``constants.py`` / ``utils/`` -- small supporting modules not present
  in the installed qiskit-ibm-runtime==0.49.0.

Usage (fully local -- no IBM Quantum credentials required, exactly like
this platform's existing ``hardware.ibm_runtime.primitives`` FAKE path)::

    from algorithms_extended.qiskit_ibm_runtime_asymmetrica.fake_provider import FakeSherbrooke
    from algorithms_extended.qiskit_ibm_runtime_asymmetrica.sampler import SamplerV2

    backend = FakeSherbrooke()
    sampler = SamplerV2(mode=backend)
    job = sampler.run([transpiled_circuit], shots=4096)
    counts = job.result()[0].data.meas.get_counts()

Nothing in this subpackage reads or contains any IBM Quantum credential
value. Real-hardware execution paths inherited from upstream (which need a
token) go through the same ``core.config.get_settings().ibm`` gate as the
rest of this platform wherever this platform's own code drives them; this
subpackage itself never reads ``os.environ`` directly.
"""

from .sampler import SamplerV2
from .estimator import EstimatorV2
from .session import Session
from .batch import Batch
from .runtime_job_v2 import RuntimeJobV2

__all__ = ["SamplerV2", "EstimatorV2", "Session", "Batch", "RuntimeJobV2"]

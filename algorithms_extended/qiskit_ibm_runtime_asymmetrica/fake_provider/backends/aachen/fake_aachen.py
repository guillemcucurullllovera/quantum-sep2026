# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fake Aachen device (156 qubit)."""

import os

from algorithms_extended.qiskit_ibm_runtime_asymmetrica.fake_provider import fake_backend  # Asymmetrica: point at the vendored copy, not the installed package


class FakeAachen(fake_backend.FakeBackendV2):
    """A fake 156 qubit backend."""

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_aachen.json"
    props_filename = "props_aachen.json"
    backend_name = "fake_aachen"

# Third-party dependencies

The authoritative declared install graph is pyproject.toml, pinned for reproduction
by uv.lock. Key installed dependencies: NumPy, SciPy, pandas, scikit-learn, hmmlearn
(BSD); Qiskit, Aer, Algorithms, IBM Runtime, mthree, Qiskit Experiments/ML/Optimization,
Braket SDK/simulator/schemas, quimb and pytket integrations (Apache-2.0);
matplotlib (Matplotlib license), Pydantic/NetworkX/joblib/fastdtw/python-sat and
provider wrappers (their own permissive terms). Check installed distribution metadata
for exact versions and transitive license expressions.

Existing upstream/adapted source retained under algorithms_extended is separately
licensed, not first-party: see the existing complete component/license table in
THIRD_PARTY_NOTICES.md and full license/NOTICE texts in LICENSES/. Its mathematical
lookup arrays and small solver fixtures belong to those permissively licensed
packages; they are not private market observations or live hardware results.

No GPL/AGPL/SSPL library is intentionally used. Mitiq and ucc were reviewed but not
incorporated by the source project. Provider data terms and paid-solver licenses
are separate; credentials and license keys are never distributed.

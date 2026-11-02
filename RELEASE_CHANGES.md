# Release changes

- Complete main/research branch union, preferring the later research implementation.
  Accepted hygiene checkpoint inspected. First-party model code, finance bridges,
  full research runners and tests remain available even without private data.
- Upstream/adapted permissively licensed components are kept distinctly attributed
  under their original Apache-2.0/MIT terms, preserving the existing notices and
  support fixtures. They are not called first-party work. The original proprietary
  root license is replaced only for owner-authorized first-party material.
- Private host/interpreter paths in two runner docstrings replaced by local-checkout
  descriptions. No production services, source clones or existing public repos changed.
- Public Python 3.12 packaging and locked dependencies replace root metadata; all
  original algorithm/package namespaces remain compatible. Declared yfinance and
  importlib-resources ranges corrected to include the source's recorded tested
  environment; secure pytest 9 replaces the unconstrained legacy test requirement.
  Qiskit/Aer/Runtime and the pytket adapter use source-tested API versions.
- Matplotlib constrained to its maintained 3.10 patch series after a local native-
  library load hang before test collection with the latest graphics stack. No OS
  security policy or production environment was modified to bypass validation.
- Original executed notebooks are not published. Code-only archival extracts retain
  their scientific content as `.py.txt`, with outputs/metadata removed. Several
  original cells have invalid Python indentation; their intended logic is not
  reconstructed or claimed runnable. The retired investor notebook extract is
  explicitly labeled RETIRED/unvalidated. Canonical model implementations remain
  executable Python modules elsewhere in the platform.
- Historical result archives, internal operational/review prose, retired slides,
  private account configuration and source-corpus prose are excluded with exact
  coverage reasons. No historical result or missing model is fabricated.
- Added comprehensive source hashes/coverage verification, release architecture,
  license-boundary and scientific-limitations documentation.
- The mixed-provenance MLAE module's original scalar likelihood grid is evaluated
  with cached NumPy log-probability arrays. Grid points, clipping, the likelihood,
  golden-section refinement and all bootstrap/trial counts remain unchanged.
  Twelve independent regressions compare every grid likelihood and selected grid
  minimum with the retained original scalar `_nll`; these and the unchanged full
  five-test noise/coverage suite pass. This is a documented performance change,
  not a reduction in validation workload or a new estimator-performance claim.

Original upstream whitespace/CRLF and documentation underline conventions are
preserved rather than silently rewriting vendors solely to satisfy `git diff
--check`; apparent conflict-marker warnings from RST headings are not merge conflicts.

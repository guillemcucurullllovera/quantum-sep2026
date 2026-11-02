"""Coupling-map utilities: re-exports `hardware.topology`'s graph primitives
under a more discoverable name for callers who think in terms of "coupling
map" rather than "topology graph," plus one genuinely new capability:
`heavy_hex_reference`, a synthetic heavy-hex-style lattice generator for
scaling experiments that need an N-qubit topology larger or smaller than any
currently-available real or fake backend provides.

`heavy_hex_reference` is NOT a copy of any real IBM chip's coupling map (real
chips' actual maps come from `hardware.fake_backends`/`hardware.backends`,
which pull genuine calibrated topology). It implements the general, published
heavy-hexagon lattice construction (hexagonal lattice with every edge
subdivided by one degree-2 "flag" vertex, giving every vertex degree <= 3) —
see Chamberland et al., "Topological and Subsystem Codes on Low-Degree Graphs
with Flag Qubits" (Phys. Rev. X 10, 011022, 2020) for the general lattice
family this follows. No source code was adapted from any library.
"""
from __future__ import annotations

from hardware.topology.graph import CouplingGraph, from_backend

__all__ = ["CouplingGraph", "from_backend", "heavy_hex_reference"]


def heavy_hex_reference(rows: int, cols: int) -> CouplingGraph:
    """Synthetic heavy-hex-style coupling graph with `rows` x `cols`
    "data" qubits arranged on a brick-wall hex lattice, with one extra
    "flag" qubit inserted on every edge (the standard heavy-hex
    edge-subdivision construction) so every vertex has degree <= 3, matching
    the qualitative shape of IBM's real device topologies.

    Returns a graph over `rows*cols` data qubits (ids 0..rows*cols-1) plus
    one flag qubit per edge (ids continuing from there) — useful for
    topology-shaped scaling/what-if experiments where no real or fake
    backend of the desired size exists.
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")

    def data_id(r: int, c: int) -> int:
        return r * cols + c

    n_data = rows * cols
    raw_edges: set[tuple[int, int]] = set()

    for r in range(rows):
        for c in range(cols):
            # horizontal neighbor
            if c + 1 < cols:
                raw_edges.add((data_id(r, c), data_id(r, c + 1)))
            # vertical neighbor on a checkerboard (row+col) parity offset —
            # a brick-wall hex pattern, not a square grid. Using (r + c) % 2
            # (rather than c % 2 alone) ensures each node has AT MOST ONE
            # vertical edge (never both an upward and a downward one, which
            # would push it to degree 4) while still giving every row at
            # least some vertical bonds, so the lattice stays connected.
            if r + 1 < rows and (r + c) % 2 == 0:
                raw_edges.add((data_id(r, c), data_id(r + 1, c)))

    edges: list[tuple[int, int]] = []
    next_flag = n_data
    for a, b in sorted(raw_edges):
        flag = next_flag
        next_flag += 1
        edges.append((a, flag))
        edges.append((flag, b))

    return CouplingGraph(edges=edges)

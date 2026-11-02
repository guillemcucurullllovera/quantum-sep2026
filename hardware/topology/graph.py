"""Coupling-map / connectivity-graph utilities. Pure-Python adjacency-list
implementation (no networkx dependency) since the operations needed
(BFS distance, degree, connectivity check) don't warrant the extra
dependency for a platform this size.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class CouplingGraph:
    edges: list[tuple[int, int]]
    _adj: dict[int, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for a, b in self.edges:
            self._adj.setdefault(a, []).append(b)
            self._adj.setdefault(b, []).append(a)

    @property
    def nodes(self) -> list[int]:
        return sorted(self._adj)

    def degree(self, node: int) -> int:
        return len(self._adj.get(node, []))

    def degree_distribution(self) -> dict[int, int]:
        """degree -> count of nodes with that degree."""
        dist: dict[int, int] = {}
        for n in self.nodes:
            d = self.degree(n)
            dist[d] = dist.get(d, 0) + 1
        return dist

    def shortest_path_distance(self, source: int, target: int) -> int | None:
        """BFS shortest-path edge count; None if disconnected."""
        if source == target:
            return 0
        visited = {source}
        queue: deque[tuple[int, int]] = deque([(source, 0)])
        while queue:
            node, dist = queue.popleft()
            for nb in self._adj.get(node, []):
                if nb == target:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return None

    def connected_subgraph_bfs(self, seed: int, n: int, priority: dict[int, float] | None = None) -> list[int] | None:
        """BFS from `seed`, greedily expanding via the neighbor with the
        highest `priority` score (defaults to arbitrary order), returning
        `n` connected nodes or None if the graph doesn't have that many
        reachable from `seed`.
        """
        priority = priority or {}
        visited = {seed}
        frontier = [seed]
        result = [seed]
        while len(result) < n and frontier:
            candidates = []
            for node in frontier:
                for nb in self._adj.get(node, []):
                    if nb not in visited:
                        candidates.append((nb, priority.get(nb, 0.0)))
            if not candidates:
                return None
            candidates.sort(key=lambda x: -x[1])
            best = candidates[0][0]
            visited.add(best)
            result.append(best)
            frontier.append(best)
        return result if len(result) == n else None

    def is_connected(self) -> bool:
        if not self.nodes:
            return True
        start = self.nodes[0]
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in self._adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return len(visited) == len(self.nodes)


def from_backend(backend) -> CouplingGraph:
    """Build a CouplingGraph from a Qiskit BackendV2's coupling_map."""
    edges = [tuple(e) for e in backend.coupling_map.get_edges()]
    return CouplingGraph(edges=edges)

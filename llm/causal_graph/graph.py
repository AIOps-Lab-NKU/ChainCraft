#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CausalGraph - First-class causal graph representation

Shared data structure used by M1 (constrained chain generation) / M2 (structural & temporal checkers) /
M3 (graph-aware RAG).

Design principles:
- Edges optionally carry PCMCI statistical features (lag / strength / p_value), still usable when missing
- Lightweight nodes: metric name + optional layer/component tags
- No networkx dependency, all graph operations use pure Python (adjacency list + sets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class NodeInfo:
    metric: str
    layer: Optional[str] = None
    component: Optional[str] = None


@dataclass(frozen=True)
class EdgeInfo:
    src: str
    dst: str
    lag: Optional[float] = None        # PCMCI tau, unit is typically sampling steps
    strength: Optional[float] = None   # ParCorr coefficient
    p_value: Optional[float] = None    # Significance p-value


@dataclass
class CausalGraph:
    """Directed causal graph (DAG, no duplicate edges)"""

    nodes: Dict[str, NodeInfo] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], EdgeInfo] = field(default_factory=dict)

    # Adjacency list cache (lazy construction)
    _out_neighbors: Dict[str, Set[str]] = field(default_factory=dict)
    _in_neighbors: Dict[str, Set[str]] = field(default_factory=dict)

    # -------- Construction / Maintenance --------

    def add_node(self, metric: str, layer: Optional[str] = None,
                 component: Optional[str] = None) -> None:
        if metric in self.nodes:
            return
        self.nodes[metric] = NodeInfo(metric=metric, layer=layer, component=component)
        self._out_neighbors.setdefault(metric, set())
        self._in_neighbors.setdefault(metric, set())

    def add_edge(self, src: str, dst: str,
                 lag: Optional[float] = None,
                 strength: Optional[float] = None,
                 p_value: Optional[float] = None) -> None:
        if src == dst:
            return  # Disallow self-loops
        if src not in self.nodes:
            self.add_node(src)
        if dst not in self.nodes:
            self.add_node(dst)

        # If edge already exists, merge (keep stronger one)
        key = (src, dst)
        if key in self.edges:
            existing = self.edges[key]
            new_strength = strength if strength is not None else existing.strength
            if (existing.strength is not None and strength is not None
                    and abs(existing.strength) >= abs(strength)):
                return  # Keep original
            self.edges[key] = EdgeInfo(
                src=src, dst=dst,
                lag=lag if lag is not None else existing.lag,
                strength=new_strength,
                p_value=p_value if p_value is not None else existing.p_value,
            )
        else:
            self.edges[key] = EdgeInfo(src, dst, lag, strength, p_value)

        self._out_neighbors[src].add(dst)
        self._in_neighbors[dst].add(src)

    # -------- Query --------

    def has_node(self, metric: str) -> bool:
        return metric in self.nodes

    def has_edge(self, src: str, dst: str) -> bool:
        return (src, dst) in self.edges

    def get_edge(self, src: str, dst: str) -> Optional[EdgeInfo]:
        return self.edges.get((src, dst))

    def get_edge_lag(self, src: str, dst: str) -> Optional[float]:
        edge = self.edges.get((src, dst))
        return edge.lag if edge else None

    def get_edge_strength(self, src: str, dst: str) -> Optional[float]:
        edge = self.edges.get((src, dst))
        return edge.strength if edge else None

    def out_neighbors(self, node: str) -> Set[str]:
        return self._out_neighbors.get(node, set())

    def in_neighbors(self, node: str) -> Set[str]:
        return self._in_neighbors.get(node, set())

    def is_valid_edge(self, src: str, dst: str) -> bool:
        """src->dst exists in graph"""
        return self.has_edge(src, dst)

    def is_valid_path(self, path: List[str]) -> bool:
        """Every hop on path must be a directed edge in graph"""
        if len(path) < 2:
            return len(path) == 1 and path[0] in self.nodes
        for i in range(len(path) - 1):
            if not self.has_edge(path[i], path[i + 1]):
                return False
        return True

    def reachable_from(self, start: str, max_hops: int = 6) -> Set[str]:
        """Set of nodes reachable from start within max_hops hops (includes start)"""
        if start not in self.nodes:
            return set()
        visited = {start}
        frontier = {start}
        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for u in frontier:
                for v in self.out_neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        next_frontier.add(v)
            if not next_frontier:
                break
            frontier = next_frontier
        return visited

    def downstream_closure(self, node: str) -> Set[str]:
        """All reachable downstream nodes (counterfactual scenario: which metrics are affected after root cause suppression)"""
        if node not in self.nodes:
            return set()
        return self.reachable_from(node, max_hops=len(self.nodes))

    # -------- Path search --------

    def shortest_path(self, src: str, dst: str, max_hops: int = 6) -> Optional[List[str]]:
        """BFS to find shortest directed path; returns None if not found"""
        if src not in self.nodes or dst not in self.nodes:
            return None
        if src == dst:
            return [src]
        from collections import deque
        parent: Dict[str, Optional[str]] = {src: None}
        queue = deque([(src, 0)])
        while queue:
            u, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for v in self.out_neighbors(u):
                if v in parent:
                    continue
                parent[v] = u
                if v == dst:
                    # Reconstruct path
                    path = [v]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])  # type: ignore[index]
                    return list(reversed(path))
                queue.append((v, depth + 1))
        return None

    # -------- Summary --------

    def summary(self) -> Dict[str, int]:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_edges_with_lag": sum(1 for e in self.edges.values() if e.lag is not None),
            "n_edges_with_strength": sum(1 for e in self.edges.values() if e.strength is not None),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"CausalGraph(nodes={s['n_nodes']}, edges={s['n_edges']}, "
            f"with_lag={s['n_edges_with_lag']}, with_strength={s['n_edges_with_strength']})"
        )

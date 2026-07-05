#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_similarity - Propagation chain graph similarity computation

Four sub-similarities:
- node:   Match by (layer, node_type) dual label, weighted by metric name (pure Python sets, no networkx)
- edge:   Edge set Jaccard, judged by (src_key, dst_key, relation_class) triple
- path:   Longest Common Subsequence (LCS) / shorter chain length
- topology: Normalized difference of numerical features like length/layer coverage

Aggregation:
    Score = α·Sim_text + β·Sim_node + γ·Sim_edge + δ·Sim_path + ε·Sim_topo
Default weights (α/β/γ/δ/ε) = (0.2, 0.25, 0.25, 0.2, 0.1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .chain_graph import ChainGraph, ChainNode


DEFAULT_WEIGHTS: Dict[str, float] = {
    "text": 0.2,
    "node": 0.25,
    "edge": 0.25,
    "path": 0.2,
    "topo": 0.1,
}


# Relation normalization: map semantically similar relation text to same class,
# avoiding "causes" / "results_in" being counted as different edges due to wording
_RELATION_CLASS_MAP = {
    "causes": "causal",
    "cause": "causal",
    "导致": "causal",
    "造成": "causal",
    "引起": "causal",
    "triggers": "causal",
    "trigger": "causal",
    "propagates_to": "propagation",
    "propagation": "propagation",
    "传播至": "propagation",
    "传播": "propagation",
    "correlates_with": "correlation",
    "correlation": "correlation",
    "相关": "correlation",
    "amplifies": "amplification",
    "放大": "amplification",
}


def _normalize_relation(relation: str) -> str:
    if not relation:
        return "unknown"
    key = relation.strip().lower()
    return _RELATION_CLASS_MAP.get(key, key or "unknown")


def _node_match_score(a: ChainNode, b: ChainNode) -> float:
    """Similarity between two nodes (same layer+same type base score; bonus for same metric name)"""
    if a.layer == b.layer and a.node_type == b.node_type:
        base = 1.0
    elif a.layer == b.layer or a.node_type == b.node_type:
        base = 0.5
    else:
        base = 0.0

    if a.metric and a.metric == b.metric:
        # Bonus when metric names match exactly, capped at 1.0
        base = min(1.0, base + 0.2)
    return base


def node_similarity(g1: ChainGraph, g2: ChainGraph) -> float:
    """Node-level similarity: for each node in g1 find best match in g2, average (then take mean with reverse alignment to mitigate asymmetry)"""
    if not g1.nodes or not g2.nodes:
        return 0.0

    def _avg_best(src: List[ChainNode], dst: List[ChainNode]) -> float:
        total = 0.0
        for sn in src:
            total += max(_node_match_score(sn, dn) for dn in dst)
        return total / len(src)

    return 0.5 * (_avg_best(g1.nodes, g2.nodes) + _avg_best(g2.nodes, g1.nodes))


def _edge_keys(g: ChainGraph, nodes_by_metric: Dict[str, ChainNode]) -> set:
    """Express edges as set of ((src_layer, src_type), (dst_layer, dst_type), relation_class) triples"""
    keys = set()
    for e in g.edges:
        src_node = nodes_by_metric.get(e.src)
        dst_node = nodes_by_metric.get(e.dst)
        if src_node is None or dst_node is None:
            continue
        keys.add(
            (
                src_node.identity_key(),
                dst_node.identity_key(),
                _normalize_relation(e.relation),
            )
        )
    return keys


def edge_similarity(g1: ChainGraph, g2: ChainGraph) -> float:
    """Edge set Jaccard similarity (judged by (src_layer+type, dst_layer+type, relation_class))"""
    if not g1.edges or not g2.edges:
        return 0.0

    nodes_by_metric_1 = {n.metric: n for n in g1.nodes}
    nodes_by_metric_2 = {n.metric: n for n in g2.nodes}

    s1 = _edge_keys(g1, nodes_by_metric_1)
    s2 = _edge_keys(g2, nodes_by_metric_2)

    if not s1 or not s2:
        return 0.0

    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0.0


def _lcs_length(a: List[Tuple[str, str]], b: List[Tuple[str, str]]) -> int:
    """LCS length of two (layer, node_type) sequences"""
    if not a or not b:
        return 0
    n, p = len(a), len(b)
    dp = [[0] * (p + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(p):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[n][p]


def path_similarity(g1: ChainGraph, g2: ChainGraph) -> float:
    """Path order similarity: LCS length of (layer, type) sequences / shorter chain length"""
    if not g1.nodes or not g2.nodes:
        return 0.0
    seq1 = [n.identity_key() for n in g1.nodes]
    seq2 = [n.identity_key() for n in g2.nodes]
    lcs = _lcs_length(seq1, seq2)
    denom = min(len(seq1), len(seq2))
    return lcs / denom if denom else 0.0


def topology_similarity(g1: ChainGraph, g2: ChainGraph) -> float:
    """Topology/scale similarity: normalized difference of structural features like chain length, layer coverage, root&symptom presence"""
    if not g1.nodes or not g2.nodes:
        return 0.0

    def features(g: ChainGraph) -> Tuple[int, int, int, int]:
        n_nodes = len(g.nodes)
        n_edges = len(g.edges)
        n_layers = len({n.layer for n in g.nodes})
        n_types = len({n.node_type for n in g.nodes})
        return (n_nodes, n_edges, n_layers, n_types)

    f1 = features(g1)
    f2 = features(g2)
    sims = []
    for a, b in zip(f1, f2):
        if a == 0 and b == 0:
            sims.append(1.0)
        else:
            sims.append(1.0 - abs(a - b) / max(a, b))
    return sum(sims) / len(sims) if sims else 0.0


@dataclass
class GraphSimilarityBreakdown:
    text: float
    node: float
    edge: float
    path: float
    topo: float
    score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "text": self.text,
            "node": self.node,
            "edge": self.edge,
            "path": self.path,
            "topo": self.topo,
            "score": self.score,
        }


def hybrid_similarity(
    g1: ChainGraph,
    g2: ChainGraph,
    text_sim: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
) -> GraphSimilarityBreakdown:
    """
    Weighted similarity for a single chain pair.

    Args:
        g1, g2: Parsed ChainGraph
        text_sim: Cosine similarity of corresponding text chunks (from SimilarityAnalyzer)
        weights: Custom weights; defaults to DEFAULT_WEIGHTS

    Returns:
        GraphSimilarityBreakdown with 4 sub-scores + weighted score
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    sim_node = node_similarity(g1, g2)
    sim_edge = edge_similarity(g1, g2)
    sim_path = path_similarity(g1, g2)
    sim_topo = topology_similarity(g1, g2)

    score = (
        w["text"] * float(text_sim)
        + w["node"] * sim_node
        + w["edge"] * sim_edge
        + w["path"] * sim_path
        + w["topo"] * sim_topo
    )

    return GraphSimilarityBreakdown(
        text=float(text_sim),
        node=sim_node,
        edge=sim_edge,
        path=sim_path,
        topo=sim_topo,
        score=score,
    )

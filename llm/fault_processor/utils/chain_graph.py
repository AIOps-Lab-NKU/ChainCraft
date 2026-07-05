#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chain_graph - Propagation chain structured representation and parsing

Provides two graph construction entry points:
- from_struct: Build graph from raw chain dict at chunk_processor stage (for current chain)
- from_chain_text: Reverse-parse graph from [CHAIN] text chunks stored in ChromaDB (for historical chain)

Nodes are uniformly tagged with (layer, node_type) dual labels:
- layer: Inferred via MetricLayerConfig.get_metric_layer
- node_type: Determined by chain position - first node=root_cause, last node=symptom, middle=intermediate
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from llm.agent.metric_layer_config import MetricLayerConfig


NODE_TYPE_ROOT = "root_cause"
NODE_TYPE_INTERMEDIATE = "intermediate"
NODE_TYPE_SYMPTOM = "symptom"


@dataclass(frozen=True)
class ChainNode:
    metric: str
    layer: str
    node_type: str

    def identity_key(self) -> Tuple[str, str]:
        return (self.layer, self.node_type)


@dataclass
class ChainEdge:
    src: str
    dst: str
    relation: str = ""
    when: str = ""
    observed: bool = False
    inferred: bool = False


@dataclass
class ChainGraph:
    chain_id: Optional[str] = None
    confidence: float = 0.0
    nodes: List[ChainNode] = field(default_factory=list)
    edges: List[ChainEdge] = field(default_factory=list)
    raw_text: str = ""

    @property
    def metric_list(self) -> List[str]:
        return [n.metric for n in self.nodes]

    @property
    def node_signature(self) -> str:
        return " > ".join(self.metric_list)


def _infer_node_type(idx: int, total: int) -> str:
    if total <= 1:
        return NODE_TYPE_ROOT
    if idx == 0:
        return NODE_TYPE_ROOT
    if idx == total - 1:
        return NODE_TYPE_SYMPTOM
    return NODE_TYPE_INTERMEDIATE


def _build_nodes(metric_sequence: List[str]) -> List[ChainNode]:
    nodes: List[ChainNode] = []
    n = len(metric_sequence)
    for i, metric in enumerate(metric_sequence):
        layer = MetricLayerConfig.get_metric_layer(metric) or "unknown_layer"
        nodes.append(
            ChainNode(
                metric=metric,
                layer=layer,
                node_type=_infer_node_type(i, n),
            )
        )
    return nodes


def from_struct(chain_info: Dict) -> Optional[ChainGraph]:
    """Build graph from original chain dict (with chain_id / confidence / chain list)"""
    if not chain_info:
        return None

    steps = chain_info.get("chain") or []
    if not steps:
        return None

    metrics: List[str] = []
    edges: List[ChainEdge] = []

    for i, step in enumerate(steps):
        src = step.get("from")
        dst = step.get("to")
        if not src or not dst:
            continue

        if i == 0:
            metrics.append(src)
        metrics.append(dst)

        edges.append(
            ChainEdge(
                src=src,
                dst=dst,
                relation=str(step.get("relation", "")),
                when=str(step.get("when", "")),
                observed=bool(step.get("observed", False)),
                inferred=bool(step.get("inferred", False)),
            )
        )

    if not metrics:
        return None

    return ChainGraph(
        chain_id=chain_info.get("chain_id"),
        confidence=float(chain_info.get("confidence", 0.0) or 0.0),
        nodes=_build_nodes(metrics),
        edges=edges,
    )


_EDGE_LINE_RE = re.compile(
    r"-\s*([^->]+?)\s*->\s*([^;]+?)\s*;\s*"
    r"relation=([^;]*?)\s*;\s*"
    r"when=([^;]*?)\s*;\s*"
    r"observed=([^;]*?)\s*;\s*"
    r"inferred=([^;]*?)\s*;"
)


def _parse_kv_value(line: str, key: str) -> Optional[str]:
    m = re.search(rf"{re.escape(key)}=([^\n]+)", line)
    if not m:
        return None
    return m.group(1).strip()


def from_chain_text(text: str, chain_id: Optional[str] = None) -> Optional[ChainGraph]:
    """Reverse-parse graph from [CHAIN] text chunks stored in ChromaDB"""
    if not text or "[CHAIN]" not in text:
        return None

    nodes_str: Optional[str] = None
    confidence = 0.0

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("nodes="):
            nodes_str = stripped[len("nodes="):].strip()
        elif stripped.startswith("confidence="):
            try:
                confidence = float(stripped[len("confidence="):].strip())
            except ValueError:
                confidence = 0.0

    metrics: List[str] = []
    if nodes_str:
        metrics = [m.strip() for m in nodes_str.split(",") if m.strip()]

    edges: List[ChainEdge] = []
    for m in _EDGE_LINE_RE.finditer(text):
        edges.append(
            ChainEdge(
                src=m.group(1).strip(),
                dst=m.group(2).strip(),
                relation=m.group(3).strip(),
                when=m.group(4).strip(),
                observed=m.group(5).strip().lower() == "true",
                inferred=m.group(6).strip().lower() == "true",
            )
        )

    # If nodes= line is missing/empty, fallback to reconstruct node chain from edge sequence
    if not metrics and edges:
        metrics = [edges[0].src] + [e.dst for e in edges]

    if not metrics:
        return None

    return ChainGraph(
        chain_id=chain_id,
        confidence=confidence,
        nodes=_build_nodes(metrics),
        edges=edges,
        raw_text=text,
    )


def parse_candidate_chain(candidate_chain: Dict) -> Optional[ChainGraph]:
    """ChromaDB returned candidate chain wrapper: prioritize content field reverse-parse, with chain_id/confidence from metadata"""
    if not candidate_chain:
        return None

    content = candidate_chain.get("content") or candidate_chain.get("page_content") or ""
    chain_id = candidate_chain.get("chain_id")

    graph = from_chain_text(content, chain_id=chain_id)
    if graph is None:
        return None

    # If metadata has more accurate confidence, prefer metadata
    if "confidence" in candidate_chain and candidate_chain["confidence"] is not None:
        try:
            graph.confidence = float(candidate_chain["confidence"])
        except (TypeError, ValueError):
            pass
    return graph


def parse_current_chain(current_chain: Dict) -> Optional[ChainGraph]:
    """
    Current inference stage chain dict wrapper:

    Input has two possible forms:
    1. {'page_content': '[CHAIN]...', 'metadata': {...}} generated by chunk_processor
    2. Upstream raw chain_info {'chain_id', 'confidence', 'chain': [...]}
    """
    if not current_chain:
        return None

    # Case 1
    if "chain" in current_chain and isinstance(current_chain["chain"], list):
        return from_struct(current_chain)

    # Case 2
    content = current_chain.get("page_content") or current_chain.get("content") or ""
    metadata = current_chain.get("metadata") or {}
    chain_id = metadata.get("chain_id") or current_chain.get("chain_id")
    graph = from_chain_text(content, chain_id=chain_id)
    if graph is None:
        return None
    if "confidence" in metadata and metadata["confidence"] is not None:
        try:
            graph.confidence = float(metadata["confidence"])
        except (TypeError, ValueError):
            pass
    return graph

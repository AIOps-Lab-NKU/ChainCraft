"""
llm.causal_graph - First-class causal graph representation

Provides unified causal graph data structure and loader across M1/M2/M3:
- graph.CausalGraph: Nodes + edges (with optional lag/strength/p_value fields)
- loader: Load from causal_edges.txt / edges dict list / influence_statistics.txt
"""

from .graph import CausalGraph, EdgeInfo, NodeInfo
from .loader import (
    load_from_edges_file,
    load_from_edges_list,
    load_from_text,
)

__all__ = [
    "CausalGraph",
    "EdgeInfo",
    "NodeInfo",
    "load_from_edges_file",
    "load_from_edges_list",
    "load_from_text",
]

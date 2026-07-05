#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CausalGraph loader - Construct causal graph from multiple sources

Supports three input types:
1. PCMCI persisted causal_edges.txt (each line "src -> dst")
2. In-memory edges dict list (with lag/strength/p_value etc. full fields, i.e. output of causal_analysis.build_causal_graph_pcmci)
3. Arbitrary "src -> dst" text (from prompt strings)

Optionally tags MetricLayerConfig node layer annotations for downstream M2/M3 reuse.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional

from .graph import CausalGraph


_EDGE_LINE_RE = re.compile(r"^\s*([^\->\s][^\->]*?)\s*->\s*([^\->\s][^\->]*?)\s*$")


def _maybe_tag_layer(metric: str) -> Optional[str]:
    try:
        from llm.agent.metric_layer_config import MetricLayerConfig
        return MetricLayerConfig.get_metric_layer(metric)
    except Exception:
        return None


def load_from_edges_list(edges: Iterable[dict]) -> CausalGraph:
    """
    Construct graph from PCMCI native output (dict list).
    Each dict expected to contain: source, target, [lag, strength, p_value]
    """
    g = CausalGraph()
    for e in edges or []:
        src = e.get("source") or e.get("src")
        dst = e.get("target") or e.get("dst")
        if not src or not dst:
            continue
        g.add_node(src, layer=_maybe_tag_layer(src))
        g.add_node(dst, layer=_maybe_tag_layer(dst))
        g.add_edge(
            src=src,
            dst=dst,
            lag=e.get("lag"),
            strength=e.get("strength"),
            p_value=e.get("p_value"),
        )
    return g


def load_from_text(text: str) -> CausalGraph:
    """
    Construct graph from "src -> dst\\nsrc -> dst..." style text.
    Used for directly consuming causal_analysis strings (falls back to None when lag/strength missing).
    """
    g = CausalGraph()
    if not text:
        return g
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _EDGE_LINE_RE.match(line)
        if not m:
            continue
        src, dst = m.group(1).strip(), m.group(2).strip()
        g.add_node(src, layer=_maybe_tag_layer(src))
        g.add_node(dst, layer=_maybe_tag_layer(dst))
        g.add_edge(src, dst)
    return g


def load_from_edges_file(path: str) -> CausalGraph:
    """Load from causal_edges.txt file"""
    if not path or not os.path.exists(path):
        return CausalGraph()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return load_from_text(text)


def autoload(causal_analysis) -> CausalGraph:
    """
    autoload(): Intelligently identify causal_analysis input type and return CausalGraph

    Supports four entry points:
    - CausalGraph instance: return directly
    - File path (string and file exists): load_from_edges_file
    - Text string: load_from_text
    - dict list (PCMCI edges native output): load_from_edges_list
    """
    if causal_analysis is None:
        return CausalGraph()
    if isinstance(causal_analysis, CausalGraph):
        return causal_analysis
    if isinstance(causal_analysis, list):
        return load_from_edges_list(causal_analysis)
    if isinstance(causal_analysis, str):
        if os.path.exists(causal_analysis):
            return load_from_edges_file(causal_analysis)
        return load_from_text(causal_analysis)
    raise TypeError(f"Unsupported causal_analysis type: {type(causal_analysis)}")

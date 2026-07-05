#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChainScorer - Latency-aware chain scoring

Score = α * structural_validity        # Whether chain is fully valid on G
      + β * lag_consistency            # Match between observed edge Δt and PCMCI lag
      + γ * strength_aggregation       # Average statistical strength (|ρ|) of chain edges
      + δ * symptom_coverage           # Tail node in observed anomaly metric set

When fields are missing (lag/strength/observed_delta), the similarity term degrades to 0.5 (neutral),
avoiding penalizing valid terms.

No external dependencies; if upstream passes (metric_name, t_anomaly) dict, lag matching can be activated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llm.causal_graph import CausalGraph


DEFAULT_WEIGHTS = {
    "structural": 0.4,
    "lag": 0.25,
    "strength": 0.2,
    "coverage": 0.15,
}


@dataclass
class ChainScoreBreakdown:
    structural: float
    lag: float
    strength: float
    coverage: float
    score: float
    detail: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "structural": self.structural,
            "lag": self.lag,
            "strength": self.strength,
            "coverage": self.coverage,
            "score": self.score,
            "detail": self.detail,
        }


def _lag_match(observed_dt: float, graph_lag: float, sigma: float = 1.5) -> float:
    """Gaussian kernel: |Δt - τ|² / σ²; returns 1 when they match, approaches 0 as gap increases"""
    if observed_dt is None or graph_lag is None:
        return 0.5
    diff = float(observed_dt) - float(graph_lag)
    return math.exp(-(diff * diff) / (sigma * sigma))


class ChainScorer:
    def __init__(
        self,
        graph: CausalGraph,
        weights: Optional[Dict[str, float]] = None,
        lag_sigma: float = 1.5,
    ):
        self.graph = graph
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.lag_sigma = lag_sigma

    def score(
        self,
        node_chain: List[str],
        observed_lags: Optional[Dict] = None,
        observed_symptoms: Optional[List[str]] = None,
    ) -> ChainScoreBreakdown:
        """
        Args:
            node_chain: Node sequence on chain [n0, n1, ..., nk]
            observed_lags: {(src, dst): observed_dt} observed edge lags, optional
            observed_symptoms: Known anomaly metric list for coverage calculation, optional
        """
        if not node_chain or len(node_chain) < 2:
            return ChainScoreBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, detail={"reason": "empty"})

        # 1) structural validity
        structural = 1.0 if self.graph.is_valid_path(node_chain) else 0.0
        if structural < 1.0:
            # Partial validity: ratio of valid edges
            ok = 0
            for i in range(len(node_chain) - 1):
                if self.graph.has_edge(node_chain[i], node_chain[i + 1]):
                    ok += 1
            structural = ok / (len(node_chain) - 1)

        # 2) lag consistency
        lag_terms = []
        for i in range(len(node_chain) - 1):
            src, dst = node_chain[i], node_chain[i + 1]
            graph_lag = self.graph.get_edge_lag(src, dst)
            observed_dt = None
            if observed_lags:
                observed_dt = observed_lags.get((src, dst))
            lag_terms.append(_lag_match(observed_dt, graph_lag, sigma=self.lag_sigma))
        lag_score = sum(lag_terms) / len(lag_terms) if lag_terms else 0.5

        # 3) strength aggregation: average |ρ|; neutral 0.5 when missing
        strengths = []
        for i in range(len(node_chain) - 1):
            s = self.graph.get_edge_strength(node_chain[i], node_chain[i + 1])
            if s is not None:
                strengths.append(min(1.0, abs(float(s))))
        strength_score = sum(strengths) / len(strengths) if strengths else 0.5

        # 4) symptom coverage: chain tail in observed anomaly set
        if observed_symptoms:
            symptom_set = set(observed_symptoms)
            tail = node_chain[-1]
            coverage_score = 1.0 if tail in symptom_set else 0.0
        else:
            coverage_score = 0.5

        w = self.weights
        score = (
            w["structural"] * structural
            + w["lag"] * lag_score
            + w["strength"] * strength_score
            + w["coverage"] * coverage_score
        )

        return ChainScoreBreakdown(
            structural=structural,
            lag=lag_score,
            strength=strength_score,
            coverage=coverage_score,
            score=score,
            detail={"weights": dict(w)},
        )


def create_chain_scorer(causal_analysis, **kwargs) -> ChainScorer:
    """Convenience factory"""
    from llm.causal_graph.loader import autoload
    return ChainScorer(graph=autoload(causal_analysis), **kwargs)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2CriticPipeline - Chain multiple checkers into a pipeline

After calling .critique(chains, **kwargs) once:
    1) Run StructuralChecker / TemporalChecker in sequence
    2) Run SemanticCritic or CriticEnsemble (if configured with K>1)
    3) Aggregate the three CritiqueResults into one AggregatedCritique

Shared by IterationController and DeterministicComparator, ensuring both sides
see the same signals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_checker import AggregatedCritique, BaseChecker, CritiqueResult
from .critic_ensemble import CriticEnsemble
from .semantic_critic import BaseSemanticCritic, NoopSemanticCritic
from .structural_checker import StructuralChecker
from .temporal_checker import TemporalChecker


class M2CriticPipeline:

    def __init__(
        self,
        graph,
        semantic_critic: Optional[BaseChecker] = None,
        semantic_critics: Optional[List[BaseChecker]] = None,
        delay_tolerance: float = 2.0,
        ensemble_min_votes: int = 1,
    ):
        """
        Args:
            graph: CausalGraph
            semantic_critic: Single SemanticCritic. Lower priority than semantic_critics
            semantic_critics: K critics (forms ensemble when K>=2)
            delay_tolerance: Used by TemporalChecker
            ensemble_min_votes: Minimum votes required for ensemble to keep a violation
        """
        self.graph = graph
        self.structural = StructuralChecker(graph)
        self.temporal = TemporalChecker(graph, delay_tolerance=delay_tolerance)

        if semantic_critics and len(semantic_critics) > 1:
            self.semantic = CriticEnsemble(
                critics=semantic_critics,
                min_votes_to_keep=ensemble_min_votes,
            )
            self._semantic_is_ensemble = True
        elif semantic_critics and len(semantic_critics) == 1:
            self.semantic = semantic_critics[0]
            self._semantic_is_ensemble = False
        elif semantic_critic is not None:
            self.semantic = semantic_critic
            self._semantic_is_ensemble = False
        else:
            self.semantic = NoopSemanticCritic()
            self._semantic_is_ensemble = False

    def critique(self, chains: List[Dict[str, Any]], **kwargs) -> AggregatedCritique:
        s = self.structural.check(chains, **kwargs)
        t = self.temporal.check(chains, **kwargs)
        if self._semantic_is_ensemble:
            sem = self.semantic.critique(chains, **kwargs)
        else:
            sem = self.semantic.check(chains, **kwargs)
        return AggregatedCritique(results=[s, t, sem])

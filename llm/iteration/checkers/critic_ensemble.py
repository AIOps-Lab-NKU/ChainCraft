#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CriticEnsemble - K-critic Voting + Entropy

Combine K SemanticCritics (or any BaseCheckers) together:
    - Majority vote on the same (chain_id, assertion, edge) dimension
    - When critics highly agree -> high confidence, violation kept and merged
    - When critics disagree -> compute 'vote entropy', optionally downgrade (skip merge or reduce severity)

Design principles:
    - Input is List[BaseChecker]; degenerates to single critic when K=1 (does not affect NoopCritic scenario)
    - Output is still a CritiqueResult for easy aggregation with other checkers
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from .base_checker import (
    BaseChecker,
    CheckerKind,
    CritiqueResult,
    Severity,
    Violation,
)


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        h -= p * math.log(p + 1e-12)
    return h


class CriticEnsemble:
    """K-critic voting ensemble"""

    def __init__(
        self,
        critics: List[BaseChecker],
        min_votes_to_keep: int = 1,
        downgrade_when_split: bool = True,
    ):
        if not critics:
            raise ValueError("critics must not be empty")
        self.critics = critics
        self.min_votes_to_keep = min_votes_to_keep
        self.downgrade_when_split = downgrade_when_split

    @property
    def k(self) -> int:
        return len(self.critics)

    def critique(self, chains: List[Dict[str, Any]], **kwargs) -> CritiqueResult:
        per_critic: List[CritiqueResult] = []
        for c in self.critics:
            per_critic.append(c.check(chains, **kwargs))

        # Single critic case: pass through directly
        if self.k == 1:
            r = per_critic[0]
            return CritiqueResult(
                checker=f"ensemble[{r.checker}]",
                passed=r.passed,
                violations=r.violations,
                meta={"k": 1, "child": r.meta, "vote_entropy": 0.0},
            )

        # Multi-critic: aggregate votes by (chain_id, assertion, edge)
        buckets: Dict[Tuple[str, str, str], List[Violation]] = defaultdict(list)
        for r in per_critic:
            for v in r.violations:
                key = (
                    v.chain_id or "",
                    v.assertion,
                    "->".join(v.edge) if v.edge else "",
                )
                buckets[key].append(v)

        merged: List[Violation] = []
        vote_entropies: List[float] = []

        for key, vs in buckets.items():
            votes = len(vs)
            if votes < self.min_votes_to_keep:
                continue

            sev_counts = {s.value: 0 for s in Severity}
            for v in vs:
                sev_counts[v.severity.value] += 1
            entropy = _entropy(list(sev_counts.values()))
            vote_entropies.append(entropy)

            # Most voted severity; take most severe on tie
            dominant = self._dominant_severity(sev_counts, vs)
            # High disagreement: severity distribution entropy exceeds threshold, or votes not full (some critics silent)
            split = entropy > 0.4 or votes < self.k
            if self.downgrade_when_split and split:
                dominant = self._downgrade(dominant)

            base = vs[0]
            merged.append(Violation(
                checker=f"ensemble[k={self.k}]",
                assertion=base.assertion,
                chain_id=base.chain_id,
                edge=base.edge,
                position=base.position,
                severity=dominant,
                detail=base.detail,
                suggested_action=base.suggested_action,
                meta={
                    "votes": votes,
                    "k": self.k,
                    "severity_counts": sev_counts,
                    "entropy": entropy,
                },
            ))

        avg_entropy = sum(vote_entropies) / len(vote_entropies) if vote_entropies else 0.0
        return CritiqueResult(
            checker=f"ensemble[k={self.k}]",
            passed=not merged,
            violations=merged,
            meta={
                "k": self.k,
                "vote_entropy": avg_entropy,
                "per_critic_meta": [r.meta for r in per_critic],
            },
        )

    # -------- Private --------

    @staticmethod
    def _dominant_severity(sev_counts: Dict[str, int], vs: List[Violation]) -> Severity:
        max_count = max(sev_counts.values()) if sev_counts else 0
        if max_count == 0:
            return Severity.LOW
        tied = [k for k, v in sev_counts.items() if v == max_count]
        if len(tied) == 1:
            return Severity(tied[0])
        order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
        for s in order:
            if s.value in tied:
                return s
        return vs[0].severity

    @staticmethod
    def _downgrade(s: Severity) -> Severity:
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        idx = order.index(s)
        return order[max(0, idx - 1)]

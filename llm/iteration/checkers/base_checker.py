#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaseChecker - M2 Checker Base Class and Shared Data Structures

Each checker (structural / temporal / semantic) produces a set of Violations;
all violations are aggregated into a CritiqueResult, serving as input for the Refiner and
scoring basis for DeterministicComparator.

Design principles:
- A violation is self-describing: knows its assertion / chain / edge / severity / one-line readable description
- assertion is a string enum for easy cross-checker aggregation statistics
- checker does not depend on any LLM; semantic layer uses separate SemanticCritic (optionally with LLM)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CheckerKind(str, Enum):
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"


SEVERITY_WEIGHT = {
    Severity.LOW: 0.05,
    Severity.MEDIUM: 0.15,
    Severity.HIGH: 0.35,
    Severity.CRITICAL: 0.6,
}


@dataclass
class Violation:
    """Single violation, shared format across all checkers"""
    checker: str                    # "structural" / "temporal" / "semantic"
    assertion: str                  # e.g. "every_edge_in_causal_graph"
    chain_id: Optional[str] = None
    edge: Optional[List[str]] = None      # [src, dst], can be None if related to whole chain
    position: Optional[int] = None        # Position on chain (hop index)
    severity: Severity = Severity.MEDIUM
    detail: str = ""                # One-line readable description
    suggested_action: Optional[str] = None  # Suggestion for Refiner (natural language)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class CritiqueResult:
    """Output of a single checker"""
    checker: str
    passed: bool                    # Whether all chains are completely violation-free
    violations: List[Violation] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checker": self.checker,
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "meta": self.meta,
        }

    def by_severity(self) -> Dict[str, int]:
        bucket = {s.value: 0 for s in Severity}
        for v in self.violations:
            bucket[v.severity.value] += 1
        return bucket

    def by_chain(self) -> Dict[str, List[Violation]]:
        grouped: Dict[str, List[Violation]] = {}
        for v in self.violations:
            grouped.setdefault(v.chain_id or "_global", []).append(v)
        return grouped


@dataclass
class AggregatedCritique:
    """Merge outputs from multiple checkers for use by Refiner / Comparator"""
    results: List[CritiqueResult] = field(default_factory=list)

    @property
    def all_violations(self) -> List[Violation]:
        out: List[Violation] = []
        for r in self.results:
            out.extend(r.violations)
        return out

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def severity_counts(self) -> Dict[str, int]:
        bucket = {s.value: 0 for s in Severity}
        for v in self.all_violations:
            bucket[v.severity.value] += 1
        return bucket

    def penalty_score(self) -> float:
        """Compute total penalty score by severity (used by DeterministicComparator)"""
        return sum(SEVERITY_WEIGHT[v.severity] for v in self.all_violations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "severity_counts": self.severity_counts(),
            "penalty_score": self.penalty_score(),
            "results": [r.to_dict() for r in self.results],
        }


class BaseChecker:
    """Common interface for all checkers"""

    kind: CheckerKind = CheckerKind.STRUCTURAL
    name: str = "base"

    def check(self, chains: List[Dict[str, Any]], **kwargs) -> CritiqueResult:
        raise NotImplementedError

    # Utility: convert chain step sequence from propagation_chains JSON to node sequence
    @staticmethod
    def steps_to_nodes(steps: List[Dict[str, Any]]) -> List[str]:
        if not steps:
            return []
        seq: List[str] = []
        for i, s in enumerate(steps):
            src = s.get("from")
            dst = s.get("to")
            if not src or not dst:
                continue
            if i == 0:
                seq.append(src)
            seq.append(dst)
        return seq

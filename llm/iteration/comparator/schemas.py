#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparator Data Schemas

Define all output data structures for the Comparator module
"""

from typing import List, Dict, Any, Literal
from dataclasses import dataclass, field, asdict


@dataclass
class FixedIssue:
    """
    Fixed issue

    Represents an issue that existed in the original chain and was fixed in the refined chain
    """
    chain_id: int
    issue_type: str
    original_severity: str
    status: Literal["fixed", "partially_fixed", "not_fixed"]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class RemainingIssue:
    """
    Remaining issue

    Represents an issue that existed in the original chain and still exists in the refined chain
    """
    chain_id: int
    issue_type: str
    status: Literal["still_exists", "partially_resolved"]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class NewIssue:
    """
    Newly introduced issue

    Represents an issue that did not exist in the original chain but appeared in the refined chain
    """
    chain_id: int
    issue_type: str
    severity: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ChainComparison:
    """
    Comparison result for a single chain
    """
    chain_id: int
    result: Literal["better", "worse", "tie"]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ComparisonSummary:
    """
    Comparison result summary
    """
    comparison_result: Literal["better", "worse", "tie"]
    keep_new_version: bool
    confidence: Literal["high", "medium", "low"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ComparatorResult:
    """
    Complete output result of Comparator
    """
    comparison_summary: ComparisonSummary
    fixed_issues: List[FixedIssue] = field(default_factory=list)
    remaining_issues: List[RemainingIssue] = field(default_factory=list)
    new_issues: List[NewIssue] = field(default_factory=list)
    chain_comparisons: List[ChainComparison] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary, suitable for JSON serialization

        Returns:
            Complete dictionary representation
        """
        return {
            'comparison_summary': self.comparison_summary.to_dict(),
            'fixed_issues': [issue.to_dict() for issue in self.fixed_issues],
            'remaining_issues': [issue.to_dict() for issue in self.remaining_issues],
            'new_issues': [issue.to_dict() for issue in self.new_issues],
            'chain_comparisons': [comp.to_dict() for comp in self.chain_comparisons],
            'reason': self.reason
        }


# Issue type constants
class IssueType:
    """Issue type enumeration"""
    # Hard violations
    REVERSE_CAUSALITY = "reverse_causality"
    UNSUPPORTED_EDGE = "unsupported_edge"
    INVALID_ROOT_START = "invalid_root_start"
    SEVERE_LAYER_VIOLATION = "severe_layer_violation"

    # Soft issues
    MISSING_INTERMEDIATE_STEP = "missing_intermediate_step"
    WEAK_OBSERVATION_SUPPORT = "weak_observation_support"
    MILD_LAYER_SKIP = "mild_layer_skip"
    WEAK_TAIL_EXTENSION = "weak_tail_extension"
    UNCLEAR_MECHANISM = "unclear_mechanism"


# Severity constants
class Severity:
    """Severity enumeration"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Comparison result constants
class ComparisonResult:
    """Comparison result enumeration"""
    BETTER = "better"
    WORSE = "worse"
    TIE = "tie"


# Confidence constants
class Confidence:
    """Confidence enumeration"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


if __name__ == "__main__":
    # Test schema
    print("=== Testing Comparator Schemas ===\n")

    # Test FixedIssue
    fixed = FixedIssue(
        chain_id=1,
        issue_type=IssueType.SEVERE_LAYER_VIOLATION,
        original_severity=Severity.HIGH,
        status="fixed",
        description="Layer violation fixed"
    )
    print("FixedIssue:", fixed.to_dict())

    # Test NewIssue
    new = NewIssue(
        chain_id=2,
        issue_type=IssueType.UNSUPPORTED_EDGE,
        severity=Severity.MEDIUM,
        description="Introduced unsupported edge"
    )
    print("\nNewIssue:", new.to_dict())

    # Test ComparisonSummary
    summary = ComparisonSummary(
        comparison_result=ComparisonResult.BETTER,
        keep_new_version=True,
        confidence=Confidence.HIGH
    )
    print("\nComparisonSummary:", summary.to_dict())

    # Test ComparatorResult
    result = ComparatorResult(
        comparison_summary=summary,
        fixed_issues=[fixed],
        new_issues=[new],
        reason="New chain fixed 1 hard violation but introduced 1 medium issue, still better overall"
    )
    print("\nComparatorResult:")
    import json
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    print("\n=== Schema Test Complete ===")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparator Module

Compare the relative quality of original_chains and refined_chains,
determine if new chains are better than old chains (better/worse/tie)
"""

from .schemas import (
    FixedIssue,
    RemainingIssue,
    NewIssue,
    ChainComparison,
    ComparisonSummary,
    ComparatorResult,
    IssueType,
    Severity,
    ComparisonResult,
    Confidence
)

# Lazy import to avoid circular dependency
def get_comparator():
    """
    Get Comparator instance

    Lazy import to avoid circular dependency
    """
    from .comparator import Comparator
    return Comparator()


def get_deterministic_comparator(critic_pipeline, **kwargs):
    """
    Get M2 DeterministicComparator instance

    Args:
        critic_pipeline: Object implementing .critique(chains, **kwargs) (e.g. M2CriticPipeline)
        **kwargs: Passed through to DeterministicComparator constructor
    """
    from .deterministic_comparator import DeterministicComparator
    return DeterministicComparator(critic_pipeline=critic_pipeline, **kwargs)


__all__ = [
    # Main classes
    'get_comparator',
    'get_deterministic_comparator',

    # Data classes
    'FixedIssue',
    'RemainingIssue',
    'NewIssue',
    'ChainComparison',
    'ComparisonSummary',
    'ComparatorResult',

    # Constants
    'IssueType',
    'Severity',
    'ComparisonResult',
    'Confidence',
]


__version__ = '1.0.0'

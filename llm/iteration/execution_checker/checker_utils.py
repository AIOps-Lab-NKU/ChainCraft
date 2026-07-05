# -*- coding: utf-8 -*-
"""
Execution Checker Utility Functions

Provide various utility functions for Execution Checker
"""

import re
from typing import List, Tuple, Dict, Any, Optional


def compute_lcs(seq1: List[Tuple], seq2: List[Tuple]) -> List[Tuple]:
    """
    Compute the Longest Common Subsequence (LCS) of two sequences

    Use dynamic programming to compute LCS for comparing edge sequences of two chains

    Args:
        seq1: First sequence, typically the edge list of the original chain [(from, to), ...]
        seq2: Second sequence, typically the edge list of the refined chain [(from, to), ...]

    Returns:
        Longest common subsequence [(from, to), ...]

    Example:
        >>> seq1 = [('A', 'B'), ('B', 'C'), ('C', 'D')]
        >>> seq2 = [('A', 'B'), ('B', 'X'), ('C', 'D')]
        >>> compute_lcs(seq1, seq2)
        [('A', 'B'), ('C', 'D')]
    """
    if not seq1 or not seq2:
        return []

    m, n = len(seq1), len(seq2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Backtrack to find LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i-1] == seq2[j-1]:
            lcs.append(seq1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return list(reversed(lcs))


def parse_location_string(location_str: str) -> Dict[str, Any]:
    """
    Parse location description string

    Supports multiple formats:
    - "step 1 to step 2"
    - "between step 1 and step 2"
    - "from step 1 to step 2"
    - "at step 3"
    - "start node"
    - "tail/end"

    Args:
        location_str: Location description string

    Returns:
        Parsed location info dictionary
        {
            'type': 'between_steps' | 'at_step' | 'start' | 'tail' | 'range',
            'from_step': int | None,
            'to_step': int | None,
            'at_step': int | None
        }

    Example:
        >>> parse_location_string("between step 2 and step 3")
        {'type': 'between_steps', 'from_step': 2, 'to_step': 3, 'at_step': None}
    """
    if not location_str:
        return {'type': 'unknown', 'from_step': None, 'to_step': None, 'at_step': None}

    location_str = location_str.lower().strip()

    # Check start node
    if 'start' in location_str or 'root' in location_str:
        return {'type': 'start', 'from_step': None, 'to_step': None, 'at_step': None}

    # Check tail/end
    if 'tail' in location_str or 'end' in location_str:
        return {'type': 'tail', 'from_step': None, 'to_step': None, 'at_step': None}

    # Parse "between step X and step Y"
    match = re.search(r'between\s+step\s+(\d+)\s+and\s+step\s+(\d+)', location_str)
    if match:
        return {
            'type': 'between_steps',
            'from_step': int(match.group(1)),
            'to_step': int(match.group(2)),
            'at_step': None
        }

    # Parse "from step X to step Y" or "step X to step Y"
    match = re.search(r'(?:from\s+)?step\s+(\d+)\s+to\s+step\s+(\d+)', location_str)
    if match:
        return {
            'type': 'range',
            'from_step': int(match.group(1)),
            'to_step': int(match.group(2)),
            'at_step': None
        }

    # Parse "at step X" or standalone "step X"
    match = re.search(r'(?:at\s+)?step\s+(\d+)', location_str)
    if match:
        return {
            'type': 'at_step',
            'from_step': None,
            'to_step': None,
            'at_step': int(match.group(1))
        }

    return {'type': 'unknown', 'from_step': None, 'to_step': None, 'at_step': None}


def extract_step_context(chain: Dict[str, Any], step_idx: int, window: int = 1) -> Dict[str, Any]:
    """
    Extract context information for a specified step

    Args:
        chain: Chain dictionary
        step_idx: Step index (1-based)
        window: Context window size (window steps before and after)

    Returns:
        Context info dictionary
        {
            'target_step': dict | None,
            'prev_steps': [dict],
            'next_steps': [dict],
            'is_start': bool,
            'is_end': bool
        }
    """
    steps = chain.get('chain', [])
    if not steps or step_idx < 1 or step_idx > len(steps):
        return {
            'target_step': None,
            'prev_steps': [],
            'next_steps': [],
            'is_start': False,
            'is_end': False
        }

    # Convert to 0-based index
    idx = step_idx - 1

    return {
        'target_step': steps[idx],
        'prev_steps': steps[max(0, idx - window):idx],
        'next_steps': steps[idx + 1:min(len(steps), idx + 1 + window)],
        'is_start': (idx == 0),
        'is_end': (idx == len(steps) - 1)
    }


def match_edge_pattern(edge: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
    """
    Check if edge matches pattern (supports wildcards)

    Args:
        edge: Edge dictionary {'from': str, 'to': str, 'relation': str, ...}
        pattern: Pattern dictionary, None value means match any

    Returns:
        Whether it matches

    Example:
        >>> edge = {'from': 'A', 'to': 'B', 'relation': 'causes'}
        >>> pattern = {'from': 'A', 'to': None}  # Only match from
        >>> match_edge_pattern(edge, pattern)
        True
    """
    for key, value in pattern.items():
        if value is None:
            continue  # None means wildcard, skip
        if key not in edge:
            return False
        if edge[key] != value:
            return False
    return True


def normalize_metric_name(metric: str) -> str:
    """
    Normalize metric name for comparison

    Processing:
    - Strip leading/trailing whitespace
    - Convert to lowercase
    - Remove special characters

    Args:
        metric: Original metric name

    Returns:
        Normalized metric name

    Example:
        >>> normalize_metric_name("  Middleware_TDDL_RT  ")
        "middleware_tddl_rt"
    """
    if not metric:
        return ""

    # Strip whitespace
    normalized = metric.strip()

    # Convert to lowercase
    normalized = normalized.lower()

    # Optional: remove or replace special characters (adjust as needed)
    # normalized = re.sub(r'[^a-z0-9_]', '_', normalized)

    return normalized


def find_edge_in_chain(chain: Dict[str, Any], from_node: str, to_node: str) -> Optional[int]:
    """
    Find the position of a specified edge in a chain

    Args:
        chain: Chain dictionary
        from_node: Source node name
        to_node: Target node name

    Returns:
        Step index (1-based), or None if not found

    Example:
        >>> chain = {'chain': [{'from': 'A', 'to': 'B'}, {'from': 'B', 'to': 'C'}]}
        >>> find_edge_in_chain(chain, 'B', 'C')
        2
    """
    steps = chain.get('chain', [])
    for idx, step in enumerate(steps):
        if step.get('from') == from_node and step.get('to') == to_node:
            return idx + 1  # Return 1-based index
    return None


def edges_equal(edge1: Dict[str, Any], edge2: Dict[str, Any],
                check_relation: bool = False) -> bool:
    """
    Compare whether two edges are equal

    Args:
        edge1: First edge
        edge2: Second edge
        check_relation: Whether to also check the relation field

    Returns:
        Whether they are equal
    """
    if edge1.get('from') != edge2.get('from'):
        return False
    if edge1.get('to') != edge2.get('to'):
        return False
    if check_relation and edge1.get('relation') != edge2.get('relation'):
        return False
    return True


def format_location_description(location: Dict[str, Any]) -> str:
    """
    Format location info into a readable string

    Args:
        location: Location info dictionary

    Returns:
        Formatted location description

    Example:
        >>> location = {'chain_id': 1, 'from_step': 2, 'to_step': 3}
        >>> format_location_description(location)
        "Chain 1, from step 2 to step 3"
    """
    parts = []

    if 'chain_id' in location and location['chain_id'] is not None:
        parts.append(f"Chain {location['chain_id']}")

    if 'at_step' in location and location['at_step'] is not None:
        parts.append(f"at step {location['at_step']}")
    elif 'from_step' in location and 'to_step' in location:
        if location['from_step'] is not None and location['to_step'] is not None:
            parts.append(f"from step {location['from_step']} to step {location['to_step']}")
        elif location['from_step'] is not None:
            parts.append(f"from step {location['from_step']}")
        elif location['to_step'] is not None:
            parts.append(f"to step {location['to_step']}")

    if 'between_step' in location and location['between_step']:
        between = location['between_step']
        if isinstance(between, list) and len(between) == 2:
            parts.append(f"between step {between[0]} and step {between[1]}")

    return ', '.join(parts) if parts else "unknown location"


if __name__ == "__main__":
    # Simple test
    print("Testing compute_lcs...")
    seq1 = [('A', 'B'), ('B', 'C'), ('C', 'D')]
    seq2 = [('A', 'B'), ('B', 'X'), ('C', 'D')]
    lcs = compute_lcs(seq1, seq2)
    print(f"LCS: {lcs}")
    assert lcs == [('A', 'B'), ('C', 'D')], "LCS test failed"

    print("\nTesting parse_location_string...")
    result = parse_location_string("between step 2 and step 3")
    print(f"Parsed: {result}")
    assert result['type'] == 'between_steps', "Location parse test failed"
    assert result['from_step'] == 2, "Location parse test failed"
    assert result['to_step'] == 3, "Location parse test failed"

    print("\nAll tests passed!")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New Issue Detector

Detect new issues introduced in refined_chains (rule-based)
Only identify most critical new issues, no LLM calls
"""

from typing import List, Dict, Any, Optional, Set
import sys
import os
import csv
from io import StringIO

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llm.iteration.comparator.schemas import NewIssue, IssueType, Severity


class NewIssueDetector:
    """
    New issue detector (rule-based)

    Detect new structural issues introduced in refined_chains
    Rule-based detection only, no LLM calls
    """

    # Layer definitions
    LAYER_HIERARCHY = {
        'dependency_layer': 0,  # Dependency layer (bottom)
        'core_layer': 1,        # Core layer (middle)
        'inbound_layer': 2,     # Inbound layer (top)
        'system_layer': -1,     # System layer (special, can affect any layer)
        'jvm_layer': -1         # JVM layer (special, can affect any layer)
    }

    def __init__(self):
        """Initialize detector"""
        pass

    def detect(self,
               original_chains: List[Dict],
               refined_chains: List[Dict],
               evaluator_result: Dict,
               causal_analysis: str,
               metric_analysis: List[Dict]) -> List[NewIssue]:
        """
        Main detection method

        Args:
            original_chains: Original chain list
            refined_chains: Refined chain list
            evaluator_result: Evaluation result (used to get existing issues)
            causal_analysis: Causal analysis (CSV format)
            metric_analysis: Metric analysis (JSON format)

        Returns:
            New issue list
        """
        new_issues = []

        # Get existing issue set (to distinguish new vs existing issues)
        existing_issues = self._extract_existing_issues(evaluator_result)

        # Build metric layer mapping
        metric_layer_map = self._build_metric_layer_map(metric_analysis)

        # Parse causal_analysis
        causal_map = self._parse_causal_analysis(causal_analysis)

        # Check each refined chain
        for refined_chain in refined_chains:
            chain_id = refined_chain.get('chain_id')

            # Find corresponding original chain
            original_chain = self._find_chain_by_id(original_chains, chain_id)

            # Check layer violations
            layer_issues = self._check_layer_violations(
                refined_chain, metric_layer_map
            )

            # Check reverse causality
            causality_issues = self._check_reverse_causality(
                refined_chain, causal_map
            )

            # Check unsupported edges
            unsupported_issues = self._check_unsupported_edges(
                refined_chain, causal_map
            )

            # Check weak observation support
            weak_obs_issues = self._check_weak_observation(
                refined_chain, metric_analysis
            )

            # Merge all issues
            chain_issues = (
                layer_issues + causality_issues +
                unsupported_issues + weak_obs_issues
            )

            # Filter out existing issues, keep only new issues
            for issue in chain_issues:
                if not self._is_existing_issue(issue, existing_issues, original_chain):
                    new_issues.append(issue)

        return new_issues

    def _check_layer_violations(self,
                                chain: Dict,
                                metric_layer_map: Dict[str, str]) -> List[NewIssue]:
        """
        Check layer violations

        Rules:
        - dependency → inbound (no core bridge) → severe_layer_violation
        - inbound → dependency (reverse) → severe_layer_violation
        -Spanning two or more layers → severe_layer_violation
        - Other minor layer skips → mild_layer_skip

        Args:
            chain: Chain
            metric_layer_map: Metric to layer mapping

        Returns:
            Layer violation issue list
        """
        issues = []
        chain_id = chain.get('chain_id')
        edges = chain.get('chain', [])

        for idx, edge in enumerate(edges):
            from_metric = edge.get('from')
            to_metric = edge.get('to')

            from_layer = metric_layer_map.get(from_metric, 'unknown')
            to_layer = metric_layer_map.get(to_metric, 'unknown')

            if from_layer == 'unknown' or to_layer == 'unknown':
                continue

            # Get layer level values
            from_level = self.LAYER_HIERARCHY.get(from_layer, -2)
            to_level = self.LAYER_HIERARCHY.get(to_layer, -2)

            # Special layers (system/jvm) can affect any layer
            if from_level == -1:
                continue

            # Check reverse propagation (inbound → dependency)
            if from_level > to_level and (from_level - to_level) >= 2:
                issues.append(NewIssue(
                    chain_id=chain_id,
                    issue_type=IssueType.SEVERE_LAYER_VIOLATION,
                    severity=Severity.HIGH,
                    description=(
                        f"Step{idx+1}: from{from_layer}reverse propagation to{to_layer}"
                        f"（{from_metric} → {to_metric}），violates layer propagation direction"
                    )
                ))

            # Check forward layer skip (dependency → inbound, skipping core)
            elif to_level > from_level and (to_level - from_level) >= 2:
                issues.append(NewIssue(
                    chain_id=chain_id,
                    issue_type=IssueType.SEVERE_LAYER_VIOLATION,
                    severity=Severity.HIGH,
                    description=(
                        f"Step{idx+1}: from{from_layer}directly jumps to{to_layer}"
                        f"（{from_metric} → {to_metric}），missing intermediate layer bridge"
                    )
                ))

            # Check mild layer skip
            elif to_level > from_level and (to_level - from_level) == 1:
                # Normal layer propagation, but may be mild_layer_skip if lacking evidence
                # Not marking here, let unsupported_edge check handle it
                pass

        return issues

    def _check_reverse_causality(self,
                                chain: Dict,
                                causal_map: Dict[str, Dict]) -> List[NewIssue]:
        """
        Check reverse causality

        Rules:
        - If from node net impact score < to node → possibly reversed
        - If from rank is much lower than to (rank diff > 3) → possibly reversed

        Args:
            chain: Chain
            causal_map: Causal analysis mapping

        Returns:
            Reverse causality issue list
        """
        issues = []
        chain_id = chain.get('chain_id')
        edges = chain.get('chain', [])

        for idx, edge in enumerate(edges):
            from_metric = edge.get('from')
            to_metric = edge.get('to')

            from_info = causal_map.get(from_metric, {})
            to_info = causal_map.get(to_metric, {})

            if not from_info or not to_info:
                continue

            from_net_impact = from_info.get('net_impact', 0)
            to_net_impact = to_info.get('net_impact', 0)

            from_rank = from_info.get('rank', 999)
            to_rank = to_info.get('rank', 999)

            # Rule 1: Net impact score clearly reversed
            if from_net_impact < to_net_impact - 1:
                issues.append(NewIssue(
                    chain_id=chain_id,
                    issue_type=IssueType.REVERSE_CAUSALITY,
                    severity=Severity.HIGH,
                    description=(
                        f"Step{idx+1}: {from_metric}（net impact score{from_net_impact}）"
                        f" → {to_metric}（net impact score{to_net_impact}），"
                        f"Net impact scores inverted, possible reverse causality"
                    )
                ))

            # Rule 2: Rank gap too large
            elif from_rank > to_rank + 3:
                issues.append(NewIssue(
                    chain_id=chain_id,
                    issue_type=IssueType.REVERSE_CAUSALITY,
                    severity=Severity.MEDIUM,
                    description=(
                        f"Step{idx+1}: {from_metric}（rank{from_rank}）"
                        f" → {to_metric}（rank{to_rank}），"
                        f"Rank gap too large, possible reverse causality"
                    )
                ))

        return issues

    def _check_unsupported_edges(self,
                                chain: Dict,
                                causal_map: Dict[str, Dict]) -> List[NewIssue]:
        """
        Check unsupported edges

        Rules:
        - Edge has no corresponding relation in causal_analysis (out-degree 0 or in-degree 0 both problematic)
        - and relation description is too generic

        Args:
            chain: Chain
            causal_map: Causal analysis mapping

        Returns:
            Unsupported edge issue list
        """
        issues = []
        chain_id = chain.get('chain_id')
        edges = chain.get('chain', [])

        # Generic relation description keywords
        generic_relations = [
            'affect', 'cause', 'induce', 'trigger', 'correlate',
            'affect', 'cause', 'trigger', 'relate', 'impact'
        ]

        for idx, edge in enumerate(edges):
            from_metric = edge.get('from')
            to_metric = edge.get('to')
            relation = edge.get('relation', '').lower()

            from_info = causal_map.get(from_metric, {})
            to_info = causal_map.get(to_metric, {})

            # Check for causal support
            has_causal_support = False

            if from_info:
                out_degree = from_info.get('out_degree', 0)
                if out_degree > 0:
                    has_causal_support = True

            if to_info:
                in_degree = to_info.get('in_degree', 0)
                if in_degree > 0:
                    has_causal_support = True

            # Check if relation is generic
            is_generic_relation = any(
                keyword in relation
                for keyword in generic_relations
            )

            # If no causal support and generic description
            if not has_causal_support and is_generic_relation:
                issues.append(NewIssue(
                    chain_id=chain_id,
                    issue_type=IssueType.UNSUPPORTED_EDGE,
                    severity=Severity.HIGH,
                    description=(
                        f"Step{idx+1}: {from_metric} → {to_metric} "
                        f"Lacks causal analysis support, and relation description is generic"
                    )
                ))

        return issues

    def _check_weak_observation(self,
                               chain: Dict,
                               metric_analysis: List[Dict]) -> List[NewIssue]:
        """
        Check weak observation support

        Rules:
        - Node operational_severity is Low or Medium
        - and statistical_severity is also Low or Medium

        Args:
            chain: Chain
            metric_analysis: Metric analysis list

        Returns:
            Weak observation support issue list
        """
        issues = []
        chain_id = chain.get('chain_id')
        edges = chain.get('chain', [])

        # Build metric → analysis mapping
        metric_map = {
            m.get('metric_name'): m
            for m in metric_analysis
        }

        # Collect all nodes in chain
        nodes = set()
        for edge in edges:
            nodes.add(edge.get('from'))
            nodes.add(edge.get('to'))

        for node in nodes:
            metric_info = metric_map.get(node)
            if not metric_info:
                continue

            operational_assessment = metric_info.get('operational_assessment', {})
            op_severity = operational_assessment.get('operational_severity', 'Medium')
            stat_severity = operational_assessment.get('statistical_severity', 'Medium')

            # If neither is High/Critical
            if op_severity in ['Low', 'Medium'] and stat_severity in ['Low', 'Medium']:
                issues.append(NewIssue(
                    chain_id=chain_id,
                    issue_type=IssueType.WEAK_OBSERVATION_SUPPORT,
                    severity=Severity.MEDIUM,
                    description=(
                        f"Node {node} has weak observation support: "
                        f"operational_severity={op_severity}, "
                        f"statistical_severity={stat_severity}"
                    )
                ))

        return issues

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _extract_existing_issues(self, evaluator_result: Dict) -> Set[tuple]:
        """
        Extract existing issue set

        Args:
            evaluator_result: evaluation result

        Returns:
            Issue signature set, format: (chain_id, issue_type, location_key)
        """
        existing = set()

        chain_evaluations = evaluator_result.get('chain_evaluations', [])
        for chain_eval in chain_evaluations:
            chain_id = chain_eval.get('chain_id')

            # Collect hard violations
            for issue in chain_eval.get('hard_violations', []):
                issue_type = issue.get('issue_type')
                location = issue.get('location', {})
                location_key = (
                    location.get('from_step'),
                    location.get('to_step')
                )
                existing.add((chain_id, issue_type, location_key))

            # Collect soft issues
            for issue in chain_eval.get('soft_issues', []):
                issue_type = issue.get('issue_type')
                location = issue.get('location', {})
                location_key = (
                    location.get('from_step'),
                    location.get('to_step')
                )
                existing.add((chain_id, issue_type, location_key))

        return existing

    def _is_existing_issue(self,
                          issue: NewIssue,
                          existing_issues: Set[tuple],
                          original_chain: Optional[Dict]) -> bool:
        """
        Determine if issue is pre-existing

        Args:
            issue: Newly detected issue
            existing_issues: Existing issue set
            original_chain: original chains

        Returns:
            True if pre-existing issue, False if new issue
        """
        # Simplified: only check chain_id and issue_type
        # More precise check could further verify position
        chain_id = issue.chain_id
        issue_type = issue.issue_type

        for existing in existing_issues:
            if existing[0] == chain_id and existing[1] == issue_type:
                return True

        return False

    def _build_metric_layer_map(self, metric_analysis: List[Dict]) -> Dict[str, str]:
        """
        Build metric to layer mapping

        Args:
            metric_analysis: Metric analysis list

        Returns:
            {metric_name: layer} mapping
        """
        metric_layer_map = {}
        for metric_info in metric_analysis:
            metric_name = metric_info.get('metric_name')
            layer = metric_info.get('layer', 'unknown')
            if metric_name:
                metric_layer_map[metric_name] = layer
        return metric_layer_map

    def _parse_causal_analysis(self, causal_csv: str) -> Dict[str, Dict]:
        """
        Parse causal analysis CSV

        Args:
            causal_csv: CSV format causal analysis

        Returns:
            {metric_name: {rank, net_impact, out_degree, in_degree}} mapping
        """
        causal_map = {}

        try:
            reader = csv.DictReader(StringIO(causal_csv))
            for row in reader:
                metric_name = row.get('metric_name', '').strip()
                if not metric_name:
                    continue

                rank = int(row.get('rank', '999'))
                net_impact = float(row.get('net impact score', '0'))

                # Parse (out_degree/in_degree)
                degree_str = row.get('(out_degree/in_degree)', '(0/0)')
                degree_str = degree_str.strip('()')
                out_deg, in_deg = 0, 0
                if '/' in degree_str:
                    parts = degree_str.split('/')
                    out_deg = int(parts[0])
                    in_deg = int(parts[1])

                causal_map[metric_name] = {
                    'rank': rank,
                    'net_impact': net_impact,
                    'out_degree': out_deg,
                    'in_degree': in_deg
                }
        except Exception as e:
            print(f"Warning: Failed to parse causal_analysis: {e}")

        return causal_map

    def _find_chain_by_id(self, chains: List[Dict], chain_id: int) -> Optional[Dict]:
        """Find chain by chain_id"""
        for chain in chains:
            if chain.get('chain_id') == chain_id:
                return chain
        return None


if __name__ == "__main__":
    print("NewIssueDetector module loaded successfully")

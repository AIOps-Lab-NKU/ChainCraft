#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Issue Detector

Detect whether issues identified by evaluator in original chain are fixed in refined chain
"""

from typing import List, Dict, Any, Optional
import sys
import os

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llm.iteration.comparator.schemas import FixedIssue, IssueType, Severity
from llm.iteration.execution_checker.chain_diff import ChainDiff


class FixedIssueDetector:
    """
    Fixed issue detector

    Detect whether issues identified by evaluator are fixed in refined chain
    """

    def __init__(self):
        """Initialize detector"""
        pass

    def detect(self,
               original_chains: List[Dict],
               refined_chains: List[Dict],
               evaluator_result: Dict,
               execution_check_result: Dict) -> List[FixedIssue]:
        """
        Main detection method

        Args:
            original_chains: Original chain list
            refined_chains: Refined chain list
            evaluator_result: Evaluation result
            execution_check_result: Execution check result

        Returns:
            Fixed issue list
        """
        fixed_issues = []

        # Get all chain evaluations
        chain_evaluations = evaluator_result.get('chain_evaluations', [])

        for chain_eval in chain_evaluations:
            chain_id = chain_eval.get('chain_id')

            # Get original and refined versions of this chain
            original_chain = self._find_chain_by_id(original_chains, chain_id)
            refined_chain = self._find_chain_by_id(refined_chains, chain_id)

            if not original_chain or not refined_chain:
                continue

            # Get execution check result of this chain
            chain_check = self._find_chain_check(execution_check_result, chain_id)

            # Check all hard violations
            for issue in chain_eval.get('hard_violations', []):
                fixed_issue = self._check_issue_fixed(
                    issue, original_chain, refined_chain, chain_check
                )
                if fixed_issue:
                    fixed_issues.append(fixed_issue)

            # Check all soft issues
            for issue in chain_eval.get('soft_issues', []):
                fixed_issue = self._check_issue_fixed(
                    issue, original_chain, refined_chain, chain_check
                )
                if fixed_issue:
                    fixed_issues.append(fixed_issue)

        return fixed_issues

    def _check_issue_fixed(self,
                          issue: Dict,
                          original_chain: Dict,
                          refined_chain: Dict,
                          chain_check: Optional[Dict]) -> Optional[FixedIssue]:
        """
        Check if a single issue is fixed

        Args:
            issue: Issue description
            original_chain: Original chain
            refined_chain: Refined chain
            chain_check: Execution check result of this chain

        Returns:
            FixedIssue object, returns None if issue does not exist
        """
        chain_id = original_chain.get('chain_id')
        issue_type = issue.get('issue_type')
        severity = issue.get('severity', 'medium')
        location = issue.get('location', {})
        edit_action = issue.get('edit_action', {})

        # Find corresponding action_check from execution_check
        action_check = self._find_action_check(chain_check, edit_action, location)

        # Determine fix status based on issue_type
        if issue_type == IssueType.SEVERE_LAYER_VIOLATION or issue_type == IssueType.MILD_LAYER_SKIP:
            status, description = self._check_layer_violation_fixed(
                issue, original_chain, refined_chain, action_check
            )
        elif issue_type == IssueType.UNSUPPORTED_EDGE:
            status, description = self._check_unsupported_edge_fixed(
                issue, original_chain, refined_chain, action_check
            )
        elif issue_type == IssueType.MISSING_INTERMEDIATE_STEP:
            status, description = self._check_missing_step_fixed(
                issue, original_chain, refined_chain, action_check
            )
        elif issue_type == IssueType.WEAK_TAIL_EXTENSION:
            status, description = self._check_weak_tail_fixed(
                issue, original_chain, refined_chain, action_check
            )
        elif issue_type == IssueType.INVALID_ROOT_START:
            status, description = self._check_invalid_root_fixed(
                issue, original_chain, refined_chain, action_check
            )
        elif issue_type == IssueType.UNCLEAR_MECHANISM:
            status, description = self._check_unclear_mechanism_fixed(
                issue, original_chain, refined_chain, action_check
            )
        else:
            # Unknown type, default to checking action_check
            status, description = self._check_generic_fix(
                issue, action_check
            )

        return FixedIssue(
            chain_id=chain_id,
            issue_type=issue_type,
            original_severity=severity,
            status=status,
            description=description
        )

    def _check_layer_violation_fixed(self,
                                     issue: Dict,
                                     original_chain: Dict,
                                     refined_chain: Dict,
                                     action_check: Optional[Dict]) -> tuple:
        """
        Check if layer violation is fixed

        Layer violations are usually fixed via insert_step, inserting bridge nodes

        Returns:
            (status, description) tuple
        """
        location = issue.get('location', {})
        from_step = location.get('from_step')
        to_step = location.get('to_step')

        # Check if action_check shows successful execution
        if action_check and action_check.get('severity') == 'pass':
            # Verify that valid bridge nodes were actually inserted
            diff = ChainDiff(original_chain, refined_chain)
            diff_result = diff.compute_diff()

            nodes_added = diff_result.get('nodes_added', [])
            if nodes_added:
                return (
                    "fixed",
                    f"Successfully inserted bridge nodes between step {from_step} and {to_step}:  {', '.join(nodes_added)}，"
                    f", fixing the layer violation"
                )
            else:
                return (
                    "partially_fixed",
                    f"action_check shows successful execution, but no new nodes detected"
                )
        elif action_check and action_check.get('severity') == 'error':
            return (
                "not_fixed",
                f"Fix action execution failed: {action_check.get('reason', 'unknown reason')}"
            )
        elif not action_check:
            # Check if fixed through other means
            diff = ChainDiff(original_chain, refined_chain)
            diff_result = diff.compute_diff()

            # Check if edges at issue position were modified
            edges_removed = diff_result.get('edges_removed', [])
            edges_added = diff_result.get('edges_added', [])

            if edges_added or edges_removed:
                return (
                    "partially_fixed",
                    f"Edge structure at issue position changed, may have partially mitigated layer violation"
                )
            else:
                return (
                    "not_fixed",
                    f"No corresponding fix action found, and chain structure unchanged"
                )
        else:
            return (
                "not_fixed",
                f"action_check status unknown or execution incomplete"
            )

    def _check_unsupported_edge_fixed(self,
                                     issue: Dict,
                                     original_chain: Dict,
                                     refined_chain: Dict,
                                     action_check: Optional[Dict]) -> tuple:
        """
        Check if unsupported edge is fixed

        Usually fixed via delete_step or replace_edge

        Returns:
            (status, description) tuple
        """
        location = issue.get('location', {})
        from_step = location.get('from_step')
        to_step = location.get('to_step')

        if action_check and action_check.get('severity') == 'pass':
            action = action_check.get('action')
            if action == 'delete_step':
                return (
                    "fixed",
                    f"Successfully deleted unsupported edge from step {from_step} to {to_step}"
                )
            elif action == 'replace_edge':
                return (
                    "fixed",
                    f"Successfully replaced edge from step {from_step} to {to_step}, using stronger causal support"
                )
            else:
                return (
                    "partially_fixed",
                    f"Executed action {action}，, may have partially mitigated unsupported edge issue"
                )
        elif action_check and action_check.get('severity') == 'error':
            return (
                "not_fixed",
                f"Fix action execution failed: {action_check.get('reason', 'unknown reason')}"
            )
        else:
            # Check if edge still exists
            diff = ChainDiff(original_chain, refined_chain)
            diff_result = diff.compute_diff()

            # Simple check: if edges were removed, issue may be mitigated
            edges_removed = diff_result.get('edges_removed', [])
            if edges_removed:
                return (
                    "partially_fixed",
                    f"Removed {len(edges_removed)} edges, possibly including unsupported edges"
                )
            else:
                return (
                    "not_fixed",
                    f"Unsupported edge still exists, no fix action detected"
                )

    def _check_missing_step_fixed(self,
                                  issue: Dict,
                                  original_chain: Dict,
                                  refined_chain: Dict,
                                  action_check: Optional[Dict]) -> tuple:
        """
        Check if missing intermediate step is fixed

        Usually fixed via insert_step

        Returns:
            (status, description) tuple
        """
        location = issue.get('location', {})
        from_step = location.get('from_step')
        to_step = location.get('to_step')

        if action_check and action_check.get('severity') == 'pass':
            # Check if nodes were actually inserted
            details = action_check.get('details', {})
            actual = details.get('actual', {})
            inserted_nodes = actual.get('inserted_nodes', [])

            if inserted_nodes:
                return (
                    "fixed",
                    f"Successfully inserted bridge nodes between step {from_step} and {to_step}:  : {', '.join(inserted_nodes)}"
                )
            else:
                return (
                    "partially_fixed",
                    f"action_check shows successful execution, but inserted nodes are unclear"
                )
        elif action_check and action_check.get('severity') == 'error':
            return (
                "not_fixed",
                f"Inserting intermediate step failed: {action_check.get('reason', 'unknown reason')}"
            )
        else:
            # Check if nodes were added
            diff = ChainDiff(original_chain, refined_chain)
            diff_result = diff.compute_diff()

            nodes_added = diff_result.get('nodes_added', [])
            if nodes_added:
                return (
                    "partially_fixed",
                    f"Detected {len(nodes_added)} new nodes, possibly filling missing intermediate steps"
                )
            else:
                return (
                    "not_fixed",
                    f"No new nodes detected, missing intermediate steps not filled"
                )

    def _check_weak_tail_fixed(self,
                               issue: Dict,
                               original_chain: Dict,
                               refined_chain: Dict,
                               action_check: Optional[Dict]) -> tuple:
        """
        Check if weak tail extension is fixed

        Usually fixed via shorten_chain

        Returns:
            (status, description) tuple
        """
        if action_check and action_check.get('severity') == 'pass':
            return (
                "fixed",
                f"Successfully shortened chain, removed weak tail extension"
            )
        elif action_check and action_check.get('severity') == 'error':
            return (
                "not_fixed",
                f"Shortening chain failed: {action_check.get('reason', 'unknown reason')}"
            )
        else:
            # Check if chain length decreased
            diff = ChainDiff(original_chain, refined_chain)
            diff_result = diff.compute_diff()

            length_change = diff_result.get('length_change', {})
            old_len = length_change.get('old', 0)
            new_len = length_change.get('new', 0)

            if new_len < old_len:
                return (
                    "partially_fixed",
                    f"Chain length reduced from {old_len} to {new_len}，, possibly removing weak tail"
                )
            else:
                return (
                    "not_fixed",
                    f"Chain length did not decrease, weak tail extension not fixed"
                )

    def _check_invalid_root_fixed(self,
                                  issue: Dict,
                                  original_chain: Dict,
                                  refined_chain: Dict,
                                  action_check: Optional[Dict]) -> tuple:
        """
        Check if invalid root start is fixed

        Usually fixed via replace_start_node

        Returns:
            (status, description) tuple
        """
        if action_check and action_check.get('severity') == 'pass':
            # Check if start node changed
            orig_edges = original_chain.get('chain', [])
            refn_edges = refined_chain.get('chain', [])

            if orig_edges and refn_edges:
                old_start = orig_edges[0].get('from')
                new_start = refn_edges[0].get('from')

                if old_start != new_start:
                    return (
                        "fixed",
                        f"Successfully replaced start node from '{old_start}' to '{new_start}'"
                    )
                else:
                    return (
                        "partially_fixed",
                        f"action_check shows successful execution, but start node unchanged"
                    )
            else:
                return (
                    "not_fixed",
                    f"Cannot compare start nodes"
                )
        elif action_check and action_check.get('severity') == 'error':
            return (
                "not_fixed",
                f"Replacing start node failed: {action_check.get('reason', 'unknown reason')}"
            )
        else:
            # Check if start node changed
            orig_edges = original_chain.get('chain', [])
            refn_edges = refined_chain.get('chain', [])

            if orig_edges and refn_edges:
                old_start = orig_edges[0].get('from')
                new_start = refn_edges[0].get('from')

                if old_start != new_start:
                    return (
                        "partially_fixed",
                        f"Start node changed from '{old_start}' to '{new_start}'，, but no corresponding fix action found"
                    )
                else:
                    return (
                        "not_fixed",
                        f"Start node unchanged, invalid root start issue not fixed"
                    )
            else:
                return (
                    "not_fixed",
                    f"Cannot determine if start node is fixed"
                )

    def _check_unclear_mechanism_fixed(self,
                                      issue: Dict,
                                      original_chain: Dict,
                                      refined_chain: Dict,
                                      action_check: Optional[Dict]) -> tuple:
        """
        Check if unclear mechanism is fixed

        Usually fixed via clarify_relation

        Returns:
            (status, description) tuple
        """
        location = issue.get('location', {})
        from_step = location.get('from_step')
        to_step = location.get('to_step')

        if action_check and action_check.get('severity') == 'pass':
            return (
                "fixed",
                f"Successfully clarified relation description from step {from_step} to {to_step} to step "
            )
        elif action_check and action_check.get('severity') == 'error':
            return (
                "not_fixed",
                f"Clarifying relation failed: {action_check.get('reason', 'unknown reason')}"
            )
        else:
            # Check if relation changed
            diff = ChainDiff(original_chain, refined_chain)
            diff_result = diff.compute_diff()

            edges_modified = diff_result.get('edges_modified', [])
            if edges_modified:
                return (
                    "partially_fixed",
                    f"Detected {len(edges_modified)} edges with changed relations, possibly clarifying mechanism"
                )
            else:
                return (
                    "not_fixed",
                    f"Relation description unchanged, mechanism still unclear"
                )

    def _check_generic_fix(self,
                          issue: Dict,
                          action_check: Optional[Dict]) -> tuple:
        """
        Generic check method

        Used when issue type is unknown

        Returns:
            (status, description) tuple
        """
        issue_type = issue.get('issue_type', 'unknown')

        if action_check and action_check.get('severity') == 'pass':
            return (
                "fixed",
                f"Issue type '{issue_type}' corresponding fix action executed successfully"
            )
        elif action_check and action_check.get('severity') == 'error':
            return (
                "not_fixed",
                f"Issue type '{issue_type}' corresponding fix action execution failed"
            )
        else:
            return (
                "not_fixed",
                f"Issue type '{issue_type}' no corresponding fix action found"
            )

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _find_chain_by_id(self, chains: List[Dict], chain_id: int) -> Optional[Dict]:
        """Find chain by chain_id"""
        for chain in chains:
            if chain.get('chain_id') == chain_id:
                return chain
        return None

    def _find_chain_check(self, execution_check_result: Dict, chain_id: int) -> Optional[Dict]:
        """Find corresponding chain check result from execution_check_result"""
        chain_checks = execution_check_result.get('chain_checks', [])
        for check in chain_checks:
            if check.get('chain_id') == chain_id:
                return check
        return None

    def _find_action_check(self,
                          chain_check: Optional[Dict],
                          edit_action: Dict,
                          location: Dict) -> Optional[Dict]:
        """
        Find corresponding action_check from chain_check

        Args:
            chain_check: Chain execution check result
            edit_action: Issue edit_action
            location: Issue location

        Returns:
            Corresponding action_check, returns None if not found
        """
        if not chain_check:
            return None

        action_checks = chain_check.get('action_checks', [])
        action_type = edit_action.get('action')

        # Simple match: find check with same action type
        for check in action_checks:
            if check.get('action') == action_type:
                # Can further check if position matches
                # Simply return first match for now
                return check

        return None


if __name__ == "__main__":
    print("FixedIssueDetector module loaded successfully")

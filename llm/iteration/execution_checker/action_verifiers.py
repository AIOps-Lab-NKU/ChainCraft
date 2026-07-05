#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action Verifiers

Implement specialized verification logic for 6 types of edit_action
"""

from typing import Dict, List, Any, Optional
import re

try:
    from .chain_diff import ChainDiff
    from .checker_utils import (
        parse_location_string,
        find_edge_in_chain,
        edges_equal,
        format_location_description
    )
except ImportError:
    from chain_diff import ChainDiff
    from checker_utils import (
        parse_location_string,
        find_edge_in_chain,
        edges_equal,
        format_location_description
    )


class ActionVerifier:
    """
    Action verifier base class

    Parent class of all concrete verifiers, defines unified interface
    """

    def __init__(self,
                 evaluator_issue: Dict[str, Any],
                 edit_plan_item: Optional[Dict[str, Any]],
                 chain_diff: ChainDiff,
                 original_chain: Dict[str, Any],
                 refined_chain: Dict[str, Any]):
        """
        Initialize verifier

        Args:
            evaluator_issue: Single issue from Evaluator (containing edit_action)
            edit_plan_item: Corresponding item from RefineAgent edit_plan (may be None)
            chain_diff: Chain diff computation result
            original_chain: Original chain
            refined_chain: Refined chain
        """
        self.issue = evaluator_issue
        self.plan = edit_plan_item
        self.diff = chain_diff
        self.original = original_chain
        self.refined = refined_chain

        # Extract key information
        self.edit_action = evaluator_issue.get('edit_action', {})
        self.action_type = self.edit_action.get('action', 'unknown')
        self.constraint = self.edit_action.get('constraint', {})
        self.location = evaluator_issue.get('location', {})

    def verify(self) -> Dict[str, Any]:
        """
        Verify whether action was executed correctly

        Returns:
            Verification result dict
            {
                'action': str,
                'executed': bool,
                'correct_location': bool,
                'meets_constraints': bool,
                'severity': 'pass' | 'warning' | 'error',
                'reason': str,
                'details': {...}
            }
        """
        raise NotImplementedError("Subclasses must implement verify()")

    def _compute_severity(self, executed: bool, correct_location: bool,
                         meets_constraints: bool) -> str:
        """Compute severity"""
        if executed and correct_location and meets_constraints:
            return 'pass'
        elif not executed:
            return 'error'
        elif not correct_location or not meets_constraints:
            return 'warning'
        return 'pass'

    def _build_base_result(self) -> Dict[str, Any]:
        """Build base result structure"""
        return {
            'action': self.action_type,
            'executed': False,
            'correct_location': False,
            'meets_constraints': False,
            'severity': 'error',
            'reason': '',
            'details': {
                'expected': {},
                'actual': {},
                'discrepancies': []
            },
            'issue_reference': {
                'issue_type': self.issue.get('issue_type', ''),
                'location': format_location_description(self.location),
                'rationale': self.edit_action.get('rationale', '')
            }
        }


class InsertStepVerifier(ActionVerifier):
    """Verify insert_step action"""

    def verify(self) -> Dict[str, Any]:
        """Verify whether step insertion was executed correctly"""
        result = self._build_base_result()

        # Step 1: Parse insertion position
        expected_location = self._parse_insertion_location()
        result['details']['expected']['location'] = expected_location

        # Step 2: Get candidate node list
        candidate_metrics = self.constraint.get('candidate_metrics', [])
        result['details']['expected']['candidates'] = candidate_metrics

        if not candidate_metrics:
            result['reason'] = "candidate_metrics not provided in constraint, cannot verify inserted node"
            result['severity'] = 'warning'
            return result

        # Step 3: Check if new nodes were inserted
        inserted_nodes = self._find_inserted_nodes()
        result['details']['actual']['inserted_nodes'] = inserted_nodes

        if not inserted_nodes:
            result['executed'] = False
            result['reason'] = "No new node insertion detected"
            result['details']['discrepancies'].append("No new nodes added to chain")
            return result

        # Step 4: Check if inserted nodes are in candidate list
        valid_inserted = [node for node in inserted_nodes if node in candidate_metrics]

        if not valid_inserted:
            result['executed'] = True
            result['correct_location'] = False
            result['meets_constraints'] = False
            result['severity'] = 'error'
            result['reason'] = f"Inserted nodes {inserted_nodes}，but not in candidate list"
            result['details']['discrepancies'].append(
                f"Inserted nodes not in candidate_metrics: {candidate_metrics}"
            )
            return result

        # Step 5: Check if insertion position is correct
        position_correct, position_details = self._verify_insertion_position(
            valid_inserted[0], expected_location
        )
        result['correct_location'] = position_correct
        result['details']['actual']['position'] = position_details

        # Step 6: Check if other parts remain unchanged
        keep_unchanged = self.constraint.get('keep_other_steps_unchanged', True)
        if keep_unchanged:
            unchanged = self._verify_other_parts_unchanged(expected_location)
            result['meets_constraints'] = unchanged
            if not unchanged:
                result['details']['discrepancies'].append("Unexpected modifications detected at other positions")
        else:
            result['meets_constraints'] = True

        # Aggregate results
        result['executed'] = True
        result['severity'] = self._compute_severity(
            result['executed'],
            result['correct_location'],
            result['meets_constraints']
        )

        if result['severity'] == 'pass':
            result['reason'] = f"Successfully inserted node at{position_details}inserted node {valid_inserted[0]}"
        else:
            reasons = []
            if not position_correct:
                reasons.append("Insertion position is incorrect")
            if not result['meets_constraints']:
                reasons.append("Other parts were unexpectedly modified")
            result['reason'] = f"Inserted nodes {valid_inserted[0]}, but " + "，".join(reasons)

        return result

    def _parse_insertion_location(self) -> Dict[str, Any]:
        """Parse insertion position"""
        # Extract from_step and to_step from location
        from_step = self.location.get('from_step')
        to_step = self.location.get('to_step')

        # Also check target_location in plan
        if self.plan:
            target_loc = self.plan.get('target_location', {})
            if 'between_step' in target_loc and target_loc['between_step']:
                between = target_loc['between_step']
                from_step = between[0]
                to_step = between[1]

        return {
            'from_step': from_step,
            'to_step': to_step,
            'type': 'between_steps' if (from_step and to_step) else 'unknown'
        }

    def _find_inserted_nodes(self) -> List[str]:
        """Find newly inserted nodes"""
        diff_result = self.diff.compute_diff()
        return diff_result.get('nodes_added', [])

    def _verify_insertion_position(self, inserted_node: str,
                                   expected_location: Dict) -> tuple:
        """
        Verify insertion position is correct

        Returns:
            (position_correct, position_description)
        """
        from_step = expected_location.get('from_step')
        to_step = expected_location.get('to_step')

        if not from_step or not to_step:
            return False, "Cannot determine expected position"

        # Find edges containing inserted_node in refined chain
        refined_steps = self.refined.get('chain', [])
        for idx, step in enumerate(refined_steps):
            if inserted_node in [step.get('from'), step.get('to')]:
                # Found, check if position is betweenfrom_stepandto_step
                # Simplified check: verify adjacent nodes
                if idx > 0:
                    prev_step = refined_steps[idx - 1]
                    # Check if previous step corresponds to from_step node
                    orig_from_node = self._get_node_at_step(self.original, from_step, 'to')
                    if prev_step.get('to') == orig_from_node or prev_step.get('from') == orig_from_node:
                        return True, f"Step{idx + 1} (between Step{from_step}and{to_step})"

                return True, f"Step{idx + 1}"

        return False, "Insertion position not found"

    def _get_node_at_step(self, chain: Dict, step_num: int, field: str) -> Optional[str]:
        """Get node at specified Step"""
        steps = chain.get('chain', [])
        if 0 < step_num <= len(steps):
            return steps[step_num - 1].get(field)
        return None

    def _verify_other_parts_unchanged(self, expected_location: Dict) -> bool:
        """Verify if other parts remain unchanged"""
        # Simplified: check for excessive edge changes
        diff_result = self.diff.compute_diff()
        edges_added = diff_result.get('edges_added', [])
        edges_removed = diff_result.get('edges_removed', [])

        # insert_step should add 2 edges (before and after inserted node), remove 1 edge (original connection)
        # If change count significantly exceeds expectation, consider other modifications
        if len(edges_added) > 3 or len(edges_removed) > 2:
            return False

        return True


class DeleteStepVerifier(ActionVerifier):
    """Verify delete_step action"""

    def verify(self) -> Dict[str, Any]:
        """Verify whether step deletion was executed correctly"""
        result = self._build_base_result()

        # Step1: Identify Target
        target = self._identify_deletion_target()
        result['details']['expected']['target'] = target

        if not target:
            result['reason'] = "Cannot determine target to delete"
            result['severity'] = 'warning'
            return result

        # Step2: Check if Target was removed
        deleted, details = self._check_deletion_occurred(target)
        result['executed'] = deleted
        result['details']['actual']['deletion'] = details

        if not deleted:
            result['reason'] = f"Target{target}still exists in refined chain"
            result['details']['discrepancies'].append("Targetedge or node was not deleted")
            return result

        # Step3: Verify only target was deleted
        only_target = self._verify_only_target_removed(target)
        result['correct_location'] = only_target
        result['meets_constraints'] = only_target

        if not only_target:
            result['details']['discrepancies'].append("Additional deletion operations detected")

        result['severity'] = self._compute_severity(
            result['executed'],
            result['correct_location'],
            result['meets_constraints']
        )

        if result['severity'] == 'pass':
            result['reason'] = f"Successfully deleted Target{target}"
        else:
            result['reason'] = f"Deleted Target{target}，but also deleted other content"

        return result

    def _identify_deletion_target(self) -> Optional[Dict]:
        """Identify Target"""
        from_step = self.location.get('from_step')
        to_step = self.location.get('to_step')

        if from_step and to_step:
            # Get edge at this position
            orig_step = self._get_step_at_position(from_step)
            if orig_step:
                return {
                    'type': 'edge',
                    'from': orig_step.get('from'),
                    'to': orig_step.get('to'),
                    'position': from_step
                }

        return None

    def _get_step_at_position(self, position: int) -> Optional[Dict]:
        """Get step at specified position"""
        steps = self.original.get('chain', [])
        if 0 < position <= len(steps):
            return steps[position - 1]
        return None

    def _check_deletion_occurred(self, target: Dict) -> tuple:
        """Check if deletion occurred"""
        if target['type'] == 'edge':
            # Check if edge still exists in refined chain
            from_node = target['from']
            to_node = target['to']

            refined_steps = self.refined.get('chain', [])
            for step in refined_steps:
                if step.get('from') == from_node and step.get('to') == to_node:
                    return False, f"edge {from_node}->{to_node} still exists"

            return True, f"edge {from_node}->{to_node} deleted"

        return False, "Unknown Target type"

    def _verify_only_target_removed(self, target: Dict) -> bool:
        """Verify only Target"""
        diff_result = self.diff.compute_diff()
        edges_removed = diff_result.get('edges_removed', [])

        # Simplified: if too many edges removed, consider extra content deleted
        if len(edges_removed) > 1:
            return False

        return True


class ReplaceedgeVerifier(ActionVerifier):
    """Verify replace_edge action"""

    def verify(self) -> Dict[str, Any]:
        """Verify whether edge replacement was executed correctly"""
        result = self._build_base_result()

        # Step1: Identify edge to replaceedge
        target_edge = self._identify_target_edge()
        result['details']['expected']['target_edge'] = target_edge

        if not target_edge:
            result['reason'] = "Cannot determine edge to replaceedge"
            result['severity'] = 'warning'
            return result

        # Step2: Check if Old edge was removed
        old_removed = self._check_old_edge_removed(target_edge)
        result['details']['actual']['old_edge_removed'] = old_removed

        # Step3: Check if New edge was added
        new_added, new_edge = self._check_new_edge_added(target_edge)
        result['details']['actual']['new_edge_added'] = new_added
        result['details']['actual']['new_edge'] = new_edge

        result['executed'] = old_removed and new_added
        result['correct_location'] = old_removed and new_added
        result['meets_constraints'] = True  # Simplified

        result['severity'] = self._compute_severity(
            result['executed'],
            result['correct_location'],
            result['meets_constraints']
        )

        if result['severity'] == 'pass':
            result['reason'] = f"Successfully replaced edge: {target_edge} -> {new_edge}"
        elif not old_removed:
            result['reason'] = "Old edge was not removed"
        elif not new_added:
            result['reason'] = "New edge was not added"
        else:
            result['reason'] = "edge replacement was not executed successfully"

        return result

    def _identify_target_edge(self) -> Optional[Dict]:
        """Identify edge to replaceedge"""
        from_step = self.location.get('from_step')
        if from_step:
            step = self._get_step_at_position(from_step)
            if step:
                return {
                    'from': step.get('from'),
                    'to': step.get('to'),
                    'position': from_step
                }
        return None

    def _get_step_at_position(self, position: int) -> Optional[Dict]:
        """Get step at specified position"""
        steps = self.original.get('chain', [])
        if 0 < position <= len(steps):
            return steps[position - 1]
        return None

    def _check_old_edge_removed(self, target_edge: Dict) -> bool:
        """Check if Old edge was removed"""
        diff_result = self.diff.compute_diff()
        edges_removed = diff_result.get('edges_removed', [])

        from_node = target_edge['from']
        to_node = target_edge['to']

        for edge in edges_removed:
            if edge.get('from') == from_node and edge.get('to') == to_node:
                return True

        return False

    def _check_new_edge_added(self, target_edge: Dict) -> tuple:
        """Check if New edge was added"""
        diff_result = self.diff.compute_diff()
        edges_added = diff_result.get('edges_added', [])

        # Simplified: return any newly added edge
        if edges_added:
            return True, edges_added[0]

        return False, None


class ReplaceStartNodeVerifier(ActionVerifier):
    """Verify replace_start_node action"""

    def verify(self) -> Dict[str, Any]:
        """Verify whether start node replacement was executed correctly"""
        result = self._build_base_result()

        # Step1: Get original start node
        orig_start = self._get_start_node(self.original)
        result['details']['expected']['original_start'] = orig_start

        # Step2: Get refined start node
        refn_start = self._get_start_node(self.refined)
        result['details']['actual']['refined_start'] = refn_start

        # Step3: Check if start node changed
        start_changed = (orig_start != refn_start)
        result['executed'] = start_changed

        if not start_changed:
            result['reason'] = f"Start node unchanged, still {orig_start}"
            return result

        # Step4: Check if new start node is valid (candidate list)
        candidate_metrics = self.constraint.get('candidate_metrics', [])
        if candidate_metrics and refn_start not in candidate_metrics:
            result['correct_location'] = False
            result['meets_constraints'] = False
            result['reason'] = f"New start node {refn_start} not in candidate list"
            result['details']['discrepancies'].append(
                f"New start nodenot in candidate_metrics: {candidate_metrics}"
            )
        else:
            result['correct_location'] = True
            result['meets_constraints'] = True

        result['severity'] = self._compute_severity(
            result['executed'],
            result['correct_location'],
            result['meets_constraints']
        )

        if result['severity'] == 'pass':
            result['reason'] = f"Successfully replaced start node: {orig_start} -> {refn_start}"
        else:
            result['reason'] = f"Replaced start node with {refn_start}，but does not meet constraints"

        return result

    def _get_start_node(self, chain: Dict) -> Optional[str]:
        """Get chain start node"""
        steps = chain.get('chain', [])
        if steps:
            return steps[0].get('from')
        return None


class ShortenChainVerifier(ActionVerifier):
    """Verify shorten_chain action"""

    def verify(self) -> Dict[str, Any]:
        """Verify whether chain shortening was executed correctly"""
        result = self._build_base_result()

        # Step1: Check if chain length decreased
        orig_len = len(self.original.get('chain', []))
        refn_len = len(self.refined.get('chain', []))
        length_decreased = refn_len < orig_len

        result['details']['expected']['action'] = 'shorten_chain'
        result['details']['actual']['length_change'] = {
            'original': orig_len,
            'refined': refn_len,
            'decreased': length_decreased
        }

        result['executed'] = length_decreased

        if not length_decreased:
            result['reason'] = f"Chain length did not decrease（{orig_len} -> {refn_len})"
            return result

        # Step2: Check if deletion was from tail
        tail_removed = self._verify_tail_removal()
        result['correct_location'] = tail_removed
        result['meets_constraints'] = tail_removed

        if not tail_removed:
            result['details']['discrepancies'].append("Deletion was not from tail Step")

        result['severity'] = self._compute_severity(
            result['executed'],
            result['correct_location'],
            result['meets_constraints']
        )

        if result['severity'] == 'pass':
            result['reason'] = f"Successfully shortened chain（{orig_len} -> {refn_len})"
        else:
            result['reason'] = f"Chain length decreased, but deletion was not from tail"

        return result

    def _verify_tail_removal(self) -> bool:
        """Verify if deletion was from tail"""
        # Simplified check: Check if front steps are mostly preserved
        orig_steps = self.original.get('chain', [])
        refn_steps = self.refined.get('chain', [])

        if len(refn_steps) >= len(orig_steps):
            return False

        # Check if first len(refn_steps) steps are roughly the same
        for i in range(min(len(refn_steps), len(orig_steps) - 1)):
            orig_edge = (orig_steps[i].get('from'), orig_steps[i].get('to'))
            refn_edge = (refn_steps[i].get('from'), refn_steps[i].get('to'))
            if orig_edge != refn_edge:
                return False  # Head changed

        return True


class ClarifyRelationVerifier(ActionVerifier):
    """Verify clarify_relation action"""

    def verify(self) -> Dict[str, Any]:
        """Verify whether relation clarification was executed correctly"""
        result = self._build_base_result()

        # Step1: Identify TargetStep
        target_step_idx = self._find_target_step()
        result['details']['expected']['target_step'] = target_step_idx

        if not target_step_idx:
            result['reason'] = "Cannot determine TargetStep"
            result['severity'] = 'warning'
            return result

        # Step2: Get original and refined Step
        orig_step = self._get_step_at_position(self.original, target_step_idx)
        refn_step = self._get_step_by_nodes(
            self.refined,
            orig_step.get('from'),
            orig_step.get('to')
        )

        result['details']['expected']['step'] = orig_step
        result['details']['actual']['step'] = refn_step

        if not refn_step:
            result['executed'] = False
            result['reason'] = "Corresponding edge not found in refined chainedge"
            return result

        # Step3: Check if from/to nodes are unchanged
        nodes_unchanged = (
            orig_step.get('from') == refn_step.get('from') and
            orig_step.get('to') == refn_step.get('to')
        )
        result['correct_location'] = nodes_unchanged

        if not nodes_unchanged:
            result['executed'] = False
            result['reason'] = "Nodes changed, not clarify_relation"
            result['details']['discrepancies'].append("from/to nodes were modified")
            return result

        # Step4: Check if relation changed
        relation_changed = (
            orig_step.get('relation', '') != refn_step.get('relation', '')
        )
        result['executed'] = relation_changed
        result['meets_constraints'] = relation_changed

        result['details']['actual']['relation_change'] = {
            'original': orig_step.get('relation', ''),
            'refined': refn_step.get('relation', ''),
            'changed': relation_changed
        }

        result['severity'] = self._compute_severity(
            result['executed'],
            result['correct_location'],
            result['meets_constraints']
        )

        if result['severity'] == 'pass':
            result['reason'] = f"Successfully clarified relation description"
        elif not relation_changed:
            result['reason'] = "relation did not change"
        else:
            result['reason'] = "relation verification failed"

        return result

    def _find_target_step(self) -> Optional[int]:
        """Find TargetStep index"""
        from_step = self.location.get('from_step')
        to_step = self.location.get('to_step')

        if from_step:
            return from_step
        elif to_step:
            return to_step

        return None

    def _get_step_at_position(self, chain: Dict, position: int) -> Optional[Dict]:
        """Get step at specified position"""
        steps = chain.get('chain', [])
        if 0 < position <= len(steps):
            return steps[position - 1]
        return None

    def _get_step_by_nodes(self, chain: Dict, from_node: str, to_node: str) -> Optional[Dict]:
        """Find step by nodesStep"""
        steps = chain.get('chain', [])
        for step in steps:
            if step.get('from') == from_node and step.get('to') == to_node:
                return step
        return None


# Verifier mapping
VERIFIER_MAP = {
    'insert_step': InsertStepVerifier,
    'delete_step': DeleteStepVerifier,
    'replace_edge': ReplaceedgeVerifier,
    'replace_start_node': ReplaceStartNodeVerifier,
    'shorten_chain': ShortenChainVerifier,
    'clarify_relation': ClarifyRelationVerifier
}


def get_verifier(action_type: str) -> Optional[type]:
    """
    Get corresponding verifier class by action type

    Args:
        action_type: Action type

    Returns:
        Verifier class, returns None if not found
    """
    return VERIFIER_MAP.get(action_type)


if __name__ == "__main__":
    print("Action verifiers module loaded successfully")
    print(f"Available verifiers: {list(VERIFIER_MAP.keys())}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execution Checker - Main Coordinator

Integrate all check logic, verify whether RefineAgent correctly executed modifications as suggested by EvaluatorAgent
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from .chain_diff import ChainDiff
    from .action_verifiers import get_verifier
    from .checker_utils import format_location_description
except ImportError:
    from chain_diff import ChainDiff
    from action_verifiers import get_verifier
    from checker_utils import format_location_description


class ExecutionChecker:
    """
    Execution checker main class

    Responsible for:
    1. Per-chain modification execution check
    2. Matching evaluator issues and refiner edit_plans
    3. Detecting unauthorized changes
    4. Verifying failed_action consistency
    5. Generating unified check report
    """

    def __init__(self):
        """Initialize execution checker"""
        pass

    def check_execution(self,
                       original_chains: List[Dict],
                       refined_chains: List[Dict],
                       evaluator_result: Dict,
                       refine_result: Dict) -> Dict[str, Any]:
        """
        Execute complete execution check

        Args:
            original_chains: Original propagation chain list
            refined_chains: Refined propagation chain list
            evaluator_result: EvaluatorAgent evaluation result
            refine_result: RefineAgent refinement result

        Returns:
            Complete check result dict
        """
        print("\n" + "="*80)
        print("Execution check started")
        print("="*80)

        # Step 1: Build chain maps
        chain_map = self._build_chain_maps(original_chains, refined_chains)
        print(f"✓ Chain map building completed, total{len(chain_map)}chains")

        # Step 2: Per-chain check
        chain_checks = []
        for chain_id, chains in chain_map.items():
            print(f"\nChecking Chain {chain_id}...")
            chain_check = self._check_single_chain(
                chain_id,
                chains['original'],
                chains['refined'],
                evaluator_result,
                refine_result
            )
            chain_checks.append(chain_check)
            print(f"  Chain {chain_id}: {len(chain_check.get('action_checks', []))} action checks completed")

        # Step 3: Aggregate results
        overall_result = self._aggregate_results(chain_checks)
        overall_result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("\n" + "="*80)
        print("execution check completed")
        print(f"Overall status: {overall_result['overall_status']}")
        print(f"Executed correctly: {overall_result['executed_correctly']}/{overall_result['total_actions']}")
        print(f"Unauthorized changes: {overall_result['unauthorized_changes']}")
        print("="*80 + "\n")

        return overall_result

    def _build_chain_maps(self,
                         original_chains: List[Dict],
                         refined_chains: List[Dict]) -> Dict[int, Dict]:
        """
        Build chain_id to original/refined chain mapping

        Args:
            original_chains: Original chain list
            refined_chains: Refined chain list

        Returns:
            {chain_id: {'original': chain, 'refined': chain}}
        """
        chain_map = {}

        # Build original chain mapping
        for chain in original_chains:
            chain_id = chain.get('chain_id')
            if chain_id is not None:
                chain_map[chain_id] = {'original': chain, 'refined': None}

        # Add refined chains
        for chain in refined_chains:
            chain_id = chain.get('chain_id')
            if chain_id is not None:
                if chain_id in chain_map:
                    chain_map[chain_id]['refined'] = chain
                else:
                    # New chains (should not appear theoretically)
                    chain_map[chain_id] = {'original': None, 'refined': chain}

        return chain_map

    def _check_single_chain(self,
                           chain_id: int,
                           original: Optional[Dict],
                           refined: Optional[Dict],
                           evaluator_result: Dict,
                           refine_result: Dict) -> Dict[str, Any]:
        """
        Check modification status of a single chain

        Args:
            chain_id: Chain ID
            original: original chains
            refined: Refined chain
            evaluator_result: evaluation result
            refine_result: refinement result

        Returns:
            Single chain check result
        """
        result = {
            'chain_id': chain_id,
            'status': 'unknown',
            'action_checks': [],
            'unauthorized_changes': [],
            'failed_action_consistency': {},
            'diff_summary': {}
        }

        # Handle missing cases
        if not original or not refined:
            result['status'] = 'missing_chain'
            result['error'] = f"Original: {original is not None}, Refined: {refined is not None}"
            return result

        # Compute chain diff
        try:
            diff = ChainDiff(original, refined)
            diff_result = diff.compute_diff()
            result['diff_summary'] = {
                'length_change': diff_result.get('length_change'),
                'confidence_change': diff_result.get('confidence_change'),
                'nodes_added': len(diff_result.get('nodes_added', [])),
                'nodes_removed': len(diff_result.get('nodes_removed', [])),
                'edges_added': len(diff_result.get('edges_added', [])),
                'edges_removed': len(diff_result.get('edges_removed', [])),
                'edges_modified': len(diff_result.get('edges_modified', []))
            }
        except Exception as e:
            result['status'] = 'diff_error'
            result['error'] = str(e)
            return result

        # Get evaluation result for this chain
        chain_eval = self._find_chain_evaluation(chain_id, evaluator_result)
        if not chain_eval:
            result['status'] = 'no_evaluation'
            return result

        # Get edit_plan for this chain
        edit_plans = self._find_edit_plans(chain_id, refine_result)

        # Extract all issues (hard_violations + soft_issues)
        all_issues = []
        all_issues.extend(chain_eval.get('hard_violations', []))
        all_issues.extend(chain_eval.get('soft_issues', []))

        # Verify each issue edit_action
        for issue in all_issues:
            if 'edit_action' not in issue:
                continue

            action_check = self._verify_single_action(
                issue, edit_plans, diff, original, refined
            )
            result['action_checks'].append(action_check)

        # Detect unauthorized changes
        result['unauthorized_changes'] = self._detect_unauthorized_changes(
            all_issues, diff_result, edit_plans
        )

        # Verify failed_action consistency
        result['failed_action_consistency'] = self._check_failed_actions(
            chain_id, refine_result, diff_result
        )

        # Determine overall status
        if result['action_checks']:
            all_passed = all(
                check.get('severity') == 'pass'
                for check in result['action_checks']
            )
            has_errors = any(
                check.get('severity') == 'error'
                for check in result['action_checks']
            )

            if all_passed and not result['unauthorized_changes']:
                result['status'] = 'pass'
            elif has_errors or result['unauthorized_changes']:
                result['status'] = 'issues'
            else:
                result['status'] = 'warnings'
        else:
            result['status'] = 'no_actions'

        return result

    def _find_chain_evaluation(self, chain_id: int, evaluator_result: Dict) -> Optional[Dict]:
        """Find evaluation result for specified chain_id"""
        chain_evals = evaluator_result.get('chain_evaluations', [])
        for eval_item in chain_evals:
            if eval_item.get('chain_id') == chain_id:
                return eval_item
        return None

    def _find_edit_plans(self, chain_id: int, refine_result: Dict) -> List[Dict]:
        """Find edit_plan items for specified chain_id"""
        edit_plan = refine_result.get('edit_plan', [])
        return [item for item in edit_plan if item.get('chain_id') == chain_id]

    def _verify_single_action(self,
                              issue: Dict,
                              edit_plans: List[Dict],
                              diff: ChainDiff,
                              original: Dict,
                              refined: Dict) -> Dict[str, Any]:
        """
        Verify execution of a single action

        Args:
            issue: Evaluator issue item
            edit_plans: Edit plan items for this chain
            diff: Chain diff calculator
            original: original chains
            refined: Refined chain

        Returns:
            Action check result
        """
        edit_action = issue.get('edit_action', {})
        action_type = edit_action.get('action')

        # Find corresponding edit_plan item
        plan_item = self._match_plan_item(issue, edit_plans)

        # Get verifier class
        verifier_class = get_verifier(action_type)

        if not verifier_class:
            return {
                'action': action_type,
                'executed': False,
                'status': 'unknown_action',
                'severity': 'error',
                'reason': f'Unknown action type: {action_type}'
            }

        # Create verifier and execute verification
        try:
            verifier = verifier_class(issue, plan_item, diff, original, refined)
            check_result = verifier.verify()
            return check_result
        except Exception as e:
            return {
                'action': action_type,
                'executed': False,
                'status': 'verification_error',
                'severity': 'error',
                'reason': f'Verification process error: {str(e)}'
            }

    def _match_plan_item(self, issue: Dict, edit_plans: List[Dict]) -> Optional[Dict]:
        """
        Match issue and edit_plan items

        Args:
            issue: Evaluator issue
            edit_plans: edit_plan list

        Returns:
            Matched edit_plan item, returns None if no match
        """
        edit_action = issue.get('edit_action', {})
        action_type = edit_action.get('action')
        location = issue.get('location', {})

        # Simple match: find plan item with same action_type
        for plan in edit_plans:
            if plan.get('action') == action_type:
                # Can further check if location matches
                return plan

        return None

    def _detect_unauthorized_changes(self,
                                    all_issues: List[Dict],
                                    diff_result: Dict,
                                    edit_plans: List[Dict]) -> List[Dict]:
        """
        Detect unauthorized changes

        Args:
            all_issues: All issues list
            diff_result: Diff computation result
            edit_plans: edit_plan list

        Returns:
            Unauthorized change list
        """
        unauthorized = []

        # Extract authorized modification zones
        authorized_zones = self._extract_authorized_zones(all_issues)

        # Check added edges
        edges_added = diff_result.get('edges_added', [])
        for edge in edges_added:
            if not self._is_authorized_addition(edge, authorized_zones):
                unauthorized.append({
                    'type': 'unauthorized_insertion',
                    'location': f"edge {edge.get('from')} -> {edge.get('to')}",
                    'edge': edge,
                    'reason': 'This edge addition was not authorized in edit_action'
                })

        # Check removed edges
        edges_removed = diff_result.get('edges_removed', [])
        for edge in edges_removed:
            if not self._is_authorized_removal(edge, authorized_zones):
                unauthorized.append({
                    'type': 'unauthorized_deletion',
                    'location': f"edge {edge.get('from')} -> {edge.get('to')}",
                    'edge': edge,
                    'reason': 'This edge removal was not authorized in edit_action'
                })

        # Check modified relations
        edges_modified = diff_result.get('edges_modified', [])
        for edge_mod in edges_modified:
            if not self._is_authorized_modification(edge_mod, authorized_zones):
                unauthorized.append({
                    'type': 'unauthorized_modification',
                    'location': f"edge {edge_mod.get('from')} -> {edge_mod.get('to')}",
                    'edge': edge_mod,
                    'reason': 'This edge relation modification was not authorized in edit_action'
                })

        return unauthorized

    def _extract_authorized_zones(self, all_issues: List[Dict]) -> Dict:
        """Extract authorized modification zones"""
        zones = {
            'insertions': [],
            'deletions': [],
            'modifications': [],
            'replacements': [],
            'start_changes': [],
            'shortenings': []
        }

        for issue in all_issues:
            edit_action = issue.get('edit_action', {})
            action_type = edit_action.get('action')
            location = issue.get('location', {})

            if action_type == 'insert_step':
                zones['insertions'].append({
                    'location': location,
                    'candidates': edit_action.get('constraint', {}).get('candidate_metrics', [])
                })
            elif action_type == 'delete_step':
                zones['deletions'].append(location)
            elif action_type == 'clarify_relation':
                zones['modifications'].append(location)
            elif action_type == 'replace_edge':
                zones['replacements'].append(location)
            elif action_type == 'replace_start_node':
                zones['start_changes'].append(location)
            elif action_type == 'shorten_chain':
                zones['shortenings'].append(location)

        return zones

    def _is_authorized_addition(self, edge: Dict, zones: Dict) -> bool:
        """Check if edge addition is authorized"""
        # Simplified: if insert_step or replace_edge action exists, consider addition authorized
        if zones['insertions'] or zones['replacements']:
            return True
        return False

    def _is_authorized_removal(self, edge: Dict, zones: Dict) -> bool:
        """Check if edge removal is authorized"""
        # Simplified: if delete_step, replace_edge or shorten_chain action exists, consider removal authorized
        if zones['deletions'] or zones['replacements'] or zones['shortenings']:
            return True
        return False

    def _is_authorized_modification(self, edge_mod: Dict, zones: Dict) -> bool:
        """Check if relation modification is authorized"""
        # Check if clarify_relation action exists
        return bool(zones['modifications'])

    def _check_failed_actions(self,
                             chain_id: int,
                             refine_result: Dict,
                             diff_result: Dict) -> Dict[str, Any]:
        """
        Check failed_action consistency

        Args:
            chain_id: Chain ID
            refine_result: RefineAgent result
            diff_result: Diff result

        Returns:
            Consistency check result
        """
        failed_actions = refine_result.get('failed_actions', [])
        chain_failed = [fa for fa in failed_actions if fa.get('chain_id') == chain_id]

        inconsistencies = []

        for failed_action in chain_failed:
            action_type = failed_action.get('action')

            # Check for evidence of corresponding modifications
            has_evidence = False

            if action_type == 'insert_step':
                # Check if nodes were added
                if diff_result.get('nodes_added'):
                    has_evidence = True
            elif action_type == 'delete_step':
                # Check if nodes were removed
                if diff_result.get('nodes_removed'):
                    has_evidence = True
            elif action_type == 'shorten_chain':
                # Check if chain length changed
                length_change = diff_result.get('length_change', {})
                if length_change.get('new', 0) < length_change.get('old', 0):
                    has_evidence = True

            if has_evidence:
                inconsistencies.append({
                    'claimed_failed': failed_action,
                    'evidence_of_execution': {
                        'nodes_added': diff_result.get('nodes_added', []),
                        'nodes_removed': diff_result.get('nodes_removed', []),
                        'length_change': diff_result.get('length_change', {})
                    },
                    'issue': f'RefineAgent claimed{action_type}failed, but related modifications detected'
                })

        return {
            'status': 'consistent' if not inconsistencies else 'inconsistent',
            'claimed_failed': len(chain_failed),
            'inconsistencies': inconsistencies
        }

    def _aggregate_results(self, chain_checks: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate check results of all chains

        Args:
            chain_checks: Check result list of all chains

        Returns:
            Aggregated result
        """
        total_actions = 0
        executed_correctly = 0
        executed_incorrectly = 0
        not_executed = 0
        unauthorized_changes = 0

        for chain_check in chain_checks:
            for action_check in chain_check.get('action_checks', []):
                total_actions += 1
                severity = action_check.get('severity', 'error')

                if severity == 'pass':
                    executed_correctly += 1
                elif action_check.get('executed', False):
                    executed_incorrectly += 1
                else:
                    not_executed += 1

            unauthorized_changes += len(chain_check.get('unauthorized_changes', []))

        # Determine overall status
        if total_actions == 0:
            overall_status = 'no_actions'
        elif executed_correctly == total_actions and unauthorized_changes == 0:
            overall_status = 'pass'
        elif not_executed > 0 or unauthorized_changes > 0:
            overall_status = 'fail'
        else:
            overall_status = 'partial'

        # Generate summary
        summary = self._generate_summary(
            total_actions, executed_correctly, executed_incorrectly,
            not_executed, unauthorized_changes
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(chain_checks)

        return {
            'overall_status': overall_status,
            'total_actions': total_actions,
            'executed_correctly': executed_correctly,
            'executed_incorrectly': executed_incorrectly,
            'not_executed': not_executed,
            'unauthorized_changes': unauthorized_changes,
            'chain_checks': chain_checks,
            'summary': summary,
            'recommendations': recommendations
        }

    def _generate_summary(self, total: int, correct: int, incorrect: int,
                         not_exec: int, unauth: int) -> str:
        """Generate summary text"""
        parts = [f"Checked{total}actions"]

        if correct > 0:
            parts.append(f"{correct}correctly executed")
        if incorrect > 0:
            parts.append(f"{incorrect}executed with issues")
        if not_exec > 0:
            parts.append(f"{not_exec}not executed")
        if unauth > 0:
            parts.append(f"found {unauth} unauthorized changes")

        return "，".join(parts)

    def _generate_recommendations(self, chain_checks: List[Dict]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        for chain_check in chain_checks:
            chain_id = chain_check.get('chain_id')

            # Check unexecuted actions
            for action_check in chain_check.get('action_checks', []):
                if not action_check.get('executed', False):
                    action = action_check.get('action', 'unknown')
                    recommendations.append(
                        f"Chain {chain_id}: {action}action not executed, recommend checking RefineAgent log"
                    )

            # Check unauthorized changes
            if chain_check.get('unauthorized_changes'):
                count = len(chain_check['unauthorized_changes'])
                recommendations.append(
                    f"Chain {chain_id}: found {count} unauthorized changes, recommend reviewing if corresponding edit_action should be added"
                )

            # Check failed_action inconsistency
            consistency = chain_check.get('failed_action_consistency', {})
            if consistency.get('status') == 'inconsistent':
                recommendations.append(
                    f"Chain {chain_id}: failed_action declaration inconsistent with actual modifications, needs investigation"
                )

        return recommendations


if __name__ == "__main__":
    print("ExecutionChecker module loaded successfully")

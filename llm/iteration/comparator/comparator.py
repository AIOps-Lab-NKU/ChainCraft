#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparator - Main Coordinator

Integrate fixed_issue_detector, new_issue_detector, rule_precheck
Output final comparison result
"""

from typing import List, Dict, Any
import sys
import os

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llm.iteration.comparator.schemas import (
    FixedIssue, RemainingIssue, NewIssue,
    ChainComparison, ComparisonSummary, ComparatorResult,
    ComparisonResult, Confidence
)
from llm.iteration.comparator.fixed_issue_detector import FixedIssueDetector
from llm.iteration.comparator.new_issue_detector import NewIssueDetector
from llm.iteration.comparator.rule_precheck import RulePrecheck


class Comparator:
    """
    Main comparator

    Integrate all detection results, output final comparison conclusion
    """

    def __init__(self):
        """Initialize comparator"""
        self.fixed_detector = FixedIssueDetector()
        self.new_detector = NewIssueDetector()
        self.rule_precheck = RulePrecheck()

    def compare(self,
                original_chains: List[Dict],
                refined_chains: List[Dict],
                evaluator_result: Dict,
                refine_result: Dict,
                execution_check_result: Dict,
                causal_analysis: str,
                metric_analysis: List[Dict]) -> ComparatorResult:
        """
        Main comparison method

        Args:
            original_chains: Original chain list
            refined_chains: Refined chain list
            evaluator_result: Evaluation result
            refine_result: Refinement result
            execution_check_result: Execution check result
            causal_analysis: Causal analysis (CSV format)
            metric_analysis: Metric analysis (JSON format)

        Returns:
            ComparatorResult object
        """
        print("\n=== Comparator: comparing original vs refined chains ===\n")

        # Step 1: Identify fixed issues
        print("[1/4] Identifying fixed issues...")
        fixed_issues = self.fixed_detector.detect(
            original_chains=original_chains,
            refined_chains=refined_chains,
            evaluator_result=evaluator_result,
            execution_check_result=execution_check_result
        )
        print(f"  Detected {len(fixed_issues)} issue fix statuses")

        # Step 2: Identify newly introduced issues
        print("[2/4] Identifying newly introduced issues...")
        new_issues = self.new_detector.detect(
            original_chains=original_chains,
            refined_chains=refined_chains,
            evaluator_result=evaluator_result,
            causal_analysis=causal_analysis,
            metric_analysis=metric_analysis
        )
        print(f"  Detected {len(new_issues)} new issues")

        # Step 3: Rule pre-check
        print("[3/4] Rule pre-check...")
        precheck_result = self.rule_precheck.precheck(
            fixed_issues=fixed_issues,
            new_issues=new_issues,
            execution_check_result=execution_check_result
        )
        print(f"  Pre-check result: {precheck_result['preliminary_result']} (Confidence: {precheck_result['confidence']})")

        # Identify remaining issues
        remaining_issues = self._identify_remaining_issues(
            fixed_issues, evaluator_result
        )

        # Step 4: Comprehensive judgement
        print("[4/4] Comprehensive judgement...")
        comparison_summary, chain_comparisons, reason = self._make_final_decision(
            fixed_issues=fixed_issues,
            new_issues=new_issues,
            remaining_issues=remaining_issues,
            precheck_result=precheck_result,
            original_chains=original_chains,
            refined_chains=refined_chains
        )
        print(f"  Final judgement: {comparison_summary.comparison_result} (Confidence: {comparison_summary.confidence})")
        print(f"  Whether to keep new version: {comparison_summary.keep_new_version}")

        # Build result
        result = ComparatorResult(
            comparison_summary=comparison_summary,
            fixed_issues=fixed_issues,
            remaining_issues=remaining_issues,
            new_issues=new_issues,
            chain_comparisons=chain_comparisons,
            reason=reason
        )

        print("\n=== Comparator: comparison completed ===\n")

        return result

    def _identify_remaining_issues(self,
                                   fixed_issues: List[FixedIssue],
                                   evaluator_result: Dict) -> List[RemainingIssue]:
        """
        Identify remaining issues

        Args:
            fixed_issues: Fixed issue list
            evaluator_result: Evaluation result

        Returns:
            Remaining issue list
        """
        remaining = []

        # Create mapping of fixed issues
        fixed_map = {
            (issue.chain_id, issue.issue_type): issue
            for issue in fixed_issues
            if issue.status == 'fixed'
        }

        # Iterate all original issues
        chain_evaluations = evaluator_result.get('chain_evaluations', [])
        for chain_eval in chain_evaluations:
            chain_id = chain_eval.get('chain_id')

            # Check hard violations
            for issue in chain_eval.get('hard_violations', []):
                issue_type = issue.get('issue_type')
                if (chain_id, issue_type) not in fixed_map:
                    # Not fixed or partially fixed
                    remaining.append(RemainingIssue(
                        chain_id=chain_id,
                        issue_type=issue_type,
                        status='still_exists',
                        description=issue.get('description', '')
                    ))

            # Check soft issues
            for issue in chain_eval.get('soft_issues', []):
                issue_type = issue.get('issue_type')
                if (chain_id, issue_type) not in fixed_map:
                    remaining.append(RemainingIssue(
                        chain_id=chain_id,
                        issue_type=issue_type,
                        status='still_exists',
                        description=issue.get('description', '')
                    ))

        return remaining

    def _make_final_decision(self,
                            fixed_issues: List[FixedIssue],
                            new_issues: List[NewIssue],
                            remaining_issues: List[RemainingIssue],
                            precheck_result: Dict,
                            original_chains: List[Dict],
                            refined_chains: List[Dict]) -> tuple:
        """
        Final judgement

        Args:
            fixed_issues: Fixed issue list
            new_issues: New issue list
            remaining_issues: Remaining issue list
            precheck_result: Pre-check result
            original_chains: Original chains
            refined_chains: Refined chains

        Returns:
            (ComparisonSummary, List[ChainComparison], reason)
        """
        # Get pre-check result
        preliminary = precheck_result['preliminary_result']
        precheck_confidence = precheck_result['confidence']
        precheck_reasons = precheck_result['reasons']
        warning_flags = precheck_result['warning_flags']

        # Statistics
        fixed_stats = precheck_result['stats']['fixed']
        new_stats = precheck_result['stats']['new']

        # Decision logic
        final_result = preliminary
        final_confidence = precheck_confidence

        # Adjustment logic: fine-tune based on specific conditions
        # If precheck judged as worse, but only few new issues
        if preliminary == ComparisonResult.WORSE:
            if new_stats['high'] == 0 and new_stats['medium'] <= 1:
                if fixed_stats['high'] > 0:
                    # Fixed severe issues, new issues not severe, possibly tie
                    final_result = ComparisonResult.TIE
                    final_confidence = Confidence.MEDIUM

        # If precheck judged as tie, but fixed many severe issues
        elif preliminary == ComparisonResult.TIE:
            if fixed_stats['high'] >= 2 and new_stats['high'] == 0:
                # Fixed multiple severe issues, no new severe issues
                final_result = ComparisonResult.BETTER
                final_confidence = Confidence.MEDIUM

        # If precheck judged as better, but with warning flags
        elif preliminary == ComparisonResult.BETTER:
            if len(warning_flags) >= 2:
                # Multiple warnings, reduce confidence
                final_confidence = Confidence.LOW

        # Determine whether to keep new version
        keep_new_version = (final_result == ComparisonResult.BETTER) or \
                          (final_result == ComparisonResult.TIE and fixed_stats['high'] > 0)

        # Generate per-chain comparison results
        chain_comparisons = self._compare_individual_chains(
            original_chains, refined_chains, fixed_issues, new_issues
        )

        # Generate detailed reason
        reason = self._generate_reason(
            final_result, fixed_issues, new_issues, remaining_issues,
            precheck_reasons, warning_flags, fixed_stats, new_stats
        )

        # Build summary
        summary = ComparisonSummary(
            comparison_result=final_result,
            keep_new_version=keep_new_version,
            confidence=final_confidence
        )

        return summary, chain_comparisons, reason

    def _compare_individual_chains(self,
                                   original_chains: List[Dict],
                                   refined_chains: List[Dict],
                                   fixed_issues: List[FixedIssue],
                                   new_issues: List[NewIssue]) -> List[ChainComparison]:
        """
        Compare each chain

        Args:
            original_chains: Original chains
            refined_chains: Refined chains
            fixed_issues: Fixed issue list
            new_issues: New issue list

        Returns:
            Per-chain comparison results
        """
        comparisons = []

        # Build issue mapping
        fixed_by_chain = {}
        for issue in fixed_issues:
            fixed_by_chain.setdefault(issue.chain_id, []).append(issue)

        new_by_chain = {}
        for issue in new_issues:
            new_by_chain.setdefault(issue.chain_id, []).append(issue)

        # Compare each chain
        for refined_chain in refined_chains:
            chain_id = refined_chain.get('chain_id')

            fixed_in_chain = fixed_by_chain.get(chain_id, [])
            new_in_chain = new_by_chain.get(chain_id, [])

            # Count severity
            fixed_high = sum(1 for i in fixed_in_chain if i.original_severity == 'high' and i.status == 'fixed')
            new_high = sum(1 for i in new_in_chain if i.severity == 'high')

            # Determine comparison result for this chain
            if new_high > 0:
                result = ComparisonResult.WORSE
                reason = f"Introduced{new_high}new high-severity issues"
            elif fixed_high > 0:
                result = ComparisonResult.BETTER
                reason = f"Fixed{fixed_high}high-severity issues"
            elif len(fixed_in_chain) > len(new_in_chain):
                result = ComparisonResult.BETTER
                reason = f"Fixed{len(fixed_in_chain)}issues, only introduced{len(new_in_chain)}new issues"
            elif len(new_in_chain) > len(fixed_in_chain):
                result = ComparisonResult.WORSE
                reason = f"Fixed{len(fixed_in_chain)}issues, but introduced{len(new_in_chain)}new issues"
            else:
                result = ComparisonResult.TIE
                reason = f"Fixed and introduced issue counts are comparable"

            comparisons.append(ChainComparison(
                chain_id=chain_id,
                result=result,
                reason=reason
            ))

        return comparisons

    def _generate_reason(self,
                        comparison_result: str,
                        fixed_issues: List[FixedIssue],
                        new_issues: List[NewIssue],
                        remaining_issues: List[RemainingIssue],
                        precheck_reasons: List[str],
                        warning_flags: List[str],
                        fixed_stats: Dict,
                        new_stats: Dict) -> str:
        """
        Generate explanatory reason

        Args:
            comparison_result: Comparison result
            fixed_issues: Fixed issue list
            new_issues: New issue list
            remaining_issues: Remaining issue list
            precheck_reasons: Pre-check reason
            warning_flags: Warning flags
            fixed_stats: Fixed issue statistics
            new_stats: New issue statistics

        Returns:
            Detailed reason string
        """
        reason_parts = []

        # Result judgement description
        result_desc = {
            ComparisonResult.BETTER: "New chains are better than old chains",
            ComparisonResult.WORSE: "New chains are worse than old chains",
            ComparisonResult.TIE: "New and old chains are comparable in quality"
        }
        reason_parts.append(f"[Comparison Conclusion] {result_desc.get(comparison_result, comparison_result)}")

        # Fixed issues description
        if fixed_stats['fixed'] > 0:
            fixed_detail = []
            if fixed_stats['high'] > 0:
                fixed_detail.append(f"{fixed_stats['high']} high-severity")
            if fixed_stats['medium'] > 0:
                fixed_detail.append(f"{fixed_stats['medium']} medium-severity")
            if fixed_stats['low'] > 0:
                fixed_detail.append(f"{fixed_stats['low']} low-severity")

            reason_parts.append(
                f"[Fixed Issues] Successfully fixed{fixed_stats['fixed']} issues"
                f"（{', '.join(fixed_detail)}）"
            )

            # List some fixed issues
            fixed_examples = [
                f"{issue.issue_type}"
                for issue in fixed_issues[:3]
                if issue.status == 'fixed'
            ]
            if fixed_examples:
                reason_parts.append(f"  Including: {', '.join(fixed_examples)}")

        # New issues description
        if new_stats['high'] > 0 or new_stats['medium'] > 0:
            new_detail = []
            if new_stats['high'] > 0:
                new_detail.append(f"{new_stats['high']} high-severity")
            if new_stats['medium'] > 0:
                new_detail.append(f"{new_stats['medium']} medium-severity")

            total_new = new_stats['high'] + new_stats['medium']
            reason_parts.append(
                f"[New Issues] Introduced{total_new} issues"
                f"（{', '.join(new_detail)}）"
            )

            # List some new issues
            new_examples = [
                f"{issue.issue_type}"
                for issue in new_issues[:3]
                if issue.severity in ['high', 'medium']
            ]
            if new_examples:
                reason_parts.append(f"  Including: {', '.join(new_examples)}")

        # Remaining issues description
        if len(remaining_issues) > 0:
            reason_parts.append(f"[Remaining Issues] Still have {len(remaining_issues)} issues not fixed")

        # Pre-check reason
        if precheck_reasons:
            reason_parts.append(f"[Rule Judgement] {'; '.join(precheck_reasons[:2])}")

        # Warning flags
        if warning_flags:
            reason_parts.append(f"[Warning] {'; '.join(warning_flags[:2])}")

        # Overall description
        if comparison_result == ComparisonResult.BETTER:
            reason_parts.append(
                "[Overall Assessment] The quantity and severity of fixed issues exceeds newly introduced issues, overall quality improved."
            )
        elif comparison_result == ComparisonResult.WORSE:
            reason_parts.append(
                "[Overall Assessment] Newly introduced issues have high severity, or modifications were not effectively executed, overall quality degraded."
            )
        else:
            reason_parts.append(
                "[Overall Assessment] Fixed and introduced issues offset each other, or modification effects are not obvious, no significant quality change."
            )

        return "\n".join(reason_parts)


if __name__ == "__main__":
    print("Comparator module loaded successfully")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule Precheck

Rule-based pre-judgment based on execution_check_result and issue statistics
"""

from typing import List, Dict, Any
import sys
import os

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llm.iteration.comparator.schemas import (
    FixedIssue, NewIssue, ComparisonResult, Confidence
)


class RulePrecheck:
    """
    Rule pre-checker

    Preliminary judgement based on execution_check_result and issue statistics
    Output better/worse/tie tendency and confidence
    """

    def __init__(self):
        """Initialize pre-checker"""
        pass

    def precheck(self,
                fixed_issues: List[FixedIssue],
                new_issues: List[NewIssue],
                execution_check_result: Dict) -> Dict[str, Any]:
        """
        Rule pre-check

        Args:
            fixed_issues: Fixed issue list
            new_issues: Newly introduced issue list
            execution_check_result: Execution check result

        Returns:
            Pre-check result dict
            {
                'preliminary_result': 'better|worse|tie',
                'confidence': 'high|medium|low',
                'reasons': [reason list],
                'warning_flags': [warning flags]
            }
        """
        reasons = []
        warning_flags = []

        # Count issues and severity
        fixed_stats = self._count_issues_by_severity(fixed_issues)
        new_stats = self._count_issues_by_severity_new(new_issues)

        # Get execution_check statistics
        exec_status = execution_check_result.get('overall_status', 'unknown')
        unauthorized_changes = execution_check_result.get('unauthorized_changes', 0)
        executed_correctly = execution_check_result.get('executed_correctly', 0)
        total_actions = execution_check_result.get('total_actions', 0)
        not_executed = execution_check_result.get('not_executed', 0)

        # Apply rule set (by priority, high to low)
        preliminary_result, confidence, rule_reasons = self._apply_rules(
            fixed_stats=fixed_stats,
            new_stats=new_stats,
            exec_status=exec_status,
            unauthorized_changes=unauthorized_changes,
            executed_correctly=executed_correctly,
            total_actions=total_actions,
            not_executed=not_executed
        )

        reasons.extend(rule_reasons)

        # Generate warning flags
        if unauthorized_changes > 0:
            warning_flags.append(f'Found {unauthorized_changes} unauthorized changes')

        if not_executed > 0:
            warning_flags.append(f'Have {not_executed} actions not executed')

        if exec_status == 'fail':
            warning_flags.append('Execution check failed')

        if new_stats['high'] > 0:
            warning_flags.append(f'Introduced {new_stats["high"]} high-severity issues')

        return {
            'preliminary_result': preliminary_result,
            'confidence': confidence,
            'reasons': reasons,
            'warning_flags': warning_flags,
            'stats': {
                'fixed': fixed_stats,
                'new': new_stats,
                'execution': {
                    'status': exec_status,
                    'unauthorized_changes': unauthorized_changes,
                    'executed_correctly': executed_correctly,
                    'total_actions': total_actions,
                    'not_executed': not_executed
                }
            }
        }

    def _apply_rules(self,
                    fixed_stats: Dict,
                    new_stats: Dict,
                    exec_status: str,
                    unauthorized_changes: int,
                    executed_correctly: int,
                    total_actions: int,
                    not_executed: int) -> tuple:
        """
        Apply rule set

        Rule priority (high to low):
        1. New high severity issues introduced → worse
        2. Many unauthorized changes → worse
        3. Fixed high severity issues with no new high severity issues → better
        4. Execution check failed → worse
        5. Many actions not executed → tie
        6. Only fixed medium/low issues with no new issues → better/tie
        7. Default → tie

        Returns:
            (preliminary_result, confidence, reasons)
        """
        reasons = []

        # Rule 1: New high severity issues introduced
        if new_stats['high'] > 0:
            reasons.append(f'Introduced {new_stats["high"]} new high-severity issues')
            return (ComparisonResult.WORSE, Confidence.HIGH, reasons)

        # Rule 2: Many unauthorized changes
        if unauthorized_changes >= 2:
            reasons.append(f'Found {unauthorized_changes} unauthorized changes，may have damaged chain structure')
            return (ComparisonResult.WORSE, Confidence.HIGH, reasons)

        # Rule 3: Fixed high severity issues with no new high severity issues
        if fixed_stats['high'] > 0 and new_stats['high'] == 0:
            reasons.append(f'Fixed {fixed_stats["high"]} high-severity issues')
            if new_stats['medium'] == 0 and new_stats['low'] == 0:
                reasons.append('and no new issues introduced')
                return (ComparisonResult.BETTER, Confidence.HIGH, reasons)
            elif new_stats['medium'] <= 1:
                reasons.append(f'Only introduced {new_stats["medium"]} medium-severity issues')
                return (ComparisonResult.BETTER, Confidence.MEDIUM, reasons)
            else:
                reasons.append(f'But introduced {new_stats["medium"]} medium-severity issues')
                return (ComparisonResult.TIE, Confidence.MEDIUM, reasons)

        # Rule 4: Execution check failed
        if exec_status == 'fail':
            reasons.append('Execution check failed，modifications may not have executed as expected')
            if unauthorized_changes > 0:
                return (ComparisonResult.WORSE, Confidence.MEDIUM, reasons)
            else:
                return (ComparisonResult.TIE, Confidence.MEDIUM, reasons)

        # Rule 5: Many actions not executed
        if total_actions > 0 and not_executed >= total_actions * 0.5:
            reasons.append(f'Have {not_executed}/{total_actions} actions not executed')
            return (ComparisonResult.TIE, Confidence.MEDIUM, reasons)

        # Rule 6: Only fixed medium/low issues with no new issues
        if fixed_stats['medium'] > 0 or fixed_stats['low'] > 0:
            total_fixed = fixed_stats['medium'] + fixed_stats['low']
            reasons.append(f'Fixed {total_fixed} medium/low severity issues')

            if new_stats['medium'] == 0 and new_stats['low'] == 0:
                reasons.append('and no new issues introduced')
                return (ComparisonResult.BETTER, Confidence.MEDIUM, reasons)
            elif new_stats['medium'] + new_stats['low'] < total_fixed:
                reasons.append(f'Introduced fewer new issues than fixed issues')
                return (ComparisonResult.BETTER, Confidence.LOW, reasons)
            else:
                reasons.append(f'But introduced same number of new issues')
                return (ComparisonResult.TIE, Confidence.LOW, reasons)

        # Rule 7: Unauthorized changes but few
        if unauthorized_changes == 1:
            reasons.append('Found 1 unauthorized changes')
            if fixed_stats['high'] > 0:
                reasons.append('but fixed high-severity issues')
                return (ComparisonResult.BETTER, Confidence.LOW, reasons)
            else:
                return (ComparisonResult.TIE, Confidence.LOW, reasons)

        # Rule 8: Execution successful but no significant improvement
        if exec_status == 'pass' and total_actions > 0:
            if fixed_stats['high'] == 0 and fixed_stats['medium'] == 0:
                reasons.append('Execution successful but no significant issues fixed')
                return (ComparisonResult.TIE, Confidence.LOW, reasons)

        # Default rule
        if total_actions == 0:
            reasons.append('No modification actions executed')
        else:
            reasons.append('Modification effects not significant')
        return (ComparisonResult.TIE, Confidence.LOW, reasons)

    def _count_issues_by_severity(self, issues: List[FixedIssue]) -> Dict[str, int]:
        """
        Count severity distribution of fixed issues

        Args:
            issues: Fixed issue list

        Returns:
            {'high': count, 'medium': count, 'low': count, 'fixed': count, 'not_fixed': count}
        """
        stats = {
            'high': 0,
            'medium': 0,
            'low': 0,
            'fixed': 0,
            'partially_fixed': 0,
            'not_fixed': 0
        }

        for issue in issues:
            # Only count actually fixed issues
            if issue.status == 'fixed':
                stats['fixed'] += 1
                severity = issue.original_severity.lower()
                if severity in stats:
                    stats[severity] += 1
            elif issue.status == 'partially_fixed':
                stats['partially_fixed'] += 1
            else:
                stats['not_fixed'] += 1

        return stats

    def _count_issues_by_severity_new(self, issues: List[NewIssue]) -> Dict[str, int]:
        """
        Count severity distribution of new issues

        Args:
            issues: New issue list

        Returns:
            {'high': count, 'medium': count, 'low': count}
        """
        stats = {
            'high': 0,
            'medium': 0,
            'low': 0
        }

        for issue in issues:
            severity = issue.severity.lower()
            if severity in stats:
                stats[severity] += 1

        return stats


if __name__ == "__main__":
    print("RulePrecheck module loaded successfully")

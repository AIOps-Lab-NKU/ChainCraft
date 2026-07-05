#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IterationController - Iteration Flow Controller

Responsible for coordinating EvaluatorAgent and RefineAgent, controlling the entire iterative refinement process
"""

import sys
import os

# Dynamically get project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from .config import IterationConfig
from .iteration_logger import IterationLogger
from agent.EvaluatorAgent import EvaluatorAgent
from agent.RefineAgent import RefineAgent
import logging


class IterationController:
    """Iteration controller, manages evaluation-refinement iteration process"""

    def __init__(self, case_id, app=None, app_group=None, config=None, base_path=None):
        """
        Initialize iteration controller

        Args:
            case_id (str): Case ID
            app (str, optional): Application name
            app_group (str, optional): Application group name
            config (IterationConfig, optional): Config class, uses IterationConfig by default
            base_path (str, optional): Data base path, defaults to user home directory
        """
        self.case_id = case_id
        self.app = app
        self.app_group = app_group
        self.config = config or IterationConfig

        # Initialize logger
        self.logger = IterationLogger(case_id, app, app_group, base_path)

        # Initialize agents
        self.evaluator = EvaluatorAgent()
        self.refiner = RefineAgent()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger(self.__class__.__name__)

        # Record configuration
        self.logger.set_config(self.config.get_config_dict())

        # State tracking variables
        self.consecutive_execution_failures = 0
        self.consecutive_tie_count = 0
        self.consecutive_worse_count = 0

    def run_iteration(self, initial_chains, causal_analysis, metric_analysis, save_dir=None):
        """
        Run iterative refinement process

        Args:
            initial_chains (list): Initial propagation chains
            causal_analysis (str): PCMCI causal analysis result (CSV file path or content)
            metric_analysis (list or str): Metric analysis results
            save_dir (str, optional): Intermediate results save directory

        Returns:
            dict: Iteration result, containing final_chains, converged, iteration_log, etc.
        """
        self.log.info(f"Starting iterative refinement process - Case: {self.case_id}")
        print(f"\n{'='*80}")
        print(f"Iterative Refinement Process - Case: {self.case_id}")
        print(f"{'='*80}")
        print(f"Config: max{self.config.MAX_ITERATIONS}iteration rounds")
        print(f"{'='*80}\n")

        current_chains = initial_chains
        iteration_num = 0

        # Track iteration history (simplified, passed to agents)
        iteration_history = {
            'evaluations': [],
            'changes': []
        }

        try:
            # Iteration loop
            while iteration_num < self.config.MAX_ITERATIONS:
                iteration_num += 1
                self.log.info(f"Starting round {iteration_num} iteration rounds")
                print(f"\n{'='*70}")
                print(f"Round {iteration_num} iteration")
                print(f"{'='*70}")

                # Build iteration_context (starting from round 2)
                iteration_context = None
                if iteration_num > 1:
                    iteration_context = self._build_iteration_context(
                        iteration_num,
                        iteration_history
                    )

                # ===== Step 1: Evaluate propagation chains =====
                print(f"\n[Step 1/3] Evaluating propagation chain quality...")
                evaluation_save_path = None
                if save_dir:
                    evaluation_save_path = f"{save_dir}/iteration_{iteration_num}_evaluation.json"

                evaluation = self.evaluator.evaluate_chains(
                    propagation_chains=current_chains,
                    causal_analysis=causal_analysis,
                    metric_analysis=metric_analysis,
                    iteration_context=iteration_context,
                    save_path=evaluation_save_path
                )

                # New format: get evaluation info from summary
                summary = evaluation.get('summary', {})
                overall_judgement = summary.get('overall_judgement', 'needs_refinement')
                hard_violation_count = summary.get('hard_violation_count', 0)
                soft_issue_count = summary.get('soft_issue_count', 0)

                self.log.info(f"Round {iteration_num} evaluation completed")
                self.log.info(f"  Overall judgement: {overall_judgement}")
                self.log.info(f"  hard_violations: {hard_violation_count}, soft_issues: {soft_issue_count}")

                # Record this iteration (without refinement results yet)
                self.logger.add_iteration(
                    iteration_num=iteration_num,
                    chains=current_chains,
                    evaluation=evaluation
                )

                # ===== Step 2: Determine if current chain quality is good enough =====
                print(f"\n[Step 2/4] Assessing current chain quality...")
                print(f"  Current round: {iteration_num}/{self.config.MAX_ITERATIONS}")
                print(f"  Overall judgement: {overall_judgement}")
                print(f"  Issue count: {hard_violation_count} hard + {soft_issue_count} soft")

                # [NEW] Check if current chains are good enough
                should_stop, quality_reason = self._should_stop_by_quality(evaluation)
                if should_stop:
                    self.log.info(f"[Decision] {quality_reason}")
                    print(f"  ✓ [Decision] {quality_reason} -> stop iteration")

                    # Record decision info
                    if self.logger.log_data['iterations']:
                        last_iteration = self.logger.log_data['iterations'][-1]
                        last_iteration['controller_decision'] = {
                            'accept_refined': False,
                            'continue_iteration': False,
                            'stop_reason': 'current_chain_good_enough',
                            'decision_basis': quality_reason
                        }

                    self.logger.set_final_result(
                        final_chains=current_chains,
                        converged=True,
                        stop_reason="Current chains are good enough"
                    )

                    return self._build_result(current_chains, True, "Current chains are good enough")

                # Check if max iteration rounds reached
                if iteration_num >= self.config.MAX_ITERATIONS:
                    self.log.info(f"Reached max iteration rounds {self.config.MAX_ITERATIONS}")
                    print(f"  ✗ Reached max iteration rounds")

                    self.logger.set_final_result(
                        final_chains=current_chains,
                        converged=False,
                        stop_reason=f"Reached max iteration rounds({self.config.MAX_ITERATIONS})"
                    )

                    return self._build_result(current_chains, False, "Reached max rounds")

                # ===== Step 3: Refine propagation chains =====
                print(f"\n[Step 3/4] Refining propagation chains...")
                refine_save_path = None
                if save_dir:
                    refine_save_path = f"{save_dir}/iteration_{iteration_num}_refine.json"

                refine_result = self.refiner.refine_chains(
                    original_chains=current_chains,
                    evaluation=evaluation,
                    causal_analysis=causal_analysis,
                    metric_analysis=metric_analysis,
                    iteration_context=iteration_context,
                    save_path=refine_save_path
                )

                # Update this iteration record, add refinement results (new format)
                refined_chains = refine_result.get('refined_chains', [])
                edit_plan = refine_result.get('edit_plan', [])  # New format: edit_plan replaces changes
                rollback = refine_result.get('rollback', False)
                rollback_to = refine_result.get('rollback_to_iteration', None)

                # ========== Execution Check + Decision ==========
                check_result = None
                execution_unreliable = False

                if self.config.ENABLE_EXECUTION_CHECKING:
                    self.log.info(f"Execution check started...")
                    print(f"\n[Step 3.5/4] Verifying modification execution...")

                    try:
                        from .execution_checker.execution_checker import ExecutionChecker
                        checker = ExecutionChecker()

                        check_result = checker.check_execution(
                            original_chains=current_chains,
                            refined_chains=refined_chains,
                            evaluator_result=evaluation,
                            refine_result=refine_result
                        )

                        # Record check results
                        self.logger.log_execution_check(iteration_num, check_result)

                        # Print summary
                        print(f"✓ Execution check completed")
                        print(f"  - Overall status: {check_result['overall_status']}")
                        print(f"  - Executed correctly: {check_result['executed_correctly']}/{check_result['total_actions']}")
                        if check_result['unauthorized_changes'] > 0:
                            print(f"  - Unauthorized changes: {check_result['unauthorized_changes']}")

                        # [NEW] Determine if execution is reliable
                        execution_unreliable = self._is_execution_unreliable(check_result)

                        if execution_unreliable:
                            action, reason = self._decide_after_execution_check(check_result, iteration_num)
                            self.log.warning(f"[Decision] Execution unreliable: {reason}")
                            print(f"  ⚠ [Decision] {reason}")

                            # Record decision
                            if self.logger.log_data['iterations']:
                                last_iteration = self.logger.log_data['iterations'][-1]
                                last_iteration['controller_decision'] = {
                                    'accept_refined': False,
                                    'continue_iteration': (action == 'continue'),
                                    'stop_reason': 'execution_failed' if action == 'stop' else None,
                                    'decision_basis': reason
                                }

                            if action == 'stop':
                                # Too many consecutive failures, stop iteration
                                self.logger.set_final_result(
                                    final_chains=current_chains,
                                    converged=False,
                                    stop_reason="Too many consecutive execution check failures"
                                )
                                return self._build_result(current_chains, False, "Too many consecutive execution check failures")
                            else:
                                # Reject new chains, continue to next round
                                print(f"  → Keeping old chains, continuing to next iteration round")
                                continue  # Skip quality comparison, go directly to next round

                    except Exception as e:
                        self.log.error(f"Execution check process error: {str(e)}")
                        print(f"  ✗ Execution check error: {str(e)}")
                # ========== End Execution Check ==========

                # ========== Quality Comparison + Decision ==========
                comparison_result = None
                if self.config.ENABLE_QUALITY_COMPARISON:
                    self.log.info(f"Quality comparison started...")
                    print(f"\n[Step 4/4] Comparing chain quality and making decision...")

                    try:
                        from .comparator import get_comparator
                        comparator = get_comparator()

                        comparison_result = comparator.compare(
                            original_chains=current_chains,
                            refined_chains=refined_chains,
                            evaluator_result=evaluation,
                            refine_result=refine_result,
                            execution_check_result=check_result,
                            causal_analysis=causal_analysis,
                            metric_analysis=metric_analysis
                        )

                        # Record comparison results
                        self.logger.log_quality_comparison(iteration_num, comparison_result.to_dict())

                        # Print summary
                        print(f"✓ Quality comparison completed")
                        print(f"  - Comparison result: {comparison_result.comparison_summary.comparison_result}")
                        print(f"  - Keep new version: {comparison_result.comparison_summary.keep_new_version}")
                        print(f"  - Confidence: {comparison_result.comparison_summary.confidence}")
                        print(f"  - Fixed issues: {len(comparison_result.fixed_issues)}")
                        print(f"  - New issues: {len(comparison_result.new_issues)}")

                        # [NEW] Use new decision function
                        accept_new, should_continue, decision_reason = self._decide_after_comparison(
                            comparison_result, current_chains, refined_chains
                        )

                        self.log.info(f"[Decision] {decision_reason}")
                        print(f"  → [Decision] {decision_reason}")

                        # Record decision
                        if self.logger.log_data['iterations']:
                            last_iteration = self.logger.log_data['iterations'][-1]
                            last_iteration['controller_decision'] = {
                                'accept_refined': accept_new,
                                'continue_iteration': should_continue,
                                'stop_reason': None if should_continue else comparison_result.comparison_summary.comparison_result,
                                'decision_basis': decision_reason
                            }
                            # Update refined_chains field
                            last_iteration['refined_chains'] = refined_chains
                            last_iteration['edit_plan'] = edit_plan

                        if not should_continue:
                            # stop iteration
                            final_chains_to_use = refined_chains if accept_new else current_chains
                            self.logger.set_final_result(
                                final_chains=final_chains_to_use,
                                converged=False,
                                stop_reason=decision_reason
                            )
                            return self._build_result(final_chains_to_use, False, decision_reason)

                        # If accepting new chains, update current_chains
                        if accept_new:
                            current_chains = refined_chains
                            print(f"  ✓ Accepting new chains, updating current_chains")
                        else:
                            print(f"  ✓ Keeping old chains, current_chains unchanged")

                    except Exception as e:
                        self.log.error(f"Quality comparison process error: {str(e)}")
                        print(f"  ✗ Quality comparison error: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Conservative handling on comparison error: keep current chains, stop iteration
                        self.logger.set_final_result(
                            final_chains=current_chains,
                            converged=False,
                            stop_reason=f"Quality comparison error: {str(e)}"
                        )
                        return self._build_result(current_chains, False, f"Quality comparison error: {str(e)}")
                else:
                    # If quality comparison not enabled, accept new chains by default and record
                    current_chains = refined_chains
                    if self.logger.log_data['iterations']:
                        last_iteration = self.logger.log_data['iterations'][-1]
                        last_iteration['refined_chains'] = refined_chains
                        last_iteration['edit_plan'] = edit_plan
                        last_iteration['controller_decision'] = {
                            'accept_refined': True,
                            'continue_iteration': True,
                            'stop_reason': None,
                            'decision_basis': 'Quality comparison not enabled, accept new chains by default'
                        }
                # ========== End Quality Comparison ==========

                # Add to iteration history (simplified)
                iteration_history['evaluations'].append({
                    'iteration': iteration_num,
                    'overall_judgement': overall_judgement,
                    'hard_violations': hard_violation_count,
                    'soft_issues': soft_issue_count,
                    'main_issues': self._extract_main_issues(evaluation)
                })

                iteration_history['changes'].append({
                    'iteration': iteration_num,
                    'changes_summary': self._summarize_changes(edit_plan)
                })

                # Handle rollback (keep old logic)
                if rollback and rollback_to:
                    self.log.info(f"Rollback request detected, rolling back to round {rollback_to}")
                    print(f"\n  ⚠ Rollback request detected, rolling back to round {rollback_to}")

                    # Find target iteration chains for rollback
                    if rollback_to <= len(self.logger.log_data['iterations']):
                        target_iteration = self.logger.log_data['iterations'][rollback_to - 1]
                        current_chains = target_iteration.get('chains', refined_chains)
                        self.log.info(f"Rolled back to round {rollback_to} chains")
                        print(f"  ✓ Rolled back to round {rollback_to} chains")

                        # stop iteration
                        self.logger.set_final_result(
                            final_chains=current_chains,
                            converged=False,
                            stop_reason=f"Rollback to round {rollback_to}"
                        )

                        return self._build_result(current_chains, False, "Rolled back to historical best")
                    else:
                        self.log.warning(f"Rollback target round {rollback_to} invalid, continuing with refined chains")
                        current_chains = refined_chains

                self.log.info(f"Round {iteration_num} iteration completed\n")
                print(f"\nRound {iteration_num} iteration completed")

            # If loop ends normally (max rounds reached)
            self.logger.set_final_result(
                final_chains=current_chains,
                converged=False,
                stop_reason=f"Reached max iteration rounds({self.config.MAX_ITERATIONS})but not converged"
            )

            return self._build_result(current_chains, False, "Reached max rounds")

        except Exception as e:
            self.log.error(f"Iteration process error: {str(e)}")
            import traceback
            traceback.print_exc()

            # Record failure
            self.logger.set_final_result(
                final_chains=current_chains,
                converged=False,
                stop_reason=f"Iteration process error: {str(e)}"
            )

            return self._build_result(current_chains, False, f"Error: {str(e)}")

    def _build_result(self, final_chains, converged, stop_reason):
        """
        Build iteration result

        Args:
            final_chains (list): Final propagation chains
            converged (bool): Whether converged
            stop_reason (str): Stop reason

        Returns:
            dict: Iteration result
        """
        # Save log
        log_path = self.logger.save()
        self.log.info(f"Iteration log saved to: {log_path}")

        # Print summary
        self.logger.print_summary()

        # Return result
        result = {
            'final_chains': final_chains,
            'converged': converged,
            'stop_reason': stop_reason,
            'iteration_summary': self.logger.get_iteration_summary(),
            'iteration_history': self.logger.get_iteration_history(),
            'log_path': log_path
        }

        return result

    def evaluate_only(self, chains, causal_analysis, metric_analysis, save_path=None):
        """
        Evaluation-only mode (no refinement)

        Args:
            chains (list): Propagation chains
            causal_analysis (str): PCMCI causal analysis
            metric_analysis (list or str): Metric analysis
            save_path (str, optional): Evaluation result save path

        Returns:
            dict: Evaluation result
        """
        self.log.info("Evaluation-only mode")
        print(f"\n{'='*70}")
        print("Evaluation-only mode (no refinement)")
        print(f"{'='*70}\n")

        evaluation = self.evaluator.evaluate_chains(
            propagation_chains=chains,
            causal_analysis=causal_analysis,
            metric_analysis=metric_analysis,
            save_path=save_path
        )

        return evaluation

    def get_iteration_log(self):
        """Get iteration log data"""
        return self.logger.log_data

    def export_markdown_report(self, file_path=None):
        """
        Export Markdown format report

        Args:
            file_path (str, optional): Save path

        Returns:
            str: Saved file path
        """
        return self.logger.export_to_markdown(file_path)

    def _build_iteration_context(self, current_iteration, iteration_history):
        """
        Build iteration context

        Args:
            current_iteration (int): Current iteration round
            iteration_history (dict): Iteration history records

        Returns:
            dict: Iteration context
        """
        previous_evaluations = iteration_history['evaluations'][:-1] if len(iteration_history['evaluations']) > 1 else iteration_history['evaluations']
        previous_changes = iteration_history['changes'][:-1] if len(iteration_history['changes']) > 1 else iteration_history['changes']

        context = {
            'current_iteration': current_iteration,
            'previous_evaluations': previous_evaluations,
            'previous_changes': previous_changes
        }

        return context

    def _extract_main_issues(self, evaluation):
        """
        Extract main issues from evaluation (new format)

        Args:
            evaluation (dict): Evaluation result

        Returns:
            list: Main issues list
        """
        main_issues = []

        for chain_eval in evaluation.get('chain_evaluations', []):
            # New format: hard_violations and soft_issues separated
            for issue in chain_eval.get('hard_violations', []):
                main_issues.append({
                    'type': issue.get('issue_type'),
                    'severity': issue.get('severity', 'high'),
                    'location': issue.get('location'),
                    'description': issue.get('description', '')[:100]
                })

            # Only collect high-priority soft_issues
            for issue in chain_eval.get('soft_issues', []):
                if issue.get('severity') in ['high', 'medium']:
                    main_issues.append({
                        'type': issue.get('issue_type'),
                        'severity': issue.get('severity'),
                        'location': issue.get('location'),
                        'description': issue.get('description', '')[:100]
                    })

        return main_issues

    def _summarize_changes(self, refine_changes):
        """
        Summarize refinement changes (new format: extract from edit_plan)

        Args:
            refine_changes (list): edit_plan list

        Returns:
            str: Changes summary
        """
        if not refine_changes:
            return "No refinements made"

        actions = [change.get('action', '') for change in refine_changes]
        return ', '.join(actions) if actions else "No refinements made"

    # ============ New Decision Functions ============

    def _should_stop_by_quality(self, evaluation):
        """
        Determine if current chains are good enough for early stopping

        Args:
            evaluation (dict): Evaluation result

        Returns:
            tuple: (should_stop: bool, reason: str)
        """
        summary = evaluation.get('summary', {})
        hard_violation_count = summary.get('hard_violation_count', 0)
        soft_issue_count = summary.get('soft_issue_count', 0)
        overall_judgement = summary.get('overall_judgement', 'needs_refinement')

        # Conditions: no hard violations, soft issues below threshold, and judgement is acceptable
        if (hard_violation_count == 0 and
            soft_issue_count <= self.config.SOFT_ISSUE_THRESHOLD and
            overall_judgement == 'acceptable_with_minor_fixes'):

            reason = f"Current chains are good enough (hard={hard_violation_count}, soft={soft_issue_count}, judgement={overall_judgement})"
            return True, reason

        return False, ""

    def _is_execution_unreliable(self, execution_check_result):
        """
        Determine if this round of modifications is unreliable

        Args:
            execution_check_result (dict): Execution check result

        Returns:
            bool: True means execution is unreliable
        """
        if execution_check_result is None:
            return False

        overall_status = execution_check_result.get('overall_status', 'unknown')
        unauthorized_changes = execution_check_result.get('unauthorized_changes', 0)
        total_actions = execution_check_result.get('total_actions', 0)
        not_executed = execution_check_result.get('not_executed', 0)

        # Conditions:
        # 1. Overall status is fail
        # 2. Has unauthorized changes
        # 3. Ratio of unexecuted actions exceeds threshold
        if overall_status == 'fail':
            return True

        if unauthorized_changes > 0:
            return True

        if total_actions > 0:
            miss_ratio = not_executed / total_actions
            if miss_ratio > self.config.EXECUTION_MISS_RATIO_THRESHOLD:
                return True

        return False

    def _decide_after_execution_check(self, check_result, iteration_num):
        """
        When execution is unreliable, decide whether to continue or stop

        Args:
            check_result (dict): Execution check result
            iteration_num (int): Current iteration round

        Returns:
            tuple: (action: str, reason: str)
                   action can be 'stop' or 'continue'
        """
        self.consecutive_execution_failures += 1

        if self.consecutive_execution_failures >= self.config.MAX_EXECUTION_FAILS:
            reason = f"Consecutive {self.consecutive_execution_failures} rounds of execution check failures, stopping iteration"
            return 'stop', reason
        else:
            reason = f"Execution check failed (round {self.consecutive_execution_failures} times), rejecting new chains but continuing"
            return 'continue', reason

    def _decide_after_comparison(self, comparison_result, current_chains, refined_chains):
        """
        Make final decision based on Comparator results

        Args:
            comparison_result: Comparator comparison result object
            current_chains: Current chains
            refined_chains: Refined chains

        Returns:
            tuple: (accept_new: bool, should_continue: bool, reason: str)
        """
        if comparison_result is None:
            # If comparison fails, use conservative strategy: reject new chains, stop iteration
            return False, False, "Quality comparison failed, keeping current chains and stopping"

        comparison_summary = comparison_result.comparison_summary
        result_type = comparison_summary.comparison_result
        keep_new_version = comparison_summary.keep_new_version
        confidence = comparison_summary.confidence

        if result_type == 'better':
            # better: accept new chains, continue iteration
            self.consecutive_tie_count = 0
            self.consecutive_worse_count = 0
            self.consecutive_execution_failures = 0  # Reset execution failure count
            reason = f"Quality comparison: better (Confidence: {confidence})，accepting new chains and continuing iteration"
            return True, True, reason

        elif result_type == 'worse':
            # worse: reject new chains, stop iteration
            self.consecutive_worse_count += 1
            reason = f"Quality comparison: worse (Confidence: {confidence})，keeping old chains and stopping iteration"

            if self.config.STOP_ON_FIRST_WORSE:
                return False, False, reason
            else:
                # If not stopping on first worse, check consecutive count
                if self.consecutive_worse_count >= 2:
                    return False, False, f"Consecutive {self.consecutive_worse_count} worse, stopping iteration"
                else:
                    return False, True, f"{reason} (allowing continued attempts)"

        elif result_type == 'tie':
            # tie: decide which version to keep based on keep_new_version, then stop
            self.consecutive_tie_count += 1

            if keep_new_version:
                reason = f"Quality comparison: tie (Confidence: {confidence})，keeping new chains and stopping (no significant improvement)"
                accept_new = True
            else:
                reason = f"Quality comparison: tie (Confidence: {confidence})，keeping old chains and stopping (no significant improvement)"
                accept_new = False

            if self.config.STOP_ON_FIRST_TIE:
                return accept_new, False, reason
            else:
                # If not stopping on first tie, check consecutive count
                if self.consecutive_tie_count >= 2:
                    return accept_new, False, f"Consecutive {self.consecutive_tie_count} tie, stopping iteration"
                else:
                    return accept_new, True, f"{reason} (allowing continued attempts)"

        else:
            # Unknown result type, conservative handling
            return False, False, f"Unknown comparison result type: {result_type}"


def create_iteration_controller(case_id, app=None, app_group=None):
    """
    Convenience function to create iteration controller

    Args:
        case_id (str): Case ID
        app (str, optional): Application name
        app_group (str, optional): Application group name

    Returns:
        IterationController: Controller instance
    """
    return IterationController(case_id, app, app_group)


# ============================================================
# M2 Entry: Verifiable iterative refinement loop (deterministic checker + deterministic comparator)
# ============================================================

def run_iteration_m2(
    case_id,
    initial_chains,
    causal_analysis,
    metric_analysis,
    observed_symptoms=None,
    observed_anomaly_ts=None,
    semantic_critic=None,
    semantic_critics=None,
    refiner=None,
    max_iterations=3,
    delay_tolerance=2.0,
    tie_margin=0.02,
    save_dir=None,
    app=None,
):
    """
    M2 main iteration loop (independent of IterationController.run_iteration, does not affect old path)

    Pipeline:
        ┌─────────────────────────────────────────────────────────────┐
        │  loop iteration_num in 1..max_iterations:                   │
        │      critique = M2CriticPipeline.critique(current_chains)   │
        │      if critique.passed: stop (converged)                   │
        │      refined = refiner.refine_from_violations(...)          │
        │      cmp = DeterministicComparator.compare(orig, refined)   │
        │      if cmp == better: current = refined                    │
        │      elif cmp == tie/worse: stop                            │
        └─────────────────────────────────────────────────────────────┘

    Args:
        case_id: Case ID
        initial_chains: Initial propagation_chains (same as AnalysisAgent output)
        causal_analysis: PCMCI causal analysis (CausalGraph / dict list / text / file path all acceptable)
        metric_analysis: Metric analysis (list or JSON string)
        observed_symptoms: Observed anomaly metric set (used by structural endpoint assertions)
        observed_anomaly_ts: {metric: timestamp} Observed anomaly timestamps (used by temporal assertions)
        semantic_critic: Single semantic critic instance (defaults to NoopSemanticCritic)
        semantic_critics: K semantic critics (forming ensemble; higher priority than semantic_critic)
        refiner: Object implementing `.refine_chains_from_violations(...)` or `.refine_chains(...)`.
                 If not provided, skip refine and report directly.
        max_iterations: Maximum iteration rounds
        delay_tolerance: TemporalChecker tolerance
        tie_margin: DeterministicComparator tie threshold
        save_dir: Trajectory log save directory
        app: Optional app name

    Returns:
        dict:
            {
              "final_chains": [...],
              "converged": bool,
              "stop_reason": str,
              "convergence_summary": {...},
              "log_path": str | None,
            }
    """
    from .checkers import M2CriticPipeline, NoopSemanticCritic
    from .comparator import get_deterministic_comparator
    from .convergence_logger import ConvergenceLogger
    from llm.causal_graph.loader import autoload

    graph = autoload(causal_analysis)

    pipeline = M2CriticPipeline(
        graph=graph,
        semantic_critic=semantic_critic or NoopSemanticCritic(),
        semantic_critics=semantic_critics,
        delay_tolerance=delay_tolerance,
    )
    comparator = get_deterministic_comparator(
        critic_pipeline=pipeline, tie_margin=tie_margin
    )

    logger = ConvergenceLogger(case_id=case_id, app=app)

    current_chains = initial_chains or []
    stop_reason = "max_iterations_reached"
    converged = False

    for it in range(1, max_iterations + 1):
        critique = pipeline.critique(
            current_chains,
            observed_symptoms=observed_symptoms,
            observed_anomaly_ts=observed_anomaly_ts,
            causal_analysis=causal_analysis,
            metric_analysis=metric_analysis,
        )
        cur_score = comparator._score(critique).score
        logger.record_round(
            iteration=it,
            chains=current_chains,
            aggregated_critique=critique,
            quality_score=cur_score,
        )

        if critique.passed:
            converged = True
            stop_reason = "all_checks_passed"
            logger.record_decision(it, "stop_converged", stop_reason, delta_score=0.0)
            break

        if refiner is None:
            stop_reason = "no_refiner_provided"
            logger.record_decision(it, "stop_no_refiner", stop_reason)
            break

        violations = [v.to_dict() for v in critique.all_violations]
        try:
            if hasattr(refiner, "refine_chains_from_violations"):
                refine_result = refiner.refine_chains_from_violations(
                    original_chains=current_chains,
                    violations=violations,
                    causal_analysis=causal_analysis,
                    metric_analysis=metric_analysis,
                )
            else:
                # Fallback: group violations by chain into old evaluator expected format
                by_chain = {}
                for v in critique.all_violations:
                    cid = v.chain_id or ""
                    by_chain.setdefault(cid, []).append(v)
                pseudo_eval = {
                    "summary": {
                        "hard_violation_count": sum(1 for v in critique.all_violations
                                                    if v.severity.value in ("high", "critical")),
                        "soft_issue_count": sum(1 for v in critique.all_violations
                                                if v.severity.value in ("low", "medium")),
                        "overall_judgement": "needs_refinement",
                    },
                    "chain_evaluations": [
                        {
                            "chain_id": cid,
                            "hard_violations": [v.to_dict() for v in vs
                                                if v.severity.value in ("high", "critical")],
                            "soft_issues":      [v.to_dict() for v in vs
                                                 if v.severity.value in ("low", "medium")],
                        }
                        for cid, vs in by_chain.items()
                    ],
                }
                refine_result = refiner.refine_chains(
                    original_chains=current_chains,
                    evaluation=pseudo_eval,
                    causal_analysis=causal_analysis,
                    metric_analysis=metric_analysis,
                )
            refined_chains = refine_result.get("refined_chains", []) or []
        except Exception as e:
            stop_reason = f"refiner_error: {e}"
            logger.record_decision(it, "stop_refiner_error", stop_reason)
            break

        cmp = comparator.compare(
            original_chains=current_chains,
            refined_chains=refined_chains,
            observed_symptoms=observed_symptoms,
            observed_anomaly_ts=observed_anomaly_ts,
            causal_analysis=causal_analysis,
            metric_analysis=metric_analysis,
        )

        new_critique = pipeline.critique(
            refined_chains,
            observed_symptoms=observed_symptoms,
            observed_anomaly_ts=observed_anomaly_ts,
            causal_analysis=causal_analysis,
            metric_analysis=metric_analysis,
        )
        new_score = comparator._score(new_critique).score
        delta = new_score - cur_score

        result_type = cmp.comparison_summary.comparison_result
        if result_type == "better":
            current_chains = refined_chains
            logger.record_decision(it, "accept_new",
                                   f"score Δ={delta:+.3f}", delta_score=delta)
            if new_critique.passed:
                converged = True
                stop_reason = "all_checks_passed_after_refine"
                logger.record_decision(it, "stop_converged",
                                       stop_reason, delta_score=delta)
                break
        else:
            stop_reason = f"comparator_{result_type}_after_refine"
            logger.record_decision(it, "reject_new", stop_reason, delta_score=delta)
            break

    logger.set_final(
        converged=converged,
        stop_reason=stop_reason,
        n_iterations=len(logger.rounds),
        final_quality=logger.rounds[-1]["quality_score"] if logger.rounds else None,
    )

    log_path = None
    if save_dir:
        log_path = logger.save(os.path.join(save_dir, f"m2_convergence_{case_id}.json"))

    return {
        "final_chains": current_chains,
        "converged": converged,
        "stop_reason": stop_reason,
        "convergence_summary": logger.summary(),
        "log_path": log_path,
    }

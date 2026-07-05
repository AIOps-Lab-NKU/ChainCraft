#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RefineAgent - Propagation chain refinement agent

Refines fault propagation chains based on EvaluatorAgent evaluation feedback.
Applies refinement strategies:
1. Causal direction correction - Based on PCMCI ordering
2. Supplement intermediate nodes - Fill logical jumps
3. Adjust layer ordering - Ensure forward propagation
4. Remove or split chains - Handle invalid or mixed chains
5. Temporal adjustment - Ensure reasonable propagation order
"""

from .BaseAgent import BaseAgent
import os
import json


class RefineAgent(BaseAgent):
    def __init__(self):
        """Initialize RefineAgent"""
        super().__init__()

        # Load prompt templates
        prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')

        with open(os.path.join(prompt_dir, 'chain_refine_system.txt'), 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()

        with open(os.path.join(prompt_dir, 'chain_refine_user.txt'), 'r', encoding='utf-8') as f:
            self.user_prompt_template = f.read()

    def refine_chains(self, original_chains, evaluation, causal_analysis, metric_analysis, iteration_context=None, save_path=None):
        """
        Refine propagation chains

        Args:
            original_chains (list): Original propagation chain list
            evaluation (dict): EvaluatorAgent evaluation result
            causal_analysis (str): PCMCI causal analysis result (CSV format string)
            metric_analysis (list or str): Metric analysis result (JSON list or string)
            iteration_context (dict, optional): Iteration context (provided from round 2 onwards)
            save_path (str, optional): Refinement result save path

        Returns:
            dict: Refinement result containing refined_chains, changes, reasoning, summary, etc.
        """
        self.logger.info("Starting propagation chain refinement")
        if iteration_context:
            self.logger.info(f"Iteration round: {iteration_context.get('current_iteration')}")

        try:
            # Prepare input data
            original_chains_str = json.dumps(original_chains, ensure_ascii=False, indent=2)
            evaluation_str = json.dumps(evaluation, ensure_ascii=False, indent=2)

            # Process metric_analysis
            if isinstance(metric_analysis, str):
                if os.path.exists(metric_analysis):
                    with open(metric_analysis, 'r', encoding='utf-8') as f:
                        metric_analysis_data = json.load(f)
                    metric_analysis_str = json.dumps(metric_analysis_data, ensure_ascii=False, indent=2)
                else:
                    metric_analysis_str = metric_analysis
            else:
                metric_analysis_str = json.dumps(metric_analysis, ensure_ascii=False, indent=2)

            # Process causal_analysis
            if isinstance(causal_analysis, str) and os.path.exists(causal_analysis):
                with open(causal_analysis, 'r', encoding='utf-8') as f:
                    causal_analysis_content = f.read()
            else:
                causal_analysis_content = str(causal_analysis)

            # Process iteration_context
            iteration_context_str = ""
            if iteration_context:
                iteration_context_str = "\n\n## Iteration Context\n\n"
                iteration_context_str += "```json\n"
                iteration_context_str += json.dumps(iteration_context, ensure_ascii=False, indent=2)
                iteration_context_str += "\n```"

            # Fill user prompt
            user_prompt = self.user_prompt_template.format(
                original_chains=original_chains_str,
                evaluation=evaluation_str,
                causal_analysis=causal_analysis_content,
                metric_analysis=metric_analysis_str
            )

            # Append iteration_context to user_prompt if present
            if iteration_context_str:
                user_prompt += iteration_context_str
                print(f"\n{'='*70}")
                print("Iteration Context")
                print(f"{'='*70}")
                print(iteration_context_str)
                print(f"\n{'='*70}\n")

            # Call LLM for refinement
            self.logger.info("Calling LLM for propagation chain refinement")
            response = self.analyze_with_prompt(self.system_prompt, user_prompt)

            if response is None:
                raise Exception("LLM refinement failed, response is empty")

            # Save raw response for debugging and analysis
            if save_path:
                raw_response_path = save_path.replace('.json', '_raw_response.txt')
                try:
                    os.makedirs(os.path.dirname(raw_response_path), exist_ok=True)
                    with open(raw_response_path, 'w', encoding='utf-8') as f:
                        f.write(response)
                    self.logger.info(f"Raw response saved to: {raw_response_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save raw response: {str(e)}")

            # Parse JSON response
            refine_result = self._parse_refine_response(response)

            # Save refinement result
            if save_path:
                self._save_refine_result(refine_result, save_path)
                self.logger.info(f"Refinement result saved to: {save_path}")

            # Print refinement summary
            self._print_refine_summary(refine_result, original_chains)

            return refine_result

        except Exception as e:
            self.logger.error(f"Error refining propagation chains: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _parse_refine_response(self, response):
        """
        Parse refinement result returned by LLM

        Args:
            response (str): Raw response from LLM

        Returns:
            dict: Parsed refinement result
        """
        try:
            # Try to extract JSON portion
            original_response = response
            response = response.strip()

            # Method 1: If response contains markdown code block, extract JSON from it
            if '```' in response:
                lines = response.split('\n')
                json_lines = []
                in_code_block = False

                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped.startswith('```'):
                        if not in_code_block:
                            in_code_block = True
                            continue  # Skip ```json line
                        else:
                            break  # Closing ```
                    elif in_code_block:
                        json_lines.append(line)

                if json_lines:
                    response = '\n'.join(json_lines).strip()
                    self.logger.info("Extracted JSON from markdown code block")

            # Method 2: Try to find JSON object start and end positions
            if not response.startswith('{'):
                start_idx = response.find('{')
                if start_idx != -1:
                    response = response[start_idx:]
                    self.logger.info(f"Extracted JSON starting from position {start_idx}")

            if not response.endswith('}'):
                end_idx = response.rfind('}')
                if end_idx != -1:
                    response = response[:end_idx+1]
                    self.logger.info(f"Ended JSON extraction at position {end_idx}")

            # Parse JSON
            self.logger.info(f"Attempting to parse JSON, length: {len(response)} chars")
            refine_result = json.loads(response)

            # Validate required fields (new format)
            required_fields = ['edit_plan', 'refined_chains']
            for field in required_fields:
                if field not in refine_result:
                    self.logger.warning(f"Refinement result missing required field: {field}")

            # failed_actions is optional, not enforced
            return refine_result

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse refinement result JSON: {str(e)}")
            self.logger.error(f"Error position: line {e.lineno}, column {e.colno}, pos {e.pos}")

            # Output content near error
            try:
                lines = response.split('\n')
                if e.lineno <= len(lines):
                    error_line = lines[e.lineno - 1]
                    self.logger.error(f"Error line content: {error_line}")
                    if e.colno < len(error_line):
                        context_start = max(0, e.colno - 20)
                        context_end = min(len(error_line), e.colno + 20)
                        context = error_line[context_start:context_end]
                        self.logger.error(f"Error context: ...{context}...")
            except:
                pass

            # Output beginning and end of response
            self.logger.error(f"Response beginning (200 chars): {response[:200]}")
            self.logger.error(f"Response end (200 chars): {response[-200:]}")

            raise Exception(f"Refinement result JSON parse failed: {str(e)}\nError position: line {e.lineno}, column {e.colno}")

    def _save_refine_result(self, refine_result, save_path):
        """
        Save refinement result to JSON file

        Args:
            refine_result (dict): Refinement result
            save_path (str): Save path
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(refine_result, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save refinement result: {str(e)}")
            raise

    def _print_refine_summary(self, refine_result, original_chains):
        """
        Print refinement summary information

        Args:
            refine_result (dict): Refinement result
            original_chains (list): Original chains
        """
        print(f"\n{'='*70}")
        print("Propagation Chain Refinement Results")
        print(f"{'='*70}")

        # Refinement statistics
        if 'summary' in refine_result:
            summary = refine_result['summary']
            print(f"\nRefinement statistics:")
            print(f"  - Total changes: {summary.get('total_changes', 0)}")
            print(f"  - Chains modified: {summary.get('chains_modified', 0)}/{len(original_chains)}")
            print(f"  - Nodes added: {summary.get('nodes_added', 0)}")
            print(f"  - Nodes removed: {summary.get('nodes_removed', 0)}")
            print(f"  - Direction reversed: {summary.get('direction_reversed', 0)}")
            print(f"  - Confidence adjusted: {summary.get('confidence_adjusted', 0)}")

        # Detailed change list
        if 'changes' in refine_result and refine_result['changes']:
            print(f"\nDetailed changes:")
            for idx, change in enumerate(refine_result['changes'], 1):
                chain_id = change.get('chain_id', 'unknown')
                action = change.get('action', 'unknown')
                position = change.get('position', '')
                detail = change.get('detail', '')
                reason = change.get('reason', '')

                action_icon = {
                    'Add intermediate node': '➕',
                    'Remove incorrect node': '➖',
                    'Reverse direction': '🔄',
                    'Adjust layer order': '🔀',
                    'Adjust confidence': '📊',
                    'Split chain': '✂️',
                    'Merge chains': '🔗'
                }.get(action, '📝')

                print(f"\n  {idx}. {action_icon} Chain {chain_id} - {action}")
                print(f"     Position: {position}")
                print(f"     Detail: {detail}")
                print(f"     Reason: {reason}")

        # Refinement reasoning
        if 'reasoning' in refine_result:
            print(f"\nRefinement summary:")
            reasoning_lines = refine_result['reasoning'].split('\n')
            for line in reasoning_lines:
                if line.strip():
                    print(f"  {line}")

        # Refined chains overview
        if 'refined_chains' in refine_result:
            print(f"\nRefined propagation chains:")
            for chain in refine_result['refined_chains']:
                chain_id = chain.get('chain_id', 'unknown')
                summary_text = chain.get('summary', '')
                confidence = chain.get('confidence', 0)
                chain_length = len(chain.get('chain', []))

                # Find original chain confidence for comparison
                original_confidence = None
                for orig_chain in original_chains:
                    if orig_chain.get('chain_id') == chain_id:
                        original_confidence = orig_chain.get('confidence', 0)
                        break

                confidence_change = ""
                if original_confidence is not None:
                    diff = confidence - original_confidence
                    if diff > 0:
                        confidence_change = f" (up {diff:+.2f})"
                    elif diff < 0:
                        confidence_change = f" (down {diff:+.2f})"

                print(f"\n  Chain {chain_id}:")
                print(f"    - Summary: {summary_text}")
                print(f"    - Confidence: {confidence:.2f}{confidence_change}")
                print(f"    - Length: {chain_length} hops")

                # Print propagation path
                if 'chain' in chain:
                    print(f"    - Path: ", end="")
                    path_parts = []
                    for step in chain['chain']:
                        if not path_parts:
                            path_parts.append(step.get('from', '?'))
                        path_parts.append(step.get('to', '?'))
                    print(" -> ".join(path_parts))

        print(f"\n{'='*70}\n")

    def refine_chains_from_violations(self, original_chains, violations,
                                       causal_analysis, metric_analysis,
                                       iteration_context=None, save_path=None):
        """
        M2 entry: Directly accepts structured violation list (from StructuralChecker /
        TemporalChecker / SemanticCritic), reorganizes them into LLM-friendly
        evaluation format and reuses refine_chains.

        Args:
            original_chains: Original propagation_chains
            violations: List[dict], each must have at least chain_id / assertion / severity / detail / suggested_action
            causal_analysis / metric_analysis / iteration_context / save_path: Same as refine_chains
        """
        by_chain = {}
        for v in violations or []:
            cid = v.get("chain_id") or ""
            by_chain.setdefault(cid, []).append(v)

        hard_set = ("high", "critical")
        chain_evaluations = []
        for cid, vs in by_chain.items():
            hard = [self._violation_to_issue(v) for v in vs if v.get("severity") in hard_set]
            soft = [self._violation_to_issue(v) for v in vs if v.get("severity") not in hard_set]
            chain_evaluations.append({
                "chain_id": cid,
                "hard_violations": hard,
                "soft_issues": soft,
            })

        evaluation = {
            "summary": {
                "hard_violation_count": sum(len(c["hard_violations"]) for c in chain_evaluations),
                "soft_issue_count":      sum(len(c["soft_issues"]) for c in chain_evaluations),
                "overall_judgement": "needs_refinement" if violations else "acceptable_with_minor_fixes",
                "source": "m2_checker_pipeline",
            },
            "chain_evaluations": chain_evaluations,
        }

        return self.refine_chains(
            original_chains=original_chains,
            evaluation=evaluation,
            causal_analysis=causal_analysis,
            metric_analysis=metric_analysis,
            iteration_context=iteration_context,
            save_path=save_path,
        )

    @staticmethod
    def _violation_to_issue(v):
        """Convert M2 Violation dict to legacy evaluator issue dict"""
        return {
            "issue_type": v.get("assertion", "unknown"),
            "severity": v.get("severity", "medium"),
            "location": v.get("edge"),
            "description": v.get("detail", ""),
            "suggested_fix": v.get("suggested_action"),
            "source_checker": v.get("checker"),
        }

    def get_changes_by_chain(self, refine_result, chain_id):
        """
        Get all changes for a specific chain

        Args:
            refine_result (dict): Refinement result
            chain_id: Chain ID

        Returns:
            list: All changes for this chain
        """
        if 'changes' not in refine_result:
            return []

        return [
            change for change in refine_result['changes']
            if change.get('chain_id') == chain_id
        ]

    def compare_chains(self, original_chain, refined_chain):
        """
        Compare original chain with refined chain

        Args:
            original_chain (dict): Original chain
            refined_chain (dict): Refined chain

        Returns:
            dict: Comparison result
        """
        result = {
            'chain_id': original_chain.get('chain_id'),
            'length_changed': False,
            'confidence_changed': False,
            'nodes_changed': [],
            'summary_changed': False
        }

        # Compare length
        orig_len = len(original_chain.get('chain', []))
        refine_len = len(refined_chain.get('chain', []))
        if orig_len != refine_len:
            result['length_changed'] = True
            result['length_change'] = f"{orig_len} -> {refine_len}"

        # Compare confidence
        orig_conf = original_chain.get('confidence', 0)
        refine_conf = refined_chain.get('confidence', 0)
        if abs(orig_conf - refine_conf) > 0.01:
            result['confidence_changed'] = True
            result['confidence_change'] = f"{orig_conf:.2f} -> {refine_conf:.2f}"

        # Compare nodes
        orig_nodes = set()
        for step in original_chain.get('chain', []):
            orig_nodes.add(step.get('from'))
            orig_nodes.add(step.get('to'))

        refine_nodes = set()
        for step in refined_chain.get('chain', []):
            refine_nodes.add(step.get('from'))
            refine_nodes.add(step.get('to'))

        added_nodes = refine_nodes - orig_nodes
        removed_nodes = orig_nodes - refine_nodes

        if added_nodes:
            result['nodes_changed'].append(f"Added: {', '.join(added_nodes)}")
        if removed_nodes:
            result['nodes_changed'].append(f"Removed: {', '.join(removed_nodes)}")

        # Compare summary
        if original_chain.get('summary') != refined_chain.get('summary'):
            result['summary_changed'] = True

        return result


def create_refiner():
    """Convenience function to create RefineAgent instance"""
    return RefineAgent()

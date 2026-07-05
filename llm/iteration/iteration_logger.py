#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iteration Log Recording Module

Responsible for recording detailed info of each iteration, generating structured JSON log
"""

import json
import os
from datetime import datetime
from typing import Dict, List


class IterationLogger:
    """Iteration logger"""

    def __init__(self, case_id: str, app: str = None, app_group: str = None,
                 base_path: str = None):
        """
        Initialize iteration logger

        Args:
            case_id: Case ID
            app: Application name (optional)
            app_group: Application group name (optional)
            base_path: Data base path
        """
        self.case_id = case_id
        self.app = app
        self.app_group = app_group
        # If base_path not specified, use current user home directory
        if base_path is None:
            import os
            base_path = os.path.join(os.path.expanduser("~"), "llm_test_data", "collected_data")
        self.base_path = base_path

        # Initialize log data structure
        self.log_data = {
            "case_id": case_id,
            "app": app,
            "app_group": app_group,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "total_time": None,
            "config": {},
            "iterations": [],
            "final_result": {},
            "summary": {
                "total_iterations": 0,
                "converged": False
            }
        }

        self._start_timestamp = datetime.now()

    def set_config(self, config_dict: Dict):
        """
        Set configuration info

        Args:
            config_dict: Config dict
        """
        self.log_data["config"] = config_dict

    def add_iteration(self, iteration_num: int, chains: List[Dict],
                     evaluation: Dict,
                     refined_chains: List[Dict] = None,
                     refine_changes: List[Dict] = None):
        """
        Add one iteration record

        Args:
            iteration_num: Iteration round (starting from 1)
            chains: Current propagation chains
            evaluation: Evaluation result
            refined_chains: Refined chains (optional)
            refine_changes: Refinement change log (optional)
        """
        iteration_record = {
            "iteration": iteration_num,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chains": chains,
            "evaluation": evaluation,
            "issues_found": self._count_issues(evaluation)
        }

        # If refinement results exist, add to record
        if refined_chains is not None:
            iteration_record["refined_chains"] = refined_chains

        if refine_changes is not None:
            iteration_record["refine_changes"] = refine_changes

        self.log_data["iterations"].append(iteration_record)
        self.log_data["summary"]["total_iterations"] = iteration_num

    def log_execution_check(self, iteration_num: int, check_result: Dict):
        """
        Record execution check result

        Args:
            iteration_num: Iteration round
            check_result: Execution check result dict
        """
        # Find corresponding iteration record
        if iteration_num <= len(self.log_data["iterations"]):
            iteration_record = self.log_data["iterations"][iteration_num - 1]
            iteration_record["execution_check"] = {
                "overall_status": check_result.get("overall_status"),
                "total_actions": check_result.get("total_actions", 0),
                "executed_correctly": check_result.get("executed_correctly", 0),
                "executed_incorrectly": check_result.get("executed_incorrectly", 0),
                "not_executed": check_result.get("not_executed", 0),
                "unauthorized_changes": check_result.get("unauthorized_changes", 0),
                "summary": check_result.get("summary", ""),
                "timestamp": check_result.get("timestamp", "")
            }

            # Can also save detailed chain_checks (optional)
            # iteration_record["execution_check"]["details"] = check_result.get("chain_checks", [])

    def log_quality_comparison(self, iteration_num: int, comparison_result: Dict):
        """
        Record quality comparison result

        Args:
            iteration_num: Iteration round
            comparison_result: Comparison result dict
        """
        # Find corresponding iteration record
        if iteration_num <= len(self.log_data["iterations"]):
            iteration_record = self.log_data["iterations"][iteration_num - 1]

            # Extract core information
            summary = comparison_result.get("comparison_summary", {})
            iteration_record["quality_comparison"] = {
                "comparison_result": summary.get("comparison_result"),
                "keep_new_version": summary.get("keep_new_version"),
                "confidence": summary.get("confidence"),
                "fixed_issues_count": len(comparison_result.get("fixed_issues", [])),
                "new_issues_count": len(comparison_result.get("new_issues", [])),
                "remaining_issues_count": len(comparison_result.get("remaining_issues", [])),
                "reason": comparison_result.get("reason", "")
            }

            # Optional: save detailed fixed_issues and new_issues
            # iteration_record["quality_comparison"]["fixed_issues"] = comparison_result.get("fixed_issues", [])
            # iteration_record["quality_comparison"]["new_issues"] = comparison_result.get("new_issues", [])

    def set_final_result(self, final_chains: List[Dict], converged: bool,
                        stop_reason: str):
        """
        Set final result

        Args:
            final_chains: Final propagation chains
            converged: Whether converged
            stop_reason: Stop reason
        """
        # Calculate total time
        end_timestamp = datetime.now()
        total_seconds = (end_timestamp - self._start_timestamp).total_seconds()

        self.log_data["end_time"] = end_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self.log_data["total_time"] = f"{total_seconds:.2f}s"

        # Set final result
        self.log_data["final_result"] = {
            "chains": final_chains,
            "converged": converged,
            "stop_reason": stop_reason
        }

        # Update summary
        self.log_data["summary"]["converged"] = converged

    def save(self, file_path: str = None) -> str:
        """
        Save log to JSON file

        Args:
            file_path: Save path (optional, uses standard path by default)

        Returns:
            str: Saved file path
        """
        if file_path is None:
            file_path = self._get_default_log_path()

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)

        return file_path

    def load(self, file_path: str = None) -> Dict:
        """
        Load log from file

        Args:
            file_path: Log file path (optional)

        Returns:
            Dict: Log data
        """
        if file_path is None:
            file_path = self._get_default_log_path()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Log file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            self.log_data = json.load(f)

        return self.log_data

    def get_iteration_summary(self) -> Dict:
        """
        Get iteration summary info

        Returns:
            Dict: Summary info
        """
        return self.log_data["summary"]

    def get_iteration_history(self) -> List[Dict]:
        """
        Get iteration history records

        Returns:
            List[Dict]: Records of each iteration
        """
        return self.log_data["iterations"]

    def print_summary(self):
        """Print iteration summary"""
        summary = self.log_data["summary"]

        print(f"\n{'='*60}")
        print(f"Iteration Log Summary - Case: {self.case_id}")
        print(f"{'='*60}")
        print(f"Total iteration rounds: {summary['total_iterations']}")
        print(f"Whether converged: {'Yes' if summary['converged'] else 'No'}")

        if self.log_data["total_time"]:
            print(f"Total time: {self.log_data['total_time']}")

        print(f"{'='*60}\n")

    def print_iteration_details(self):
        """Print detailed info of each iteration"""
        print(f"\n{'='*60}")
        print("Iteration Detailed Records")
        print(f"{'='*60}")

        for record in self.log_data["iterations"]:
            iteration = record["iteration"]
            issues_found = record["issues_found"]

            print(f"\nRound {iteration} iteration:")
            print(f"  Time: {record['timestamp']}")
            print(f"  Issues found: {issues_found}")

            # Print key info from evaluation results
            evaluation = record.get("evaluation", {})
            if "chain_evaluations" in evaluation:
                print(f"  Propagation chain count: {len(evaluation['chain_evaluations'])}")

        print(f"{'='*60}\n")

    def export_to_markdown(self, file_path: str = None) -> str:
        """
        Export as Markdown format report

        Args:
            file_path: Save path (optional)

        Returns:
            str: Saved file path
        """
        if file_path is None:
            default_path = self._get_default_log_path()
            file_path = default_path.replace(".json", ".md")

        summary = self.log_data["summary"]

        # Build Markdown content
        md_lines = [
            f"# Iteration Refinement Report - {self.case_id}\n",
            f"**Application**: {self.app or 'N/A'} ({self.app_group or 'N/A'})\n",
            f"**Start Time**: {self.log_data['start_time']}\n",
            f"**End Time**: {self.log_data['end_time']}\n",
            f"**Total time**: {self.log_data['total_time']}\n",
            "\n## Summary\n",
            f"- **Total iteration rounds**: {summary['total_iterations']}",
            f"- **Whether converged**: {'Yes' if summary['converged'] else 'No'}",
            "\n## Iteration History\n"
        ]

        # Add details for each iteration
        for record in self.log_data["iterations"]:
            iteration = record["iteration"]
            issues = record["issues_found"]

            md_lines.append(f"\n### Round {iteration}\n")
            md_lines.append(f"- **Issues found**: {issues}")

            # Add propagation chain evaluation
            evaluation = record.get("evaluation", {})
            if "chain_evaluations" in evaluation:
                md_lines.append(f"\n**Propagation chain count**: {len(evaluation['chain_evaluations'])}")

        # Add final result
        if self.log_data["final_result"]:
            md_lines.append("\n## Final Result\n")
            final_result = self.log_data["final_result"]
            md_lines.append(f"- **Stop reason**: {final_result.get('stop_reason', 'N/A')}")
            md_lines.append(f"- **Whether converged**: {'Yes' if final_result.get('converged') else 'No'}")

        # Save file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(line for line in md_lines if line))

        return file_path

    def _get_default_log_path(self) -> str:
        """Get default log save path"""
        if self.app and self.app_group:
            case_dir = os.path.join(
                self.base_path,
                self.case_id,
                f"{self.app}_{self.app_group}"
            )
        else:
            case_dir = os.path.join(self.base_path, self.case_id)

        summary_dir = os.path.join(case_dir, "summary")
        return os.path.join(summary_dir, "iteration_log.json")

    def _count_issues(self, evaluation: Dict) -> int:
        """
        Count issues in evaluation result

        Args:
            evaluation: Evaluation result dict

        Returns:
            int: Total issues
        """
        total_issues = 0

        if "chain_evaluations" in evaluation:
            for chain_eval in evaluation["chain_evaluations"]:
                if "issues" in chain_eval:
                    total_issues += len(chain_eval["issues"])

        return total_issues


# Convenience function
def create_logger(case_id: str, app: str = None, app_group: str = None) -> IterationLogger:
    """
    Convenience function to create iteration logger

    Args:
        case_id: Case ID
        app: Application name
        app_group: Application group name

    Returns:
        IterationLogger: Logger instance
    """
    return IterationLogger(case_id, app, app_group)

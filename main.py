#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChainCraft — Fault Root Cause Analysis and Risk Prediction System

This system collects application metric data and log data, combined with large language models for:
  1. Anomaly Detection (Prophet algorithm)
  2. Causal Analysis (PCMCI)
  3. LLM Intelligent Analysis (root cause attribution / inference prediction)
  4. Vector Knowledge Base Construction and Retrieval (ChromaDB)

Usage:
  # Option 1: Run directly (modify the WORKFLOWS section below)
  python main.py

  # Option 2: Import as module
  from main import process_historical_cases, process_prediction_cases
  process_historical_cases(['case1'], enable_iteration=True)

  # Option 3: Single case debugging
  from main import process_case_complete
  process_case_complete('case1', item_index=0)

Workflow Description:
  - Historical case processing: Analyze case -> Process fault report (build vector DB)
  - Prediction case processing: Pull data and inference -> Process inference report (similarity matching + risk judgment)
"""

import sys
import os
import logging
import time

# ============================================================
# Path setup: ensure project root is in sys.path
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from batch_executor import execute_case_tasks

# ============================================================
# Import core modules
# ============================================================
from config import Config, setup_logging

# Data collection and analysis
from data_handle.integrated_data_collector import (
    analyze_single_case,
    inference_single_case,
)

# Fault report processing (vector DB storage / retrieval)
from llm.fault_processor.main import (
    deal_fault_report,
    deal_inference_report,
)

# Case data
from data_handle import case_table

# ============================================================
# Logging configuration
# ============================================================
setup_logging(level="INFO")
logger = logging.getLogger(__name__)

# ============================================================
# Batch processing parallel configuration
# ============================================================
# Workflows are still executed sequentially; this switch only controls
# case-level parallelism within batch_* functions.
BATCH_PARALLEL = False  # True: parallel case execution, False: sequential case execution
BATCH_MAX_WORKERS = 4


# ============================================================
# Utility functions
# ============================================================

def _iterate_case_items(case_ids):
    """
    Generator: iterate over case ID list, yielding (case_id, item_index) pairs

    Args:
        case_ids: list of case IDs

    Yields:
        tuple: (case_id, item_index, case_info)
    """
    for case_id in case_ids:
        case_info = case_table.get(case_id)
        if not case_info:
            logger.warning("Case %s not found, skipping", case_id)
            continue
        for item_index in range(len(case_info['app_name'])):
            yield case_id, item_index, case_info


def _print_section(title):
    """Print section separator title"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def _format_elapsed(seconds):
    """Format seconds into a human-readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.2f}s"


def _timed_run(name, func, *args, **kwargs):
    """Execute a function and print elapsed time, also tracking token consumption for this workflow (bucketed by model)"""
    from llm.agent.BaseAgent import BaseAgent

    # Snapshot before execution
    before = BaseAgent.get_total_token_usage()

    wf_start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - wf_start
    print(f"\n[Timing] {name} elapsed: {_format_elapsed(elapsed)}")

    # Snapshot after execution, compute diff
    after = BaseAgent.get_total_token_usage()
    total_diff_prompt = after['total']['prompt_tokens'] - before['total']['prompt_tokens']
    total_diff_completion = after['total']['completion_tokens'] - before['total']['completion_tokens']
    total_diff_total = after['total']['total_tokens'] - before['total']['total_tokens']

    if total_diff_total > 0:
        print(f"[TOKEN Summary] {name} consumption:")
        # Print diff per model (excluding 'total' key)
        for model in after:
            if model == 'total':
                continue
            before_model = before.get(model, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
            d_prompt = after[model]['prompt_tokens'] - before_model['prompt_tokens']
            d_completion = after[model]['completion_tokens'] - before_model['completion_tokens']
            d_total = after[model]['total_tokens'] - before_model['total_tokens']
            if d_total > 0:
                print(f"  Model [{model}]: Input: {d_prompt:,}, Output: {d_completion:,}, Total: {d_total:,}")
        print(f"  Overall: Input: {total_diff_prompt:,}, Output: {total_diff_completion:,}, Total: {total_diff_total:,}")
    else:
        print(f"[TOKEN Summary] {name} no token consumption this run")

    return result


def _execute_batch_case_items(case_ids, worker):
    """Execute case tasks with global parallel configuration, preserving input order."""
    tasks = [
        (case_id, item_index)
        for case_id, item_index, _ in _iterate_case_items(case_ids)
    ]
    mode = "Parallel" if BATCH_PARALLEL else "Sequential"
    logger.info(
        "Batch execution mode: %s, task count: %d, max workers: %d",
        mode,
        len(tasks),
        BATCH_MAX_WORKERS,
    )
    return execute_case_tasks(
        tasks,
        worker,
        parallel=BATCH_PARALLEL,
        max_workers=BATCH_MAX_WORKERS,
    )


# ============================================================
# Core workflows
# ============================================================

def process_historical_cases(case_ids, enable_iteration=False,
                             run_anomaly_detection=True,
                             run_metric_analysis=True,
                             run_causal_analysis=True,
                             use_causal_analysis=True):
    """
    Process historical cases and build vector knowledge base

    Flow: Analyze case -> Process fault report -> Store in ChromaDB

    Args:
        case_ids: list of historical case IDs
        enable_iteration: whether to enable iterative refinement (default False)
        run_anomaly_detection: whether to run anomaly detection (False to reuse existing results from ANOMALY_DETECTION_READ_PATH, default True)
        run_metric_analysis: whether to run metric analysis (False to reuse existing results from METRIC_ANALYSIS_READ_PATH, default True)
        run_causal_analysis: whether to run causal analysis (False to reuse existing results from ANALYSIS_READ_PATH, default True)
        use_causal_analysis: whether to use causal analysis information (default True).
            False uses prompt template without causal information.

    Returns:
        dict: {'success': int, 'failed': list}
    """
    _print_section(f"Processing {len(case_ids)} historical cases, building vector knowledge base")

    success_count = 0
    failed_cases = []

    for case_id, item_index, _ in _iterate_case_items(case_ids):
        logger.info("Processing historical case: %s [index %d]", case_id, item_index)

        try:
            # Step 1: Analyze case
            logger.info("  -> Step 1/2: Analyzing case")
            analyze_single_case(
                case_id, item_index,
                collect_data=False,
                enable_iteration=enable_iteration,
                run_anomaly_detection=run_anomaly_detection,
                run_metric_analysis=run_metric_analysis,
                run_causal_analysis=run_causal_analysis,
                use_causal_analysis=use_causal_analysis,
            )

            # Step 2: Process fault report and store in vector DB
            logger.info("  -> Step 2/2: Processing fault report and storing in vector DB")
            deal_fault_report(case_id, item_index)

            success_count += 1
            logger.info("  ✓ Case %s [index %d] processing complete", case_id, item_index)

        except Exception as e:
            logger.error("  ✗ Case %s [index %d] processing failed: %s", case_id, item_index, e)
            failed_cases.append(f"{case_id}_{item_index}")

    _print_section(
        f"Historical case processing complete — Success: {success_count}, Failed: {len(failed_cases)}"
    )
    if failed_cases:
        logger.warning("Failed cases: %s", failed_cases)

    return {'success': success_count, 'failed': failed_cases}


def process_prediction_cases(case_ids, enable_iteration=False,
                             run_anomaly_detection=True,
                             run_metric_analysis=True,
                             run_causal_analysis=True,
                             use_structure_rag=True,
                             use_chain_rerank=True,
                             use_causal_analysis=True):
    """
    Process cases that require prediction

    Flow: Pull data and inference -> Process inference report -> Output risk judgment

    Args:
        case_ids: list of case IDs requiring prediction
        enable_iteration: whether to enable iterative refinement (default False)
        run_anomaly_detection: whether to run anomaly detection (False to reuse existing results from ANOMALY_DETECTION_READ_PATH, default True)
        run_metric_analysis: whether to run metric analysis (False to reuse existing results from METRIC_ANALYSIS_READ_PATH, default True)
        run_causal_analysis: whether to run causal analysis (False to reuse existing results from ANALYSIS_READ_PATH, default True)
        use_structure_rag: whether to use structure RAG for chain matching (default True)
        use_chain_rerank: whether to enable propagation chain reranking (default True)
        use_causal_analysis: whether to use causal analysis information (default True).
            False uses prompt template without causal information.

    Returns:
        dict: inference result dictionary
    """
    _print_section(f"Processing {len(case_ids)} prediction cases")

    results = {}

    for case_id, item_index, _ in _iterate_case_items(case_ids):
        logger.info("Processing prediction case: %s [index %d]", case_id, item_index)

        try:
            # Step 1: Pull data and perform inference analysis
            logger.info("  -> Step 1/2: Pulling data and performing inference analysis")
            inference_result = inference_single_case(
                case_id, item_index,
                enable_iteration=enable_iteration,
                run_anomaly_detection=run_anomaly_detection,
                run_metric_analysis=run_metric_analysis,
                run_causal_analysis=run_causal_analysis,
                use_causal_analysis=use_causal_analysis,
            )

            # Step 2: Process inference report and make prediction
            logger.info("  -> Step 2/2: Processing inference report and making prediction")
            prediction_result = deal_inference_report(case_id, item_index, use_structure_rag=use_structure_rag, use_chain_rerank=use_chain_rerank)

            results[f"{case_id}_{item_index}"] = {
                'inference': inference_result,
                'prediction': prediction_result,
            }
            logger.info("  ✓ Case %s [index %d] processing complete", case_id, item_index)

        except Exception as e:
            logger.error("  ✗ Case %s [index %d] processing failed: %s", case_id, item_index, e)

    _print_section(f"Prediction case processing complete — Success: {len(results)}")
    return results


def batch_analyze_cases(case_ids, collect_data=True, enable_iteration=False,
                        run_anomaly_detection=True,
                        run_metric_analysis=True,
                        run_causal_analysis=True,
                        use_causal_analysis=True):
    """
    Batch analyze cases (analysis step only, no fault report processing)

    Args:
        case_ids: list of case IDs
        collect_data: whether to collect data (True will run data collection first)
        enable_iteration: whether to enable iterative refinement
        run_anomaly_detection: whether to run anomaly detection (False to reuse existing results from ANOMALY_DETECTION_READ_PATH, default True)
        run_metric_analysis: whether to run metric analysis (False to reuse existing results from METRIC_ANALYSIS_READ_PATH, default True)
        run_causal_analysis: whether to run causal analysis (False to reuse existing results from ANALYSIS_READ_PATH, default True)
        use_causal_analysis: whether to use causal analysis information (default True).
            False uses prompt template without causal information.

    Returns:
        dict: analysis result dictionary
    """
    _print_section(f"Batch analyzing {len(case_ids)} cases"
                   f" ({'with data collection' if collect_data else 'inference only'})")

    def analyze_worker(case_id, item_index):
        return analyze_single_case(
            case_id, item_index, collect_data,
            enable_iteration=enable_iteration,
            run_anomaly_detection=run_anomaly_detection,
            run_metric_analysis=run_metric_analysis,
            run_causal_analysis=run_causal_analysis,
            use_causal_analysis=use_causal_analysis,
        )

    results = {}
    for outcome in _execute_batch_case_items(case_ids, analyze_worker):
        case_id = outcome.case_id
        item_index = outcome.item_index
        if outcome.success:
            results[f"{case_id}_{item_index}"] = outcome.value
            logger.info("  ✓ Case %s [index %d] analysis complete", case_id, item_index)
        else:
            logger.error(
                "  ✗ Case %s [index %d] analysis failed: %s",
                case_id, item_index, outcome.error,
            )

    _print_section(f"Batch analysis complete — Success: {len(results)}")
    return results


def batch_inference_cases(case_ids, collect_data=True, enable_iteration=False,
                          run_anomaly_detection=True,
                          run_metric_analysis=True,
                          run_causal_analysis=True,
                          use_causal_analysis=True):
    """
    Batch inference cases (inference step only, no inference report processing)

    Args:
        case_ids: list of case IDs
        collect_data: whether to collect data
        enable_iteration: whether to enable iterative refinement
        run_anomaly_detection: whether to run anomaly detection (False to reuse existing results from ANOMALY_DETECTION_READ_PATH, default True)
        run_metric_analysis: whether to run metric analysis (False to reuse existing results from METRIC_ANALYSIS_READ_PATH, default True)
        run_causal_analysis: whether to run causal analysis (False to reuse existing results from ANALYSIS_READ_PATH, default True)
        use_causal_analysis: whether to use causal analysis information (default True).
            False uses prompt template without causal information.

    Returns:
        dict: inference result dictionary
    """
    _print_section(f"Batch inference for {len(case_ids)} cases"
                   f" ({'with data collection' if collect_data else 'inference only'})")

    def inference_worker(case_id, item_index):
        return inference_single_case(
            case_id, item_index, collect_data,
            enable_iteration=enable_iteration,
            run_anomaly_detection=run_anomaly_detection,
            run_metric_analysis=run_metric_analysis,
            run_causal_analysis=run_causal_analysis,
            use_causal_analysis=use_causal_analysis,
        )

    results = {}
    for outcome in _execute_batch_case_items(case_ids, inference_worker):
        case_id = outcome.case_id
        item_index = outcome.item_index
        if outcome.success:
            results[f"{case_id}_{item_index}"] = outcome.value
            logger.info("  ✓ Case %s [index %d] inference complete", case_id, item_index)
        else:
            logger.error(
                "  ✗ Case %s [index %d] inference failed: %s",
                case_id, item_index, outcome.error,
            )

    _print_section(f"Batch inference complete — Success: {len(results)}")
    return results


def batch_deal_fault_reports(case_ids):
    """
    Batch process fault reports (store to vector database)

    Args:
        case_ids: list of case IDs
    """
    _print_section(f"Batch processing fault reports for {len(case_ids)} cases")

    outcomes = _execute_batch_case_items(case_ids, deal_fault_report)
    success_count = sum(outcome.success for outcome in outcomes)
    for outcome in outcomes:
        if not outcome.success:
            logger.error(
                "  ✗ Fault report %s [index %d] processing failed: %s",
                outcome.case_id, outcome.item_index, outcome.error,
            )

    _print_section(f"Batch processing complete — Success: {success_count}")


def batch_deal_inference_reports(case_ids, use_structure_rag=True, use_chain_rerank=True):
    """
    Batch process inference reports

    Args:
        case_ids: list of case IDs
        use_structure_rag: whether to use structure RAG for chain matching (default True)
        use_chain_rerank: whether to enable propagation chain reranking (default True)

    Returns:
        dict: processing result dictionary
    """
    _print_section(f"Batch processing inference reports for {len(case_ids)} cases")

    def inference_report_worker(case_id, item_index):
        return deal_inference_report(
            case_id,
            item_index,
            use_structure_rag=use_structure_rag,
            use_chain_rerank=use_chain_rerank,
        )

    results = {}
    for outcome in _execute_batch_case_items(case_ids, inference_report_worker):
        case_id = outcome.case_id
        item_index = outcome.item_index
        if outcome.success:
            results[f"{case_id}_{item_index}"] = outcome.value
        else:
            logger.error(
                "  ✗ Inference report %s [index %d] processing failed: %s",
                case_id, item_index, outcome.error,
            )

    _print_section(f"Batch processing complete — Success: {len(results)}")
    return results


def process_case_complete(case_id, item_index=0, enable_iteration=False,
                          run_anomaly_detection=True,
                          run_metric_analysis=True,
                          run_causal_analysis=True,
                          use_causal_analysis=True):
    """
    Complete processing of a single case: from data collection to inference analysis (for debugging or single case processing)

    Full flow: Data collection -> Fault report storage -> Inference analysis -> Inference report processing

    Args:
        case_id: case ID
        item_index: application index
        enable_iteration: whether to enable iterative refinement
        run_anomaly_detection: whether to run anomaly detection (False to reuse existing results from ANOMALY_DETECTION_READ_PATH, default True)
        run_metric_analysis: whether to run metric analysis (False to reuse existing results from METRIC_ANALYSIS_READ_PATH, default True)
        run_causal_analysis: whether to run causal analysis (False to reuse existing results from ANALYSIS_READ_PATH, default True)
        use_causal_analysis: whether to use causal analysis information (default True).
            False uses prompt template without causal information.

    Returns:
        dict: dictionary containing results from each phase, None on failure
    """
    _print_section(f"Complete processing of case: {case_id} [index {item_index}]")

    try:
        # Step 1: Data collection and analysis
        logger.info("  -> Step 1/4: Data collection and analysis")
        analysis_result = analyze_single_case(
            case_id, item_index, collect_data=True,
            enable_iteration=enable_iteration,
            run_anomaly_detection=run_anomaly_detection,
            run_metric_analysis=run_metric_analysis,
            run_causal_analysis=run_causal_analysis,
            use_causal_analysis=use_causal_analysis,
        )

        # Step 2: Fault report storage
        logger.info("  -> Step 2/4: Processing fault report and storing in vector DB")
        fault_report_result = deal_fault_report(case_id, item_index)

        # Step 3: Inference analysis
        logger.info("  -> Step 3/4: Performing inference analysis")
        inference_result = inference_single_case(
            case_id, item_index, collect_data=False,
            enable_iteration=enable_iteration,
            run_anomaly_detection=run_anomaly_detection,
            run_metric_analysis=run_metric_analysis,
            run_causal_analysis=run_causal_analysis,
            use_causal_analysis=use_causal_analysis,
        )

        # Step 4: Process inference report
        logger.info("  -> Step 4/4: Processing inference report and performing similar case analysis")
        deal_inference_result = deal_inference_report(case_id, item_index)

        _print_section(f"Case {case_id} processing complete!")

        return {
            'analysis': analysis_result,
            'fault_report': fault_report_result,
            'inference': inference_result,
            'deal_inference': deal_inference_result,
        }

    except Exception as e:
        logger.error("  ✗ Case %s processing failed: %s", case_id, e)
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# Case group constants (for easy switching during debugging)
# ============================================================

# Historical cases (for building vector knowledge base)
# HISTORICAL_CASES = ['case1', 'case17', 'case20', 'case21', 'case29', 'case31', 'case39', 'case47', 'case48', 'case52', 'case53', 'case56', 'case60']
HISTORICAL_DEMO_CASES = ['case1']


# Prediction cases (for validating inference capability)
# PREDICTION_CASES = ['case2', 'case14', 'case26', 'case33', 'case41', 'case90', 'risk55', 'case92', 'risk1', 'risk2', 'risk3', 'risk4', 'case96', 'case97', 'case98', 'case100', 'case24', 'case25', 'case34', 'case35', 'risk46', 'case40', 'case42', 'case43', 'case46', 'case49', 'case50', 'risk47', 'risk48', 'case55', 'case57', 'case58', 'case59', 'case61', 'case62', 'case63', 'case64', 'case65', 'case66', 'case67', 'case68', 'case69', 'case70', 'case71', 'case72', 'case74', 'risk49', 'case76', 'case77', 'risk50', 'case79', 'case80', 'case81', 'risk51', 'case83', 'risk52', 'case85', 'case86', 'case87', 'risk53', 'risk54', 'case93', 'risk5', 'risk6', 'risk7', 'risk8', 'risk9', 'risk10', 'case95', 'risk12', 'risk13', 'risk14', 'risk15', 'risk17', 'risk18', 'risk19', 'risk20', 'risk21', 'risk22', 'risk23', 'risk24', 'risk25', 'risk28', 'risk29', 'risk31', 'risk32', 'risk33', 'case99', 'risk35', 'risk36', 'risk37', 'risk38', 'risk40', 'risk41', 'risk42', 'risk43', 'risk44', 'risk45', 'risk56', 'risk57']
PREDICTION_DEMO_CASES = ['case2']

# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    # ========================================================
    # WORKFLOWS — Configure workflows to execute in this section
    # ========================================================

    from llm.agent.BaseAgent import BaseAgent
    BaseAgent.reset_token_count()

    overall_start = time.time()

    # Print configuration info (optional)
    Config.print_path_config()

    # ---------- Workflow 1: Batch analyze historical cases ----------
    _timed_run("Workflow 1 (Batch analyze historical cases)", batch_analyze_cases, HISTORICAL_DEMO_CASES, collect_data=False, run_anomaly_detection=True, run_metric_analysis=True, run_causal_analysis=True, use_causal_analysis=True, enable_iteration=True)

    # ---------- Workflow 2: Batch inference prediction cases ----------
    _timed_run("Workflow 2 (Batch inference prediction cases)", batch_inference_cases, PREDICTION_DEMO_CASES, collect_data=False, run_anomaly_detection=True, run_metric_analysis=True, run_causal_analysis=True, use_causal_analysis=True, enable_iteration=True)

    # ---------- Workflow 3: Build historical case database ----------
    _timed_run("Workflow 3 (Build historical case database)", batch_deal_fault_reports, HISTORICAL_DEMO_CASES)

    # ---------- Workflow 4: Batch predict cases ----------
    _timed_run("Workflow 4 (Batch predict cases)", batch_deal_inference_reports, PREDICTION_DEMO_CASES, use_structure_rag=True, use_chain_rerank=True)

    print(f"\n[Timing] Total elapsed time: {_format_elapsed(time.time() - overall_start)}")

    # ---------- Token overall consumption summary ----------
    token_summary = BaseAgent.get_total_token_usage()
    total = token_summary.get('total', {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
    print(f"\n[TOKEN Summary] Overall consumption statistics:")
    for model, usage in token_summary.items():
        if model == 'total':
            continue
        print(f"  Model [{model}]:")
        print(f"    - Input Token: {usage['prompt_tokens']:,}")
        print(f"    - Output Token: {usage['completion_tokens']:,}")
        print(f"    - Total Token: {usage['total_tokens']:,}")
    print(f"  --- Overall ---")
    print(f"    - Input Token: {total['prompt_tokens']:,}")
    print(f"    - Output Token: {total['completion_tokens']:,}")
    print(f"    - Total Token: {total['total_tokens']:,}")

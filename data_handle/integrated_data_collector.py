#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
from datetime import datetime

# Data collector (for anomaly detection)
from data_process.pipeline import IntegratedDataCollector

# LLM related imports
from llm.agent.MetricAnalysisAgent import MetricAnalysisAgent
from llm.agent.AnalysisAgent import AnalysisAgent
from llm.agent.InferenceAgent import InferenceAgent

from data_handle.data_config import case_table
from data_process.analysis.causal_analysis import case_causal_relations_process
from config import Config
from config.path_manager import path_manager


def _ensure_metric_data_ready(case_id, app, app_group, save_dir=None):
    """
    Ensure metric data file is ready.

    Check whether all_metrics.csv exists under the COLLECTED_DATA_PATH,
    if not, attempt to merge from the raw_data/ subdirectory.

    Args:
        case_id: case ID
        app: application name
        app_group: application group name
        save_dir: base save directory (compatible with data_handle mode)

    Returns:
        tuple: (metric_data_path, summary_path)

    Raises:
        FileNotFoundError: when data file does not exist
    """
    if save_dir is not None:
        app_path = os.path.join(save_dir, f"{app}_{app_group}")
    else:
        app_path = path_manager.get_collected_app_path(case_id, app, app_group)

    metric_dir = os.path.join(app_path, "metric")
    all_metrics_path = os.path.join(metric_dir, "all_metrics.csv")
    raw_data_dir = os.path.join(metric_dir, "raw_data")
    summary_path = os.path.join(app_path, "summary")

    if os.path.exists(all_metrics_path):
        print(f"✓ Metric data file already exists: {all_metrics_path}")
        return all_metrics_path, summary_path

    # Try merging from raw_data/
    if os.path.isdir(raw_data_dir) and os.listdir(raw_data_dir):
        print(f"✓ all_metrics.csv does not exist, merging from raw_data/ directory...")
        from data_handle.data_merge_and_clean import DataMergeAndClean
        processor = DataMergeAndClean(missing_threshold=0.2, verbose=True)
        processor.process_directory(raw_data_dir, all_metrics_path)
        if os.path.exists(all_metrics_path):
            print(f"✓ Metric data merge complete: {all_metrics_path}")
            return all_metrics_path, summary_path

    raise FileNotFoundError(
        f"Metric data file does not exist: {all_metrics_path}, "
        f"and raw_data/ directory is empty or does not exist: {raw_data_dir}.\n"
        f"Please place metric data in one of the following paths:\n"
        f"  1. Direct placement: {all_metrics_path}\n"
        f"  2. Multiple CSVs in: {raw_data_dir}/ (will be auto-merged)"
    )


def analyze_single_case(case_id, item_index, collect_data=True,
                        severity_filter='Medium', enable_iteration=False,
                        run_anomaly_detection=True,
                        run_metric_analysis=True,
                        run_causal_analysis=True,
                        use_causal_analysis=True):
    """
    Analyze metric data for a single case

    Args:
        case_id: case ID
        item_index: application index
        collect_data: whether to prepare data (True validates data file existence under COLLECTED_DATA_PATH,
                      merges from raw_data/ if available, but does not perform remote data pulling)
        severity_filter: severity filter level
        enable_iteration: whether to enable iterative refinement (default False)
        run_anomaly_detection: whether to run anomaly detection (False to reuse existing results from ANOMALY_DETECTION_READ_PATH, default True)
        run_metric_analysis: whether to run metric analysis (False to reuse existing results from METRIC_ANALYSIS_READ_PATH, default True)
        run_causal_analysis: whether to run causal analysis (False to reuse existing results from ANALYSIS_READ_PATH, default True)
        use_causal_analysis: whether to use causal analysis information (default True).
            False uses prompt template without causal information.

    Returns:
        dict: dictionary containing analysis results and related paths
    """
    case_info = case_table[case_id]
    app = case_info['app_name'][item_index]
    app_group = case_info['app_groups'][item_index][0]

    anomaly_start_time = case_info['fault_start']
    anomaly_end_time = case_info['fault_end']

    # display_start_time is 3 days before anomaly_start_time
    display_start_time = (datetime.strptime(anomaly_start_time, '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')
    display_end_time = anomaly_end_time
    save_dir = path_manager.get_collected_case_path(case_id)

    print(f"Display time range: {display_start_time} - {display_end_time}")
    print(f"Anomaly time range: {anomaly_start_time} - {anomaly_end_time}")
    print(f"Data save path: {save_dir}")

    # Data preparation + anomaly detection
    if collect_data:
        # Verify data file exists (or merge from raw_data)
        metric_data_path, summary_path = _ensure_metric_data_ready(
            case_id, app, app_group, save_dir=save_dir
        )

        if run_anomaly_detection:
            # Run anomaly detection on prepared data
            collector = IntegratedDataCollector()
            metric_first_20_anomaly_times_path, metric_top_20_anomalous_metrics_path = collector.detect_anomalies(
                metric_data_path=metric_data_path,
                summary_dir=summary_path,
                anomaly_start_time=anomaly_start_time,
                anomaly_end_time=anomaly_end_time
            )
            print(f"✓ Anomaly detection complete, results written to {summary_path}")
        else:
            # Read anomaly detection results from ANOMALY_DETECTION_READ_PATH
            read_app_path = path_manager.get_anomaly_detection_read_app_path(case_id, app, app_group)
            read_summary = os.path.join(read_app_path, "summary")
            metric_first_20_anomaly_times_path = f"{read_summary}/metrics_detection_results/first_20_anomaly_times.csv"
            metric_top_20_anomalous_metrics_path = f"{read_summary}/metrics_detection_results/top_20_anomalous_metrics.csv"
            print(f"✓ Skipping anomaly detection, reusing existing results from {read_summary}")
    else:
        # Read existing metric data from DATA_READ_PATH
        read_app_path = path_manager.get_data_read_app_path(case_id, app, app_group)
        metric_data_path = os.path.join(read_app_path, "metric", "all_metrics.csv")
        if run_anomaly_detection:
            # Reuse metric data, re-run anomaly detection, write results to COLLECTED_DATA_PATH
            collector = IntegratedDataCollector()
            save_app_path = path_manager.get_collected_app_path(case_id, app, app_group)
            summary_path = os.path.join(save_app_path, "summary")
            metric_first_20_anomaly_times_path, metric_top_20_anomalous_metrics_path = collector.detect_anomalies(
                metric_data_path=metric_data_path,
                summary_dir=summary_path,
                anomaly_start_time=anomaly_start_time,
                anomaly_end_time=anomaly_end_time
            )
            print(f"✓ Reusing metric data and re-running anomaly detection, results written to {summary_path}")
        else:
            # Reuse anomaly detection results
            read_app_path = path_manager.get_anomaly_detection_read_app_path(case_id, app, app_group)
            summary_path = os.path.join(read_app_path, "summary")
            metric_first_20_anomaly_times_path = f"{summary_path}/metrics_detection_results/first_20_anomaly_times.csv"
            metric_top_20_anomalous_metrics_path = f"{summary_path}/metrics_detection_results/top_20_anomalous_metrics.csv"
            print(f"✓ Skipping anomaly detection, reusing existing results from {summary_path}")

    # Build result output paths
    analysis_dir = Config.get_result_analysis_path(case_id, app, app_group)
    iteration_dir = Config.get_result_iteration_path(case_id, app, app_group)

    # Perform metric analysis
    given_root_cause = case_info['given_root_cause']
    hypothesis = case_info['hypothesis']
    suspected_component = case_info['suspected_component']
    metric_analysis_save_path = os.path.join(analysis_dir, "metric_analysis_result.txt")

    if run_metric_analysis:
        metric_analysis_agent = MetricAnalysisAgent()
        metric_analysis_result = metric_analysis_agent.analyze_case_with_layers(
            metric_order_csv_path=metric_first_20_anomaly_times_path,
            metric_feature_csv_path=metric_top_20_anomalous_metrics_path,
            save_path=metric_analysis_save_path
        )
    else:
        read_metric_dir = path_manager.get_metric_analysis_read_path(case_id, app, app_group)
        read_metric_analysis_path = os.path.join(read_metric_dir, "metric_analysis_result.txt")
        if os.path.exists(read_metric_analysis_path):
            metric_analysis_save_path = read_metric_analysis_path
            metric_analysis_result = None
            print(f"✓ Skipping metric analysis, reusing existing results: {read_metric_analysis_path}")
        else:
            print(f"⚠ Metric analysis results not found at {read_metric_analysis_path}, running metric analysis")
            metric_analysis_agent = MetricAnalysisAgent()
            metric_analysis_result = metric_analysis_agent.analyze_case_with_layers(
                metric_order_csv_path=metric_first_20_anomaly_times_path,
                metric_feature_csv_path=metric_top_20_anomalous_metrics_path,
                save_path=metric_analysis_save_path
            )

    # Perform causal analysis
    if run_causal_analysis:
        causal_analysis = case_causal_relations_process(base_collected_data_path=Config.DATA_READ_PATH, case_id=case_id, item_index=item_index)
        with open(os.path.join(analysis_dir, "causal_result.json"), "w", encoding="utf-8") as f:
            json.dump(causal_analysis, f, ensure_ascii=False, indent=2)
    else:
        read_analysis_dir = path_manager.get_analysis_read_path(case_id, app, app_group)
        causal_json_file = os.path.join(read_analysis_dir, "causal_result.json")
        influence_stats_file = os.path.join(read_analysis_dir, "influence_statistics.txt")
        if os.path.exists(causal_json_file):
            with open(causal_json_file, "r", encoding="utf-8") as f:
                causal_analysis = json.load(f)
            print(f"✓ Skipping causal analysis, reusing existing results: {causal_json_file}")
        elif os.path.exists(influence_stats_file):
            with open(influence_stats_file, "r", encoding="utf-8") as f:
                causal_analysis = f.read()
            print(f"✓ Skipping causal analysis, reusing existing results: {influence_stats_file}")
        else:
            print(f"⚠ Causal analysis results not found, running causal analysis")
            causal_analysis = case_causal_relations_process(base_collected_data_path=Config.DATA_READ_PATH, case_id=case_id, item_index=item_index)

    # Perform case analysis
    case_analysis_save_path = os.path.join(analysis_dir, "case_analysis_result.txt")
    analysis_agent = AnalysisAgent()
    case_analysis_result = analysis_agent.analyze_case(
        given_root_cause, hypothesis, suspected_component,
        metric_analysis_save_path, causal_analysis,
        case_analysis_save_path,
        severity_filter,
        enable_iteration=enable_iteration,
        case_id=case_id,
        app=app,
        app_group=app_group,
        iteration_save_dir=iteration_dir,
        use_causal_analysis=use_causal_analysis
    )

    result = {
        'metric_analysis_result': metric_analysis_result,
        'case_analysis_result': case_analysis_result,
        'metric_analysis_path': metric_analysis_save_path,
        'case_analysis_path': case_analysis_save_path,
        'summary_path': summary_path
    }

    if enable_iteration and isinstance(case_analysis_result, dict):
        result['iteration_result'] = case_analysis_result.get('iteration_result')
        result['iteration_enabled'] = True

    return result


def inference_single_case(case_id, item_index, collect_data=True, severity_filter='Medium', enable_iteration=False,
                          run_anomaly_detection=True,
                          run_metric_analysis=True,
                          run_causal_analysis=True,
                          use_causal_analysis=True):
    """
    Perform inference analysis on a single case (supports independent data preparation)

    Args:
        case_id: case ID
        item_index: application index
        collect_data: whether to prepare data (True validates data file existence under COLLECTED_DATA_PATH,
                      merges from raw_data/ if available, but does not perform remote data pulling)
        severity_filter: severity filter level
        enable_iteration: whether to enable iterative refinement (default False)
        run_anomaly_detection: whether to run anomaly detection (False to reuse existing results from ANOMALY_DETECTION_READ_PATH, default True)
        run_metric_analysis: whether to run metric analysis (False to reuse existing results from METRIC_ANALYSIS_READ_PATH, default True)
        run_causal_analysis: whether to run causal analysis (False to reuse existing results from ANALYSIS_READ_PATH, default True)
        use_causal_analysis: whether to use causal analysis information (default True).
            False uses prompt template without causal information.

    Returns:
        Inference result (str or dict, depending on whether iteration is enabled)
    """
    case_info = case_table[case_id]
    app = case_info['app_name'][item_index]
    app_group = case_info['app_groups'][item_index][0]

    summary_path = os.path.join(path_manager.get_data_read_app_path(case_id, app, app_group), "summary")
    analysis_dir = Config.get_result_analysis_path(case_id, app, app_group)
    iteration_dir = Config.get_result_iteration_path(case_id, app, app_group)
    metric_analysis_save_path = os.path.join(analysis_dir, "metric_analysis_result.txt")

    anomaly_start_time = case_info['fault_start']
    anomaly_end_time = case_info['fault_end']

    # display_start_time is 3 days before anomaly_start_time
    display_start_time = (datetime.strptime(anomaly_start_time, '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')
    display_end_time = anomaly_end_time

    if collect_data:
        print(f"Validating data files...")
        save_dir = path_manager.get_collected_case_path(case_id)

        metric_data_path, summary_path = _ensure_metric_data_ready(
            case_id, app, app_group, save_dir=save_dir
        )

        if run_anomaly_detection:
            collector = IntegratedDataCollector()
            metric_first_20_anomaly_times_path, metric_top_20_anomalous_metrics_path = collector.detect_anomalies(
                metric_data_path=metric_data_path,
                summary_dir=summary_path,
                anomaly_start_time=anomaly_start_time,
                anomaly_end_time=anomaly_end_time
            )
            print(f"✓ Anomaly detection complete, results written to {summary_path}")
        else:
            read_app_path = path_manager.get_anomaly_detection_read_app_path(case_id, app, app_group)
            read_summary = os.path.join(read_app_path, "summary")
            metric_first_20_anomaly_times_path = f"{read_summary}/metrics_detection_results/first_20_anomaly_times.csv"
            metric_top_20_anomalous_metrics_path = f"{read_summary}/metrics_detection_results/top_20_anomalous_metrics.csv"
            print(f"✓ Skipping anomaly detection, reusing existing results from {read_summary}")

        print("✓ Data validation complete")
    else:
        read_app_path = path_manager.get_data_read_app_path(case_id, app, app_group)
        metric_data_path = os.path.join(read_app_path, "metric", "all_metrics.csv")
        if run_anomaly_detection:
            collector = IntegratedDataCollector()
            save_app_path = path_manager.get_collected_app_path(case_id, app, app_group)
            summary_path = os.path.join(save_app_path, "summary")
            metric_first_20_anomaly_times_path, metric_top_20_anomalous_metrics_path = collector.detect_anomalies(
                metric_data_path=metric_data_path,
                summary_dir=summary_path,
                anomaly_start_time=anomaly_start_time,
                anomaly_end_time=anomaly_end_time
            )
            print(f"✓ Reusing metric data and re-running anomaly detection, results written to {summary_path}")
        else:
            read_app_path = path_manager.get_anomaly_detection_read_app_path(case_id, app, app_group)
            summary_path = os.path.join(read_app_path, "summary")
            metric_first_20_anomaly_times_path = f"{summary_path}/metrics_detection_results/first_20_anomaly_times.csv"
            metric_top_20_anomalous_metrics_path = f"{summary_path}/metrics_detection_results/top_20_anomalous_metrics.csv"
            print(f"✓ Skipping anomaly detection, reusing existing results from {summary_path}")

    # Perform metric analysis (required for inference)
    if run_metric_analysis:
        metric_analysis_agent = MetricAnalysisAgent()
        metric_analysis_agent.analyze_case_with_layers(
            metric_order_csv_path=metric_first_20_anomaly_times_path,
            metric_feature_csv_path=metric_top_20_anomalous_metrics_path,
            save_path=metric_analysis_save_path
        )
    else:
        read_metric_dir = path_manager.get_metric_analysis_read_path(case_id, app, app_group)
        read_metric_analysis_path = os.path.join(read_metric_dir, "metric_analysis_result.txt")
        if os.path.exists(read_metric_analysis_path):
            metric_analysis_save_path = read_metric_analysis_path
            print(f"✓ Skipping metric analysis, reusing existing results: {read_metric_analysis_path}")
        else:
            print(f"⚠ Metric analysis results not found at {read_metric_analysis_path}, running metric analysis")
            metric_analysis_agent = MetricAnalysisAgent()
            metric_analysis_agent.analyze_case_with_layers(
                metric_order_csv_path=metric_first_20_anomaly_times_path,
                metric_feature_csv_path=metric_top_20_anomalous_metrics_path,
                save_path=metric_analysis_save_path
            )

    # Perform causal graph construction
    if run_causal_analysis:
        causal_analysis = case_causal_relations_process(base_collected_data_path=Config.DATA_READ_PATH, case_id=case_id, item_index=item_index)
        with open(os.path.join(analysis_dir, "causal_result.json"), "w", encoding="utf-8") as f:
            json.dump(causal_analysis, f, ensure_ascii=False, indent=2)
    else:
        read_analysis_dir = path_manager.get_analysis_read_path(case_id, app, app_group)
        causal_json_file = os.path.join(read_analysis_dir, "causal_result.json")
        influence_stats_file = os.path.join(read_analysis_dir, "influence_statistics.txt")
        if os.path.exists(causal_json_file):
            with open(causal_json_file, "r", encoding="utf-8") as f:
                causal_analysis = json.load(f)
            print(f"✓ Skipping causal analysis, reusing existing results: {causal_json_file}")
        elif os.path.exists(influence_stats_file):
            with open(influence_stats_file, "r", encoding="utf-8") as f:
                causal_analysis = f.read()
            print(f"✓ Skipping causal analysis, reusing existing results: {influence_stats_file}")
        else:
            print(f"⚠ Causal analysis results not found, running causal analysis")
            causal_analysis = case_causal_relations_process(base_collected_data_path=Config.DATA_READ_PATH, case_id=case_id, item_index=item_index)

    # Execute inference analysis
    inference_agent = InferenceAgent()
    inference_case_save_path = os.path.join(analysis_dir, "inference_case_result.txt")
    inference_result = inference_agent.inference_case(
        metric_analysis_save_path,
        causal_analysis,
        inference_case_save_path,
        severity_filter,
        enable_iteration=enable_iteration,
        case_id=case_id,
        app=app,
        app_group=app_group,
        iteration_save_dir=iteration_dir,
        use_causal_analysis=use_causal_analysis
    )

    return inference_result


# Usage example
if __name__ == "__main__":
    case_ids = ['risk12', 'risk13', 'risk14', 'risk15', 'risk16', 'risk17', 'risk18', 'risk19', 'risk20', 'risk21']

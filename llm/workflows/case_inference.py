#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference analysis workflow module

Handles risk cases without known root causes, performing data collection,
metric analysis, causal analysis, and inference prediction.
"""

from datetime import datetime
import os
import pandas as pd

from data_process.configs.cases import case_table
from config.config import Config
from llm.workflows.data_collector import collect_case_data_and_detection
from llm.agent.MetricAnalysisAgent import MetricAnalysisAgent
from llm.agent.InferenceAgent import InferenceAgent
from data_process.analysis.causal_analysis import case_causal_relations_process


def inference_single_case(case_id, item_index, collect_data=True,detection_data=True,
                         severity_filter='Medium', enable_iteration=False):
    """
    Perform inference analysis on a single case (can independently collect data)

    Args:
        case_id: Case ID
        item_index: Application index
        collect_data: Whether to collect data (True triggers data collection and metric analysis first)
        severity_filter: Severity filter level
        enable_iteration: Whether to enable iterative refinement mechanism (default False)

    Returns:
        Inference result (str or dict, depending on whether iteration is enabled)
    """
    case_info = case_table[case_id]
    app = case_info['app_name'][item_index]
    app_group = case_info['app_groups'][item_index][0]

    anomaly_start_time = case_info['fault_start']
    anomaly_end_time = case_info['fault_end']

    # display_start_time is 3 days before anomaly_start_time
    display_start_time = (datetime.strptime(anomaly_start_time, '%Y-%m-%d %H:%M:%S')
                         - pd.Timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')
    display_end_time = anomaly_end_time

    # Use result write path
    save_dir = Config.get_result_write_case_path(case_id)

    print(f"Display time range: {display_start_time} - {display_end_time}")
    print(f"Anomaly time range: {anomaly_start_time} - {anomaly_end_time}")
    print(f"Result save path: {save_dir}")

    # Collect data
    dashboard_ids = None  # Use default collection method when set to None

    _, metric_first_20_anomaly_times_path, \
    metric_top_20_anomalous_metrics_path, \
    _, _, summary_path = collect_case_data_and_detection(
            app=app, app_group=app_group, display_start_time=display_start_time, display_end_time=display_end_time,
            anomaly_start_time=anomaly_start_time, anomaly_end_time=anomaly_end_time, case_id=case_id, save_dir=save_dir,
            collect_data=collect_data, detection_data=detection_data,
            plugin_ids=None, dashboard_ids=dashboard_ids
        )

    print("✓ Data collection completed")
    analysis_dir = Config.get_result_analysis_path(case_id, app, app_group)
    iteration_dir = Config.get_result_iteration_path(case_id, app, app_group)
    metric_analysis_save_path = os.path.join(analysis_dir, "metric_analysis_result.txt")

    # Perform metric analysis (required for inference)
    metric_analysis_agent = MetricAnalysisAgent()
    metric_analysis_agent.analyze_case_with_layers(
        metric_order_csv_path=metric_first_20_anomaly_times_path,
        metric_feature_csv_path=metric_top_20_anomalous_metrics_path,
        save_path=metric_analysis_save_path
    )

    # Build causal graph (read raw data from data read path)
    causal_analysis = case_causal_relations_process(
        base_collected_data_path=Config.DATA_READ_PATH,
        case_id=case_id,
        item_index=item_index
    )

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
        iteration_save_dir=iteration_dir
    )

    return inference_result

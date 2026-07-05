#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data collection workflow module

Encapsulates the complete data collection pipeline, including metric data
validation and anomaly detection.

Open-source version notes:
Remote data fetching has been removed. When collect_data=True, it validates
that local data files exist; if raw_data/ is present, it performs merge and
cleanup without remote fetching.
"""
import os
from data_process.pipeline import IntegratedDataCollector

from config.config import Config


def collect_case_data_and_detection(app, app_group, display_start_time, display_end_time,
                     anomaly_start_time, anomaly_end_time, case_id, save_dir, collect_data=True, detection_data=True,
                     plugin_ids=None, dashboard_ids=None):
    """
    Execute the complete data collection and anomaly detection pipeline

    Open-source version notes:
    When collect_data=True, validates that data files exist under COLLECTED_DATA_PATH;
    if raw_data/ is present, performs merge and cleanup without remote fetching.
    plugin_ids and dashboard_ids parameters are deprecated, kept for backward compatibility.

    Args:
        app: Application name
        app_group: Application group name
        display_start_time: Display start time, format: 'YYYY-MM-DD HH:MM:SS'
        display_end_time: Display end time, format: 'YYYY-MM-DD HH:MM:SS'
        anomaly_start_time: Anomaly start time, format: 'YYYY-MM-DD HH:MM:SS'
        anomaly_end_time: Anomaly end time, format: 'YYYY-MM-DD HH:MM:SS'
        case_id: Case ID
        save_dir: Base save directory
        collect_data: Whether to validate data files exist (True checks and optionally merges raw_data, False reads from DATA_READ_PATH)
        detection_data: Whether to execute anomaly detection
        plugin_ids: (Deprecated, kept for backward compatibility)
        dashboard_ids: (Deprecated, kept for backward compatibility)

    Returns:
        tuple: (results, metric_first_20_anomaly_times_path,
                metric_top_20_anomalous_metrics_path, summary_path)
    """
    collector = IntegratedDataCollector()

    # Step 1: Data preparation
    if collect_data:
        results, metric_data_path, summary_dir = collector.collect_all_data(
            app=app,
            app_group=app_group,
            case_id=case_id,
            display_start_time=display_start_time,
            display_end_time=display_end_time,
            anomaly_start_time=anomaly_start_time,
            anomaly_end_time=anomaly_end_time,
        )
        print(f"\nCollection summary:")
        print(f"Metric data: {'✓' if results['collection_success']['metrics_data'] else '✗'}")

    else:
        paths = collector.create_directory_structure(case_id=case_id, app=app, app_group=app_group)
        metric_data_path = os.path.join(paths['metric'], "all_metrics.csv")
        summary_dir = paths['summary']
        results = {
            'collection_success': {
                'metrics_data': True
            }
        }

    # Step 2: Execute anomaly detection
    if detection_data:
        print("\nStarting anomaly detection...")
        metric_first_20_anomaly_times_path, metric_top_20_anomalous_metrics_path = collector.detect_anomalies(
            metric_data_path=metric_data_path,
            summary_dir=summary_dir,
            anomaly_start_time=anomaly_start_time,
            anomaly_end_time=anomaly_end_time
        )
    else:
        print("\nSkipping anomaly detection step...")
        metric_first_20_anomaly_times_path = os.path.join(summary_dir, "metrics_detection_results/first_20_anomaly_times.csv")
        metric_top_20_anomalous_metrics_path = os.path.join(summary_dir, "metrics_detection_results/top_20_anomalous_metrics.csv")
        print(f"Using existing data paths:")
        print(f"Metric anomaly time path: {metric_first_20_anomaly_times_path}")

    return (results, metric_first_20_anomaly_times_path,
            metric_top_20_anomalous_metrics_path, summary_dir)

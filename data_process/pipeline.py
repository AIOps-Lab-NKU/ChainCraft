#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import traceback
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from data_handle.data_merge_and_clean import DataMergeAndClean
from data_process.utils.visualizer import DataFrameMetricsVisualizer, MetricsVisualizationTool
import sys
from config.config import Config
from data_process.analysis.detection_prophet import detection


class IntegratedDataCollector:
    """
    Integrated data collector class for managing application metric data

    Open source version note:
    Data collection functionality (Sunfire remote pulling) has been removed, replaced by local file reading.
    Users need to pre-place metric data in the correct directory structure.
    """

    def __init__(self, max_workers: int = 10):
        """
        Initialize integrated data collector

        Args:
            max_workers: max thread pool size (parameter retained for backward compatibility)
        """
        self._max_workers = max_workers

    def create_directory_structure(self, case_id=None, app: str = '',
                                    app_group: str = '',
                                    save_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Create directory structure for saving data

        Supports two modes:
        - Via case_id + Config path (used by data_process workflow)
        - Via save_dir directly specifying base directory (data_handle compatible mode)

        Args:
            case_id: case ID, used with Config path
            app: application name
            app_group: application group name
            save_dir: base save directory, when provided uses Path-based path construction (compatible with data_handle)

        Returns:
            Dict[str, str]: dictionary containing various data save paths
        """
        if save_dir is not None:
            base_path = Path(save_dir)
            app_path = base_path / f"{app}_{app_group}"
            paths = {
                'base': str(app_path),
                'metric': str(app_path / 'metric'),
                'summary': str(app_path / 'summary')
            }
        else:
            data_base_path = Config.get_data_read_app_path(case_id, app, app_group)
            result_base_path = Config.get_result_write_app_path(case_id, app, app_group)
            paths = {
                'base': data_base_path,
                'metric': os.path.join(data_base_path, 'metric'),
                'summary': os.path.join(result_base_path, 'summary')
            }

        for path in paths.values():
            os.makedirs(path, exist_ok=True)

        return paths

    def datetime_to_timestamp(self, datetime_str: str) -> int:
        """Convert time string to millisecond timestamp"""
        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        return int(dt.timestamp() * 1000)

    def collect_all_data(self, app: str, app_group: str, case_id: str = None,
                         display_start_time: str = '', display_end_time: str = '',
                         anomaly_start_time: str = '', anomaly_end_time: str = '',
                         plugin_ids: List[str] = None,
                         dashboard_ids: List[str] = None,
                         save_dir: Optional[str] = None) -> tuple:
        """
        Execute data preparation flow (excluding anomaly detection)

        Open source version note:
        Remote data pulling has been removed. This method checks whether all_metrics.csv
        exists under the target path. If not, it attempts to merge from the raw_data/
        subdirectory (using DataMergeAndClean). If raw_data/ also does not exist,
        a FileNotFoundError is raised.

        Args:
            app: application name
            app_group: application group name
            case_id: case ID
            display_start_time: display start time (retained for backward compatibility)
            display_end_time: display end time (retained for backward compatibility)
            anomaly_start_time: anomaly start time (retained for backward compatibility)
            anomaly_end_time: anomaly end time (retained for backward compatibility)
            plugin_ids: (deprecated, parameter retained for backward compatibility)
            dashboard_ids: (deprecated, parameter retained for backward compatibility)
            save_dir: base save directory (compatible with data_handle)

        Returns:
            tuple: (results, metric_data_path, summary_dir)

        Raises:
            FileNotFoundError: when both all_metrics.csv and raw_data/ do not exist
        """
        print("\n" + "=" * 80)
        print("Integrated Data Collector - Checking Local Data")
        print("=" * 80)
        print(f"Application: {app}")
        print(f"Application group: {app_group}")
        if save_dir:
            print(f"Data directory: {save_dir}")
        print("=" * 80)

        paths = self.create_directory_structure(
            case_id=case_id, app=app, app_group=app_group, save_dir=save_dir
        )
        print(f"✓ Directory structure: {paths['base']}")

        all_metrics_output_path = os.path.join(paths['metric'], "all_metrics.csv")
        raw_data_dir = os.path.join(paths['metric'], "raw_data")
        metrics_ready = False

        if os.path.exists(all_metrics_output_path):
            print(f"✓ Metric data file already exists: {all_metrics_output_path}")
            metrics_ready = True
        elif os.path.isdir(raw_data_dir) and os.listdir(raw_data_dir):
            print(f"✓ all_metrics.csv does not exist, merging from raw_data/ directory...")
            processor = DataMergeAndClean(missing_threshold=0.2, verbose=True)
            processor.process_directory(raw_data_dir, all_metrics_output_path)
            if os.path.exists(all_metrics_output_path):
                print(f"✓ Metric data merge complete: {all_metrics_output_path}")
                metrics_ready = True
            else:
                raise FileNotFoundError(
                    f"all_metrics.csv still not generated after merging from raw_data/, "
                    f"please check raw_data/ directory contents: {raw_data_dir}"
                )
        else:
            raise FileNotFoundError(
                f"Metric data file does not exist: {all_metrics_output_path}, "
                f"and raw_data/ directory is empty or does not exist: {raw_data_dir}.\n"
                f"Please place metric data in one of the following paths:\n"
                f"  1. Direct placement: {all_metrics_output_path}\n"
                f"  2. Multiple CSVs in: {raw_data_dir}/ (will be auto-merged)"
            )

        summary_dir = paths['summary']
        os.makedirs(summary_dir, exist_ok=True)

        results = {
            'app': app,
            'app_group': app_group,
            'display_time_range': {'start': display_start_time, 'end': display_end_time},
            'anomaly_time_range': {'start': anomaly_start_time, 'end': anomaly_end_time},
            'paths': paths,
            'metrics_results': {},
            'collection_success': {'metrics_data': metrics_ready}
        }

        print("\n" + "=" * 80)
        print("Data preparation complete")
        print("=" * 80)
        print(f"✓ Metric data: {'Ready' if metrics_ready else 'Not found'}")
        print(f"  - Metric data: {paths['metric']}")
        print(f"  - Summary report: {paths['summary']}")
        print("=" * 80)

        return results, all_metrics_output_path, summary_dir

    def detect_anomalies(self, metric_data_path: str,
                         summary_dir: str,
                         anomaly_start_time: str,
                         anomaly_end_time: str) -> tuple:
        """
        Perform anomaly detection on collected data

        Args:
            metric_data_path: metric data CSV path
            summary_dir: result save directory
            anomaly_start_time: anomaly start time
            anomaly_end_time: anomaly end time

        Returns:
            tuple: (metric_first_20_anomaly_times_path,
                    metric_top_20_anomalous_metrics_path)
        """
        print("\n" + "=" * 80)
        print("Starting anomaly detection")
        print("=" * 80)

        metrics_detection_output_dir = os.path.join(summary_dir, "metrics_detection_results")
        os.makedirs(metrics_detection_output_dir, exist_ok=True)

        print(f"\n=== Detecting metric data ===")
        print(f"Data path: {metric_data_path}")
        print(f"Results saved to: {metrics_detection_output_dir}")

        metric_first_20_anomaly_times_path, metric_top_20_anomalous_metrics_path = detection(
            metric_data_path, metrics_detection_output_dir,
            anomaly_start_time, anomaly_end_time,
            use_parallel=False
        )

        print("\n" + "=" * 80)
        print("Anomaly detection complete")
        print("=" * 80)
        print(f"✓ Metric detection results:")
        print(f"  - {metric_first_20_anomaly_times_path}")
        print(f"  - {metric_top_20_anomalous_metrics_path}")
        print("=" * 80)

        return (metric_first_20_anomaly_times_path,
                metric_top_20_anomalous_metrics_path)

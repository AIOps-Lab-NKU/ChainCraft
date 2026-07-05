#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import glob
from typing import List, Tuple
import math
from config import Config

# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

class MetricsVisualizationTool:
    """
    Monitoring Metric Visualization Tool - Simplified Version
    """
    
    def __init__(self, data_dir: str = "./output", output_dir: str = "./charts_simple",
                 fault_start_time: str = None, fault_end_time: str = None, 
                 detection_start_time: str = None, detection_end_time: str = None, 
                 metric_name: List[str] = None):
        """
        Initialize tool

        Args:
            data_dir: data directory
            output_dir: chart output directory
            fault_start_time: fault start time (format: 'YYYY-MM-DD HH:MM:SS')
            fault_end_time: fault end time (format: 'YYYY-MM-DD HH:MM:SS')
            detection_start_time: anomaly detection start time (format: 'YYYY-MM-DD HH:MM:SS')
            detection_end_time: anomaly detection end time (format: 'YYYY-MM-DD HH:MM:SS')
            metric_name: metric name filter list; if None, all metrics are shown; if a list is provided, only metrics containing any substring in the list are displayed
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_metrics_per_file = 5  # Max number of metrics per file
        self.charts_per_row = 5        # Number of charts per row
        self.max_rows_per_figure = 6   # Max rows per figure
        self.metric_name = metric_name  # Metric name filter list

        # Fault time period
        self.fault_start_time = None
        self.fault_end_time = None
        if fault_start_time and fault_end_time:
            try:
                fault_start_naive = pd.to_datetime(fault_start_time)
                fault_end_naive = pd.to_datetime(fault_end_time)

                self.fault_start_time = fault_start_naive
                self.fault_end_time = fault_end_naive

                print(f"Fault period (Beijing Time): {self.fault_start_time} to {self.fault_end_time}")
            except Exception as e:
                print(f"Error parsing fault time: {e}")

        # Anomaly detection time period
        self.detection_start_time = None
        self.detection_end_time = None
        if detection_start_time and detection_end_time:
            try:
                detection_start_naive = pd.to_datetime(detection_start_time)
                detection_end_naive = pd.to_datetime(detection_end_time)

                self.detection_start_time = detection_start_naive
                self.detection_end_time = detection_end_naive

                print(f"Detection period (Beijing Time): {self.detection_start_time} to {self.detection_end_time}")
            except Exception as e:
                print(f"Error parsing detection time: {e}")
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def should_include_file(self, filename: str) -> bool:
        """
        Determine whether a file should be included
        
        Args:
            filename: filename
            
        Returns:
            bool: whether this file should be included
        """
        if self.metric_name is None:
            return True
        
        # Extract metric name corresponding to the file (only the name part, excluding displayName)
        metric_name = self.extract_metric_name_from_filename(filename)
        
        # Check if any filter keyword is a substring of the metric name
        for filter_name in self.metric_name:
            if filter_name.lower() in metric_name.lower():
                return True
        
        return False
    
    def load_csv_files(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        Load all CSV files
        
        Returns:
            List[Tuple[str, pd.DataFrame]]: list of (filename, DataFrame) tuples
        """
        csv_files = []
        pattern = os.path.join(self.data_dir, "*.csv")
        
        for file_path in glob.glob(pattern):
            try:
                # Skip summary files
                if "summary" in os.path.basename(file_path):
                    continue
                
                file_name = os.path.basename(file_path)
                
                # Check whether this file should be included
                if not self.should_include_file(file_name):
                    # print(f"Skipped: {file_name} (filtered by metric_name)")
                    continue
                    
                df = pd.read_csv(file_path)
                if not df.empty and 'timestamp' in df.columns:
                    # Convert timestamp to Beijing time (UTC+8)
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
                    
                    csv_files.append((file_name, df))
                    print(f"Loaded: {file_name}, Shape: {df.shape}")
                else:
                    print(f"Skipped: {file_path} (empty or no timestamp)")
                    
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
        
        return csv_files
    
    def extract_metric_name_from_filename(self, filename: str) -> str:
        """
        Extract plugin ID or machine metric name from filename as title
        
        Args:
            filename: filename
            
        Returns:
            str: plugin ID or machine metric name
        """
        # Remove .csv suffix
        name = filename.replace('.csv', '')
        
        # Split filename
        parts = name.split('_')
        
        # Check if it's a machine metric
        if 'machine_metric' in name and len(parts) >= 4:
            # Machine metric format: alsc-pos-order_machine_metric_name&displayName
            # Extract all parts from the third part onward as metric name
            metric_part = '_'.join(parts[3:])
            # If contains & symbol, only take the name part (before &)
            if '&' in metric_part:
                metric_name = metric_part.split('&')[0]
            else:
                metric_name = metric_part
            return metric_name
        elif len(parts) >= 4:
            # Plugin metric format: app_plugin_id_description
            # Example: alsc-pos-order_1030_MM_1909_hsf service exception - pre-release
            # Extract plugin ID: combination of second, third, fourth parts (1030_MM_1909)
            plugin_id = '_'.join(parts[1:4])
            return plugin_id
        elif len(parts) >= 3:
            # If only 3 parts, take the last two
            plugin_id = '_'.join(parts[1:3])
            return plugin_id
        elif len(parts) >= 2:
            # If only 2 parts, take the second part
            return parts[1]
        
        # If format doesn't match, return original filename (limited length)
        return filename[:30]
    
    def select_top_metrics(self, df: pd.DataFrame) -> List[str]:
        """
        Select top N metric columns (based on data volume or variance)
        
        Args:
            df: DataFrame
            
        Returns:
            List[str]: selected metric column names
        """
        # Get numeric columns excluding timestamp and datetime
        metric_columns = [col for col in df.columns 
                         if col not in ['timestamp', 'datetime'] and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(metric_columns) <= self.max_metrics_per_file:
            return metric_columns
        
        # Calculate variance for each metric, select top N with highest variance
        variances = {}
        for col in metric_columns:
            try:
                values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) > 0:
                    variances[col] = values.var()
                else:
                    variances[col] = 0
            except:
                variances[col] = 0
        
        # Sort by variance, select top N
        sorted_metrics = sorted(variances.items(), key=lambda x: x[1], reverse=True)
        selected_metrics = [metric[0] for metric in sorted_metrics[:self.max_metrics_per_file]]
        
        # print(f"Selected metrics: {len(selected_metrics)}")
        return selected_metrics
    
    def create_subplot(self, ax, df: pd.DataFrame, selected_metrics: List[str], title: str):
        """
        Create subplot for a single metric
        
        Args:
            ax: matplotlib axes object
            df: DataFrame
            selected_metrics: selected metric list
            title: chart title
        """
        try:
            if not selected_metrics:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=8, pad=2)
                return
            
            # If only one metric, plot directly
            if len(selected_metrics) == 1:
                metric = selected_metrics[0]
                # Ensure data is valid and explicitly convert to numpy array
                valid_data = df[['datetime', metric]].dropna()
                
                if len(valid_data) == 0:
                    ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title, fontsize=8, pad=2)
                    return
                
                # Ensure data is numpy array
                x_data = valid_data['datetime'].values
                y_data = valid_data[metric].replace([np.inf, -np.inf], np.nan).fillna(0).values
                
                ax.plot(x_data, y_data, linewidth=1, alpha=0.8, color='blue')
            else:
                # Multiple metrics: plot normalized data
                valid_metrics = []
                for metric in selected_metrics:
                    # Check if each metric has valid data
                    if df[metric].notna().any() and not np.isinf(df[metric]).all():
                        valid_metrics.append(metric)
                
                if not valid_metrics:
                    ax.text(0.5, 0.5, 'No Valid Metrics', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title, fontsize=8, pad=2)
                    return
                
                valid_data = df[['datetime'] + valid_metrics].copy()
                
                if len(valid_data) == 0:
                    ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title, fontsize=8, pad=2)
                    return
                
                colors = ['blue', 'red', 'green', 'orange', 'purple']
                for i, metric in enumerate(valid_metrics):
                    try:
                        # Handle invalid values and convert to numpy array
                        values = valid_data[metric].replace([np.inf, -np.inf], np.nan)
                        
                        # Skip if all NaN
                        if values.isna().all():
                            continue
                            
                        # Fill NaN to avoid plotting issues
                        values = values.fillna(method='ffill').fillna(method='bfill').fillna(0)
                        
                        # Only normalize if there are enough non-zero values
                        if values.std() > 0 and (values != 0).sum() > len(values) * 0.1:
                            normalized_values = (values - values.mean()) / values.std()
                        else:
                            normalized_values = values
                        
                        # Ensure data is numpy array
                        x_data = valid_data['datetime'].values
                        y_data = normalized_values.values
                        
                        color = colors[i % len(colors)]
                        ax.plot(x_data, y_data, label=f'M{i+1}', linewidth=1, alpha=0.8, color=color)
                    except Exception as e:
                        print(f"Error plotting metric {metric}: {e}")
                
                if any(ax.get_lines()):  # Only add legend when at least one line exists
                    ax.legend(fontsize=5, loc='upper right')
            
            # Add anomaly detection period and fault period background colors
            if not df.empty and 'datetime' in df.columns:
                data_start = df['datetime'].min()
                data_end = df['datetime'].max()

                #print(f"Debug - Data time range: {data_start} to {data_end}")

                # Add anomaly detection period (detection window 1 hour before fault)
                if self.detection_start_time and self.detection_end_time:
                    detection_start_display = max(self.detection_start_time, data_start)
                    detection_end_display = min(self.detection_end_time, data_end)

                    if detection_start_display <= detection_end_display:
                        ax.axvspan(detection_start_display, detection_end_display,
                                 alpha=0.5, color='blue', zorder=0, label='Anomaly detection period')
                        #print(f"Debug - Added detection period background for {title}")
                    else:
                        print(f"Debug - No overlap for detection period in {title}")

                # Add fault period (actual fault occurrence time)
                if self.fault_start_time and self.fault_end_time:
                    # Ensure fault period doesn't overlap with anomaly detection period
                    fault_start_display = max(self.fault_start_time, data_start)
                    fault_end_display = min(self.fault_end_time, data_end)

                    # If fault start time is same or close to detection end time, slightly offset to avoid overlap
                    # Add a very small time interval to prevent exact overlap
                    time_epsilon = pd.Timedelta(seconds=1)  # 1 second interval
                    if (self.detection_end_time and 
                        fault_start_display <= self.detection_end_time and
                        fault_end_display > self.detection_end_time):
                        # Start marking fault time after anomaly detection end time
                        fault_start_display = max(fault_start_display, self.detection_end_time + time_epsilon)
                    
                    if fault_start_display <= fault_end_display:
                        ax.axvspan(fault_start_display, fault_end_display,
                                 alpha=0.5, color='red', zorder=0, label='Fault occurrence period')
                        #print(f"Debug - Added fault background for {title}")
                    else:
                        print(f"Debug - No overlap for fault in {title}")

                # If both time periods exist, add legend
                if (self.detection_start_time and self.detection_end_time) or (self.fault_start_time and self.fault_end_time):
                    # Create custom legend
                    from matplotlib.patches import Patch
                    legend_elements = []
                    if self.detection_start_time and self.detection_end_time:
                        legend_elements.append(Patch(facecolor='blue', alpha=0.5, label='Anomaly detection period'))
                    if self.fault_start_time and self.fault_end_time:
                        legend_elements.append(Patch(facecolor='red', alpha=0.5, label='Fault occurrence period'))

                    if legend_elements:
                        ax.legend(handles=legend_elements, fontsize=6, loc='upper right')
            
            # Set title
            ax.set_title(title, fontsize=8, pad=2)

            # Remove x-axis numeric labels, keep only y-axis
            ax.tick_params(axis='x', labelbottom=False)  # Don't show x-axis labels
            ax.tick_params(axis='y', labelsize=6)

            # Add grid, but only show horizontal lines (remove vertical lines)
            ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
            
        except Exception as e:
            print(f"Error creating subplot for {title}: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=8, pad=2)
    
    def create_charts(self):
        """
        Create all charts
        """
        csv_files = self.load_csv_files()
        
        if not csv_files:
            print("No valid CSV files found")
            return
        
        print(f"Total files found: {len(csv_files)}")
        
        # Calculate number of figures needed
        total_charts = len(csv_files)
        charts_per_figure = self.charts_per_row * self.max_rows_per_figure
        num_figures = math.ceil(total_charts / charts_per_figure)
        
        print(f"Creating {num_figures} figures, max {charts_per_figure} charts per figure")
        
        for fig_idx in range(num_figures):
            start_idx = fig_idx * charts_per_figure
            end_idx = min(start_idx + charts_per_figure, total_charts)
            current_files = csv_files[start_idx:end_idx]
            
            print(f"Creating figure {fig_idx + 1}, files {start_idx + 1} to {end_idx}")
            
            self.create_single_figure(current_files, fig_idx + 1)
    
    def create_single_figure(self, csv_files: List[Tuple[str, pd.DataFrame]], fig_num: int):
        """
        Create a single figure
        
        Args:
            csv_files: list of files to include in current figure
            fig_num: figure number
        """
        num_charts = len(csv_files)
        num_rows = math.ceil(num_charts / self.charts_per_row)
        
        # Create figure
        fig_width = 20  # Increase figure width
        fig_height = max(12, num_rows * 2.5)  # Adjust height based on row count
        
        fig, axes = plt.subplots(num_rows, self.charts_per_row, 
                                figsize=(fig_width, fig_height))
        
        # Ensure axes is 2D array
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        elif self.charts_per_row == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Metrics Analysis Dashboard - Page {fig_num}', fontsize=16, y=0.98)
        
        # Plot each file's chart
        for idx, (filename, df) in enumerate(csv_files):
            row = idx // self.charts_per_row
            col = idx % self.charts_per_row
            ax = axes[row, col]
            
            # Extract metric name
            metric_name = self.extract_metric_name_from_filename(filename)
            
            # Select metrics to display
            selected_metrics = self.select_top_metrics(df)
            
            # Create subplot
            self.create_subplot(ax, df, selected_metrics, metric_name)
        
        # Hide empty subplots
        for idx in range(num_charts, num_rows * self.charts_per_row):
            row = idx // self.charts_per_row
            col = idx % self.charts_per_row
            axes[row, col].set_visible(False)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Save figure
        output_path = os.path.join(self.output_dir, f'metrics_dashboard_page_{fig_num}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved chart: {output_path}")
    
    def generate_summary_report(self):
        """
        Generate analysis report
        """
        csv_files = self.load_csv_files()
        
        report_path = os.path.join(self.output_dir, 'metrics_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Metrics Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Directory: {self.data_dir}\n")
            f.write(f"Total Files: {len(csv_files)}\n\n")
            
            f.write("File Details:\n")
            f.write("-" * 30 + "\n")
            
            for filename, df in csv_files:
                metric_name = self.extract_metric_name_from_filename(filename)
                selected_metrics = self.select_top_metrics(df)
                
                f.write(f"\nFile: {filename}\n")
                f.write(f"Metric Name: {metric_name}\n")
                f.write(f"Data Shape: {df.shape}\n")
                if not df.empty:
                    f.write(f"Time Range: {df['datetime'].min()} to {df['datetime'].max()}\n")
                f.write(f"Available Metrics: {len([col for col in df.columns if col not in ['timestamp', 'datetime']])}\n")
                f.write(f"Selected Metrics: {len(selected_metrics)}\n")
        
        print(f"Summary report saved: {report_path}")

class DataFrameMetricsVisualizer:
    """
    DataFrame Metric Visualization Class
    Specialized for visualizing DataFrame-format metric data
    """
    
    def __init__(self, output_dir: str = "./charts_df"):
        """
        Initialize visualizer
        
        Args:
            output_dir: chart output directory
        """
        self.output_dir = output_dir
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def visualize_metrics(self, df: pd.DataFrame,
                         display_start_time: str = None,
                         display_end_time: str = None,
                         anomaly_start_time: str = None,
                         anomaly_end_time: str = None,
                         detection_start_time: str = None,
                         detection_end_time: str = None,
                         subplot_rows: int = 4,
                         subplot_cols: int = 4,
                         output_filename: str = "metrics_dashboard.png"):
        """
        Visualize metrics in a DataFrame

        Args:
            df: pandas.DataFrame, first column is timestamp, other columns are metric data
            display_start_time: display start time (format: 'YYYY-MM-DD HH:MM:SS')
            display_end_time: display end time (format: 'YYYY-MM-DD HH:MM:SS')
            anomaly_start_time: anomaly start time (format: 'YYYY-MM-DD HH:MM:SS')
            anomaly_end_time: anomaly end time (format: 'YYYY-MM-DD HH:MM:SS')
            detection_start_time: anomaly detection start time (format: 'YYYY-MM-DD HH:MM:SS')
            detection_end_time: anomaly detection end time (format: 'YYYY-MM-DD HH:MM:SS')
            subplot_rows: subplot row count
            subplot_cols: subplot column count
            output_filename: output filename
        """
        if df is None or df.empty:
            print("Input DataFrame is empty")
            return
        
        try:
            # Process DataFrame
            print("display_start_time:", display_start_time)
            df_processed = self._preprocess_dataframe(df, display_start_time, display_end_time)
            
            if df_processed is None or df_processed.empty:
                print("Processed DataFrame is empty")
                return
            
            # Get metric columns
            metric_columns = self._get_metric_columns(df_processed)
            
            if not metric_columns:
                print("No valid metric columns found")
                return
            
            print(f"Found {len(metric_columns)} metric columns")
            
            # Parse anomaly periods and anomaly detection periods
            anomaly_start_dt, anomaly_end_dt, detection_start_dt, detection_end_dt = self._parse_anomaly_times(
                anomaly_start_time, anomaly_end_time, detection_start_time, detection_end_time
            )

            # Create visualization
            self._create_visualization(
                df_processed, metric_columns,
                anomaly_start_dt, anomaly_end_dt,
                detection_start_dt, detection_end_dt,
                subplot_rows, subplot_cols, output_filename
            )
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _preprocess_dataframe(self, df: pd.DataFrame, start_time: str = None, end_time: str = None):
        """
        Preprocess DataFrame
        
        Args:
            df: original DataFrame
            start_time: start time
            end_time: end time
            
        Returns:
            Processed DataFrame
        """
        try:
            df_copy = df.copy()
            
            # Determine time column name (first column should be timestamp)
            time_column = 'timestamp'
            print(f"Using time column: {time_column}")
            
            # Convert time column
            if pd.api.types.is_numeric_dtype(df_copy[time_column]):
                # If numeric, assume it's a timestamp (milliseconds or seconds)
                if df_copy[time_column].max() > 1e10:  # Millisecond timestamp
                    df_copy['datetime'] = pd.to_datetime(df_copy[time_column], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
                else:  # Second timestamp
                    df_copy['datetime'] = pd.to_datetime(df_copy[time_column], unit='s', utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            else:
                # If string, parse directly
                df_copy['datetime'] = pd.to_datetime(df_copy[time_column])
            
            # Set datetime as index
            df_copy = df_copy.set_index('datetime').sort_index()
            print(f"Data time range: {df_copy.index.min()} to {df_copy.index.max()}")
            # Time range filtering
            if start_time or end_time:
                if start_time:
                    start_dt = pd.to_datetime(start_time)
                    print(f"Filter start time: {start_dt}")
                    df_copy = df_copy[df_copy.index >= start_dt]
                
                if end_time:
                    end_dt = pd.to_datetime(end_time)
                    print(f"Filter end time: {end_dt}")
                    df_copy = df_copy[df_copy.index <= end_dt]
                
                print(f"Data range after time filter: {df_copy.index.min()} to {df_copy.index.max()}")
            
            return df_copy
            
        except Exception as e:
            print(f"DataFrame preprocessing failed: {e}")
            return None
    
    def _get_metric_columns(self, df: pd.DataFrame):
        """
        Get metric column names
        
        Args:
            df: DataFrame
            
        Returns:
            List of metric column names
        """
        # Exclude non-numeric columns, keep metric columns
        metric_columns = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check for valid data
                if not df[col].isna().all() and not np.isinf(df[col]).all():
                    metric_columns.append(col)
        
        return metric_columns
    
    def _parse_anomaly_times(self, anomaly_start_time: str, anomaly_end_time: str, 
                           detection_start_time: str, detection_end_time: str):
        """
        Parse anomaly period and anomaly detection period

        Args:
            anomaly_start_time: anomaly start time
            anomaly_end_time: anomaly end time
            detection_start_time: anomaly detection start time
            detection_end_time: anomaly detection end time

        Returns:
            (anomaly_start_dt, anomaly_end_dt, detection_start_dt, detection_end_dt) or (None, None, None, None)
        """
        try:
            # Parse anomaly period
            if anomaly_start_time and anomaly_end_time:
                anomaly_start_dt = pd.to_datetime(anomaly_start_time)
                anomaly_end_dt = pd.to_datetime(anomaly_end_time)
                print(f"Fault period: {anomaly_start_dt} to {anomaly_end_dt}")
            else:
                anomaly_start_dt, anomaly_end_dt = None, None
            
            # Parse anomaly detection period
            if detection_start_time and detection_end_time:
                detection_start_dt = pd.to_datetime(detection_start_time)
                detection_end_dt = pd.to_datetime(detection_end_time)
                print(f"Anomaly detection period: {detection_start_dt} to {detection_end_dt}")
            else:
                detection_start_dt, detection_end_dt = None, None

            return anomaly_start_dt, anomaly_end_dt, detection_start_dt, detection_end_dt
        except Exception as e:
            print(f"Failed to parse anomaly times: {e}")
            return None, None, None, None
    
    def _create_visualization(self, df: pd.DataFrame, metric_columns: list,
                            anomaly_start: pd.Timestamp, anomaly_end: pd.Timestamp,
                            detection_start: pd.Timestamp, detection_end: pd.Timestamp,
                            rows: int, cols: int, filename: str):
        """
        Create visualization charts

        Args:
            df: processed DataFrame
            metric_columns: list of metric column names
            anomaly_start: anomaly start time
            anomaly_end: anomaly end time
            detection_start: anomaly detection start time
            detection_end: anomaly detection end time
            rows: subplot row count
            cols: subplot column count
            filename: output filename
        """
        try:
            total_subplots = rows * cols
            num_pages = math.ceil(len(metric_columns) / total_subplots)

            print(f"Total {len(metric_columns)} metrics, requires {num_pages} pages of charts")

            for page in range(num_pages):
                start_idx = page * total_subplots
                end_idx = min(start_idx + total_subplots, len(metric_columns))
                page_metrics = metric_columns[start_idx:end_idx]

                page_filename = filename.replace('.png', f'_page_{page+1}.png')
                self._create_single_page(df, page_metrics, anomaly_start, anomaly_end, 
                                       detection_start, detection_end,
                                       rows, cols, page_filename, page+1)

        except Exception as e:
            print(f"Failed to create visualization: {e}")
    
    def _create_single_page(self, df: pd.DataFrame, metrics: list,
                           anomaly_start: pd.Timestamp, anomaly_end: pd.Timestamp,
                           detection_start: pd.Timestamp, detection_end: pd.Timestamp,
                           rows: int, cols: int, filename: str, page_num: int):
        """
        Create single page of charts

        Args:
            df: DataFrame
            metrics: metric list for current page
            anomaly_start: anomaly start time
            anomaly_end: anomaly end time
            detection_start: anomaly detection start time
            detection_end: anomaly detection end time
            rows: row count
            cols: column count
            filename: filename
            page_num: page number
        """
        try:
            # Create figure
            fig_width = cols * 4
            fig_height = rows * 3
            fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

            # Ensure axes is 2D array
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)

            fig.suptitle(f'Metrics Dashboard - Page {page_num}', fontsize=16, y=0.98)

            # Plot each metric
            for i, metric in enumerate(metrics):
                row = i // cols
                col = i % cols
                ax = axes[row, col]

                self._plot_single_metric(ax, df, metric, anomaly_start, anomaly_end, 
                                       detection_start, detection_end)

            # Hide empty subplots
            for i in range(len(metrics), rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].set_visible(False)

            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])

            # Save figure
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved chart: {output_path}")

        except Exception as e:
            print(f"Failed to create single page: {e}")
    
    def _plot_single_metric(self, ax, df: pd.DataFrame, metric: str,
                           anomaly_start: pd.Timestamp, anomaly_end: pd.Timestamp,
                           detection_start: pd.Timestamp, detection_end: pd.Timestamp):
        """
        Plot line chart for a single metric

        Args:
            ax: matplotlib axes object
            df: DataFrame
            metric: metric name
            anomaly_start: anomaly start time
            anomaly_end: anomaly end time
            detection_start: anomaly detection start time
            detection_end: anomaly detection end time
        """
        try:
            # Get valid data
            valid_data = df[metric].replace([np.inf, -np.inf], np.nan).dropna()

            if valid_data.empty:
                ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(metric, fontsize=10, pad=5)
                return

            # Plot line chart
            # Ensure data is 1D
            y_values = valid_data.values
            if y_values.ndim > 1:
                y_values = y_values.flatten()

            # Ensure index is also 1D
            x_values = valid_data.index.values
            if x_values.ndim > 1:
                x_values = x_values.flatten()

            ax.plot(x_values, y_values,
                   linewidth=1.5, alpha=0.8, color='blue', label=metric)

            # Add anomaly detection period and fault period background
            if not valid_data.empty:
                data_start = pd.Timestamp(valid_data.index.min())
                data_end = pd.Timestamp(valid_data.index.max())

                # Add anomaly detection period (detection window 1 hour before fault)
                if detection_start and detection_end:
                    det_start_display = max(detection_start, data_start)
                    det_end_display = min(detection_end, data_end)

                    if det_start_display <= det_end_display:
                        ax.axvspan(det_start_display, det_end_display,
                                 alpha=0.5, color='blue', zorder=0, label='Anomaly detection period')

                # Add fault period (actual fault occurrence time)
                if anomaly_start and anomaly_end:
                    anom_start_display = max(anomaly_start, data_start)
                    anom_end_display = min(anomaly_end, data_end)

                    # If fault start time is same or close to detection end time, slightly offset to avoid overlap
                    # Add a very small time interval to prevent exact overlap
                    time_epsilon = pd.Timedelta(seconds=1)  # 1 second interval
                    if (detection_end and 
                        anom_start_display <= detection_end and
                        anom_end_display > detection_end):
                        # Start marking fault time after anomaly detection end time
                        anom_start_display = max(anom_start_display, detection_end + time_epsilon)

                    if anom_start_display <= anom_end_display:
                        ax.axvspan(anom_start_display, anom_end_display,
                                 alpha=0.5, color='red', zorder=0, label='Fault occurrence period')

            # Set title and labels
            ax.set_title(metric, fontsize=10, pad=5)
            ax.set_ylabel('Value', fontsize=8)

            # Remove x-axis numeric labels, keep only y-axis
            ax.tick_params(axis='x', labelbottom=False)  # Don't show x-axis labels
            ax.tick_params(axis='y', labelsize=8)

            # Add grid, but only show horizontal lines (remove vertical lines)
            ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)

            # Add legend showing anomaly detection period and fault occurrence period
            legend_elements = []
            if detection_start and detection_end:
                from matplotlib.patches import Patch
                legend_elements.append(Patch(facecolor='blue', alpha=0.3, label='Anomaly detection period'))
            if anomaly_start and anomaly_end:
                from matplotlib.patches import Patch
                legend_elements.append(Patch(facecolor='red', alpha=0.3, label='Fault occurrence period'))

            if legend_elements:
                ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

        except Exception as e:
            print(f"Failed to plot metric {metric}: {e}")
            ax.text(0.5, 0.5, f'Plot failed: {str(e)[:20]}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=8)
            ax.set_title(metric, fontsize=10, pad=5)

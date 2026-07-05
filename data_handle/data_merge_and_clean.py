#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataMergeAndClean:
    """
    Monitoring Data Merge and Clean Tool

    Features:
    1. Read all CSV files from a directory
    2. Merge into a unified dataset by timestamp
    3. Handle missing values and data cleaning
    4. Output cleaned data and processing report
    """
    
    def __init__(self, missing_threshold: float = 0.2, verbose: bool = False):
        """
        Initialize data merge and cleaner
        
        Args:
            missing_threshold: missing value threshold, columns exceeding this ratio will be dropped
            verbose: whether to output detailed information
        """
        self.missing_threshold = missing_threshold
        self.verbose = verbose
        
        # Internal state
        self.dataframes = {}
        self.file_info = []
        self.merged_df = None
        self.merge_info = []
        self.missing_analysis = []
        self.columns_to_keep = []
        self.columns_to_drop = []
        
    def _log(self, message: str) -> None:
        """Internal log output method"""
        if self.verbose:
            print(message)
    
    def load_csv_files(self, data_dir: str) -> None:
        """
        Load all CSV files from the specified directory
        
        Args:
            data_dir: path to directory containing CSV files
        """
        self._log(f"Reading directory: {data_dir}")
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        self._log(f"Found {len(csv_files)} CSV files")
        
        # Reset internal state
        self.dataframes = {}
        self.file_info = []
        
        # Read all CSV files
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            # Generate metric name (remove file extension and prefix)
            metric_name = filename.replace('.csv', '').replace('machine_metric_', '')
            
            try:
                df = pd.read_csv(file_path)
                # Ensure timestamp column exists
                if 'timestamp' in df.columns:
                    self.dataframes[metric_name] = df
                    
                    # Record file information
                    self.file_info.append({
                        'metric': metric_name,
                        'rows': len(df),
                        'columns': list(df.columns),
                        'time_range': f"{df['timestamp'].min()} - {df['timestamp'].max()}"
                    })
                    self._log(f"✓ Successfully read: {filename} ({len(df)} rows)")
                else:
                    self._log(f"⚠ Warning: {filename} does not have a timestamp column")
            except Exception as e:
                self._log(f"✗ Failed to read {filename}: {e}")
        
        self._log(f"Successfully loaded {len(self.dataframes)} data files")
    
    def create_unified_timeline(self) -> None:
        """
        Create unified minute-level time series
        """
        self._log("Creating unified time series...")
        
        # Collect all timestamps
        all_timestamps = set()
        for df in self.dataframes.values():
            all_timestamps.update(df['timestamp'].values)
        
        all_timestamps = sorted(list(all_timestamps))
        
        # Convert to datetime and create minute-level sequence
        start_time = min(all_timestamps)
        end_time = max(all_timestamps)
        
        start_dt = pd.to_datetime(start_time, unit='ms')
        end_dt = pd.to_datetime(end_time, unit='ms')
        
        # Create minute-level time series
        minute_timestamps = pd.date_range(
            start=start_dt.floor('min'), 
            end=end_dt.ceil('min'), 
            freq='1min'
        )
        minute_timestamps_ms = (minute_timestamps.astype(np.int64) // 10**6).values
        
        # Create main dataframe
        self.merged_df = pd.DataFrame({'timestamp': minute_timestamps_ms})
        
        self._log(f"Time range: {start_dt} to {end_dt}")
        self._log(f"Generated {len(minute_timestamps)} minute-level time points")
    
    def merge_metrics_data(self) -> None:
        """
        Merge all metric data into unified time series (memory optimized version)

        Optimization strategies:
        1. Incremental merge - directly add columns to merged_df, avoiding intermediate copies from reduce
        2. Data type optimization - float64 to float32, reducing memory by 50%
        3. Explicit memory management - promptly delete temporary variables and trigger garbage collection
        """
        import gc

        self._log("Starting metric data merge (memory optimized version)...")
        self.merge_info = []

        # Try importing psutil for memory monitoring (optional)
        try:
            import psutil
            process = psutil.Process(os.getpid())
            initial_mem = process.memory_info().rss / 1024 / 1024
            self._log(f"Initial memory: {initial_mem:.2f} MB")
            use_memory_monitor = True
        except ImportError:
            use_memory_monitor = False
            self._log("Tip: install psutil to monitor memory usage")

        metric_count = 0
        total_metrics = len(self.dataframes)

        for metric_name, df in self.dataframes.items():
            metric_count += 1
            self._log(f"  [{metric_count}/{total_metrics}] Processing metric: {metric_name}")

            # Get metric column names (all columns except timestamp)
            metric_columns = [col for col in df.columns if col != 'timestamp']

            if not metric_columns:
                continue

            # Only copy needed columns to reduce memory usage
            temp_df = df[['timestamp'] + metric_columns].copy()
            temp_df['datetime'] = pd.to_datetime(temp_df['timestamp'], unit='ms')
            temp_df = temp_df.set_index('datetime')

            # Resample to minute level
            resampled = temp_df[metric_columns].resample('1min').ffill()

            # Convert back to timestamp format
            resampled_df = pd.DataFrame({
                'timestamp': (resampled.index.astype(np.int64) // 10**6).values
            })

            # Data type optimization: use float32 instead of float64, saving 50% memory
            for col in metric_columns:
                values = resampled[col].values
                # If floating point, convert to float32
                if values.dtype == np.float64:
                    resampled_df[col] = values.astype(np.float32)
                elif values.dtype == np.int64:
                    # Choose appropriate integer type based on range
                    if values.max() < 2147483647 and values.min() > -2147483648:
                        resampled_df[col] = values.astype(np.int32)
                    else:
                        resampled_df[col] = values
                else:
                    resampled_df[col] = values

            # Incremental merge: directly merge into self.merged_df, avoiding intermediate copies from reduce
            self.merged_df = pd.merge(
                self.merged_df,
                resampled_df,
                on='timestamp',
                how='left'
            )

            # Explicitly delete temporary variables to free memory
            del temp_df, resampled, resampled_df

            # Record merge information
            for col in metric_columns:
                missing_count = self.merged_df[col].isna().sum() if col in self.merged_df.columns else 0
                self.merge_info.append({
                    'metric': col,
                    'original_points': len(df),
                    'resampled_points': len(self.merged_df),
                    'missing_after_merge': missing_count
                })

            # Every 2 metrics processed, force garbage collection and report memory
            if metric_count % 2 == 0:
                gc.collect()
                if use_memory_monitor:
                    current_mem = process.memory_info().rss / 1024 / 1024
                    self._log(f"    Processed {metric_count} metrics, current memory: {current_mem:.2f} MB (+{current_mem - initial_mem:.2f} MB)")

        # Final garbage collection
        gc.collect()

        if use_memory_monitor:
            final_mem = process.memory_info().rss / 1024 / 1024
            self._log(f"Final memory: {final_mem:.2f} MB (peak increase: {final_mem - initial_mem:.2f} MB)")

        self._log(f"Merge complete: {self.merged_df.shape[0]} rows × {self.merged_df.shape[1]} columns")
        self._log(f"Total metrics: {len(self.merge_info)}")
    
    def analyze_data_quality(self) -> None:
        """
        Analyze data quality, identify high-missing-rate columns for removal
        """
        self._log("Analyzing data quality...")
        
        self.missing_analysis = []
        total_rows = len(self.merged_df)
        
        # Analyze missing values for each column
        for col in self.merged_df.columns:
            if col != 'timestamp':
                missing_count = self.merged_df[col].isna().sum()
                missing_pct = missing_count / total_rows
                
                self.missing_analysis.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct,
                    'keep': missing_pct <= self.missing_threshold
                })
        
        # Sort and classify
        self.missing_analysis.sort(key=lambda x: x['missing_percentage'], reverse=True)
        
        self.columns_to_keep = []
        self.columns_to_drop = []
        
        for analysis in self.missing_analysis:
            if analysis['keep']:
                self.columns_to_keep.append(analysis['column'])
            else:
                self.columns_to_drop.append(analysis['column'])
        
        self._log(f"Quality analysis complete:")
        self._log(f"  - Columns to keep: {len(self.columns_to_keep)}")
        self._log(f"  - Columns to drop: {len(self.columns_to_drop)} (missing rate > {self.missing_threshold*100}%)")
    
    def clean_and_fill_data(self) -> None:
        """
        Data cleaning: drop high-missing-rate columns and fill remaining missing values
        """
        self._log("Starting data cleaning...")

        # Drop columns with high missing rate
        if self.columns_to_drop:
            self.merged_df = self.merged_df.drop(columns=self.columns_to_drop)
            self._log(f"Dropped {len(self.columns_to_drop)} high-missing-rate columns")

        # Count missing values before filling
        before_fill_na = self.merged_df.isna().sum().sum()

        # Forward fill
        for col in self.columns_to_keep:
            if col in self.merged_df.columns:
                self.merged_df[col] = self.merged_df[col].ffill()

        after_fill_na = self.merged_df.isna().sum().sum()

        # If there are still missing values, fill with 0 (typically the first few rows)
        if after_fill_na > 0:
            self.merged_df = self.merged_df.fillna(0)
            final_na = self.merged_df.isna().sum().sum()
        else:
            final_na = after_fill_na

        self._log(f"Missing value handling:")
        self._log(f"  - Before filling: {before_fill_na}")
        self._log(f"  - After filling: {final_na}")
        self._log(f"  - Total filled: {before_fill_na - final_na} missing values")
    
    def save_results(self, output_file: str, create_report: bool = True) -> None:
        """
        Save processing results and generate report
        
        Args:
            output_file: output CSV file path
            create_report: whether to generate processing report
        """
        self._log(f"Saving results to: {output_file}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save merged data
        self.merged_df.to_csv(output_file, index=False)
        
        # Verify saved file
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        self._log(f"File saved successfully! Size: {file_size:.2f} MB")
        
        # Create processing report
        if create_report:
            report_file = output_file.replace('.csv', '_processing_report.txt')
            self._create_processing_report(report_file)
            self._log(f"Processing report saved: {report_file}")
    
    def _create_processing_report(self, report_file: str) -> None:
        """
        Create detailed processing report
        
        Args:
            report_file: report file path
        """
        start_time_readable = pd.to_datetime(self.merged_df['timestamp'].min(), unit='ms')
        end_time_readable = pd.to_datetime(self.merged_df['timestamp'].max(), unit='ms')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Data Processing Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing date: {datetime.now()}\n")
            f.write(f"Output file: {os.path.basename(report_file).replace('_processing_report.txt', '.csv')}\n\n")
            
            f.write(f"Input files processed: {len(self.dataframes)}\n")
            f.write(f"Final dataset shape: {self.merged_df.shape}\n")
            f.write(f"Time range: {start_time_readable} to {end_time_readable}\n")
            f.write(f"Missing data threshold: {self.missing_threshold*100}%\n\n")
            
            f.write(f"Columns kept: {len(self.columns_to_keep)}\n")
            f.write(f"Columns dropped: {len(self.columns_to_drop)}\n\n")
            
            if self.columns_to_drop:
                f.write("Dropped columns:\n")
                for col in self.columns_to_drop:
                    # Find corresponding missing rate info
                    missing_info = next((item for item in self.missing_analysis if item['column'] == col), None)
                    if missing_info:
                        f.write(f"  - {col} ({missing_info['missing_percentage']:.1%} missing)\n")
                    else:
                        f.write(f"  - {col}\n")
                f.write("\n")
            
            f.write("Kept columns:\n")
            for col in self.columns_to_keep:
                f.write(f"  - {col}\n")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get processing summary information
        
        Returns:
            Dictionary containing processing summary
        """
        if self.merged_df is None:
            return {}
        
        start_time_readable = pd.to_datetime(self.merged_df['timestamp'].min(), unit='ms')
        end_time_readable = pd.to_datetime(self.merged_df['timestamp'].max(), unit='ms')
        
        return {
            'input_files': len(self.dataframes),
            'final_shape': self.merged_df.shape,
            'time_range': {
                'start': start_time_readable,
                'end': end_time_readable,
                'duration_minutes': len(self.merged_df)
            },
            'columns_kept': len(self.columns_to_keep),
            'columns_dropped': len(self.columns_to_drop),
            'missing_threshold': self.missing_threshold,
            'total_data_points': self.merged_df.shape[0] * (self.merged_df.shape[1] - 1)
        }
    
    def process_directory(self, data_dir: str, output_file: str) -> None:
        """
        Complete data processing flow: from directory read to output results (streaming version)

        Args:
            data_dir: input data directory path
            output_file: output CSV file path
        """
        import gc

        self._log("=" * 60)
        self._log("Starting data merge and cleaning process (streaming)")
        self._log("=" * 60)

        try:
            # Try importing psutil for memory monitoring (optional)
            try:
                import psutil
                process = psutil.Process(os.getpid())
                initial_mem = process.memory_info().rss / 1024 / 1024
                self._log(f"Initial memory: {initial_mem:.2f} MB")
                use_memory_monitor = True
            except ImportError:
                use_memory_monitor = False
                self._log("Tip: install psutil to monitor memory usage")

            # 1. Get all CSV file paths (only store paths, don't load data)
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            self._log(f"Found {len(csv_files)} CSV files")

            if not csv_files:
                raise ValueError("No valid CSV files found")

            # 2. First pass scan: only read timestamp range (don't load full data)
            self._log("First pass scan: determining time range...")
            min_timestamp = float('inf')
            max_timestamp = float('-inf')

            for file_path in csv_files:
                # Only read first row of timestamp column
                df_first = pd.read_csv(file_path, usecols=['timestamp'], nrows=1)
                start_ts = df_first['timestamp'].iloc[0]

                # Read last row (using tail trick)
                with open(file_path, 'rb') as f:
                    f.seek(0, 2)  # Move to end of file
                    file_size = f.tell()
                    # Read up to 200 bytes from end (enough to contain last row)
                    f.seek(max(0, file_size - 200), 0)
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].decode('utf-8')
                    else:
                        last_line = lines[0].decode('utf-8')
                    end_ts = int(last_line.split(',')[0])

                min_timestamp = min(min_timestamp, start_ts)
                max_timestamp = max(max_timestamp, end_ts)

                # Release immediately
                del df_first

            gc.collect()

            # 3. Create unified time series (small memory footprint)
            self._log("Creating unified time series...")
            start_dt = pd.to_datetime(min_timestamp, unit='ms')
            end_dt = pd.to_datetime(max_timestamp, unit='ms')

            minute_timestamps = pd.date_range(
                start=start_dt.floor('min'),
                end=end_dt.ceil('min'),
                freq='1min'
            )
            # Convert DatetimeIndex to millisecond timestamps
            # Note: pd.date_range returns datetime64[ms] precision by default, astype(int64) gives millisecond values directly
            minute_timestamps_ms = minute_timestamps.astype(np.int64).values

            # Initialize merged_df (set timestamp as index for easier column addition later)
            self.merged_df = pd.DataFrame({'timestamp': minute_timestamps_ms})

            # Check and remove duplicate timestamps (if any)
            if self.merged_df['timestamp'].duplicated().any():
                self._log(f"Warning: found {self.merged_df['timestamp'].duplicated().sum()} duplicate timestamps, deduplicating")
                self.merged_df = self.merged_df.drop_duplicates(subset=['timestamp'], keep='first')

            self.merged_df = self.merged_df.set_index('timestamp')

            self._log(f"Time range: {start_dt} to {end_dt}")
            self._log(f"Generated {len(minute_timestamps)} minute-level time points")

            if use_memory_monitor:
                current_mem = process.memory_info().rss / 1024 / 1024
                self._log(f"Memory after creating time series: {current_mem:.2f} MB (+{current_mem - initial_mem:.2f} MB)")

            # 4. Streaming merge: process CSV files one by one, release immediately
            self._log("Starting streaming metric data merge...")
            self.merge_info = []
            processed_files = 0

            for idx, file_path in enumerate(csv_files, 1):
                filename = os.path.basename(file_path)
                metric_name = filename.replace('.csv', '').replace('machine_metric_', '')

                self._log(f"  [{idx}/{len(csv_files)}] Processing metric: {metric_name}")

                # Only load current file
                df = pd.read_csv(file_path)

                # Get metric column names (all columns except timestamp)
                metric_columns = [col for col in df.columns if col != 'timestamp']

                if metric_columns:
                    # Resample processing
                    temp_df = df[['timestamp'] + metric_columns].copy()
                    temp_df['datetime'] = pd.to_datetime(temp_df['timestamp'], unit='ms')
                    temp_df = temp_df.set_index('datetime')

                    # Resample to minute level
                    resampled = temp_df[metric_columns].resample('1min').ffill()

                    # Convert to DataFrame indexed by timestamp
                    # Note: after resample, DatetimeIndex precision is datetime64[ms], astype(int64) gives millisecond values directly
                    resampled_timestamps = resampled.index.astype(np.int64).values

                    # Create temporary DataFrame for join
                    temp_join_df = pd.DataFrame(index=resampled_timestamps)

                    # Add all columns to temporary DataFrame
                    for col in metric_columns:
                        values = resampled[col].values
                        # Data type optimization
                        if values.dtype == np.float64:
                            values = values.astype(np.float32)
                        elif values.dtype == np.int64:
                            if values.max() < 2147483647 and values.min() > -2147483648:
                                values = values.astype(np.int32)

                        temp_join_df[col] = values

                    # Check for duplicate indices in temp_join_df
                    if temp_join_df.index.duplicated().any():
                        self._log(f"  Warning: metric {metric_name} has {temp_join_df.index.duplicated().sum()} duplicate timestamps after resampling, deduplicating")
                        temp_join_df = temp_join_df[~temp_join_df.index.duplicated(keep='first')]

                    # Use join for merging (based on index, auto-aligned)
                    self.merged_df = self.merged_df.join(temp_join_df, how='left')

                    # Record merge information
                    for col in metric_columns:
                        missing_count = self.merged_df[col].isna().sum() if col in self.merged_df.columns else 0
                        self.merge_info.append({
                            'metric': col,
                            'original_points': len(df),
                            'resampled_points': len(self.merged_df),
                            'missing_after_merge': missing_count
                        })

                    # Immediately delete temporary variables
                    del df, temp_df, resampled, temp_join_df
                    processed_files += 1

                # Force garbage collection every 5 files processed
                if idx % 5 == 0:
                    gc.collect()
                    if use_memory_monitor:
                        current_mem = process.memory_info().rss / 1024 / 1024
                        self._log(f"    Processed {idx} metrics, current memory: {current_mem:.2f} MB (+{current_mem - initial_mem:.2f} MB)")

            # Restore timestamp column (from index back to column)
            self.merged_df = self.merged_df.reset_index()

            # Final garbage collection
            gc.collect()

            if use_memory_monitor:
                final_mem = process.memory_info().rss / 1024 / 1024
                self._log(f"Streaming merge complete, final memory: {final_mem:.2f} MB (peak increase: {final_mem - initial_mem:.2f} MB)")

            self._log(f"Merge complete: {self.merged_df.shape[0]} rows × {self.merged_df.shape[1]} columns")
            self._log(f"Total metrics: {len(self.merge_info)}")

            # 5. Analyze data quality
            self.analyze_data_quality()

            # 6. Data cleaning
            self.clean_and_fill_data()

            # 7. Save results
            self.save_results(output_file)

            # Output summary
            summary = self.get_processing_summary()
            self._log("\n" + "=" * 60)
            self._log("Processing complete summary:")
            self._log("=" * 60)
            self._log(f"Input files: {processed_files}")
            self._log(f"Final dataset: {summary['final_shape'][0]} rows × {summary['final_shape'][1]} columns")
            self._log(f"Time span: {summary['time_range']['duration_minutes']} minutes")
            self._log(f"Metrics kept: {summary['columns_kept']}")
            self._log(f"Metrics dropped: {summary['columns_dropped']}")
            self._log(f"Total data points: {summary['total_data_points']:,}")
            self._log("=" * 60)

        except Exception as e:
            self._log(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise


# Usage example
if __name__ == "__main__":
    # Create data processor
    processor = DataMergeAndClean(missing_threshold=0.2, verbose=True)
    
    # Set paths
    data_dir = "/home/kuanjunhua/data_handle/collected_data/case1/wdkreverse_wdkreversehost/metric/raw_data"
    output_file = "/home/kuanjunhua/data_handle/collected_data/case1/wdkreverse_wdkreversehost/metric/merged_metrics.csv"
    
    # Execute complete processing flow
    processor.process_directory(data_dir, output_file)

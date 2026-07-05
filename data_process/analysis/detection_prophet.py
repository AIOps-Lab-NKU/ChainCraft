#!/usr/bin/env python3
"""
Anomaly Detection Analysis Script Based on Prophet Algorithm

Performs anomaly detection analysis using cleaned monitoring data, identifying system anomalies based on the Prophet time series prediction model.
"""

import sys
from config import Config
# sys.path.append(f'{Config.BASE_PATH}/data_handle')



from data_handle.data_config import case_table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import Prophet
from prophet import Prophet
import plotly.graph_objects as go
import os

# Import parallel processing module
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import multiprocessing as mp
from functools import partial
import time
from config import Config

# Import event window analysis module
# sys.path.append(f'{Config.BASE_PATH}/exam')
from data_process.analysis.event_window_analyzer import analyze_event_window_features

# Configuration parameters
PRINT_PROGRESS = True       # Whether to print progress information
PRINT_DATA_INFO = True      # Whether to print data information
PRINT_DETECTION_PARAMS = True  # Whether to print detection parameters
PRINT_RESULTS = True        # Whether to print result analysis
SHOW_PLOTS = True          # Whether to display charts
SAVE_RESULTS = True        # Whether to save results


def setup_plotting():
    """Set up plotting style"""
    plt.rcParams['figure.figsize'] = (15, 8)
    plt.rcParams['font.size'] = 12
    if PRINT_PROGRESS:
        print("Plotting style setup complete")


def load_data(data_file):
    """Load and preprocess data"""
    if PRINT_PROGRESS:
        print("Loading data...")
    
    df = pd.read_csv(data_file)
    
    # Convert timestamps to datetime index
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')+pd.Timedelta(hours=8)  # Convert to Beijing time
    df = df.set_index('datetime')
    df = df.drop('timestamp', axis=1)
    
    if PRINT_DATA_INFO:
        print(f"Data loaded:")
        print(f"  Shape: {df.shape}")
        print(f"  Time range: {df.index.min()} to {df.index.max()}")
        print(f"  Duration: {df.index.max() - df.index.min()}")
        print(f"  Metric count: {len(df.columns)}")
    
    return df


def check_data_quality(df):
    """Check data quality"""
    if PRINT_DATA_INFO:
        print("\nData quality check:")
        print("=" * 50)
        print(f"  Total rows: {len(df)}")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Missing values: {df.isnull().sum().sum()}")
        print(f"  Infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
    
    return df


def filter_consecutive_anomalies(anomalies_series, min_consecutive=3):
    """
    Filter anomaly points, only keeping those that appear consecutively for min_consecutive or more points
    
    Parameters:
    - anomalies_series: pandas Series containing 0/1 anomaly markers
    - min_consecutive: minimum number of consecutive anomaly points
    
    Returns:
    - filtered_anomalies: filtered anomaly point Series
    """
    if len(anomalies_series) == 0 or anomalies_series.sum() == 0:
        return anomalies_series
    
    # Convert to numpy array for easier processing
    anomaly_array = anomalies_series.values.astype(int)
    filtered_array = np.zeros_like(anomaly_array)
    
    # Find all consecutive segments of anomaly points
    anomaly_indices = np.where(anomaly_array == 1)[0]
    
    if len(anomaly_indices) == 0:
        return pd.Series(filtered_array, index=anomalies_series.index)
    
    # Identify consecutive segments
    consecutive_groups = []
    current_group = [anomaly_indices[0]]
    
    for i in range(1, len(anomaly_indices)):
        if anomaly_indices[i] == anomaly_indices[i-1] + 1:
            # Consecutive anomaly point
            current_group.append(anomaly_indices[i])
        else:
            # Not consecutive, save current group and start new group
            consecutive_groups.append(current_group)
            current_group = [anomaly_indices[i]]
    
    # Add last group
    consecutive_groups.append(current_group)
    
    # Only keep consecutive groups with length >= min_consecutive
    for group in consecutive_groups:
        if len(group) >= min_consecutive:
            filtered_array[group] = 1
    
    return pd.Series(filtered_array, index=anomalies_series.index)


def detect_single_metric_worker(args):
    """
    Worker function for single metric detection, used for parallel processing
    
    Args:
        args: tuple containing all required parameters
    
    Returns:
        tuple: (metric_index, metric_name, anomalies, z_scores, residuals, anomaly_starts, has_anomaly_chart)
    """
    (metric_index, metric_name, df_values, df_index, normal_start_time, normal_end_time,
     abnormal_start_time, abnormal_end_time, interval_width, save_chart, 
     chart_save_path, min_consecutive, anomaly_start_interval_width) = args
    
    # Reconstruct DataFrame (containing only current metric)
    df_single = pd.DataFrame({metric_name: df_values}, index=df_index)
    
    # Call the original single metric detection function
    anomalies, z_scores, residuals, anomaly_starts = prophet_anomaly_detection_single_metric(
        df_single, metric_name, normal_start_time, normal_end_time,
        abnormal_start_time, abnormal_end_time, interval_width,
        save_chart=save_chart, chart_save_path=chart_save_path,
        min_consecutive=min_consecutive,
        anomaly_start_interval_width=anomaly_start_interval_width
    )
    
    # Determine if an anomaly chart was generated
    has_anomaly_chart = save_chart and len(anomalies[anomalies > 0]) > 0
    
    return (metric_index, metric_name, anomalies, z_scores, residuals, anomaly_starts, has_anomaly_chart)


def setup_detection_params(df, event_start=None, event_end=None, t0=None):
    """Set up anomaly detection parameters"""
    # Set detection parameters
    if t0 is None:
        total_duration = df.index.max() - df.index.min()
        t0 = df.index.min() + total_duration / 2
    else:
        # If t0 is a string, convert to pd.Timestamp
        if isinstance(t0, str):
            t0 = pd.Timestamp(t0)
    print(f"\nSetting detection center time: {t0}")
    print('t0:', t0)
    
    detection_params = {
        'event_start': event_start,
        'event_end': event_end,
        't0': None,               # Detection center time point
        'pre': '60min',         # Pre-event window
        'post': '30min',        # Post-event window
        'guard': '30min',       # Guard period
        'resample_freq': None,  # Already minute-level data, no resampling needed
        'short_win': '60min',   # Short window for detrending
        'rolling_mad_window': '12H',  # Rolling MAD window
        'tau': 4.2,             # z-threshold
        'min_consecutive': 3,   # Minimum consecutive anomaly points
        'use_abs_z': True,      # Use absolute z-scores
        'interval_width': 0.9995, # Prophet confidence interval width (for final anomaly judgment)
        'anomaly_start_interval_width': 0.99,  # Confidence interval width for detecting anomaly start time
        'min_consecutive_prophet': 3,  # Prophet consecutive anomaly point requirement
        'use_parallel': True,    # Whether to use parallel processing
        'n_workers': None        # Number of parallel worker processes (None for auto-selection)
    }
    
    if PRINT_DETECTION_PARAMS:
        print(f"\nDetection parameter settings:")
        print(f"  Detection center time: {detection_params['t0']}")
        print(f"  Event window: {event_start} to {event_end}")
        print(f"  Other parameters: {detection_params}")
    
    return detection_params


def prophet_anomaly_detection_single_metric(df, metric, normal_start_time, normal_end_time, 
                                           abnormal_start_time, abnormal_end_time, interval_width=0.95, 
                                           save_chart=False, chart_save_path=None, min_consecutive=3,
                                           anomaly_start_interval_width=0.99):
    """
    Perform anomaly detection on a single metric using Prophet
    
    Parameters:
    - df: DataFrame containing time index and metric data
    - metric: name of the metric to detect
    - normal_start_time: normal period start time, used for training
    - normal_end_time: normal period end time, used for training
    - abnormal_start_time: anomaly detection period start time
    - abnormal_end_time: anomaly detection period end time
    - interval_width: Prophet confidence interval width (for final anomaly judgment)
    - anomaly_start_interval_width: confidence interval width for detecting anomaly start time
    
    Returns:
    - anomalies: anomaly markers (only within detection period)
    - z_scores: z-scores (calculated based on confidence interval)
    - residuals: residuals
    - anomaly_start_times: anomaly start times detected based on narrow confidence interval
    """
    
    # Ensure time parameters are pandas.Timestamp objects
    if isinstance(normal_start_time, str):
        normal_start_time = pd.Timestamp(normal_start_time)
    if isinstance(normal_end_time, str):
        normal_end_time = pd.Timestamp(normal_end_time)
    if isinstance(abnormal_start_time, str):
        abnormal_start_time = pd.Timestamp(abnormal_start_time)
    if isinstance(abnormal_end_time, str):
        abnormal_end_time = pd.Timestamp(abnormal_end_time)
    
    # Prepare Prophet data format
    prophet_data = pd.DataFrame({
        'ds': df.index,
        'y': df[metric]
    })
    
    # Filter normal period data for training
    train_data = prophet_data[
        (prophet_data['ds'] >= normal_start_time) & 
        (prophet_data['ds'] <= normal_end_time)
    ].copy()
    
    # Check if training data is sufficient
    if len(train_data) < 10:
        # If training data insufficient, return empty results
        anomalies = pd.Series(0, index=df.index, name=metric)
        z_scores = pd.Series(0.0, index=df.index, name=metric)
        residuals = pd.Series(0.0, index=df.index, name=metric)
        return anomalies, z_scores, residuals
    
    # Remove NaN and infinite values
    train_data = train_data.dropna()
    train_data = train_data[np.isfinite(train_data['y'])]
    
    if len(train_data) < 10:
        # If cleaned training data insufficient, return empty results
        anomalies = pd.Series(0, index=df.index, name=metric)
        z_scores = pd.Series(0.0, index=df.index, name=metric)
        residuals = pd.Series(0.0, index=df.index, name=metric)
        return anomalies, z_scores, residuals
    
    try:
        # Train Prophet model (using wide confidence interval for final anomaly judgment)
        model = Prophet(interval_width=interval_width, daily_seasonality=True, 
                       yearly_seasonality=False, weekly_seasonality=False)
        model.fit(train_data)
        
        # Train second model (using narrow confidence interval for detecting anomaly start time)
        model_narrow = Prophet(interval_width=anomaly_start_interval_width, daily_seasonality=True, 
                              yearly_seasonality=False, weekly_seasonality=False)
        model_narrow.fit(train_data)
        
        # Predict for the entire time range
        forecast = model.predict(prophet_data[['ds']])
        forecast_narrow = model_narrow.predict(prophet_data[['ds']])
        
        # Merge forecast results with original data
        result = pd.concat([
            prophet_data.set_index('ds'), 
            forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]
        ], axis=1)
        
        # Add narrow confidence interval forecast results
        result = pd.concat([
            result,
            forecast_narrow.set_index('ds')[['yhat_lower', 'yhat_upper']].rename(
                columns={'yhat_lower': 'yhat_lower_narrow', 'yhat_upper': 'yhat_upper_narrow'})
        ], axis=1)
        
        # Calculate residuals
        residuals = result['y'] - result['yhat']
        
        # Calculate z-scores (based on wide confidence interval)
        # When actual value exceeds confidence interval, calculate z-score
        z_scores = pd.Series(0.0, index=result.index)
        
        # Upper anomaly: actual value > yhat_upper
        upper_anomaly_mask = result['y'] > result['yhat_upper']
        z_scores[upper_anomaly_mask] = (result.loc[upper_anomaly_mask, 'y'] - 
                                       result.loc[upper_anomaly_mask, 'yhat_upper']) / \
                                      (result.loc[upper_anomaly_mask, 'yhat_upper'] - 
                                       result.loc[upper_anomaly_mask, 'yhat']).abs()
        
        # Lower anomaly: actual value < yhat_lower
        lower_anomaly_mask = result['y'] < result['yhat_lower']
        z_scores[lower_anomaly_mask] = (result.loc[lower_anomaly_mask, 'yhat_lower'] - 
                                       result.loc[lower_anomaly_mask, 'y']) / \
                                      (result.loc[lower_anomaly_mask, 'yhat'] - 
                                       result.loc[lower_anomaly_mask, 'yhat_lower']).abs()
        
        # Mark potential anomalies (based on wide confidence interval)
        potential_anomalies = (result['y'] > result['yhat_upper']) | (result['y'] < result['yhat_lower'])
        
        # Mark potential anomaly starts (based on narrow confidence interval)
        potential_anomaly_starts = (result['y'] > result['yhat_upper_narrow']) | (result['y'] < result['yhat_lower_narrow'])
        
        # Final anomaly points: potential anomalies + within specified detection period
        final_anomalies = potential_anomalies & \
                         (result.index >= abnormal_start_time) & \
                         (result.index <= abnormal_end_time)
        
        # Anomaly start time points: anomalies based on narrow confidence interval + within specified detection period
        anomaly_start_points = potential_anomaly_starts & \
                              (result.index >= abnormal_start_time) & \
                              (result.index <= abnormal_end_time)
        
        # Convert to 0/1 markers
        anomalies = final_anomalies.astype(int)
        anomaly_starts = anomaly_start_points.astype(int)
        
        # Re-index to match original DataFrame
        anomalies = anomalies.reindex(df.index, fill_value=0)
        anomaly_starts = anomaly_starts.reindex(df.index, fill_value=0) 
        z_scores = z_scores.reindex(df.index, fill_value=0.0)
        residuals = residuals.reindex(df.index, fill_value=0.0)
        
        # Apply consecutive anomaly point filtering
        anomalies = filter_consecutive_anomalies(anomalies, min_consecutive)
        
        # Plot chart (if needed)
        if save_chart and chart_save_path and len(anomalies[anomalies > 0]) > 0:
            # Plot from data start to event_end time range
            plot_start = df.index.min()  # Start from the very beginning of data
            plot_end = abnormal_end_time
            
            # Filter data for plotting time range
            plot_mask = (result.index >= plot_start) & (result.index <= plot_end)
            plot_result = result[plot_mask]
            # Use filtered anomaly points for plotting
            filtered_anomalies_in_detection = anomalies[(anomalies.index >= abnormal_start_time) & 
                                                       (anomalies.index <= abnormal_end_time)]
            plot_anomalies = filtered_anomalies_in_detection.reindex(plot_result.index, fill_value=0)
            
            if len(plot_result) > 0:
                # Create plotly chart
                fig = go.Figure()
                
                # Mark "training period" and "detection period" with semi-transparent regions
                fig.add_vrect(
                    x0=plot_start, x1=abnormal_start_time,
                    fillcolor="LightGreen", opacity=0.3,
                    layer="below", line_width=0,
                    annotation_text="Training Period", annotation_position="top left"
                )
                fig.add_vrect(
                    x0=abnormal_start_time, x1=abnormal_end_time,
                    fillcolor="LightPink", opacity=0.3,
                    layer="below", line_width=0,
                    annotation_text="Anomaly Detection Period", annotation_position="top left"
                )
                
                # Plot confidence interval
                fig.add_trace(go.Scatter(
                    x=plot_result.index, y=plot_result['yhat_upper'], 
                    mode='lines', line=dict(color='rgba(0,100,80,0.2)'), 
                    name='Upper Bound', showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=plot_result.index, y=plot_result['yhat_lower'], 
                    mode='lines', fill='tonexty', fillcolor='rgba(0,100,80,0.2)', 
                    line=dict(color='rgba(0,100,80,0.2)'), name='Confidence Interval'
                ))
                
                # Plot predicted values
                fig.add_trace(go.Scatter(
                    x=plot_result.index, y=plot_result['yhat'], 
                    mode='lines', line=dict(color='orange', width=2), 
                    name='Prophet Prediction'
                ))
                
                # Plot actual values
                fig.add_trace(go.Scatter(
                    x=plot_result.index, y=plot_result['y'], 
                    mode='lines', line=dict(color='royalblue', width=2), 
                    name='Actual Value'
                ))
                
                # Mark detected anomaly points (filtered consecutive anomaly points)
                anomaly_points = plot_anomalies[plot_anomalies > 0]
                if len(anomaly_points) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomaly_points.index, 
                        y=plot_result.loc[anomaly_points.index, 'y'],
                        mode='markers', 
                        marker=dict(color='red', size=10, symbol='x'), 
                        name=f'Consecutive Anomaly Points ({len(anomaly_points)})'
                    ))
                
                fig.update_layout(
                    title=f'{metric} - Prophet Anomaly Detection Results',
                    xaxis_title='Time', 
                    yaxis_title='Metric Value',
                    legend_title='Legend',
                    height=500,
                    showlegend=True
                )
                
                # Save chart
                chart_filename = f"{metric.replace('/', '_').replace(':', '_')}_detection.html"
                chart_path = os.path.join(chart_save_path, chart_filename)
                fig.write_html(chart_path)
        
        return anomalies, z_scores, residuals, anomaly_starts
        
    except Exception as e:
        if PRINT_PROGRESS:
            print(f"  Prophet training failed for {metric}: {e}")
        # Return empty results
        anomalies = pd.Series(0, index=df.index, name=metric)
        z_scores = pd.Series(0.0, index=df.index, name=metric)
        residuals = pd.Series(0.0, index=df.index, name=metric)
        anomaly_starts = pd.Series(0, index=df.index, name=metric)
        return anomalies, z_scores, residuals, anomaly_starts


def run_prophet_anomaly_detection(df, detection_params, save_charts=False, chart_save_path=None, 
                                 use_parallel=True, n_workers=None):
    """
    Run Prophet-based anomaly detection algorithm
    
    Args:
        df: DataFrame
        detection_params: detection parameter dictionary
        save_charts: whether to save charts
        chart_save_path: chart save path
        use_parallel: whether to use parallel processing (default True)
        n_workers: number of parallel worker processes (default is CPU cores - 1)
    """
    if PRINT_PROGRESS:
        print("\nRunning Prophet-based anomaly detection...")
        if use_parallel:
            actual_workers = n_workers if n_workers else max(1, cpu_count() - 1)
            print(f"Using parallel processing, worker count: {actual_workers}")
        else:
            print("Using serial processing")
        print("This may take a few minutes...")
    
    event_start = detection_params['event_start']
    event_end = detection_params['event_end']
    
    # Create charts folder
    if save_charts and chart_save_path:
        charts_dir = os.path.join(chart_save_path, 'charts')
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
            if PRINT_PROGRESS:
                print(f"Created chart save directory: {charts_dir}")
        chart_save_path = charts_dir
    
    # Ensure event_start and event_end are pandas.Timestamp objects
    if isinstance(event_start, str):
        event_start = pd.Timestamp(event_start)
    if isinstance(event_end, str):
        event_end = pd.Timestamp(event_end)
    
    # Define training period (normal period)
    # Use all data from start to event start for training
    normal_start_time = df.index.min()  # Use the very beginning of data
    normal_end_time = event_start  # Training ends at event start
    
    # Ensure training period is within data range
    normal_end_time = min(normal_end_time, df.index.max())
    
    if PRINT_PROGRESS:
        print(f"  Training period: {normal_start_time} to {normal_end_time}")
        print(f"  Detection period: {event_start} to {event_end}")
        print(f"  Data index type: {type(df.index[0])}")
        print(f"  Event time type: {type(event_start)}, {type(event_end)}")
    
    # Record start time
    start_time = time.time()
    
    if use_parallel:
        # Parallel processing
        # Prepare parameters for all tasks
        tasks = []
        for i, metric in enumerate(df.columns):
            task_args = (
                i, metric, df[metric].values, df.index,
                normal_start_time, normal_end_time,
                event_start, event_end, 
                detection_params['interval_width'],
                save_charts, chart_save_path,
                detection_params.get('min_consecutive_prophet', 3),
                detection_params.get('anomaly_start_interval_width', 0.99)
            )
            tasks.append(task_args)
        
        # Initialize result lists (preserve original order)
        n_metrics = len(df.columns)
        all_anomalies = [None] * n_metrics
        all_z_scores = [None] * n_metrics
        all_residuals = [None] * n_metrics
        all_anomaly_starts = [None] * n_metrics
        chart_count = 0
        completed_count = 0
        
        # Set worker process count
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)
        
        # Execute parallel tasks using process pool
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(detect_single_metric_worker, task): task for task in tasks}
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                try:
                    # Get results
                    metric_index, metric_name, anomalies, z_scores, residuals, anomaly_starts, has_chart = future.result()
                    
                    # Store results in original order
                    all_anomalies[metric_index] = anomalies
                    all_z_scores[metric_index] = z_scores
                    all_residuals[metric_index] = residuals
                    all_anomaly_starts[metric_index] = anomaly_starts
                    
                    if has_chart:
                        chart_count += 1
                    
                    completed_count += 1
                    if PRINT_PROGRESS and completed_count % 50 == 0:
                        print(f"  Completed {completed_count}/{n_metrics} metrics...")
                        
                except Exception as e:
                    task = future_to_task[future]
                    metric_name = task[1]
                    if PRINT_PROGRESS:
                        print(f"  Error processing metric {metric_name}: {e}")
                    # Create empty result
                    metric_index = task[0]
                    all_anomalies[metric_index] = pd.Series(0, index=df.index, name=metric_name)
                    all_z_scores[metric_index] = pd.Series(0.0, index=df.index, name=metric_name)
                    all_residuals[metric_index] = pd.Series(0.0, index=df.index, name=metric_name)
                    all_anomaly_starts[metric_index] = pd.Series(0, index=df.index, name=metric_name)
        
    else:
        # Serial processing (preserve original logic)
        all_anomalies = []
        all_z_scores = []
        all_residuals = []
        all_anomaly_starts = []
        chart_count = 0
        
        for i, metric in enumerate(df.columns):
            if PRINT_PROGRESS and i % 50 == 0:
                print(f"  Processing metric {i+1}/{len(df.columns)}: {metric}")
            
            anomalies, z_scores, residuals, anomaly_starts = prophet_anomaly_detection_single_metric(
                df, metric, normal_start_time, normal_end_time, 
                event_start, event_end, detection_params['interval_width'],
                save_chart=save_charts, chart_save_path=chart_save_path,
                min_consecutive=detection_params.get('min_consecutive_prophet', 3),
                anomaly_start_interval_width=detection_params.get('anomaly_start_interval_width', 0.99)
            )
            
            # Count generated charts
            if save_charts and len(anomalies[anomalies > 0]) > 0:
                chart_count += 1
            
            all_anomalies.append(anomalies)
            all_z_scores.append(z_scores)
            all_residuals.append(residuals)
            all_anomaly_starts.append(anomaly_starts)
    
    # Combine all results
    anomalies_df = pd.DataFrame(all_anomalies).T
    anomalies_df.columns = df.columns
    
    z_scores_df = pd.DataFrame(all_z_scores).T
    z_scores_df.columns = df.columns
    
    residuals_df = pd.DataFrame(all_residuals).T
    residuals_df.columns = df.columns
    
    anomaly_starts_df = pd.DataFrame(all_anomaly_starts).T
    anomaly_starts_df.columns = df.columns
    
    # Calculate runtime
    elapsed_time = time.time() - start_time
    
    # Consecutive anomaly point filtering already done in single metric detection function
    min_consecutive = detection_params.get('min_consecutive_prophet', 3)
    if PRINT_PROGRESS:
        final_anomaly_count = anomalies_df.sum().sum()
        print(f"Consecutive anomaly point filtering complete (min consecutive: {min_consecutive})")
        print(f"  Final anomaly point total: {final_anomaly_count}")
        print(f"\nPerformance statistics:")
        print(f"  Total elapsed: {elapsed_time:.2f} seconds")
        print(f"  Average per metric: {elapsed_time/len(df.columns):.2f} seconds")
        if use_parallel:
            print(f"  Parallel speedup: ~{len(df.columns)/(elapsed_time/(elapsed_time/len(df.columns))):.1f}x")
    
    # Create masks (compatible with original algorithm)
    masks = {
        'baseline_fit_mask': (df.index >= normal_start_time) & (df.index <= normal_end_time),
        'event_eval_mask': (df.index >= event_start) & (df.index <= event_end),
        'quality_mask': ~(df.isnull().any(axis=1) | np.isinf(df).any(axis=1))
    }
    
    if PRINT_PROGRESS:
        print("\nProphet anomaly detection complete!")
        print(f"  Residuals shape: {residuals_df.shape}")
        print(f"  Z-scores shape: {z_scores_df.shape}")
        print(f"  Anomaly markers shape: {anomalies_df.shape}")
        if save_charts:
            print(f"  Generated {chart_count} anomaly metric charts, saved to: {chart_save_path}")
    
    # Return results in the same format as original algorithm
    result = {
        'r': residuals_df,
        'z': z_scores_df,
        'a': anomalies_df,
        'anomaly_starts': anomaly_starts_df,
        'masks': masks
    }
    
    return result


def run_anomaly_detection(df, detection_params, save_charts=False, chart_save_path=None, 
                         use_parallel=True, n_workers=None):
    """
    Run anomaly detection algorithm (Prophet version)
    
    Args:
        df: DataFrame
        detection_params: detection parameter dictionary
        save_charts: whether to save charts
        chart_save_path: chart save path
        use_parallel: whether to use parallel processing (default True)
        n_workers: number of parallel worker processes (default is CPU cores - 1)
    """
    try:
        result = run_prophet_anomaly_detection(df, detection_params, save_charts, 
                                             chart_save_path, use_parallel, n_workers)
        
        # Extract results
        residuals = result['r']    # Residuals
        z_scores = result['z']     # Z-scores
        anomalies = result['a']    # Anomaly markers
        anomaly_starts = result['anomaly_starts']  # Anomaly start times based on narrow confidence interval
        masks = result['masks']    # Various masks
        
        return residuals, z_scores, anomalies, masks, anomaly_starts
        
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        raise


def analyze_results(df, anomalies, masks, detection_params, anomaly_starts=None):
    """Analyze anomaly detection results"""
    if not PRINT_RESULTS:
        return None, None, None, None
    
    print("\nAnomaly detection result analysis:")
    print("=" * 50)
    
    # Calculate anomaly point count for each metric
    anomaly_counts = anomalies.sum()
    total_points = len(anomalies)
    
    print(f"Total time points: {total_points}")
    
    # Time window analysis
    baseline_mask = masks['baseline_fit_mask']
    event_mask = masks['event_eval_mask']
    quality_mask = masks['quality_mask']
    
    print(f"\nTime window analysis:")
    print(f"  Total time points: {len(df)}")
    print(f"  Baseline fit points: {baseline_mask.sum()} ({baseline_mask.mean()*100:.1f}%)")
    print(f"  Event evaluation points: {event_mask.sum()} ({event_mask.mean()*100:.1f}%)")
    print(f"  Quality points (no NaN/Inf): {quality_mask.sum()} ({quality_mask.mean()*100:.1f}%)")
    
    # Anomaly statistics within event window
    event_anomalies = anomalies[event_mask].sum()
    event_total_anomalies = event_anomalies.sum()
    event_possible = event_mask.sum() * len(anomalies.columns)
    event_anomaly_rate = (event_total_anomalies / event_possible) * 100 if event_possible > 0 else 0
    top_anomalous_metrics = event_anomalies.nlargest(20)
    print(f"\nTop 10 most anomalous metrics:")
    for i, (metric, count) in enumerate(top_anomalous_metrics.items(), 1):
        percentage = (count / total_points) * 100
        print(f"  {i:2d}. {metric}: {count} ({percentage:.2f}%)")
    
    # Analyze first anomaly time for each metric in event window
    print(f"\nFirst anomaly time analysis for each metric in event window:")
    print("=" * 50)
    
    # Get event window time range
    event_times = df.index[event_mask]
    event_start_time = event_times.min()
    
    # Store first anomaly times
    first_anomaly_times = {}
    
    # Use narrow confidence interval-based anomaly start times (if provided)
    if anomaly_starts is not None:
        # Important: only analyze metrics that are also flagged as anomalous under wide confidence interval
        # This ensures logical consistency: only metrics ultimately judged as anomalous have their start times computed
        metrics_with_final_anomalies = event_anomalies[event_anomalies > 0].index
        
        for metric in metrics_with_final_anomalies:
            # Get narrow confidence interval-based anomaly points for this metric in event window
            metric_event_anomaly_starts = anomaly_starts.loc[event_mask, metric]
            anomaly_indices = metric_event_anomaly_starts[metric_event_anomaly_starts == 1].index
            
            if len(anomaly_indices) > 0:
                # Use first anomaly point found with narrow confidence interval as anomaly start time
                first_anomaly_time = anomaly_indices[0]
                time_from_event_start = first_anomaly_time - event_start_time
                first_anomaly_times[metric] = {
                    'time': first_anomaly_time,
                    'offset_from_event_start': time_from_event_start
                }
            else:
                # If no anomaly detected under narrow confidence interval but anomaly under wide confidence interval
                # Use wide confidence interval's first anomaly time
                metric_event_anomalies = anomalies.loc[event_mask, metric]
                wide_anomaly_indices = metric_event_anomalies[metric_event_anomalies == 1].index
                if len(wide_anomaly_indices) > 0:
                    first_anomaly_time = wide_anomaly_indices[0]
                    time_from_event_start = first_anomaly_time - event_start_time
                    first_anomaly_times[metric] = {
                        'time': first_anomaly_time,
                        'offset_from_event_start': time_from_event_start
                    }
    else:
        # If anomaly_starts not provided, fall back to using original anomalies
        metrics_with_anomalies = event_anomalies[event_anomalies > 0].index
        
        for metric in metrics_with_anomalies:
            # Get anomaly points for this metric in event window
            metric_event_anomalies = anomalies.loc[event_mask, metric]
            anomaly_indices = metric_event_anomalies[metric_event_anomalies == 1].index
            
            if len(anomaly_indices) > 0:
                first_anomaly_time = anomaly_indices[0]
                time_from_event_start = first_anomaly_time - event_start_time
                first_anomaly_times[metric] = {
                    'time': first_anomaly_time,
                    'offset_from_event_start': time_from_event_start
                }
    
    # Sort by first anomaly time
    sorted_first_anomalies = sorted(first_anomaly_times.items(), 
                                  key=lambda x: x[1]['time'])
    
    print(f"  Event window start time: {event_start_time}")
    print(f"  Metrics with anomalies: {len(sorted_first_anomalies)} (only includes metrics ultimately judged as anomalous)")
    
    # If narrow confidence interval was used, note this
    if anomaly_starts is not None:
        print(f"  Note: anomaly judgment uses confidence interval {detection_params.get('interval_width', 0.99995)}")
        print(f"        anomaly start time detection uses confidence interval {detection_params.get('anomaly_start_interval_width', 0.99)}")
    
    print(f"\n  First anomaly time per metric (in chronological order):")
    
    for i, (metric, info) in enumerate(sorted_first_anomalies[:15], 1):  # Show first 15
        offset_minutes = info['offset_from_event_start'].total_seconds() / 60
        print(f"    {i:2d}. {metric}")
        print(f"        First anomaly time: {info['time']}")
        print(f"        Offset from event start: {offset_minutes:.1f} minutes")
        
    if len(sorted_first_anomalies) > 15:
        print(f"    ... and {len(sorted_first_anomalies) - 15} more metrics")
    
    print(f"\nEvent window anomaly analysis:")
    print(f"  Total anomalies in event window: {event_total_anomalies}")
    print(f"  Event window anomaly rate: {event_anomaly_rate:.2f}%")
    
    return anomaly_counts, top_anomalous_metrics, event_anomaly_rate, sorted_first_anomalies


def visualize_results(df, anomalies, z_scores, detection_params, top_metrics,event_start,event_end, output_dir=f"{Config.BASE_PATH}/exam"):
    """Visualize anomaly detection results"""
    if top_metrics is None or len(top_metrics) == 0:
        return
    
    if PRINT_PROGRESS:
        print("\nGenerating visualization charts...")
    
    # Select top 4 most anomalous metrics for visualization
    metrics_to_plot = top_metrics.head(4).index.tolist()
    
    # Figure 1: Raw data and anomaly points
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(15, 4*len(metrics_to_plot)))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Plot raw data
        ax.plot(df.index.to_numpy(), df[metric].to_numpy(), label='Raw data', alpha=0.7, color='blue')
        
        # Mark anomaly points
        anomaly_times = anomalies.index[anomalies[metric] == 1]
        if len(anomaly_times) > 0:
            ax.scatter(anomaly_times.to_numpy(), df.loc[anomaly_times, metric].to_numpy(), 
                      color='red', s=50, alpha=0.8, label=f'Anomaly points ({len(anomaly_times)})', zorder=5)
        
        # Mark event window
        ax.axvspan(event_start, event_end, alpha=0.2, color='yellow', label='Event window')
        ax.axvline(event_start, color='orange', linestyle='--', alpha=0.8, label='Event start')
        ax.axvline(event_end, color='red', linestyle='--', alpha=0.8, label='Event end')        
        ax.set_title(f'{metric} - Prophet Anomaly Detection Results')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.xlabel('Time')
    plt.tight_layout()
    
    # Save first figure
    plt.savefig(f"{output_dir}/anomaly_detection_plot.png", dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    
    # Figure 2: Z-score distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot[:4]):
        ax = axes[i]
        
        # Plot z-score time series
        ax.plot(z_scores.index.to_numpy(), z_scores[metric].to_numpy(), alpha=0.7, color='green')
        ax.axhline(y=detection_params.get('tau', 2.0), color='red', linestyle='--', alpha=0.8, 
                  label=f'Threshold ({detection_params.get("tau", 2.0)})')
        ax.axhline(y=-detection_params.get('tau', 2.0), color='red', linestyle='--', alpha=0.8)
        
        # Mark anomaly points
        anomaly_times = anomalies.index[anomalies[metric] == 1]
        if len(anomaly_times) > 0:
            ax.scatter(anomaly_times.to_numpy(), z_scores.loc[anomaly_times, metric].to_numpy(), 
                      color='red', s=30, alpha=0.8, zorder=5)
        
        # Mark event window
        ax.axvspan(event_start, event_end, alpha=0.2, color='yellow')
        ax.axvline(event_start, color='orange', linestyle='--', alpha=0.8, label='Event start')
        ax.axvline(event_end, color='red', linestyle='--', alpha=0.8, label='Event end')

        ax.set_title(f'{metric} - Prophet Z-Score Time Series')
        ax.set_ylabel('Z-Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.xlabel('Time')
    plt.tight_layout()
    
    # Save second figure
    plt.savefig(f"{output_dir}/z_score_plot.png", dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    
    if PRINT_PROGRESS:
        print("Chart generation complete")
        print(f"  Figures saved:")
        print(f"    - {output_dir}/anomaly_detection_plot.png")
        print(f"    - {output_dir}/z_score_plot.png")


def generate_summary_report(df, anomalies, masks, detection_params, anomaly_counts, event_anomaly_rate):
    """Generate summary report"""
    if not PRINT_RESULTS:
        return ""
    
    # Basic statistics
    total_anomalies = anomalies.sum().sum()
    total_possible = len(anomalies) * len(anomalies.columns)
    anomaly_rate = (total_anomalies / total_possible) * 100
    
    # Event window parameters
    event_start = detection_params['event_start']
    event_end = detection_params['event_end']
    
    report = []
    report.append("Prophet-based Anomaly Detection Summary Report")
    report.append("=" * 60)
    report.append(f"Detection time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Data period: {df.index.min()} to {df.index.max()}")
    report.append(f"Total duration: {df.index.max() - df.index.min()}")
    report.append(f"Event window: {event_start} to {event_end}")
    
    report.append(f"\nDetection parameters:")
    for param, value in detection_params.items():
        report.append(f"  {param}: {value}")
    
    report.append(f"\nDataset summary:")
    report.append(f"  Total metrics: {len(df.columns)}")
    report.append(f"  Total time points: {len(df)}")
    quality_mask = masks['quality_mask']
    report.append(f"  Data quality points: {quality_mask.sum()} ({quality_mask.mean()*100:.1f}%)")
    
    report.append(f"\nAnomaly summary:")
    report.append(f"  Total anomalies detected: {total_anomalies}")
    report.append(f"  Overall anomaly rate: {anomaly_rate:.2f}%")
    report.append(f"  Metrics with anomalies: {(anomaly_counts > 0).sum()}/{len(anomaly_counts)}")
    
    # Event window analysis
    event_mask = masks['event_eval_mask']
    report.append(f"\nEvent window analysis:")
    report.append(f"  Event window duration: {event_end - event_start}")
    report.append(f"  Event window points: {event_mask.sum()}")
    report.append(f"  Event window anomaly rate: {event_anomaly_rate:.2f}%")
    
    # Top 10 most anomalous metrics
    top_anomalous_metrics = anomaly_counts.nlargest(10)
    report.append(f"\nTop 10 most anomalous metrics:")
    for i, (metric, count) in enumerate(top_anomalous_metrics.items(), 1):
        percentage = (count / len(anomalies)) * 100
        report.append(f"  {i:2d}. {metric}: {count} ({percentage:.2f}%)")
    
    # Recommendations
    report.append(f"\nRecommendations:")
    if event_anomaly_rate > 5:
        report.append(f"  ⚠️  Event window anomaly rate is high ({event_anomaly_rate:.1f}%) - recommend investigating potential issues")
    elif event_anomaly_rate > 1:
        report.append(f"  ⚡ Event window anomaly rate is moderate ({event_anomaly_rate:.1f}%) - recommend close monitoring")
    else:
        report.append(f"  ✅ Event window anomaly rate is low ({event_anomaly_rate:.1f}%) - system appears stable")
    
    if (anomaly_counts > 0).sum() > len(anomaly_counts) * 0.5:
        report.append(f"  ⚠️  Many metrics show anomalies - consider system-level investigation")
    
    report.append("=" * 60)
    
    report_text = "\\n".join(report)
    print(report_text)
    
    return report_text


def save_results(residuals, z_scores, anomalies, masks, detection_params,  
                top_anomalous_metrics=None, first_anomaly_times=None, df=None, 
                top_metrics_features=None, output_dir=f"{Config.BASE_PATH}/exam"):
    """Save detection results"""
    if not SAVE_RESULTS:
        return
    
    if PRINT_PROGRESS:
        print(f"\nSaving results to {output_dir}...")
    
    # # Save detection parameters and summary
    # with open(f"{output_dir}/detection_summary.txt", 'w', encoding='utf-8') as f:
    #     f.write(report_text)
    
    # Save detailed feature information for top 20 most anomalous metrics within event
    # Format data: 1) round numeric values to 2 decimal places 2) remove np.float64 format
    if top_metrics_features is not None:
        formatted_features = top_metrics_features.copy()

        # Round all numeric columns to 2 decimal places
        numeric_columns = formatted_features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            formatted_features[col] = formatted_features[col].round(2)

        # Process shape_sequence column, ensure it's a pure Python float list (double protection)
        if 'shape_sequence' in formatted_features.columns:
            def format_shape_sequence(seq):
                if isinstance(seq, list):
                    # Normal case: already a list, convert to Python float (1 decimal place)
                    return [float(round(float(x), 1)) for x in seq]
                elif isinstance(seq, str):
                    # Fallback: if already a string (saved and re-read), try cleaning
                    import re
                    # Extract all numbers
                    numbers = re.findall(r'-?\d+\.?\d*', seq)
                    if numbers:
                        return [float(round(float(n), 1)) for n in numbers]
                return seq

            formatted_features['shape_sequence'] = formatted_features['shape_sequence'].apply(format_shape_sequence)

        formatted_features.to_csv(f"{output_dir}/top_20_anomalous_metrics.csv", index=False)

    if PRINT_PROGRESS:
        print(f"  - {output_dir}/top_20_anomalous_metrics.csv")
    
    # Save first 20 metrics with earliest anomaly times
    if first_anomaly_times is not None:
        first_20_anomalies = first_anomaly_times[:20]
        event_mask = masks['event_eval_mask']
        event_start_time = df.index[event_mask].min() if event_mask.any() else None
        
        first_anomaly_data = []
        for i, (metric, info) in enumerate(first_20_anomalies, 1):
            offset_minutes = info['offset_from_event_start'].total_seconds() / 60
            offset_seconds = info['offset_from_event_start'].total_seconds()
            
            first_anomaly_data.append({
                'rank': i,
                'metric': metric,
                'first_anomaly_time': info['time'].strftime('%Y-%m-%d %H:%M:%S'),
                'event_start_time': event_start_time.strftime('%Y-%m-%d %H:%M:%S') if event_start_time else None,
                'offset_from_event_start_minutes': round(offset_minutes, 1),
                'offset_from_event_start_seconds': round(offset_seconds, 1)
            })
        
        first_anomaly_df = pd.DataFrame(first_anomaly_data)
        first_anomaly_df.to_csv(f"{output_dir}/first_20_anomaly_times.csv", index=False)
        
    if PRINT_PROGRESS:
        print("Results saved!")
        print(f"Files saved:")
        print(f"  - {output_dir}/detection_summary.txt")
        if top_metrics_features is not None and len(top_metrics_features) > 0:
            print(f"  - {output_dir}/top_20_anomalous_metrics.csv")
        if first_anomaly_times is not None:
            print(f"  - {output_dir}/first_20_anomaly_times.csv")
    top_20_anomalous_metrics_path = f"{output_dir}/top_20_anomalous_metrics.csv"
    first_20_anomaly_times_path = f"{output_dir}/first_20_anomaly_times.csv"
    return first_20_anomaly_times_path,top_20_anomalous_metrics_path


def detection(data_file, save_path, event_start, event_end, use_parallel=False):
    """Main function (Prophet-based anomaly detection)

    Args:
        data_file: metric data CSV file path
        save_path: result save directory
        event_start: anomaly start time
        event_end: anomaly end time
        use_parallel: whether to enable Prophet internal multiprocessing (default False).
            In multi-threaded scenarios like batch tasks, enabling ProcessPoolExecutor
            may trigger deadlocks, so it's disabled by default.
    """
    try:
        # 1. Set up plotting style
        setup_plotting()
        
        # 2. Load data
        df = load_data(data_file)
        
        # 3. Check data quality
        df = check_data_quality(df)
        
        # 4. Set up detection parameters
        # Data has been converted to Beijing time, directly convert input time strings
        # Set anomaly detection interval to one hour before the anomaly event
        anomaly_start = pd.Timestamp(event_start)  # Anomaly start time
        event_start = anomaly_start - pd.Timedelta(hours=1)  # Detection start: 1 hour before anomaly
        event_end = anomaly_start  # Detection end: anomaly start time
        
        if PRINT_PROGRESS:
            print(f"Event time settings:")
            print(f"  Anomaly start time: {anomaly_start}")
            print(f"  Detection start time: {event_start} (1 hour before anomaly)")
            print(f"  Detection end time: {event_end}")
            print(f"  Data time range: {df.index.min()} to {df.index.max()}")
        
        detection_params = setup_detection_params(df, event_start=event_start, event_end=event_end)
        
        # 5. Run Prophet anomaly detection (with chart saving enabled)
        residuals, z_scores, anomalies, masks, anomaly_starts = run_anomaly_detection(
            df, detection_params,
            save_charts=True,
            chart_save_path=save_path,
            use_parallel=use_parallel,
            n_workers=detection_params.get('n_workers', None)
        )
        
        # 6. Analyze results
        anomaly_counts, top_anomalous_metrics, event_anomaly_rate, first_anomaly_times = analyze_results(
            df, anomalies, masks, detection_params, anomaly_starts)
        
        # 6.1. Event window feature analysis
        if PRINT_PROGRESS:
            print("\nPerforming event window feature analysis...")
        
        # Compute features for the top 20 most anomalous metrics within event window
        top_metrics_features = analyze_event_window_features(df, event_start, event_end, top_anomalous_metrics)
        
        # 7. Visualize results (optional)
        #visualize_results(df, anomalies, z_scores, detection_params, top_anomalous_metrics, event_start, event_end, save_path)

        # # 8. Generate summary report
        # report_text = generate_summary_report(
        #     df, anomalies, masks, detection_params, anomaly_counts, event_anomaly_rate)
        
        # 9. Save results
        first_20_anomaly_times_path,top_20_anomalous_metrics_path = save_results(residuals, z_scores, anomalies, masks, detection_params, 
                    top_anomalous_metrics, first_anomaly_times, df, top_metrics_features, 
                    output_dir=save_path)

        if PRINT_PROGRESS:
            print("\nProphet-based anomaly detection analysis complete!")
        
        return first_20_anomaly_times_path,top_20_anomalous_metrics_path
            
    except Exception as e:
        print(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    case_id = 'risk2' 
    case_info = case_table[case_id]
    app = case_info['app_name'][0]
    app_group = case_info['app_groups'][0][0]
    
    anomaly_start_time = case_info['fault_start']
    anomaly_end_time = case_info['fault_end']
    data_file = f"/home/kuanjunhua/data_handle/collected_data_old/{case_id}/{app}_{app_group}/metric/all_metrics.csv"
    save_path = f"/home/kuanjunhua/data_handle/collected_data_old/{case_id}/{app}_{app_group}/summary"
    event_start = anomaly_start_time
    event_end = anomaly_end_time
    detection(data_file, save_path, event_start, event_end)

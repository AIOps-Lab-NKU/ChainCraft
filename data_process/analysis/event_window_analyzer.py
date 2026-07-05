#!/usr/bin/env python3
"""
Event Window Feature Analysis Module

Specialized in feature analysis of anomalous metrics within the event window (event_start to event_end)
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_baseline_mean(data_series: pd.Series, event_start: pd.Timestamp, 
                          baseline_window_min: int = 15, fallback_points: int = 50) -> float:
    """
    Calculate baseline mean before the event
    
    Args:
        data_series: data series
        event_start: event start time
        baseline_window_min: baseline window length (minutes)
        fallback_points: number of fallback points
        
    Returns:
        baseline mean
    """
    # Try to get data from 15 minutes before the event
    baseline_start = event_start - pd.Timedelta(minutes=baseline_window_min)
    baseline_mask = (data_series.index >= baseline_start) & (data_series.index < event_start)
    baseline_data = data_series[baseline_mask].dropna()
    
    if len(baseline_data) >= 3:
        return baseline_data.mean()
    
    # Fallback strategy: use the most recent N valid points before the event
    pre_event_data = data_series[data_series.index < event_start].dropna()
    if len(pre_event_data) >= fallback_points:
        return pre_event_data.tail(fallback_points).mean()
    elif len(pre_event_data) > 0:
        return pre_event_data.mean()
    
    # Final fallback: use global mean
    return data_series.dropna().mean()


def calculate_post_event_mean(data_series: pd.Series, event_end: pd.Timestamp, 
                             post_window_min: int = 10) -> Optional[float]:
    """
    Calculate mean after the event ends
    
    Args:
        data_series: data series
        event_end: event end time
        post_window_min: post-event window length (minutes)
        
    Returns:
        post-event mean, or None if no data available
    """
    post_start = event_end + pd.Timedelta(minutes=1)  # Avoid including the event end point
    post_end = event_end + pd.Timedelta(minutes=post_window_min + 1)
    post_mask = (data_series.index >= post_start) & (data_series.index <= post_end)
    post_data = data_series[post_mask].dropna()
    
    if len(post_data) >= 1:
        return post_data.mean()
    return None


def calculate_slope(data_series: pd.Series, event_start: pd.Timestamp, 
                   event_end: pd.Timestamp) -> float:
    """
    Calculate OLS slope within the event window
    
    Args:
        data_series: data series
        event_start: event start time
        event_end: event end time
        
    Returns:
        slope value (unit: value/minute)
    """
    event_mask = (data_series.index >= event_start) & (data_series.index <= event_end)
    event_data = data_series[event_mask].dropna()
    
    if len(event_data) < 2:
        return 0.0
    
    # Convert time to minutes
    time_minutes = [(t - event_start).total_seconds() / 60.0 for t in event_data.index]
    values = event_data.values
    
    try:
        slope, _, _, _, _ = stats.linregress(time_minutes, values)
        return slope
    except:
        return 0.0


def calculate_segmented_trends(data_series: pd.Series, event_start: pd.Timestamp, 
                              event_end: pd.Timestamp, min_segment_size: int = 3) -> Dict:
    """
    Adaptive segmented trend analysis - dynamically determine number of segments based on data length
    
    Args:
        data_series: data series
        event_start: event start time
        event_end: event end time
        min_segment_size: minimum number of points per segment
        
    Returns:
        dict containing trend information for each segment
    """
    event_mask = (data_series.index >= event_start) & (data_series.index <= event_end)
    event_data = data_series[event_mask].dropna()
    
    if len(event_data) < min_segment_size * 2:
        return {"segment_slopes": [], "trend_changes": 0, "pattern": "insufficient_data"}
    
    # Adaptively determine number of segments: ensure each segment has at least min_segment_size points
    n_segments = max(3, min(10, len(event_data) // min_segment_size))
    
    # Calculate slopes per segment
    segment_size = len(event_data) // n_segments
    segment_slopes = []
    segment_info = []  # Store detailed info for each segment
    
    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(event_data)
        
        segment_data = event_data.iloc[start_idx:end_idx]
        if len(segment_data) < 2:
            continue
            
        # Convert time to minutes
        time_minutes = [(t - segment_data.index[0]).total_seconds() / 60.0 
                       for t in segment_data.index]
        
        try:
            slope, _, r_value, _, _ = stats.linregress(time_minutes, segment_data.values)
            segment_slopes.append(slope)
            
            # Record detailed info for each segment
            segment_info.append({
                'slope': slope,
                'r_squared': r_value**2,
                'start_value': segment_data.iloc[0],
                'end_value': segment_data.iloc[-1],
                'max_value': segment_data.max(),
                'min_value': segment_data.min(),
                'range_ratio': (segment_data.max() - segment_data.min()) / abs(segment_data.mean()) if segment_data.mean() != 0 else 0
            })
        except:
            segment_slopes.append(0.0)
            segment_info.append({
                'slope': 0.0,
                'r_squared': 0.0,
                'start_value': segment_data.iloc[0] if len(segment_data) > 0 else 0,
                'end_value': segment_data.iloc[-1] if len(segment_data) > 0 else 0,
                'max_value': segment_data.max() if len(segment_data) > 0 else 0,
                'min_value': segment_data.min() if len(segment_data) > 0 else 0,
                'range_ratio': 0
            })
    
    # Analyze trend changes
    trend_changes = 0
    significant_changes = []
    
    if len(segment_slopes) >= 2:
        for i in range(1, len(segment_slopes)):
            # Detect trend direction changes (from positive to negative or vice versa)
            if len(segment_slopes) > i-1:
                # Dynamic threshold: consider absolute slope values of adjacent segments
                prev_slope = segment_slopes[i-1]
                curr_slope = segment_slopes[i]
                
                if abs(prev_slope) > 0.01 or abs(curr_slope) > 0.01:
                    # Direction change
                    if (prev_slope > 0) != (curr_slope > 0):
                        trend_changes += 1
                        significant_changes.append({
                            'position': i,
                            'from_slope': prev_slope,
                            'to_slope': curr_slope,
                            'change_magnitude': abs(curr_slope - prev_slope)
                        })
    
    # Detect short-term spikes (find in sub-segments)
    spike_segments = detect_spike_segments(segment_info)
    
    # Enhanced pattern recognition
    pattern = classify_segmented_pattern_enhanced(segment_slopes, trend_changes, 
                                                 significant_changes, spike_segments, n_segments)
    
    return {
        "segment_slopes": segment_slopes,
        "trend_changes": trend_changes,
        "significant_changes": significant_changes,
        "spike_segments": spike_segments,
        "n_segments": n_segments,
        "pattern": pattern
    }


def detect_spike_segments(segment_info: List[Dict]) -> List[Dict]:
    """
    Detect short-term spike segments
    
    Args:
        segment_info: detailed info for each segment
        
    Returns:
        detected spike segment info
    """
    spikes = []
    
    for i, seg in enumerate(segment_info):
        # Spike detection conditions:
        # 1. Large amplitude within segment (range_ratio > 1.0)
        # 2. Similar start/end values but significant change in between
        # 3. Low R² indicates poor linear fit (possibly a spike)
        
        range_ratio = seg.get('range_ratio', 0)
        r_squared = seg.get('r_squared', 1.0)
        start_val = seg.get('start_value', 0)
        end_val = seg.get('end_value', 0)
        max_val = seg.get('max_value', 0)
        min_val = seg.get('min_value', 0)
        
        # Start/end value similarity
        if abs(start_val) > 0:
            return_ratio = abs((end_val - start_val) / start_val)
        else:
            return_ratio = abs(end_val - start_val)
        
        # Spike detection conditions
        spike_conditions = [
            range_ratio > 0.5,           # Large intra-segment variation
            r_squared < 0.7,             # Poor linear fit
            return_ratio < 0.3           # Similar start/end values (regression characteristic)
        ]
        
        if sum(spike_conditions) >= 2:  # Satisfy at least 2 conditions
            spikes.append({
                'segment_index': i,
                'range_ratio': range_ratio,
                'r_squared': r_squared,
                'return_ratio': return_ratio,
                'spike_type': 'upward' if max_val > max(start_val, end_val) else 'downward',
                'spike_magnitude': max_val - min_val
            })
    
    return spikes


def calculate_monotonicity(data_series: pd.Series, event_start: pd.Timestamp, 
                          event_end: pd.Timestamp) -> Dict:
    """
    Monotonicity analysis - detect whether overall trend is monotonically increasing/decreasing
    
    Args:
        data_series: data series
        event_start: event start time
        event_end: event end time
        
    Returns:
        monotonicity analysis results
    """
    event_mask = (data_series.index >= event_start) & (data_series.index <= event_end)
    event_data = data_series[event_mask].dropna()
    
    if len(event_data) < 3:
        return {"monotonic": False, "direction": "none", "monotonic_ratio": 0.0}
    
    values = event_data.values
    
    # Calculate change direction between adjacent points
    increases = 0
    decreases = 0
    
    for i in range(1, len(values)):
        diff = values[i] - values[i-1]
        if diff > 0:
            increases += 1
        elif diff < 0:
            decreases += 1
    
    total_changes = increases + decreases
    
    if total_changes == 0:
        return {"monotonic": True, "direction": "constant", "monotonic_ratio": 1.0}
    
    # Determine monotonicity
    increase_ratio = increases / total_changes
    decrease_ratio = decreases / total_changes
    
    if increase_ratio >= 0.8:
        return {"monotonic": True, "direction": "increasing", "monotonic_ratio": increase_ratio}
    elif decrease_ratio >= 0.8:
        return {"monotonic": True, "direction": "decreasing", "monotonic_ratio": decrease_ratio}
    else:
        return {"monotonic": False, "direction": "mixed", "monotonic_ratio": max(increase_ratio, decrease_ratio)}


def classify_segmented_pattern(segment_slopes: List[float], trend_changes: int) -> str:
    """
    Classify pattern based on segmented slopes (compatibility function)
    """
    return classify_segmented_pattern_enhanced(segment_slopes, trend_changes, [], [], len(segment_slopes))


def classify_segmented_pattern_enhanced(segment_slopes: List[float], trend_changes: int,
                                       significant_changes: List[Dict], spike_segments: List[Dict],
                                       n_segments: int) -> str:
    """
    Enhanced segmented pattern classification
    
    Args:
        segment_slopes: list of slopes for each segment
        trend_changes: number of trend changes
        significant_changes: list of significant changes
        spike_segments: list of spike segments
        n_segments: total number of segments
        
    Returns:
        detailed pattern type
    """
    if len(segment_slopes) == 0:
        return "unknown"
    
    # Prioritize spike pattern detection
    if len(spike_segments) > 0:
        if len(spike_segments) == 1:
            spike = spike_segments[0]
            if spike['spike_type'] == 'upward':
                return "short_spike"  # Short-term upward spike
            else:
                return "short_dip"    # Short-term downward dip
        else:
            return "multiple_spikes"  # Multiple spikes
    
    # Threshold setting - adaptive based on number of segments
    slope_threshold = max(0.005, 0.05 / n_segments)  # More segments, smaller threshold
    
    # Analyze characteristics of significant changes
    if len(significant_changes) > 0:
        # Sort by change magnitude, find the most significant change
        max_change = max(significant_changes, key=lambda x: x['change_magnitude'])
        
        # If there is a very significant single change
        if len(significant_changes) == 1 and max_change['change_magnitude'] > 1.0:
            pos = max_change['position']
            if pos <= n_segments * 0.3:  # Change occurs in first 30%
                return "early_transition"
            elif pos >= n_segments * 0.7:  # Change occurs in last 30%
                return "late_transition"
            else:
                return "mid_transition"   # Change occurs in middle
    
    # Traditional classification logic - enhanced version
    if trend_changes >= 3:
        return "highly_oscillatory"  # Highly oscillatory
    elif trend_changes == 2:
        # Analyze whether it's W-shaped or M-shaped
        if len(segment_slopes) >= 4:
            first_half_trend = sum(1 for s in segment_slopes[:len(segment_slopes)//2] if s > slope_threshold)
            second_half_trend = sum(1 for s in segment_slopes[len(segment_slopes)//2:] if s > slope_threshold)
            
            if first_half_trend > second_half_trend:
                return "m_shaped"  # M-shaped: rise then fall then minor change
            else:
                return "w_shaped"  # W-shaped: fall then rise then change
        return "multi_phase"
    elif trend_changes == 1:
        if len(segment_slopes) >= 2:
            # More precise turning point analysis
            mid_point = len(segment_slopes) // 2
            first_half_avg = np.mean(segment_slopes[:mid_point]) if mid_point > 0 else 0
            second_half_avg = np.mean(segment_slopes[mid_point:])
            
            if first_half_avg > slope_threshold and second_half_avg < -slope_threshold:
                return "rise_then_fall"  # Rise then fall
            elif first_half_avg < -slope_threshold and second_half_avg > slope_threshold:
                return "fall_then_rise"  # Fall then rise
            else:
                return "gradual_transition"  # Gradual transition
        return "single_turning"
    elif all(s > slope_threshold for s in segment_slopes):
        # Check if it's an accelerating/decelerating pattern
        if len(segment_slopes) >= 3:
            slopes_increasing = all(segment_slopes[i] < segment_slopes[i+1] 
                                  for i in range(len(segment_slopes)-1))
            if slopes_increasing:
                return "accelerating_rise"  # Accelerating rise
        return "consistent_rise"  # Consistent rise
    elif all(s < -slope_threshold for s in segment_slopes):
        # Check if it's an accelerating/decelerating fall
        if len(segment_slopes) >= 3:
            slopes_decreasing = all(abs(segment_slopes[i]) < abs(segment_slopes[i+1]) 
                                  for i in range(len(segment_slopes)-1))
            if slopes_decreasing:
                return "accelerating_fall"  # Accelerating fall
        return "consistent_fall"  # Consistent fall
    elif all(abs(s) <= slope_threshold for s in segment_slopes):
        return "stable"  # Stable
    else:
        # Check if it's a gradual drift pattern
        slope_variance = np.var(segment_slopes) if len(segment_slopes) > 1 else 0
        if slope_variance < 0.01:
            return "gentle_drift"  # Gentle drift
        else:
            return "irregular"     # Irregular variation


def calculate_oscillation_score(data_series: pd.Series, event_start: pd.Timestamp, 
                               event_end: pd.Timestamp) -> float:
    """
    Calculate oscillation score (local extrema count / duration in minutes)
    
    Args:
        data_series: data series
        event_start: event start time
        event_end: event end time
        
    Returns:
        oscillation score
    """
    event_mask = (data_series.index >= event_start) & (data_series.index <= event_end)
    event_data = data_series[event_mask].dropna()
    
    if len(event_data) < 3:
        return 0.0
    
    # Calculate number of local extrema
    values = event_data.values
    local_extrema = 0
    
    for i in range(1, len(values) - 1):
        # Local maximum or minimum
        if ((values[i] > values[i-1] and values[i] > values[i+1]) or 
            (values[i] < values[i-1] and values[i] < values[i+1])):
            local_extrema += 1
    
    duration_min = (event_end - event_start).total_seconds() / 60.0
    if duration_min <= 0:
        return 0.0
    
    return local_extrema / max(duration_min, 1.0)


def calculate_anomaly_ratio(data_series: pd.Series, event_start: pd.Timestamp, 
                           event_end: pd.Timestamp, threshold: float) -> float:
    """
    Calculate ratio of data points exceeding threshold within the window
    
    Args:
        data_series: data series
        event_start: event start time
        event_end: event end time
        threshold: anomaly threshold
        
    Returns:
        anomaly ratio (0.0-1.0)
    """
    event_mask = (data_series.index >= event_start) & (data_series.index <= event_end)
    event_data = data_series[event_mask].dropna()
    
    if len(event_data) == 0:
        return 0.0
    
    anomaly_count = sum(1 for v in event_data.values if v > threshold)
    return anomaly_count / len(event_data)


def calculate_max_continuous_anomaly_duration(data_series: pd.Series, event_start: pd.Timestamp, 
                                            event_end: pd.Timestamp, threshold: float) -> int:
    """
    Calculate maximum continuous duration exceeding threshold (assumes data frequency of 1 point per minute)
    
    Args:
        data_series: data series
        event_start: event start time
        event_end: event end time
        threshold: anomaly threshold
        
    Returns:
        maximum continuous anomaly duration (minutes)
    """
    event_mask = (data_series.index >= event_start) & (data_series.index <= event_end)
    event_data = data_series[event_mask].dropna()
    
    if len(event_data) == 0:
        return 0
    
    max_duration = 0
    current_duration = 0
    
    for i, (idx, value) in enumerate(event_data.items()):
        if value > threshold:
            if i > 0:
                time_diff = (idx - event_data.index[i-1]).total_seconds() / 60.0
                if time_diff <= 1.5:
                    current_duration += 1
                else:
                    current_duration = 1
            else:
                current_duration = 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_duration


def calculate_trend_slope_in_window(data_series: pd.Series, event_start: pd.Timestamp, 
                                   event_end: pd.Timestamp) -> float:
    """
    Slope from linear regression on data within the window
    
    Args:
        data_series: data series
        event_start: event start time
        event_end: event end time
        
    Returns:
        linear regression slope k
    """
    event_mask = (data_series.index >= event_start) & (data_series.index <= event_end)
    event_data = data_series[event_mask].dropna()
    
    if len(event_data) < 2:
        return 0.0
    
    time_minutes = [(t - event_start).total_seconds() / 60.0 for t in event_data.index]
    values = event_data.values
    
    try:
        slope, _, _, _, _ = stats.linregress(time_minutes, values)
        return slope
    except:
        return 0.0


def calculate_volatility_in_window(data_series: pd.Series, event_start: pd.Timestamp, 
                                  event_end: pd.Timestamp) -> float:
    """
    Calculate coefficient of variation (std/mean), describing fluctuation degree
    
    Args:
        data_series: data series
        event_start: event start time
        event_end: event end time
        
    Returns:
        coefficient of variation
    """
    event_mask = (data_series.index >= event_start) & (data_series.index <= event_end)
    event_data = data_series[event_mask].dropna()
    
    if len(event_data) < 2:
        return 0.0
    
    mean_val = event_data.mean()
    std_val = event_data.std()
    
    if abs(mean_val) < 1e-6:
        return float('inf') if std_val > 0 else 0.0
    
    return abs(std_val / mean_val)


def calculate_shape_sequence(data_series: pd.Series, event_start: pd.Timestamp, 
                            event_end: pd.Timestamp, n_points: int = 6) -> List[float]:
    """
    Resample 30 minutes of data into 6 points (5-minute averages), rounded to 1 decimal place
    
    Args:
        data_series: data series
        event_start: event start time
        event_end: event end time
        n_points: number of resampled points, default 6
        
    Returns:
        shape sequence, list of length n_points
    """
    event_mask = (data_series.index >= event_start) & (data_series.index <= event_end)
    event_data = data_series[event_mask].dropna()
    
    if len(event_data) == 0:
        return [0.0] * n_points
    
    if len(event_data) <= n_points:
        # Convert to Python float type, avoid numpy types
        result = [float(x) for x in event_data.values[:n_points]]
        while len(result) < n_points:
            result.append(result[-1] if result else 0.0)
        return [round(x, 1) for x in result]
    
    duration = (event_end - event_start).total_seconds()
    interval_seconds = duration / n_points
    
    shape_sequence = []
    
    for i in range(n_points):
        interval_start = event_start + pd.Timedelta(seconds=i * interval_seconds)
        interval_end = event_start + pd.Timedelta(seconds=(i + 1) * interval_seconds)
        
        interval_mask = (event_data.index >= interval_start) & (event_data.index < interval_end)
        interval_data = event_data[interval_mask]
        
        if len(interval_data) > 0:
            # Convert to Python float type
            shape_sequence.append(float(round(interval_data.mean(), 1)))
        else:
            shape_sequence.append(float(round(event_data.mean(), 1)))
    
    return shape_sequence


def calculate_seasonality_change(data_series: pd.Series, event_start: pd.Timestamp,
                                event_end: pd.Timestamp, days_back: int) -> float:
    """
    Calculate change rate compared to a past day
    
    Args:
        data_series: data series
        event_start: event start time
        event_end: event end time
        days_back: lookback days (1=yesterday, 2=day before yesterday)
        
    Returns:
        change rate (current_mean - past_mean) / past_mean
    """
    event_mask = (data_series.index >= event_start) & (data_series.index <= event_end)
    event_data = data_series[event_mask].dropna()
    
    if len(event_data) == 0:
        return 0.0
    
    current_mean = event_data.mean()
    
    past_start = event_start - pd.Timedelta(days=days_back)
    past_end = event_end - pd.Timedelta(days=days_back)
    past_mask = (data_series.index >= past_start) & (data_series.index <= past_end)
    past_data = data_series[past_mask].dropna()
    
    if len(past_data) == 0:
        return 0.0
    
    past_mean = past_data.mean()
    
    if abs(past_mean) < 1e-6:
        return 0.0
    
    return (current_mean - past_mean) / abs(past_mean)


def calculate_severity_quantile(data_series: pd.Series, event_start: pd.Timestamp,
                               event_end: pd.Timestamp, historical_days: int = 7) -> float:
    """
    Calculate percentile rank of the maximum value in current window among historical values
    
    Args:
        data_series: data series
        event_start: event start time
        event_end: event end time
        historical_days: number of historical days for percentile calculation
        
    Returns:
        percentile rank (0-100)
    """
    event_mask = (data_series.index >= event_start) & (data_series.index <= event_end)
    event_data = data_series[event_mask].dropna()
    
    if len(event_data) == 0:
        return 0.0
    
    current_max = event_data.max()
    
    historical_start = event_start - pd.Timedelta(days=historical_days)
    historical_mask = (data_series.index >= historical_start) & (data_series.index < event_start)
    historical_data = data_series[historical_mask].dropna()
    
    if len(historical_data) == 0:
        return 50.0
    
    from scipy.stats import percentileofscore
    
    percentile = percentileofscore(historical_data.values, current_max, kind='rank')
    return round(percentile, 2)


def classify_anomaly_pattern(features: Dict) -> str:
    """
    Classify anomaly patterns based on enhanced features
    
    Args:
        features: feature dictionary
        
    Returns:
        anomaly pattern type
    """
    # Original features
    duration = features.get('duration_min', 0)
    slope = features.get('slope_in_window', 0)
    volatility = features.get('volatility_in_window', 0)
    oscillation = features.get('oscillation_score', 0)
    amplitude_percent = abs(features.get('amplitude_percent', 0))
    
    # New features
    segmented_info = features.get('segmented_info', {})
    monotonic_info = features.get('monotonic_info', {})
    
    segmented_pattern = segmented_info.get('pattern', 'unknown')
    trend_changes = segmented_info.get('trend_changes', 0)
    
    # Prioritize segmented pattern recognition
    if segmented_pattern == "rise_then_fall":
        return "inverted_spike"  # Inverted spike (rise then fall)
    elif segmented_pattern == "fall_then_rise":
        return "u_recovery"  # U-shaped recovery (fall then rise)
    elif segmented_pattern == "multi_phase":
        return "complex_oscillation"  # Complex oscillation
    elif segmented_pattern == "consistent_rise":
        return "stair_step"  # Stair-step rise
    elif segmented_pattern == "consistent_fall":
        return "degradation"  # Performance degradation
    
    # Combined with monotonicity analysis
    is_monotonic = monotonic_info.get('monotonic', False)
    direction = monotonic_info.get('direction', 'none')
    monotonic_ratio = monotonic_info.get('monotonic_ratio', 0.0)
    
    if is_monotonic and monotonic_ratio > 0.8:
        if direction == "increasing":
            return "monotonic_rise"
        elif direction == "decreasing":
            return "monotonic_fall"
        elif direction == "constant":
            return "flat"
    
    # Detect high-frequency oscillation
    if trend_changes >= 2 and oscillation > 0.5:
        return "high_freq_oscillation"
    
    # Traditional classification logic (as fallback)
    slope_threshold = volatility * 0.1 if volatility > 0 else 0.1
    high_volatility_threshold = volatility > features.get('baseline_std', 1.0) * 2
    high_oscillation_threshold = oscillation > 0.5
    short_duration_threshold = duration <= 5
    high_amplitude_threshold = amplitude_percent > 50
    
    # Classification rules
    if short_duration_threshold and high_amplitude_threshold:
        return "spike"
    elif high_volatility_threshold and high_oscillation_threshold:
        return "oscillation"
    elif abs(slope) < slope_threshold and not high_volatility_threshold:
        return "flat"
    elif slope > slope_threshold and features.get('amplitude_percent', 0) > 0:
        return "stair_step"
    elif slope < -slope_threshold and features.get('amplitude_percent', 0) < 0:
        return "drop"
    else:
        return "drift"


def analyze_event_window_features(df: pd.DataFrame, event_start: pd.Timestamp, 
                                event_end: pd.Timestamp, top_anomalous_metrics: pd.Series) -> pd.DataFrame:
    """
    Analyze features of the most anomalous metrics within the event window
    
    Args:
        df: original data DataFrame
        event_start: event start time
        event_end: event end time
        top_anomalous_metrics: Series of most anomalous metrics (metric_name -> anomaly count)
        
    Returns:
        DataFrame containing specified features
    """
    results = []
    
    for metric_name in top_anomalous_metrics.index:
        if metric_name not in df.columns:
            continue
            
        data_series = df[metric_name]
        
        # Extract data within the event window
        event_mask = (data_series.index >= event_start) & (data_series.index <= event_end)
        event_data = data_series[event_mask].dropna()
        
        if len(event_data) == 0:
            continue
        
        # 1. baseline_mean - baseline mean
        baseline_mean = calculate_baseline_mean(data_series, event_start)
        
        # Get baseline window data for calculating baseline_std
        baseline_start = event_start - pd.Timedelta(minutes=15)
        baseline_mask = (data_series.index >= baseline_start) & (data_series.index < event_start)
        baseline_data = data_series[baseline_mask].dropna()
        
        # If baseline data is insufficient, use same fallback strategy as baseline_mean
        if len(baseline_data) < 3:
            pre_event_data = data_series[data_series.index < event_start].dropna()
            if len(pre_event_data) >= 50:
                baseline_data = pre_event_data.tail(50)
            elif len(pre_event_data) > 0:
                baseline_data = pre_event_data
            else:
                baseline_data = data_series.dropna()
        
        # 2. anomaly_mean - mean within anomaly window
        anomaly_mean = event_data.mean()
        
        # 3. peak_value - maximum value within anomaly window
        peak_value = event_data.max()
        
        # 4. amplitude_percent - (anomaly_mean - baseline_mean) / max(|baseline_mean|, eps) * 100
        eps = 1e-6
        amplitude_percent = (anomaly_mean - baseline_mean) / max(abs(baseline_mean), eps) * 100
        
        # 5. peak_diff - peak_value - baseline_mean
        peak_diff = peak_value - baseline_mean
        
        # 6. baseline_std - standard deviation of baseline window
        baseline_std = baseline_data.std() if len(baseline_data) > 1 else 0.0
        
        # 7. anomaly_std - standard deviation of anomaly window
        anomaly_std = event_data.std()
        
        # 8. var_ratio - anomaly_std / max(baseline_std, eps)
        var_ratio = anomaly_std / max(baseline_std, eps)
        
        # 9. Persistence features
        threshold = baseline_mean + 3 * baseline_std
        anomaly_ratio = calculate_anomaly_ratio(data_series, event_start, event_end, threshold)
        max_continuous_anomaly_duration = calculate_max_continuous_anomaly_duration(
            data_series, event_start, event_end, threshold)
        
        # 10. Trend & Shape features
        trend_slope = calculate_trend_slope_in_window(data_series, event_start, event_end)
        volatility = calculate_volatility_in_window(data_series, event_start, event_end)
        shape_sequence = calculate_shape_sequence(data_series, event_start, event_end)
        
        # 11. Context features
        seasonality_change_yesterday = calculate_seasonality_change(data_series, event_start, event_end, 1)
        seasonality_change_day_before = calculate_seasonality_change(data_series, event_start, event_end, 2)
        severity_quantile = calculate_severity_quantile(data_series, event_start, event_end)
        
        # Build feature dictionary - includes all features
        features = {
            'metric_name': metric_name,
            'anomaly_count_in_event': top_anomalous_metrics[metric_name],
            'baseline_mean': baseline_mean,
            'anomaly_mean': anomaly_mean,
            'peak_value': peak_value,
            'amplitude_percent': amplitude_percent,
            'peak_diff': peak_diff,
            'baseline_std': baseline_std,
            'anomaly_std': anomaly_std,
            'var_ratio': var_ratio,
            # Persistence features
            'anomaly_ratio': anomaly_ratio,
            'max_continuous_anomaly_duration': max_continuous_anomaly_duration,
            # Trend & Shape features
            'trend_slope': trend_slope,
            'volatility': volatility,
            'shape_sequence': shape_sequence,
            # Context features
            'seasonality_change_yesterday': seasonality_change_yesterday,
            'seasonality_change_day_before': seasonality_change_day_before,
            'severity_quantile': severity_quantile
        }
        # Only keep entries where anomaly_count_in_event is not 0
        if features['anomaly_count_in_event'] == 0:
            continue
        
        results.append(features)
    
    # Convert to DataFrame
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    
    # Sort by anomaly count
    result_df = result_df.sort_values('anomaly_count_in_event', ascending=False).reset_index(drop=True)
    
    return result_df


if __name__ == "__main__":
    # Test code
    print("Event window feature analysis module loaded")
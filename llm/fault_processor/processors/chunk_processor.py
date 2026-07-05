from datetime import datetime
from typing import Dict, List, Any


class ChunkProcessor:
    """Process various text chunks in fault reports (events, symptoms, propagation chains)"""
    
    def __init__(self):
        self.root_cause_type_map = {
            '资源瓶颈/耗尽': 'resource_exhaustion',
            '依赖故障（DB/TDDL/RDS）': 'dependency_failure',
            '网络问题': 'network_issue',
            '配置错误': 'configuration_error',
            '代码缺陷': 'code_defect'
        }
        
        self.layer_display_names = {
            'dependency_layer': 'Dependency',
            'core_layer': 'Core', 
            'inbound_layer': 'Inbound'
        }
        
        self.severity_map = {
            'Low': 'Low',
            'Medium': 'Medium',
            'High': 'High',
            'Critical': 'Critical'
        }
    
    def process_event_chunk(self, report_json: Dict, case_id: str) -> List[Dict]:
        """
        Process event chunk
        
        Args:
            report_json: Fault analysis report dict
            case_id: Case unique identifier
            
        Returns:
            List containing event chunk dicts
        """
        fault_analysis = report_json.get('fault_analysis', {})
        
        given_root_cause = fault_analysis.get('given_root_cause', {})
        assessment = fault_analysis.get('assessment', {})
        abnormal_metrics = fault_analysis.get('abnormal_metrics', [])
        layered_analysis = fault_analysis.get('layered_analysis', {})
        abnormal_metrics_by_layer = layered_analysis.get('abnormal_metrics_by_layer', {})
        
        root_cause_type = given_root_cause.get('type', 'Unknown')
        suspected_component = given_root_cause.get('suspected_component', 'Unknown component')
        hypothesis = given_root_cause.get('hypothesis', 'No hypothesis')
        rationale = assessment.get('rationale', 'No assessment rationale')
        confidence = assessment.get('confidence', 0.0)
        first_failure_time = fault_analysis.get('first_failure_time')
        
        # Calculate time window
        time_info = self._calculate_time_window(first_failure_time, abnormal_metrics)
        
        # Get affected layers
        affected_layers = self._get_affected_layers(abnormal_metrics_by_layer)
        affected_layer_str = ','.join(affected_layers) if affected_layers else 'unknown'
        
        root_cause_type_en = self.root_cause_type_map.get(root_cause_type, 'unknown')
        
        event_content = f"""[EVENT]
case_id={case_id}
start={time_info['start_time']}; end={time_info['end_time']}; duration={time_info['duration']}
root_cause_type={root_cause_type_en}; component={suspected_component};
affected_layer={affected_layer_str}

recovery_time={time_info['end_time']};
description={hypothesis}
summary={rationale}"""
        
        event_chunk = {
            'page_content': event_content,
            'metadata': {
                'case_id': case_id,
                'start_time': time_info['start_time'],
                'end_time': time_info['end_time'],
                'duration': time_info['duration'],
                'root_cause_type': root_cause_type,
                'root_cause_type_en': root_cause_type_en,
                'suspected_component': suspected_component,
                'severity': "None",
                'affected_layers': affected_layers,
                'confidence': confidence,
                'mitigation': "Unknown operation",
                'final_state': "recovered"
            }
        }
        
        return [event_chunk]
    
    def process_symptom_chunks(self, report_json: Dict, case_id: str) -> List[Dict]:
        """
        Process symptom chunks
        
        Args:
            report_json: Fault analysis report dict
            case_id: Case unique identifier
            
        Returns:
            List containing symptom chunk dicts
        """
        fault_analysis = report_json.get('fault_analysis', {})
        abnormal_metrics = fault_analysis.get('abnormal_metrics', [])
        layered_analysis = fault_analysis.get('layered_analysis', {})
        abnormal_metrics_by_layer = layered_analysis.get('abnormal_metrics_by_layer', {})
        print("report_json in process_symptom_chunks:", report_json)
        print(f"abnormal_metrics: {abnormal_metrics}")
        print(f"abnormal_metrics_by_layer: {abnormal_metrics_by_layer}")
        
        symptom_chunks = []
        
        # Build metric details mapping
        metric_details = {
            metric_info.get('metric'): metric_info 
            for metric_info in abnormal_metrics
        }
        
        # Check if any layer has abnormal metrics
        has_any_metrics = any(layer_metrics for layer_metrics in abnormal_metrics_by_layer.values())
        
        if not has_any_metrics:
            # Return default content when no abnormal metrics found
            default_symptom_content = """[SYMPTOM_GROUP]
layer=none; window=Unknown time window; group=No abnormal metrics
items:
 • No abnormal metrics found
"""
            symptom_chunks.append({
                'page_content': default_symptom_content,
                'metadata': {
                    'case_id': case_id,
                    'layer': 'none',
                    'layer_display_name': 'None',
                    'metrics_count': 0,
                    'time_window': 'Unknown time window',
                    'group_name': 'No abnormal metrics'
                }
            })
        else:
            # Normal processing for cases with abnormal metrics
            for layer_name, layer_metrics in abnormal_metrics_by_layer.items():
                if not layer_metrics:
                    continue
                
                symptom_chunk = self._build_symptom_chunk(
                    layer_name, 
                    layer_metrics, 
                    metric_details, 
                    fault_analysis, 
                    case_id
                )
                symptom_chunks.append(symptom_chunk)
        
        return symptom_chunks
    
    def process_chain_chunks(self, report_json: Dict, case_id: str) -> List[Dict]:
        """
        Process propagation chain chunks
        
        Args:
            report_json: Fault analysis report dict
            case_id: Case unique identifier
            
        Returns:
            List containing propagation chain chunk dicts
        """
        fault_analysis = report_json.get('fault_analysis', {})
        layered_analysis = fault_analysis.get('layered_analysis', {})
        propagation_chains = layered_analysis.get('propagation_chains', [])
        
        chain_chunks = []
        
        if not propagation_chains:
            # Return default content when no propagation chains found
            default_chain_content = """[CHAIN]
signature=No propagation chain
nodes=none
confidence=0.0
edges:
 - No propagation chains analyzed
notes=No propagation chains analyzed"""
            chain_chunks.append({
                'page_content': default_chain_content,
                'metadata': {
                    'case_id': case_id,
                    'chain_id': 'none',
                    'chain_type': 'none',
                    'signature': 'No propagation chain',
                    'nodes_count': 0,
                    'confidence': 0.0
                }
            })
        else:
            # Normal processing for cases with propagation chains
            for chain_info in propagation_chains:
                chain_chunk = self._build_chain_chunk(chain_info, case_id)
                if chain_chunk:
                    chain_chunks.append(chain_chunk)
        
        return chain_chunks
    
    def process_fault_report(self, report_json: Dict, case_id: str) -> Dict[str, List]:
        """
        Process complete fault analysis report
        
        Args:
            report_json: Fault analysis report dict
            case_id: Case unique identifier
            
        Returns:
            Dict containing events, symptoms, chains
        """
        return {
            'events': self.process_event_chunk(report_json, case_id),
            'symptoms': self.process_symptom_chunks(report_json, case_id),
            'chains': self.process_chain_chunks(report_json, case_id)
        }
    
    def process_inference_report(self, report_json: Dict, case_id: str) -> Dict[str, List]:
        """
        Process inference report
        
        Args:
            report_json: Inference report dict
            case_id: Case unique identifier
            
        Returns:
            Dict containing symptoms, chains
        """
        return {
            'symptoms': self.process_symptom_chunks(report_json, case_id),
            'chains': self.process_chain_chunks(report_json, case_id)
        }
    
    # Private helper methods
    def _calculate_time_window(self, first_failure_time: str, abnormal_metrics: List) -> Dict:
        """Calculate time window information"""
        start_time = "Unknown"
        end_time = "Unknown"
        duration = "Unknown"
        
        if first_failure_time:
            all_times = [first_failure_time]
            for metric in abnormal_metrics:
                if metric.get('first_anomaly_time'):
                    all_times.append(metric.get('first_anomaly_time'))
            
            earliest = min(all_times)
            latest = max(all_times)
            
            try:
                earliest_dt = datetime.fromisoformat(earliest.replace('T', ' ').replace('Z', ''))
                latest_dt = datetime.fromisoformat(latest.replace('T', ' ').replace('Z', ''))
                
                start_time = earliest_dt.strftime('%Y-%m-%d %H:%M')
                end_time = latest_dt.strftime('%H:%M')
                
                duration_minutes = int((latest_dt - earliest_dt).total_seconds() / 60)
                if duration_minutes >= 60:
                    hours = duration_minutes // 60
                    mins = duration_minutes % 60
                    duration = f"{hours}h{mins}m" if mins > 0 else f"{hours}h"
                else:
                    duration = f"{duration_minutes}min"
            except:
                pass
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration
        }
    
    def _get_affected_layers(self, abnormal_metrics_by_layer: Dict) -> List[str]:
        """Get affected layers"""
        affected_layers = []
        for layer_name, metrics in abnormal_metrics_by_layer.items():
            if metrics:
                layer_short = layer_name.replace('_layer', '')
                affected_layers.append(layer_short)
        return affected_layers
    
    def _build_symptom_chunk(self, layer_name: str, layer_metrics: List, 
                            metric_details: Dict, fault_analysis: Dict, case_id: str) -> Dict:
        """Build symptom chunk"""
        earliest_time = None
        latest_time = None
        items_text = []
        
        for metric_info in layer_metrics:
            metric_name = metric_info.get('metric')
            detailed_info = metric_details.get(metric_name, {})
            
            direction = metric_info.get('direction', 'Unknown')
            first_anomaly_time = metric_info.get('first_anomaly_time')
            
            # Calculate time delta
            first_failure_time = fault_analysis.get('first_failure_time')
            delta_t = self._calculate_delta_time(first_failure_time, first_anomaly_time)
            
            # Update time window
            if first_anomaly_time:
                if earliest_time is None or first_anomaly_time < earliest_time:
                    earliest_time = first_anomaly_time
                if latest_time is None or first_anomaly_time > latest_time:
                    latest_time = first_anomaly_time
            
            # Get strength info (if exists)
            strength_info = detailed_info.get('strength', {})
            severity = strength_info.get('severity', 'Unknown')
            
            item_text = f" • metric={metric_name}; dir={direction}; Δt={delta_t}; severity={severity};"
            items_text.append(item_text)
        
        # Format time window
        time_window = self._format_time_window(earliest_time, latest_time)
        
        group_name = f"{self.layer_display_names.get(layer_name, layer_name)}_Metrics"
        
        symptom_group_content = f"""[SYMPTOM_GROUP]
layer={layer_name}; window={time_window}; group={group_name}
items:
{chr(10).join(items_text)}
"""
        
        return {
            'page_content': symptom_group_content,
            'metadata': {
                'case_id': case_id,
                'layer': layer_name,
                'layer_display_name': self.layer_display_names.get(layer_name, layer_name),
                'metrics_count': len(layer_metrics),
                'time_window': time_window,
                'group_name': group_name
            }
        }
    
    def _build_chain_chunk(self, chain_info: Dict, case_id: str) -> Dict:
        """Build propagation chain chunk"""
        chain_id = chain_info.get('chain_id')
        chain_summary = chain_info.get('summary', 'No summary')
        chain_confidence = chain_info.get('confidence', 0.0)
        chain_steps = chain_info.get('chain', [])
        
        if not chain_steps:
            return None
        
        nodes = []
        edges_text = []
        
        for i, step in enumerate(chain_steps):
            from_metric = step.get('from', 'Unknown source')
            to_metric = step.get('to', 'Unknown target')
            
            if i == 0:
                nodes.append(from_metric)
            nodes.append(to_metric)
            
            relation = step.get('relation', 'Unknown relation')
            when = step.get('when', 'Unknown time')
            observed = step.get('observed', False)
            inferred = step.get('inferred', False)
            evidence_ref = step.get('evidence_ref', [])
            
            # Handle time, support "same_as_previous" type
            if when == "same_as_previous" and i > 0 and edges_text:
                # Get time from previous edge
                time_part = edges_text[-1].split("when=")[1].split(";")[0]
            else:
                time_part = when.split('T')[1][:5] if 'T' in str(when) else str(when)
            
            edge_text = f" - {from_metric}->{to_metric}; relation={relation}; when={time_part}; observed={observed}; inferred={inferred};"
            if evidence_ref:
                edge_text += f" evidence={','.join(evidence_ref)};"
            edges_text.append(edge_text)
        
        signature = ' > '.join(nodes)
        nodes_str = ','.join(nodes)
        
        complete_chain_content = f"""[CHAIN]
        signature={signature}
        nodes={nodes_str}
        confidence={chain_confidence}
        edges:
{chr(10).join(edges_text)}
        notes={chain_summary}"""
        
        return {
            'page_content': complete_chain_content,
            'metadata': {
                'case_id': case_id,
                'chain_id': chain_id,
                'chain_type': 'complete',
                'signature': signature,
                'nodes_count': len(nodes),
                'confidence': chain_confidence
            }
        }
    
    def _calculate_delta_time(self, first_failure_time: str, first_anomaly_time: str) -> str:
        """Calculate time delta"""
        if not first_failure_time or not first_anomaly_time:
            return "Unknown"
        
        try:
            failure_dt = datetime.fromisoformat(first_failure_time.replace('T', ' ').replace('Z', ''))
            anomaly_dt = datetime.fromisoformat(first_anomaly_time.replace('T', ' ').replace('Z', ''))
            delta_minutes = int((anomaly_dt - failure_dt).total_seconds() / 60)
            return f"+{delta_minutes}m" if delta_minutes >= 0 else f"{delta_minutes}m"
        except:
            return "Unknown"
    
    def _format_time_window(self, earliest_time: str, latest_time: str) -> str:
        """Format time window"""
        if not earliest_time or not latest_time:
            return "Unknown time window"
        
        try:
            early_date = earliest_time.split('T')[0]
            early_time = earliest_time.split('T')[1][:5]
            late_time = latest_time.split('T')[1][:5]
            return f"{early_date} {early_time}~{late_time}"
        except:
            return f"{earliest_time}~{latest_time}"

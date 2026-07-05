from .BaseAgent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from .metric_layer_config import MetricLayerConfig
import os
import json
import re
from config import Config

class MetricAnalysisAgent(BaseAgent):
    def __init__(self):
        # Initialize base class, use OpenAI compatible URL
        super().__init__()
    
    def analyze_case(self,
                    metric_order_csv_path=None, metric_feature_csv_path=None,
                    log_order_csv_path=None, log_feature_csv_path=None,
                    save_path=None):
        """
        Complete case analysis workflow, including reading prompt templates, preparing data, calling LLM and saving results
        
        Args:
            given_root_cause (str): Given root cause
            hypothesis (str): Hypothesis
            suspected_component (str): Suspected component
            metric_order_csv_path (str): Metric anomaly order CSV file path
            metric_feature_csv_path (str): Metric features CSV file path
            log_order_csv_path (str): Log anomaly order CSV file path
            log_feature_csv_path (str): Log features CSV file path
            save_path (str): Result save path
            
        Returns:
            str: Analysis result
        """
        # Read prompt templates
        with open(f"{Config.BASE_PATH}/llm/prompts/metric_analysis_user.txt", "r") as file:
            user_prompt = file.read()
        with open(f"{Config.BASE_PATH}/llm/prompts/metric_analysis_system.txt", "r") as file:
            system_prompt = file.read()
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(user_prompt),
        ])
        
        # Read CSV file contents
        if metric_order_csv_path and os.path.exists(metric_order_csv_path):
            with open(metric_order_csv_path, "r") as file:
                metric_order_csv = file.read()
            # If file is empty, assign "None"
            if not metric_order_csv.strip():
                metric_order_csv = "None"
        else:
            metric_order_csv = "None"
            
        if metric_feature_csv_path and os.path.exists(metric_feature_csv_path):
            with open(metric_feature_csv_path, "r") as file:
                metric_feature_csv = file.read()
            # If file is empty, assign "None"
            if not metric_feature_csv.strip():
                metric_feature_csv = "None"
        else:
            metric_feature_csv = "None"
            
        if log_order_csv_path and os.path.exists(log_order_csv_path):
            with open(log_order_csv_path, "r") as file:
                log_order_csv = file.read()
        else:
            log_order_csv = "None"
            
        if log_feature_csv_path and os.path.exists(log_feature_csv_path):
            with open(log_feature_csv_path, "r") as file:
                log_feature_csv = file.read()
        else:
            log_feature_csv = "None"
        
        # Prepare input data
        input_data = {
            "metric_order_csv": metric_order_csv,
            "metric_feature_csv": metric_feature_csv,
        }
        
        # Format prompt
        formatted_prompt = prompt.format(**input_data)
        
        # Call base class analysis method
        result = self.analyze_with_prompt(system_prompt, formatted_prompt)
        
        # Save result if save path is specified
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as file:
                file.write(result)
            self.logger.info(f"Analysis result saved to: {save_path}")
        
        return result
    
    def clean_json_string(self, json_str):
        """
        Clean common formatting errors in JSON strings
        
        Args:
            json_str (str): JSON string to clean
            
        Returns:
            str: Cleaned JSON string
        """
        try:
            # Remove possible Markdown code block markers
            json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'^```\s*$', '', json_str, flags=re.MULTILINE)
            
            # Remove trailing commas (commas before } or ])
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            
            # Remove extra newlines and whitespace
            json_str = json_str.strip()
            
            return json_str
        except Exception as e:
            self.logger.warning(f"Error cleaning JSON string: {e}")
            return json_str
    
    def aggressive_clean_json(self, json_str):
        """
        More aggressive JSON cleaning method, handles complex formatting errors
        
        Args:
            json_str (str): JSON string to clean
            
        Returns:
            str: Cleaned JSON string
        """
        try:
            # Remove comments (// and /* */ forms)
            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            
            # Handle more complex comma issues: commas before object or array end
            # Use more precise regex for nested structures
            lines = json_str.split('\n')
            cleaned_lines = []
            
            for i, line in enumerate(lines):
                # Check if current line ends with comma and next line is } or ]
                stripped_line = line.strip()
                if stripped_line.endswith(','):
                    # Find next non-empty line
                    next_non_empty = None
                    for j in range(i + 1, len(lines)):
                        next_stripped = lines[j].strip()
                        if next_stripped:
                            next_non_empty = next_stripped
                            break
                    
                    # If next non-empty line is } or ], remove comma
                    if next_non_empty and (next_non_empty.startswith('}') or next_non_empty.startswith(']')):
                        line = line.rstrip().rstrip(',') + '\n' if line.endswith('\n') else line.rstrip().rstrip(',')
                
                cleaned_lines.append(line)
            
            result = '\n'.join(cleaned_lines)
            
            # Finally apply basic cleaning again
            result = self.clean_json_string(result)
            
            return result
        except Exception as e:
            self.logger.error(f"Error during aggressive JSON cleaning: {e}")
            return json_str
    
    def process_metric_analysis_result(self, analysis_result):
        """
        Process result from analyze_with_prompt, adding layer field
        
        Args:
            analysis_result (str): JSON string returned by analyze_with_prompt
            
        Returns:
            list: Metric info list with added layer field
        """
        
        try:
            # Parse JSON result
            if isinstance(analysis_result, str):
                # Clean JSON string first
                cleaned_json = self.clean_json_string(analysis_result)
                # self.logger.debug(f"Original JSON: {analysis_result[:200]}...")
                # self.logger.debug(f"Cleaned JSON: {cleaned_json[:200]}...")
                
                try:
                    metrics_data = json.loads(cleaned_json)
                except json.JSONDecodeError as first_error:
                    # If still unable to parse after cleaning, try more aggressive method
                    self.logger.warning(f"First JSON parse attempt failed: {first_error}, trying more aggressive cleaning")
                    
                    # Try to fix more complex JSON format issues
                    aggressive_cleaned = self.aggressive_clean_json(cleaned_json)
                    metrics_data = json.loads(aggressive_cleaned)
            else:
                metrics_data = analysis_result
                
            # Ensure list format
            if not isinstance(metrics_data, list):
                self.logger.error("Analysis result is not in list format")
                return []
            
            # Process each metric
            processed_metrics = []
            for metric in metrics_data:
                if isinstance(metric, dict) and 'metric_name' in metric:
                    # Get metric layer
                    metric_name = metric['metric_name']
                    layer = MetricLayerConfig.get_metric_layer(metric_name)
                    
                    # Add layer field
                    metric['layer'] = layer
                    
                    # Log
                    #self.logger.info(f"Metric {metric_name} classified to {layer}")
                    
                    # Special case checks and warnings
                    if 'middleware_hsf_consumer_' in metric_name and layer != 'dependency_layer':
                        self.logger.warning(f"Warning: {metric_name} should belong to dependency_layer, but was classified to {layer}")
                    elif 'middleware_hsf_provider_' in metric_name and layer != 'inbound_layer':
                        self.logger.warning(f"Warning: {metric_name} should belong to inbound_layer, but was classified to {layer}")
                    
                    processed_metrics.append(metric)
                else:
                    self.logger.warning(f"Skipping invalid metric data: {metric}")
            
            # Group statistics by layer
            layer_stats = {}
            for metric in processed_metrics:
                layer = metric.get('layer', 'unknown')
                if layer not in layer_stats:
                    layer_stats[layer] = []
                layer_stats[layer].append(metric['metric_name'])
            
            # Print layered statistics
            self.logger.info("Metric layer statistics:")
            for layer, metrics in layer_stats.items():
                layer_desc = MetricLayerConfig.get_layer_description(layer)
                self.logger.info(f"  {layer} ({layer_desc}): {len(metrics)} metrics")
                for metric_name in metrics:
                    self.logger.info(f"    - {metric_name}")
            
            return processed_metrics
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON result: {e}")
            self.logger.error(f"Original result: {analysis_result}")
            return []
        except Exception as e:
            self.logger.error(f"Error processing analysis result: {e}")
            self.logger.error(f"Original result: {analysis_result}")
            return []
    
    def analyze_case_with_layers(self, 
                               metric_order_csv_path=None, 
                               metric_feature_csv_path=None,
                               log_order_csv_path=None, 
                               log_feature_csv_path=None,
                               save_path=None):
        """
        Complete workflow for case analysis with layer information
        
        Args: same as analyze_case
        
        Returns:
            list: Metric analysis result with added layer field
        """
        # Call original analyze_case method to get analysis result
        raw_result = self.analyze_case(
            metric_order_csv_path=metric_order_csv_path,
            metric_feature_csv_path=metric_feature_csv_path,
            log_order_csv_path=log_order_csv_path,
            log_feature_csv_path=log_feature_csv_path,
            save_path=None  # Don't save yet, save after processing
        )
        
        # Process analysis result, add layer field
        processed_result = self.process_metric_analysis_result(raw_result)
        
        # Save processed result if save path is specified
        if save_path and processed_result:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(processed_result, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Analysis result with layer info saved to: {save_path}")
        
        return processed_result

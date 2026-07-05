import json
from .BaseAgent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import os
from config import Config

class InferenceAgent(BaseAgent):
    def __init__(self):
        # Initialize base class, use parent class default logic (read URL from config)
        super().__init__()
    
    def _filter_metrics_by_severity(self, metric_data, severity_filter):
        """
        Filter metric data by operational_severity
        
        Args:
            metric_data (str or list): Metric data, can be JSON string or list
            severity_filter (str): Filter level ['Low', 'Medium', 'High', 'Critical']
            
        Returns:
            str: Filtered metric data in JSON string format
        """
        if not severity_filter:
            return metric_data if isinstance(metric_data, str) else json.dumps(metric_data, ensure_ascii=False)
        
        # Define severity level priorities
        severity_hierarchy = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        
        # If filter level is not in predefined range, return original data
        if severity_filter not in severity_hierarchy:
            self.logger.warning(f"Unknown severity level: {severity_filter}, returning original data")
            return metric_data if isinstance(metric_data, str) else json.dumps(metric_data, ensure_ascii=False)
        
        try:
            # If input is string, try to parse as JSON
            if isinstance(metric_data, str):
                metrics = json.loads(metric_data)
            else:
                metrics = metric_data
            
            # If not list format, return original data
            if not isinstance(metrics, list):
                return metric_data if isinstance(metric_data, str) else json.dumps(metric_data, ensure_ascii=False)
            
            # Get threshold for filter level
            filter_threshold = severity_hierarchy[severity_filter]
            
            # Filter metrics
            filtered_metrics = []
            for metric in metrics:
                if isinstance(metric, dict) and 'operational_assessment' in metric:
                    operational_assessment = metric['operational_assessment']
                    if isinstance(operational_assessment, dict) and 'operational_severity' in operational_assessment:
                        metric_severity = operational_assessment['operational_severity']
                        
                        # Keep metric if severity level >= filter threshold
                        if metric_severity in severity_hierarchy:
                            if severity_hierarchy[metric_severity] >= filter_threshold:
                                filtered_metrics.append(metric)
                        else:
                            # Keep metric if severity level is unknown
                            filtered_metrics.append(metric)
                    else:
                        # Keep metric if no operational_severity field
                        filtered_metrics.append(metric)
                else:
                    # Keep metric if format does not match expectations
                    filtered_metrics.append(metric)
            
            self.logger.info(f"Severity filtering: {len(metrics)} metrics originally, filter level {severity_filter}, kept {len(filtered_metrics)} metrics")
            return json.dumps(filtered_metrics, ensure_ascii=False)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON data: {e}, returning original data")
            return metric_data if isinstance(metric_data, str) else json.dumps(metric_data, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error filtering metrics: {e}, returning original data")
            return metric_data if isinstance(metric_data, str) else json.dumps(metric_data, ensure_ascii=False)
    
    def inference_case(self, metric_analysis_path=None, causal_analysis=None,
                    save_path=None, severity_filter=None,
                    enable_iteration=False,
                    case_id=None,
                    app=None,
                    app_group=None,
                    iteration_save_dir=None,
                    use_causal_analysis=True):
        """
        Complete case inference workflow, including reading prompt templates, preparing data, calling LLM and saving results

        Args:
            metric_analysis_path (str): Metric analysis result file path
            causal_analysis (str): Causal relation statistics CSV
            save_path (str): Result save path
            severity_filter (str): Severity filter level, options: ['Low', 'Medium', 'High', 'Critical']
                                  'Low' keeps all metrics, 'Medium' keeps all except Low, etc.
            enable_iteration (bool): Whether to enable iterative correction mechanism (default False)
            case_id (str): Case ID (required when iteration is enabled)
            app (str): Application name (required when iteration is enabled)
            app_group (str): Application group name (required when iteration is enabled)
            use_causal_analysis (bool): Whether to use causal analysis information (default True).
                When False, uses prompt template without causal info (case_inference_*_no_causal.txt).

        Returns:
            str or dict: Analysis result. Returns str if enable_iteration=False; otherwise returns dict with iteration info
        """
        # Select prompt template based on whether to use causal analysis
        if use_causal_analysis:
            user_prompt_file = "case_inference_user.txt"
            system_prompt_file = "case_inference_system.txt"
        else:
            user_prompt_file = "case_inference_user_no_causal.txt"
            system_prompt_file = "case_inference_system_no_causal.txt"

        # Read prompt templates
        with open(f"{Config.BASE_PATH}/llm/prompts/{user_prompt_file}", "r") as file:
            user_prompt = file.read()
        with open(f"{Config.BASE_PATH}/llm/prompts/{system_prompt_file}", "r") as file:
            system_prompt = file.read()
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(user_prompt),
        ])
        
        # Read metric analysis file and apply severity filtering
        if metric_analysis_path and os.path.exists(metric_analysis_path):
            with open(metric_analysis_path, "r") as file:
                raw_metric_analysis = file.read()
            
            # Apply severity filter
            metric_analysis = self._filter_metrics_by_severity(raw_metric_analysis, severity_filter)
        else:
            metric_analysis = "None"
        
        # Read causal relation statistics and convert to model-friendly JSON format
        # causal_analysis = self._parse_causal_statistics(causal_statistics_path)
        
        
        # Prepare input data
        input_data = {
            "metric_analysis": metric_analysis,
        }
        if use_causal_analysis:
            input_data["causal_analysis"] = causal_analysis
        
        # Format prompt
        formatted_prompt = prompt.format(**input_data)
        
        # Call base class analysis method
        result = self.analyze_with_prompt(system_prompt,formatted_prompt)

        # Save result if save path is specified
        if save_path:
            # Ensure directory exists
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, "w") as file:
                file.write(result)
            self.logger.info(f"Analysis result saved to: {save_path}")

            # Also save causal_analysis (CSV format) to same directory
            if causal_analysis:
                causal_csv_path = os.path.join(save_dir, "causal_analysis.csv")
                with open(causal_csv_path, "w") as file:
                    file.write(causal_analysis)
                self.logger.info(f"Causal analysis CSV saved to: {causal_csv_path}")

        # If iterative correction mechanism is enabled
        if enable_iteration:
            if not case_id:
                self.logger.warning("Iteration enabled but no case_id provided, skipping iteration")
                return result

            self.logger.info("Enabling iterative correction mechanism")
            print(f"\n{'='*70}")
            print("Enabling iterative correction mechanism")
            print(f"{'='*70}")

            try:
                # Parse propagation chains from initial inference result
                print("Raw inference result:")
                print(result)

                # Clean markdown code block markers
                result_clean = result.strip()
                if result_clean.startswith('```'):
                    # Find first newline (skip ```json or ``` line)
                    first_newline = result_clean.find('\n')
                    if first_newline != -1:
                        result_clean = result_clean[first_newline + 1:]
                    # Remove trailing ```
                    if result_clean.endswith('```'):
                        result_clean = result_clean[:-3].strip()

                result_json = json.loads(result_clean)

                # Extract propagation chains
                if 'fault_analysis' in result_json and 'layered_analysis' in result_json['fault_analysis']:
                    layered_analysis = result_json['fault_analysis']['layered_analysis']
                    if 'propagation_chains' in layered_analysis:
                        initial_chains = layered_analysis['propagation_chains']
                    else:
                        self.logger.warning("propagation_chains not found in inference result")
                        return result
                else:
                    self.logger.warning("Inference result format unexpected, cannot extract propagation chains")
                    return result

                # Create IterationController
                from llm.iteration.iteration_controller import IterationController

                controller = IterationController(
                    case_id=case_id,
                    app=app,
                    app_group=app_group
                )

                # Prepare data for iteration
                # Read metric_analysis data
                if metric_analysis_path and os.path.exists(metric_analysis_path):
                    with open(metric_analysis_path, "r") as file:
                        metric_data = json.load(file)
                else:
                    metric_data = []

                # Run iterative correction
                iteration_result = controller.run_iteration(
                    initial_chains=initial_chains,
                    causal_analysis=causal_analysis,
                    metric_analysis=metric_data,
                    save_dir=iteration_save_dir if iteration_save_dir else (os.path.dirname(save_path) if save_path else None)
                )

                # Update propagation chains in result with corrected version
                result_json['fault_analysis']['layered_analysis']['propagation_chains'] = iteration_result['final_chains']

                # Add iteration info
                result_json['iteration_info'] = {
                    'enabled': True,
                    'converged': iteration_result['converged'],
                    'stop_reason': iteration_result['stop_reason'],
                    'summary': iteration_result['iteration_summary'],
                    'log_path': iteration_result['log_path']
                }

                # Save updated result
                updated_result = json.dumps(result_json, ensure_ascii=False, indent=2)
                if save_path:
                    with open(save_path, "w") as file:
                        file.write(updated_result)
                    self.logger.info(f"Iteration-corrected result saved to: {save_path}")

                # Return result with iteration info
                return {
                    'inference_result': updated_result,
                    'iteration_result': iteration_result
                }

            except Exception as e:
                self.logger.error(f"Error during iterative correction: {str(e)}")
                import traceback
                traceback.print_exc()
                self.logger.warning("Iteration correction failed, returning original inference result")
                return result

        return result

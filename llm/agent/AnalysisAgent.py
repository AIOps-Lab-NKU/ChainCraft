from .BaseAgent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import os
import json
from config import Config

class AnalysisAgent(BaseAgent):
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
    
    def _apply_m1_constraint(self, result, causal_analysis, system_prompt,
                              formatted_prompt, max_resamples=0, drop_invalid=False):
        """
        M1 causal constraint: validate and repair propagation_chains returned by LLM.

        - Validate: every edge in a chain must exist in the PCMCI causal graph
        - Repair: remove unknown nodes / deduplicate / insert missing edges / truncate unrepairable segments
        - Resample: when max_resamples > 0, re-invoke LLM for completely invalid chains
        - Report: append m1_report field to result JSON with processing summary

        Returns: processed JSON string
        """
        try:
            from llm.agent.causal_constrained_generator import create_constrained_generator
        except Exception as e:
            self.logger.warning(f"M1 module load failed, skipping: {e}")
            return result

        try:
            result_json = json.loads(result)
        except json.JSONDecodeError as e:
            self.logger.warning(f"M1 failed to parse LLM JSON, skipping: {e}")
            return result

        # Locate propagation_chains
        try:
            layered = result_json["fault_analysis"]["layered_analysis"]
            chains = layered.get("propagation_chains") or []
        except (KeyError, TypeError):
            self.logger.info("M1: No propagation_chains in result, skipping")
            return result

        if not chains:
            self.logger.info("M1: propagation_chains is empty, skipping")
            return result

        try:
            generator = create_constrained_generator(causal_analysis)
        except Exception as e:
            self.logger.warning(f"M1 failed to construct generator: {e}")
            return result

        # First validation + repair
        outcome = generator.constrain_propagation_chains(chains)
        constrained = outcome["constrained_chains"]
        report = outcome["report"]

        # Optional: resample completely unrepairable chains (default 0 times, avoid extra LLM calls)
        resample_attempts = 0
        if max_resamples > 0 and report["dropped"] > 0:
            for attempt in range(max_resamples):
                self.logger.info(f"M1: Resampling attempt {attempt + 1}")
                try:
                    new_result_raw = self.analyze_with_prompt(system_prompt, formatted_prompt)
                    new_result_raw = self.clean_json_string(new_result_raw)
                    new_json = json.loads(new_result_raw)
                    new_chains = (
                        new_json.get("fault_analysis", {})
                        .get("layered_analysis", {})
                        .get("propagation_chains", [])
                    )
                except Exception as e:
                    self.logger.warning(f"M1 resampling failed: {e}")
                    break
                new_outcome = generator.constrain_propagation_chains(new_chains)
                resample_attempts += 1
                if new_outcome["report"]["dropped"] < report["dropped"]:
                    outcome = new_outcome
                    constrained = outcome["constrained_chains"]
                    report = outcome["report"]
                    if report["dropped"] == 0:
                        break

        # Write back results
        if drop_invalid:
            layered["propagation_chains"] = constrained
        else:
            # Keep all chains (valid + repaired overwrite original; invalid marked invalid but keep original nodes)
            merged = []
            constrained_by_id = {c.get("chain_id"): c for c in constrained}
            for original in chains:
                cid = original.get("chain_id")
                if cid in constrained_by_id:
                    merged.append(constrained_by_id[cid])
                else:
                    flagged = dict(original)
                    flagged["m1_status"] = "invalid_unrepaired"
                    merged.append(flagged)
            layered["propagation_chains"] = merged

        result_json["m1_report"] = {
            "enabled": True,
            "graph_summary": generator.graph.summary(),
            "chain_summary": {
                "total": report["total"],
                "fully_valid_before": report["fully_valid_before"],
                "fully_valid_after": report["fully_valid_after"],
                "repaired": report["repaired"],
                "dropped": report["dropped"],
            },
            "resample_attempts": resample_attempts,
            "per_chain": report["per_chain"],
        }

        self.logger.info(
            "M1 completed: total={total} before_valid={before} after_valid={after} "
            "repaired={rep} dropped={drop}".format(
                total=report["total"],
                before=report["fully_valid_before"],
                after=report["fully_valid_after"],
                rep=report["repaired"],
                drop=report["dropped"],
            )
        )

        return json.dumps(result_json, ensure_ascii=False, indent=2)

    def analyze_case(self, given_root_cause, hypothesis, suspected_component,
                    metric_analysis_path=None,
                    causal_analysis=None,
                    save_path=None,
                    severity_filter=None,
                    enable_iteration=False,
                    case_id=None,
                    app=None,
                    app_group=None,
                    use_m1=False,
                    m1_max_resamples=0,
                    m1_drop_invalid=False,
                    iteration_save_dir=None,
                    use_causal_analysis=True):
        """
        Complete case analysis workflow, including reading prompt templates, preparing data, calling LLM and saving results

        Args:
            given_root_cause (str): Given root cause
            hypothesis (str): Hypothesis
            suspected_component (str): Suspected component
            metric_analysis_path (str): Metric analysis result file path
            causal_analysis (str): Causal relation statistics CSV
            save_path (str): Result save path
            severity_filter (str): Severity filter level, options: ['Low', 'Medium', 'High', 'Critical']
                                  'Low' keeps all metrics, 'Medium' keeps all except Low, etc.
            enable_iteration (bool): Whether to enable iterative correction mechanism (default False)
            case_id (str): Case ID (required when iteration is enabled)
            app (str): Application name (required when iteration is enabled)
            app_group (str): Application group name (required when iteration is enabled)
            use_m1 (bool): Whether to enable M1 causal constrained chain generation (default False).
                When enabled, validates and repairs LLM output propagation_chains against PCMCI causal graph.
            m1_max_resamples (int): Max resample attempts for invalid chains when M1 is enabled. 0 means
                repair only without resampling (default when no external LLM re-call dependency).
            m1_drop_invalid (bool): Whether to drop chains that remain invalid after repair when M1 is enabled.
                False keeps original LLM output but marks as invalid.
            use_causal_analysis (bool): Whether to use causal analysis information (default True).
                When False, uses prompt template without causal info (case_analysis_*_no_causal.txt).

        Returns:
            str or dict: Analysis result. Returns str if enable_iteration=False; otherwise returns dict with iteration info
        """
        # Select prompt template based on whether to use causal analysis
        if use_causal_analysis:
            user_prompt_file = "case_analysis_user.txt"
            system_prompt_file = "case_analysis_system.txt"
        else:
            user_prompt_file = "case_analysis_user_no_causal.txt"
            system_prompt_file = "case_analysis_system_no_causal.txt"

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
            print(metric_analysis)
        else:
            metric_analysis = "None"
        
        # Read causal relation statistics and convert to model-friendly JSON format
        # causal_analysis = self._parse_causal_statistics(causal_statistics_path)
        
        
        
        # Prepare input data
        input_data = {
            "given_root_cause": given_root_cause,
            "hypothesis": hypothesis,
            "suspected_component": suspected_component,
            "metric_analysis": metric_analysis,
        }
        if use_causal_analysis:
            input_data["causal_analysis"] = causal_analysis
        
        # Format prompt
        formatted_prompt = prompt.format(**input_data)


        
        # Call base class analysis method
        result = self.analyze_with_prompt(system_prompt, formatted_prompt)
        result = self.clean_json_string(result)

        # M1: Causal constrained chain generation (soft mode: validate + repair + optional resampling)
        if use_m1:
            result = self._apply_m1_constraint(
                result=result,
                causal_analysis=causal_analysis,
                system_prompt=system_prompt,
                formatted_prompt=formatted_prompt,
                max_resamples=m1_max_resamples,
                drop_invalid=m1_drop_invalid,
            )

        # Save result if save path is specified
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as file:
                file.write(result)
            self.logger.info(f"Analysis result saved to: {save_path}")

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
                # Parse propagation chains from initial analysis result
                import json
                result_json = json.loads(result)

                # Extract propagation chains
                if 'fault_analysis' in result_json and 'layered_analysis' in result_json['fault_analysis']:
                    layered_analysis = result_json['fault_analysis']['layered_analysis']
                    if 'propagation_chains' in layered_analysis:
                        initial_chains = layered_analysis['propagation_chains']
                    else:
                        self.logger.warning("propagation_chains not found in analysis result")
                        return result
                else:
                    self.logger.warning("Analysis result format unexpected, cannot extract propagation chains")
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
                    'analysis_result': updated_result,
                    'iteration_result': iteration_result
                }

            except Exception as e:
                self.logger.error(f"Error during iterative correction: {str(e)}")
                import traceback
                traceback.print_exc()
                self.logger.warning("Iteration correction failed, returning original analysis result")
                return result

        return result

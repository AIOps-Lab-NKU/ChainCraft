#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iteration Configuration Module

Define all configuration parameters for the iterative refinement mechanism
"""


class IterationConfig:
    """Iteration config class, manages all parameters for the iteration process"""

    # ============ Iteration Control Parameters ============

    # Maximum iteration rounds
    MAX_ITERATIONS = 3

    # ============ Evaluation Weight Parameters ============

    # Weights for four evaluation dimensions (sum to 1.0)
    # Strategy: significantly reduce coverage-driven weight, strengthen causal logic quality
    WEIGHT_CAUSAL_LOGIC = 0.40           # Causal logic consistency weight（35% → 40%）
    WEIGHT_COMPLETENESS = 0.35           # Chain completeness weight（30% → 35%）
    WEIGHT_LAYER_RATIONALITY = 0.20      # Layer propagation rationality weight (unchanged)
    WEIGHT_EXPLANATION_POWER = 0.05      # Anomaly explanation power weight（15% → 5%，significantly reduced）

    # ============ Evaluation Criteria Parameters ============

    # Score thresholds for each dimension
    CAUSAL_LOGIC_THRESHOLD = 0.80          # Minimum causal logic consistency score
    COMPLETENESS_THRESHOLD = 0.80          # Minimum chain completeness score
    LAYER_RATIONALITY_THRESHOLD = 0.75     # Minimum layer propagation rationality score
    EXPLANATION_POWER_THRESHOLD = 0.70     # Minimum anomaly explanation power score (new)

    # ============ Issue Severity Definition ============

    SEVERITY_LEVELS = ['low', 'medium', 'high', 'critical']

    # Impact of different severity levels on score (penalty)
    SEVERITY_PENALTIES = {
        'low': 0.05,
        'medium': 0.10,
        'high': 0.20,
        'critical': 0.35
    }

    # ============ Execution Mode Parameters ============

    # Whether to evaluate only without refinement (fast mode)
    EVALUATION_ONLY = False

    # Whether to save intermediate results for each iteration
    SAVE_INTERMEDIATE_RESULTS = True

    # Timeout for a single iteration round (seconds)
    ITERATION_TIMEOUT = 300

    # ============ Logging Parameters ============

    # Whether to enable verbose logging
    VERBOSE_LOGGING = True

    # Log save path template
    LOG_PATH_TEMPLATE = "{case_dir}/summary/iteration_log.json"

    # ============ LLM Call Parameters ============

    # Evaluator Agent temperature
    EVALUATOR_TEMPERATURE = 0.3

    # Refiner Agent temperature
    REFINER_TEMPERATURE = 0.5

    # LLM call timeout (seconds)
    LLM_TIMEOUT = 60

    # ============ Execution Checker Parameters ============

    # Whether to enable execution checking
    ENABLE_EXECUTION_CHECKING = True

    # Strict mode (whether to rollback on failure)
    EXECUTION_CHECK_STRICT_MODE = False

    # Whether to allow minor deviations
    ALLOW_MINOR_DEVIATIONS = True

    # Whether to save execution check results
    SAVE_EXECUTION_CHECK_RESULTS = True

    # ============ Quality Comparator Parameters ============

    # Whether to enable quality comparison
    ENABLE_QUALITY_COMPARISON = True

    # Quality comparison strict mode (stop iteration if new version is worse)
    COMPARISON_STRICT_MODE = True

    # Whether to save comparison results
    SAVE_COMPARISON_RESULTS = True

    # ============ Decision Control Parameters ============

    # Soft issue count threshold (below this value chains are considered good enough)
    SOFT_ISSUE_THRESHOLD = 1

    # Execution miss ratio threshold (above this ratio execution is considered unreliable)
    EXECUTION_MISS_RATIO_THRESHOLD = 0.5

    # Maximum consecutive execution failures
    MAX_EXECUTION_FAILS = 2

    # Whether to stop on first tie (True: stop once, False: allow multiple)
    STOP_ON_FIRST_TIE = True

    # Whether to stop on first worse (True: stop once, False: allow multiple)
    STOP_ON_FIRST_WORSE = True

    # ============ Method Definitions ============

    @classmethod
    def validate(cls):
        """Validate configuration parameter reasonability"""
        errors = []

        # Check iteration round count
        if cls.MAX_ITERATIONS < 1:
            errors.append("MAX_ITERATIONSmust be >= 1")

        # Check weight sum
        weight_sum = (cls.WEIGHT_CAUSAL_LOGIC +
                     cls.WEIGHT_COMPLETENESS +
                     cls.WEIGHT_LAYER_RATIONALITY +
                     cls.WEIGHT_EXPLANATION_POWER)
        if abs(weight_sum - 1.0) > 0.001:
            errors.append(f"Sum of four evaluation dimension weights must be 1.0, current is {weight_sum}")

        # Check dimension thresholds
        for threshold in [cls.CAUSAL_LOGIC_THRESHOLD,
                         cls.COMPLETENESS_THRESHOLD,
                         cls.LAYER_RATIONALITY_THRESHOLD,
                         cls.EXPLANATION_POWER_THRESHOLD]:
            if not (0 <= threshold <= 1):
                errors.append("Dimension thresholds must be in [0, 1] range")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return True

    @classmethod
    def get_config_dict(cls):
        """Get config dict (for logging)"""
        return {
            "max_iterations": cls.MAX_ITERATIONS,
            "weights": {
                "causal_logic": cls.WEIGHT_CAUSAL_LOGIC,
                "completeness": cls.WEIGHT_COMPLETENESS,
                "layer_rationality": cls.WEIGHT_LAYER_RATIONALITY,
                "explanation_power": cls.WEIGHT_EXPLANATION_POWER
            },
            "thresholds": {
                "causal_logic": cls.CAUSAL_LOGIC_THRESHOLD,
                "completeness": cls.COMPLETENESS_THRESHOLD,
                "layer_rationality": cls.LAYER_RATIONALITY_THRESHOLD,
                "explanation_power": cls.EXPLANATION_POWER_THRESHOLD
            },
            "evaluation_only": cls.EVALUATION_ONLY,
            "save_intermediate": cls.SAVE_INTERMEDIATE_RESULTS,
            "enable_execution_checking": cls.ENABLE_EXECUTION_CHECKING,
            "enable_quality_comparison": cls.ENABLE_QUALITY_COMPARISON,
            "decision_control": {
                "soft_issue_threshold": cls.SOFT_ISSUE_THRESHOLD,
                "execution_miss_ratio_threshold": cls.EXECUTION_MISS_RATIO_THRESHOLD,
                "max_execution_fails": cls.MAX_EXECUTION_FAILS,
                "stop_on_first_tie": cls.STOP_ON_FIRST_TIE,
                "stop_on_first_worse": cls.STOP_ON_FIRST_WORSE
            }
        }



# Validate configuration on module load
try:
    IterationConfig.validate()
except ValueError as e:
    print(f"Warning: Configuration validation failed - {e}")

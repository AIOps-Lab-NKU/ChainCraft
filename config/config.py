"""
Unified Configuration Management Module

Centralized management of all configuration parameters, including:
- API keys (injected via environment variables, no hardcoding)
- LLM model configuration
- Data path management
- ChromaDB configuration
- Business constants

Usage:
    from config import Config
    Config.MODEL          # Model name
    Config.get_summary_path(case_id, app, app_group)  # Path utilities
"""

import os
import logging

from dotenv import load_dotenv
load_dotenv()  # Auto-load .env file from project root

# ============================================================
# API key configuration (must be injected via environment variables, no hardcoding)
# ============================================================

def _get_env_or_raise(name: str, default: str = None) -> str:
    """Get environment variable, raise a clear error if not set and no default provided"""
    value = os.environ.get(name, default)
    if value is None:
        raise EnvironmentError(
            f"Environment variable {name} is not set. "
            f"Please set it via `export {name}=xxx` or configure it in the .env file."
        )
    return value


# ============================================================
# Main configuration class
# ============================================================

class Config:
    """Unified configuration class, all configuration parameters accessed through this class"""

    # ---------- LLM / Embedding configuration ----------
    OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

    MODEL = os.environ.get("LLM_MODEL", "gpt-4o-0806")

    EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "text-embedding-v4")
    EMBEDDING_ENCODING_FORMAT = "float"
    EMBEDDING_DIMENSIONS = 512

    # ---------- ChromaDB collection names ----------
    COLLECTION_EVENTS = "fault_events"
    COLLECTION_SYMPTOMS = "fault_symptoms"
    COLLECTION_CHAINS = "fault_chains"

    # ---------- Search & similarity configuration ----------
    DEFAULT_SEARCH_RESULTS = 3
    SYMPTOM_WEIGHT = 0.6
    CHAIN_WEIGHT = 0.4
    MAX_ITEMS_PER_SECTION = 5

    # ---------- Iterative refinement configuration ----------
    MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "3"))
    ITERATION_THRESHOLD = float(os.environ.get("ITERATION_THRESHOLD", "0.8"))

    # ---------- Data path configuration ----------
    # Project root directory (parent of parent of config/config.py)
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Data base path
    DATA_BASE_PATH = os.environ.get("DATA_BASE_PATH", os.path.join(ROOT_PATH, "data"))

    # ===== Path configuration (concise naming, read-write separation) =====

    # Data collection write path (collected data + anomaly detection results written here)
    COLLECTED_DATA_PATH = os.environ.get(
        "COLLECTED_DATA_PATH",
        os.environ.get("HISTORICAL_COLLECTED_DATA_PATH",  # Backward compatible with old variable
                       os.path.join(DATA_BASE_PATH, "collected_data"))
    )

    # Data read path (read metric raw data from this path when collect_data=False)
    DATA_READ_PATH = os.environ.get(
        "DATA_READ_PATH",
        os.environ.get("HISTORICAL_DATA_READ_PATH",  # Backward compatible with old variable
                       COLLECTED_DATA_PATH)  # Default same as write path
    )

    # Anomaly detection results read path (read anomaly detection results from this path when run_anomaly_detection=False)
    # Default same as DATA_READ_PATH, can be configured separately to reuse anomaly detection results from different experiments
    ANOMALY_DETECTION_READ_PATH = os.environ.get(
        "ANOMALY_DETECTION_READ_PATH",
        DATA_READ_PATH  # Default fallback to DATA_READ_PATH
    )

    # Analysis results write path (metric analysis, causal analysis, case analysis results written here)
    ANALYSIS_RESULT_PATH = os.environ.get(
        "ANALYSIS_RESULT_PATH",
        os.environ.get("RESULT_WRITE_PATH",  # Backward compatible with old variable
                       os.path.join(ROOT_PATH, "result"))
    )

    # Analysis results read path (read causal analysis results from this path when run_causal_analysis=False)
    ANALYSIS_READ_PATH = os.environ.get(
        "ANALYSIS_READ_PATH",
        ANALYSIS_RESULT_PATH  # Default same as write path
    )

    # Metric analysis results read path (read metric analysis results from this path when run_metric_analysis=False)
    # Default same as ANALYSIS_READ_PATH, can be configured separately to read from different experiments
    METRIC_ANALYSIS_READ_PATH = os.environ.get(
        "METRIC_ANALYSIS_READ_PATH",
        ANALYSIS_READ_PATH  # Default fallback to ANALYSIS_READ_PATH
    )

    # ChromaDB path (without date suffix)
    CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", os.path.join(ROOT_PATH, "chroma_db"))
    TEST_CHROMA_DB_PATH = os.path.join(ROOT_PATH, "test_chroma_db")

    # Log path
    LOG_PATH = os.environ.get("LOG_PATH", os.path.join(ROOT_PATH, "log"))

    # Statistics data path
    STATISTICS_PATH = os.environ.get("STATISTICS_PATH", os.path.join(ROOT_PATH, "statistics"))

    # Prompt file path (unchanged)
    PROMPT_PATH = os.path.join(ROOT_PATH, "prompt")

    # ===== Backward compatible aliases (old code can still use these) =====
    HISTORICAL_COLLECTED_DATA_PATH = COLLECTED_DATA_PATH
    HISTORICAL_DATA_READ_PATH = DATA_READ_PATH
    RESULT_WRITE_PATH = ANALYSIS_RESULT_PATH
    BASE_PATH = ROOT_PATH

    # ============================================================
    # Path utility methods
    # ============================================================

    # ----- Data read paths (read-only) -----
    @classmethod
    def get_data_summary(cls, case_id: str, app: str, app_group: str) -> str:
        """Get the case summary path for data reading"""
        os.makedirs(os.path.join(cls.HISTORICAL_DATA_READ_PATH, case_id, f"{app}_{app_group}", "summary"), exist_ok=True)
        return os.path.join(cls.HISTORICAL_DATA_READ_PATH, case_id, f"{app}_{app_group}", "summary")

    @classmethod
    def get_data_read_case_path(cls, case_id: str) -> str:
        os.makedirs(os.path.join(cls.HISTORICAL_DATA_READ_PATH, case_id), exist_ok=True)
        return os.path.join(cls.HISTORICAL_DATA_READ_PATH, case_id)

    @classmethod
    def get_data_read_app_path(cls, case_id: str, app: str, app_group: str) -> str:
        os.makedirs(os.path.join(cls.get_data_read_case_path(case_id), f"{app}_{app_group}"), exist_ok=True)
        return os.path.join(cls.get_data_read_case_path(case_id), f"{app}_{app_group}")

    @classmethod
    def get_data_read_metric_path(cls, case_id: str, app: str, app_group: str) -> str:
        return os.path.join(cls.get_data_read_app_path(case_id, app, app_group), "metric", "all_metrics.csv")

    # ----- Result write paths (writable) -----
    @classmethod
    def get_result_write_case_path(cls, case_id: str) -> str:
        os.makedirs(os.path.join(cls.ANALYSIS_RESULT_PATH, case_id), exist_ok=True)
        return os.path.join(cls.ANALYSIS_RESULT_PATH, case_id)

    @classmethod
    def get_result_summary(cls, case_id: str, app: str, app_group: str) -> str:
        """Get analysis report directory (backward compatible, actually returns analysis path)"""
        return cls.get_result_analysis_path(case_id, app, app_group)

    @classmethod
    def get_result_write_app_path(cls, case_id: str, app: str, app_group: str) -> str:
        os.makedirs(os.path.join(cls.get_result_write_case_path(case_id), f"{app}_{app_group}"), exist_ok=True)
        return os.path.join(cls.get_result_write_case_path(case_id), f"{app}_{app_group}")

    @classmethod
    def get_result_write_summary_path(cls, case_id: str, app: str, app_group: str) -> str:
        """Get result summary path (backward compatible, actually returns analysis path)"""
        return cls.get_result_analysis_path(case_id, app, app_group)

    # ----- Compatibility methods (used by data_handle and llm) -----
    @classmethod
    def get_case_path(cls, case_id: str) -> str:
        return cls.get_result_write_case_path(case_id)

    @classmethod
    def get_app_path(cls, case_id: str, app: str, app_group: str) -> str:
        return cls.get_result_write_app_path(case_id, app, app_group)

    @classmethod
    def get_summary_path(cls, case_id: str, app: str, app_group: str) -> str:
        return cls.get_result_write_summary_path(case_id, app, app_group)

    @classmethod
    def get_case_analysis_path(cls, case_id: str, app: str, app_group: str) -> str:
        """Get case analysis result path"""
        return os.path.join(cls.get_result_analysis_path(case_id, app, app_group), "case_analysis_result.txt")

    @classmethod
    def get_inference_result_path(cls, case_id: str, app: str, app_group: str) -> str:
        """Get inference result file path"""
        return os.path.join(cls.get_result_analysis_path(case_id, app, app_group), "inference_case_result.txt")

    @classmethod
    def get_result_analysis_path(cls, case_id: str, app: str, app_group: str) -> str:
        """Get analysis report output directory (result/{case}/{app}_{appgroup}/analysis/)"""
        path = os.path.join(cls.ANALYSIS_RESULT_PATH, case_id, f"{app}_{app_group}", "analysis")
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_result_iteration_path(cls, case_id: str, app: str, app_group: str) -> str:
        """Get iteration result output directory (result/{case}/{app}_{appgroup}/iteration/)"""
        path = os.path.join(cls.ANALYSIS_RESULT_PATH, case_id, f"{app}_{app_group}", "iteration")
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_judgment_result_path(cls, case_id: str, app: str, app_group: str) -> str:
        """Get final prediction result path (placed directly under app directory)"""
        path = os.path.join(cls.ANALYSIS_RESULT_PATH, case_id, f"{app}_{app_group}")
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, "judgment_case_result.txt")

    # ----- Path management utilities -----
    @classmethod
    def ensure_result_path_exists(cls, case_id: str, app: str, app_group: str) -> str:
        """Ensure result path exists, create directory automatically"""
        summary_path = cls.get_result_write_summary_path(case_id, app, app_group)
        os.makedirs(summary_path, exist_ok=True)
        return summary_path

    @classmethod
    def ensure_case_paths(cls, case_id: str, app: str, app_group: str) -> dict:
        """Ensure complete directory structure exists for a case (including log/metric/summary)"""
        base = cls.get_result_write_app_path(case_id, app, app_group)
        paths = {
            'base': base,
            'log': os.path.join(base, 'log'),
            'metric': os.path.join(base, 'metric'),
            'summary': cls.get_result_write_summary_path(case_id, app, app_group),
        }
        for p in paths.values():
            os.makedirs(p, exist_ok=True)
        return paths

    @classmethod
    def validate_path_config(cls) -> bool:
        """Validate whether path configuration is correct"""
        issues = []
        if not os.path.exists(cls.HISTORICAL_DATA_READ_PATH):
            issues.append(f"Historical data read path does not exist: {cls.HISTORICAL_DATA_READ_PATH}")
        if not os.path.exists(cls.ANOMALY_DETECTION_READ_PATH):
            issues.append(f"Anomaly detection results read path does not exist: {cls.ANOMALY_DETECTION_READ_PATH}")
        result_parent = os.path.dirname(cls.ANALYSIS_RESULT_PATH)
        if result_parent and not os.access(result_parent, os.W_OK):
            issues.append(f"Analysis result output path parent directory is not writable: {result_parent}")
        if issues:
            print("\n⚠️  Path configuration warning:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        return True

    @classmethod
    def print_path_config(cls):
        """Print current path configuration information"""
        print("\n" + "=" * 60)
        print("Path Configuration")
        print("=" * 60)
        print(f"Project root: {cls.ROOT_PATH}")
        print(f"Data collection write path (COLLECTED_DATA_PATH): {cls.COLLECTED_DATA_PATH}")
        print(f"Data read path (DATA_READ_PATH): {cls.DATA_READ_PATH}")
        print(f"Anomaly detection read path (ANOMALY_DETECTION_READ_PATH): {cls.ANOMALY_DETECTION_READ_PATH}")
        print(f"Analysis result write path (ANALYSIS_RESULT_PATH): {cls.ANALYSIS_RESULT_PATH}")
        print(f"Analysis result read path (ANALYSIS_READ_PATH): {cls.ANALYSIS_READ_PATH}")
        print(f"Metric analysis read path (METRIC_ANALYSIS_READ_PATH): {cls.METRIC_ANALYSIS_READ_PATH}")
        print(f"ChromaDB path: {cls.CHROMA_DB_PATH}")
        print("=" * 60 + "\n")


# ============================================================
# Business constants
# ============================================================

ROOT_CAUSE_TYPE_MAP = {
    '资源瓶颈/耗尽': 'resource_exhaustion',
    '依赖故障（DB/TDDL/RDS）': 'dependency_failure',
    '网络问题': 'network_issue',
    '配置错误': 'configuration_error',
    '代码缺陷': 'code_defect'
}

LAYER_DISPLAY_NAMES = {
    'dependency_layer': '依赖层',
    'core_layer': '核心层',
    'inbound_layer': '入口层'
}

SEVERITY_LEVELS = {
    'critical': '严重',
    'major': '重大',
    'minor': '次要',
    'none': '无'
}


# ============================================================
# Logging configuration
# ============================================================

def setup_logging(level: str = "INFO", log_file: str = None):
    """
    Configure unified logging output

    Args:
        level: log level (DEBUG/INFO/WARNING/ERROR)
        log_file: optional, log file path
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True,
    )

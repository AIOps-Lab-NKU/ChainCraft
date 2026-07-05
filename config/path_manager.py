"""
Unified Path Manager
Provides unified access interfaces for all output paths in the project, with automatic directory structure creation.
Supports read-write path separation: write paths for data collection, read paths for reusing historical experiment data.
"""
import os
from config.config import Config


class PathManager:
    """Unified path manager, encapsulating all path retrieval and directory creation logic"""
    
    # ===== Write path methods (used during data collection) =====
    
    @staticmethod
    def get_collected_case_path(case_id: str) -> str:
        """Get the write directory for data collection (case level)"""
        path = os.path.join(Config.COLLECTED_DATA_PATH, case_id)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_collected_app_path(case_id: str, app: str, app_group: str) -> str:
        """Get the write directory for data collection (application level)"""
        path = os.path.join(Config.COLLECTED_DATA_PATH, case_id, f"{app}_{app_group}")
        os.makedirs(path, exist_ok=True)
        return path

    # ===== Read path methods (used when reusing historical data) =====
    
    @staticmethod
    def get_data_read_app_path(case_id: str, app: str, app_group: str) -> str:
        """Get the metric raw data read directory (application level) — used when collect_data=False"""
        path = os.path.join(Config.DATA_READ_PATH, case_id, f"{app}_{app_group}")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_anomaly_detection_read_app_path(case_id: str, app: str, app_group: str) -> str:
        """Get the anomaly detection results read directory (application level) — used when run_anomaly_detection=False"""
        path = os.path.join(Config.ANOMALY_DETECTION_READ_PATH, case_id, f"{app}_{app_group}")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_analysis_read_path(case_id: str, app: str, app_group: str) -> str:
        """Get the analysis results read directory — used when run_causal_analysis=False"""
        path = os.path.join(Config.ANALYSIS_READ_PATH, case_id, f"{app}_{app_group}", "analysis")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_metric_analysis_read_path(case_id: str, app: str, app_group: str) -> str:
        """Get the metric analysis results read directory — used when run_metric_analysis=False"""
        path = os.path.join(Config.METRIC_ANALYSIS_READ_PATH, case_id, f"{app}_{app_group}", "analysis")
        os.makedirs(path, exist_ok=True)
        return path

    # ===== Result output path methods =====
    
    @staticmethod
    def get_case_result_paths(case_id: str, app: str, app_group: str) -> dict:
        """
        Get the complete result directory structure for a case
        
        Args:
            case_id: case ID
            app: application name
            app_group: application group name
            
        Returns:
            dict: dictionary containing base/analysis/iteration paths
        """
        base = os.path.join(Config.ANALYSIS_RESULT_PATH, case_id, f"{app}_{app_group}")
        paths = {
            'base': base,
            'analysis': os.path.join(base, 'analysis'),
            'iteration': os.path.join(base, 'iteration'),
        }
        for p in paths.values():
            os.makedirs(p, exist_ok=True)
        return paths
    
    @staticmethod
    def get_log_path() -> str:
        """Get the system log directory"""
        os.makedirs(Config.LOG_PATH, exist_ok=True)
        return Config.LOG_PATH
    
    @staticmethod
    def get_statistics_path() -> str:
        """Get the global statistics data directory"""
        os.makedirs(Config.STATISTICS_PATH, exist_ok=True)
        return Config.STATISTICS_PATH
    
    @staticmethod
    def get_analysis_result_path() -> str:
        """Get the analysis results root directory"""
        os.makedirs(Config.ANALYSIS_RESULT_PATH, exist_ok=True)
        return Config.ANALYSIS_RESULT_PATH
    
    @staticmethod
    def ensure_all_paths():
        """Ensure all base directories exist"""
        for path in [
            Config.ANALYSIS_RESULT_PATH,
            Config.LOG_PATH,
            Config.STATISTICS_PATH,
            Config.CHROMA_DB_PATH,
        ]:
            os.makedirs(path, exist_ok=True)

    # ===== Backward compatible aliases =====
    get_historical_case_path = get_collected_case_path
    get_historical_app_path = get_collected_app_path


# Global singleton
path_manager = PathManager()

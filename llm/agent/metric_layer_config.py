import re

class MetricLayerConfig:
    """Metric layer configuration class"""
    
    # External dependency layer configuration
    DEPENDENCY_LAYER = {
        "layer_name": "dependency_layer",
        "description": "Measures the impact received from external dependency systems or performance when calling external dependencies",
        "metrics": [
            "middleware_hsf_consumer_",
            "middleware_tddl_",
            "middleware_tair_",
            "MySQL_",
            "metaq_receive_",
            "notify_receive_",
            "metaq_send_",
            "notify_send_",
            "pod_nginx_qps",
            "pod_nginx_rt"
        ]
    }
    
    # Application core layer configuration
    CORE_LAYER = {
        "layer_name": "core_layer", 
        "description": "Measures internal runtime status and resource consumption of this service",
        "metrics": [
            "jvm_gc_",
            "provider_biz_pool_usage",
            "pod_cpu_",
            "pod_memory_",
            "pod_root_disk_",
            "pod_network_retran_util",
            "pod_event_oom",
            "middleware_metaq_clnt_receive_qps",
            'app_error_cnt'
        ]
    }
    
    # Service inbound layer configuration
    INBOUND_LAYER = {
        "layer_name": "inbound_layer",
        "description": "Measures externally visible performance of this service as a service provider",
        "metrics": [
            "middleware_hsf_provider_",
            "middleware_metaq_clnt_send_qps",
            "ExternalDetect_"
        ]
    }
    
    @classmethod
    def get_metric_layer(cls, metric_name):
        """
        Determine layer membership based on metric name
        
        Args:
            metric_name (str): Metric name
            
        Returns:
            str: Layer name (dependency_layer/core_layer/inbound_layer)
        """
        if not metric_name:
            return None
            
        # Check external dependency layer
        for pattern in cls.DEPENDENCY_LAYER["metrics"]:
            if re.search(re.escape(pattern), metric_name):
                return cls.DEPENDENCY_LAYER["layer_name"]
                
        # Check application core layer first (contains more specific patterns)
        for pattern in cls.CORE_LAYER["metrics"]:
            if re.search(re.escape(pattern), metric_name):
                return cls.CORE_LAYER["layer_name"]
                
        # Finally check service inbound layer
        for pattern in cls.INBOUND_LAYER["metrics"]:
            if re.search(re.escape(pattern), metric_name):
                return cls.INBOUND_LAYER["layer_name"]
                
        # Default to application core layer
        return cls.CORE_LAYER["layer_name"]
    
    @classmethod
    def get_layer_description(cls, layer_name):
        """
        Get layer description
        
        Args:
            layer_name (str): Layer name
            
        Returns:
            str: Layer description
        """
        layer_map = {
            cls.DEPENDENCY_LAYER["layer_name"]: cls.DEPENDENCY_LAYER["description"],
            cls.CORE_LAYER["layer_name"]: cls.CORE_LAYER["description"],
            cls.INBOUND_LAYER["layer_name"]: cls.INBOUND_LAYER["description"]
        }
        return layer_map.get(layer_name, "")

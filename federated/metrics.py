"""
Metrics Calculation Module
Calculates key metrics matching the research paper methodology
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class MetricsCalculator:
    """
    Calculates metrics for federated vs centralized comparison
    Matches research paper methodology exactly
    """
    
    @staticmethod
    def calculate_data_transfer(client_data_sizes: List[int], model_size_bytes: int, 
                                num_rounds: int) -> Dict:
        """
        Calculate data transfer for centralized vs federated approaches
        
        Centralized: Transfers ALL raw data from all clients to central server
        Federated: Only transfers model updates (model parameters)
        
        Returns comparison matching paper's Table 3
        """
        # Centralized approach: transfers all raw data
        total_raw_data_bytes = sum(client_data_sizes)
        
        # Federated approach: only model updates
        # Per round: send model to clients + receive updates from clients
        model_transfer_per_round = model_size_bytes * len(client_data_sizes) * 2  # to clients + from clients
        total_federated_transfer = model_transfer_per_round * num_rounds
        
        # Calculate reduction percentage
        reduction_percent = ((total_raw_data_bytes - total_federated_transfer) / 
                           total_raw_data_bytes) * 100 if total_raw_data_bytes > 0 else 0
        
        return {
            'centralized_bytes': total_raw_data_bytes,
            'federated_bytes': total_federated_transfer,
            'reduction_percent': reduction_percent,
            'centralized_mb': total_raw_data_bytes / (1024 * 1024),
            'federated_mb': total_federated_transfer / (1024 * 1024)
        }
    
    @staticmethod
    def calculate_computational_efficiency(centralized_time: float, federated_time: float,
                                         centralized_resources: float, federated_resources: float) -> Dict:
        """
        Calculate computational efficiency metrics
        Paper shows 52% reduction in computational resources
        """
        time_reduction = ((centralized_time - federated_time) / centralized_time) * 100 if centralized_time > 0 else 0
        resource_reduction = ((centralized_resources - federated_resources) / centralized_resources) * 100 if centralized_resources > 0 else 0
        
        return {
            'time_reduction_percent': time_reduction,
            'resource_reduction_percent': resource_reduction,
            'centralized_time': centralized_time,
            'federated_time': federated_time,
            'centralized_resources': centralized_resources,
            'federated_resources': federated_resources
        }
    
    @staticmethod
    def calculate_accuracy_improvement(centralized_mape: float, federated_mape: float) -> Dict:
        """
        Calculate accuracy improvement percentage
        Paper shows 37% improvement (MAPE reduction)
        Formula: ((Centralized_MAPE - Federated_MAPE) / Centralized_MAPE) * 100
        """
        improvement_percent = ((centralized_mape - federated_mape) / centralized_mape) * 100 if centralized_mape > 0 else 0
        
        return {
            'centralized_mape': centralized_mape,
            'federated_mape': federated_mape,
            'improvement_percent': improvement_percent,
            'mape_reduction': centralized_mape - federated_mape
        }
    
    @staticmethod
    def get_paper_benchmarks() -> Dict:
        """
        Returns benchmark values from the research paper for comparison
        """
        return {
            'accuracy_improvement_target': 37.1,  # Average from Table 2
            'data_transfer_reduction_target': 65.0,  # From Table 3
            'computational_reduction_target': 52.0,  # From Section 3.2
            'paper_centralized_mape': 20.3,  # Average from Table 2
            'paper_federated_mape': 12.6  # Average from Table 2
        }


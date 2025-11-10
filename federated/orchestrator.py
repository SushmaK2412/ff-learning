"""
Federated Learning Orchestrator
Coordinates model training across multiple client nodes using PySyft
"""

import torch
import numpy as np
import time
from typing import List, Dict
import sys
import os

# Python 3.7 compatibility
try:
    from typing import List, Dict
except ImportError:
    # Fallback for older Python
    pass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated.metrics import MetricsCalculator

class FederatedOrchestrator:
    """
    Central orchestrator for federated learning
    
    Responsibilities:
    - Coordinates training rounds across financial institutions
    - Aggregates model parameters using Federated Averaging (FedAvg)
    - Collects metrics for visualization
    - NEVER receives or stores raw data (privacy preserved)
    """
    
    def __init__(self, model, num_rounds=10):
        self.model = model
        self.num_rounds = num_rounds
        
        # Track clients
        self.clients = []
        self.client_metrics = {}
        self.client_data_sizes = []  # Track raw data sizes for metrics
        
        # Training history
        self.history = {
            'round': [],
            'aggregated_loss': [],
            'aggregated_mape': [],
            'client_losses': [],
            'data_transfer': [],
            'training_times': []
        }
        
        # Computational efficiency tracking
        self.computational_metrics = {
            'federated_time': 0.0,
            'federated_resources': 0.0
        }
    
    def connect_client(self, client_worker):
        """Connect a client worker to the orchestrator"""
        self.clients.append(client_worker)
        # Get client ID - use client_id attribute
        client_id = getattr(client_worker, 'client_id', getattr(client_worker, 'id', f'client_{len(self.clients)}'))
        self.client_metrics[client_id] = {
            'loss': [],
            'mape': [],
            'data_sent': 0,
            'data_received': 0
        }
        # Track client's raw data size for centralized comparison
        if hasattr(client_worker, 'data') and client_worker.data is not None:
            data_size_bytes = client_worker.data.memory_usage(deep=True).sum()
            self.client_data_sizes.append(data_size_bytes)
        print(f"✓ Connected client: {client_id}")
    
    def federated_averaging(self, model_states: List[Dict], sample_sizes: List[int]):
        """
        Federated Averaging (FedAvg) algorithm
        Aggregates model states from clients weighted by their sample sizes
        Matches paper's horizontal federated learning approach
        """
        # Filter out None states and invalid sample sizes
        valid_pairs = [(state, size) for state, size in zip(model_states, sample_sizes) 
                      if state is not None and size is not None and size > 0]
        
        if not valid_pairs:
            raise ValueError("No valid model states to aggregate - all states are None or have zero samples")
        
        model_states, sample_sizes = zip(*valid_pairs)
        model_states = list(model_states)
        sample_sizes = list(sample_sizes)
        
        total_samples = sum(sample_sizes)
        if total_samples == 0:
            raise ValueError("Total sample size cannot be zero")
        
        # Initialize aggregated state dict
        aggregated_state = {}
        
        # Get model parameter names from first valid state
        if not model_states or not model_states[0]:
            raise ValueError("No valid model states to aggregate")
        
        param_names = list(model_states[0].keys())
        
        for param_name in param_names:
            # Weighted average of model parameters
            weighted_sum = None
            
            for state, sample_size in zip(model_states, sample_sizes):
                if state is None or param_name not in state:
                    continue
                    
                weight = sample_size / total_samples
                param_value = state[param_name]
                
                if weighted_sum is None:
                    weighted_sum = param_value * weight
                else:
                    weighted_sum += param_value * weight
            
            if weighted_sum is not None:
                aggregated_state[param_name] = weighted_sum
        
        return aggregated_state
    
    def train_round(self, round_num: int):
        """
        Execute one round of federated training
        """
        print(f"\n{'='*60}")
        print(f"Federated Learning Round {round_num + 1}/{self.num_rounds}")
        print(f"{'='*60}")
        
        # Send model to clients
        model_state = self.model.state_dict()
        client_models = []
        sample_sizes = []
        round_losses = []
        round_mapes = []
        round_data_transfer = 0
        
        # Calculate model size for data transfer tracking
        model_size_bytes = sum(p.numel() * 4 for p in self.model.parameters())  # 4 bytes per float32
        
        for client in self.clients:
            try:
                # Get client ID
                client_id = getattr(client, 'client_id', getattr(client, 'id', f'client_{self.clients.index(client)}'))
                
                # Send model to client (simulated - in real PySyft, this would be done via pointers)
                print(f"  → Sending model to {client_id}...")
                round_data_transfer += model_size_bytes
                self.client_metrics[client_id]['data_received'] += model_size_bytes
                
                # Get trained model from client (only model parameters, no raw data)
                # This preserves privacy - raw data never leaves the client
                client_trained_state, client_metrics = client.train_local_model(model_state.copy())
                
                # Only add if we got a valid model state and metrics
                if client_trained_state is not None and client_metrics is not None:
                    client_models.append(client_trained_state)
                    sample_sizes.append(client_metrics.get('sample_size', 0))
                    round_losses.append(client_metrics.get('loss', float('inf')))
                    round_mapes.append(client_metrics.get('mape', 100.0))
                    
                    # Track data transfer
                    round_data_transfer += model_size_bytes
                    self.client_metrics[client_id]['data_sent'] += model_size_bytes
                    
                    print(f"    ✓ Received update from {client_id}")
                    print(f"      Loss: {client_metrics.get('loss', 0):.4f}, MAPE: {client_metrics.get('mape', 0):.2f}%")
                else:
                    print(f"    ⚠ Skipping {client_id}: No valid model state returned (data may be empty or error occurred)")
                
            except Exception as e:
                client_id = getattr(client, 'client_id', getattr(client, 'id', 'unknown'))
                print(f"    ✗ Error with client {client_id}: {str(e)}")
                continue
        
        if not client_models:
            print("  ✗ No client updates received")
            return None
        
        # Aggregate model updates
        print(f"\n  → Aggregating model updates...")
        aggregated_state = self.federated_averaging(client_models, sample_sizes)
        
        # Update global model
        self.model.load_state_dict(aggregated_state)
        
        # Calculate aggregated metrics
        total_samples = sum(sample_sizes)
        weighted_loss = sum(loss * size for loss, size in zip(round_losses, sample_sizes)) / total_samples
        weighted_mape = sum(mape * size for mape, size in zip(round_mapes, sample_sizes)) / total_samples
        
        # Update history
        self.history['round'].append(round_num + 1)
        self.history['aggregated_loss'].append(weighted_loss)
        self.history['aggregated_mape'].append(weighted_mape)
        self.history['client_losses'].append(round_losses)
        self.history['data_transfer'].append(round_data_transfer)
        
        print(f"  ✓ Aggregation complete")
        print(f"    Aggregated Loss: {weighted_loss:.4f}")
        print(f"    Aggregated MAPE: {weighted_mape:.2f}%")
        print(f"    Data Transfer: {round_data_transfer / (1024*1024):.2f} MB")
        
        return {
            'loss': weighted_loss,
            'mape': weighted_mape,
            'data_transfer': round_data_transfer
        }
    
    def train(self):
        """
        Execute full federated training process
        Tracks computational efficiency (time and resources)
        """
        print(f"\n{'='*60}")
        print(f"Starting Federated Learning Training")
        print(f"Number of clients: {len(self.clients)}")
        print(f"Number of rounds: {self.num_rounds}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for round_num in range(self.num_rounds):
            round_start = time.time()
            result = self.train_round(round_num)
            round_time = time.time() - round_start
            self.history['training_times'].append(round_time)
            
            if result is None:
                break
        
        total_time = time.time() - start_time
        self.computational_metrics['federated_time'] = total_time
        
        # Estimate resource usage (simplified: proportional to time)
        # In real implementation, would measure CPU/GPU utilization
        self.computational_metrics['federated_resources'] = total_time * len(self.clients) * 0.3  # Distributed load
        
        print(f"\n{'='*60}")
        print(f"Federated Training Complete!")
        print(f"Total Training Time: {total_time:.2f} seconds")
        print(f"{'='*60}")
        
        return self.history
    
    def get_metrics(self):
        """Get comprehensive training metrics matching paper methodology"""
        total_federated_transfer = sum(self.history['data_transfer'])
        model_size_bytes = sum(p.numel() * 4 for p in self.model.parameters())
        
        # Calculate data transfer comparison
        data_transfer_metrics = MetricsCalculator.calculate_data_transfer(
            self.client_data_sizes,
            model_size_bytes,
            self.num_rounds
        )
        
        return {
            'history': self.history,
            'client_metrics': self.client_metrics,
            'computational_metrics': self.computational_metrics,
            'data_transfer_metrics': data_transfer_metrics,
            'total_federated_transfer_mb': total_federated_transfer / (1024 * 1024),
            'final_loss': self.history['aggregated_loss'][-1] if self.history['aggregated_loss'] else None,
            'final_mape': self.history['aggregated_mape'][-1] if self.history['aggregated_mape'] else None
        }


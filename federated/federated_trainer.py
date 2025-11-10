"""
Main Federated Learning Trainer
Simulates federated learning across multiple financial institutions
"""

import torch
import numpy as np
import pandas as pd
import time
import os
import sys
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.forecasting_model import FinancialForecastingModel, ModelTrainer
from federated.client import FederatedClient
from federated.orchestrator import FederatedOrchestrator
from federated.metrics import MetricsCalculator

class FederatedTrainer:
    """
    Main class to orchestrate federated learning simulation
    """
    
    def __init__(self, task: str = 'cash_flow_30d', num_rounds: int = 10, 
                 local_epochs: int = 5):
        self.task = task
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        
        # Initialize clients
        self.clients = self._initialize_clients()
        
        # Initialize global model
        self.global_model = self._initialize_global_model()
        
        # Initialize orchestrator
        self.orchestrator = FederatedOrchestrator(
            model=self.global_model,
            num_rounds=num_rounds
        )
        
        # Connect clients to orchestrator
        for client in self.clients:
            self.orchestrator.connect_client(client)
    
    def _initialize_clients(self) -> List[FederatedClient]:
        """Initialize client nodes for each financial institution"""
        institutions = ['Institution_A', 'Institution_B', 'Institution_C']
        clients = []
        
        for inst in institutions:
            data_path = f'data/processed/{inst}_{self.task}.csv'
            if os.path.exists(data_path):
                client = FederatedClient(
                    client_id=inst,
                    data_path=data_path,
                    target_task=self.task,
                    local_epochs=self.local_epochs
                )
                clients.append(client)
            else:
                print(f"Warning: Data file not found for {inst}: {data_path}")
        
        return clients
    
    def _initialize_global_model(self):
        """Initialize global model architecture"""
        # Load a sample dataset to determine input size
        sample_path = f'data/processed/Institution_A_{self.task}.csv'
        if os.path.exists(sample_path):
            # Handle CSV parsing errors (e.g., malformed rows)
            try:
                df = pd.read_csv(sample_path, index_col=0, error_bad_lines=False, warn_bad_lines=False)
            except TypeError:
                # For newer pandas versions
                df = pd.read_csv(sample_path, index_col=0, on_bad_lines='skip')
            target_col = self._get_target_column()
            exclude_cols = [target_col, f'{target_col}_target', 'institution', 'Date']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            input_size = len(feature_cols)
        else:
            # Default input size
            input_size = 20
        
        model = FinancialForecastingModel(input_size=input_size)
        return model
    
    def _get_target_column(self):
        """Determine target column based on task"""
        task_mapping = {
            'cash_flow_30d': 'cash_flow',
            'cash_flow_60d': 'cash_flow',
            'cash_flow_90d': 'cash_flow',
            'default_risk': 'default_risk',
            'investment_return': 'investment_return'
        }
        return task_mapping.get(self.task, 'cash_flow')
    
    def train(self):
        """Execute federated training"""
        print(f"\n{'='*70}")
        print(f"FEDERATED LEARNING FOR FINANCIAL FORECASTING")
        print(f"{'='*70}")
        print(f"Task: {self.task}")
        print(f"Clients: {len(self.clients)}")
        print(f"Training Rounds: {self.num_rounds}")
        print(f"Local Epochs per Round: {self.local_epochs}")
        print(f"{'='*70}\n")
        
        # Execute federated training
        history = self.orchestrator.train()
        
        return history
    
    def compare_with_centralized(self):
        """
        Compare federated vs centralized approach
        Matches paper methodology exactly
        """
        print(f"\n{'='*70}")
        print(f"COMPARING FEDERATED vs CENTRALIZED APPROACH")
        print(f"{'='*70}\n")
        
        # Load all data for centralized training
        all_data = []
        for client in self.clients:
            if client.data is not None:
                all_data.append(client.data)
        
        if not all_data:
            print("No data available for comparison")
            return None
        
        # Combine all data (centralized approach requires all data in one place)
        centralized_data = pd.concat(all_data, ignore_index=True)
        print(f"Centralized dataset size: {len(centralized_data)} samples")
        
        # Prepare data
        target_col = self._get_target_column()
        exclude_cols = [target_col, f'{target_col}_target', 'institution', 'Date']
        feature_cols = [col for col in centralized_data.columns if col not in exclude_cols]
        
        X = centralized_data[feature_cols].values
        y = centralized_data[f'{target_col}_target'].values
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train centralized model (track time for computational efficiency)
        print("Training centralized model...")
        centralized_start = time.time()
        
        centralized_model = FinancialForecastingModel(input_size=X_train.shape[1])
        trainer = ModelTrainer(centralized_model, device='cpu')
        trainer.scaler = scaler
        
        X_val = X_test[:len(X_test)//2]
        y_val = y_test[:len(y_test)//2]
        X_test_final = X_test[len(X_test)//2:]
        y_test_final = y_test[len(y_test)//2:]
        
        trainer.train(X_train, y_train, X_val, y_val, epochs=50)
        centralized_time = time.time() - centralized_start
        
        # Evaluate centralized
        centralized_results = trainer.evaluate(X_test_final, y_test_final)
        
        # Evaluate federated model
        federated_results = self._evaluate_federated_model(X_test_final, y_test_final, scaler)
        
        # Calculate metrics using MetricsCalculator
        accuracy_metrics = MetricsCalculator.calculate_accuracy_improvement(
            centralized_results['mape'],
            federated_results['mape']
        )
        
        # Get computational metrics
        federated_time = self.orchestrator.computational_metrics['federated_time']
        federated_resources = self.orchestrator.computational_metrics['federated_resources']
        centralized_resources = centralized_time * 1.0  # Centralized uses full resources
        
        computational_metrics = MetricsCalculator.calculate_computational_efficiency(
            centralized_time,
            federated_time,
            centralized_resources,
            federated_resources
        )
        
        comparison = {
            'centralized': {
                'mape': centralized_results['mape'],
                'rmse': centralized_results['rmse'],
                'mae': centralized_results['mae'],
                'training_time': centralized_time
            },
            'federated': {
                'mape': federated_results['mape'],
                'rmse': federated_results['rmse'],
                'mae': federated_results['mae'],
                'training_time': federated_time
            },
            'accuracy_improvement': accuracy_metrics,
            'computational_efficiency': computational_metrics
        }
        
        # Print results matching paper format
        print(f"\n{'='*70}")
        print(f"RESULTS COMPARISON (Matching Research Paper)")
        print(f"{'='*70}")
        print(f"\n📊 Forecasting Accuracy:")
        print(f"  Centralized MAPE: {centralized_results['mape']:.2f}%")
        print(f"  Federated MAPE:   {federated_results['mape']:.2f}%")
        print(f"  Improvement:      {accuracy_metrics['improvement_percent']:.1f}%")
        
        print(f"\n💻 Computational Efficiency:")
        print(f"  Centralized Time: {centralized_time:.2f}s")
        print(f"  Federated Time:   {federated_time:.2f}s")
        print(f"  Resource Reduction: {computational_metrics['resource_reduction_percent']:.1f}%")
        
        print(f"\n📡 Data Transfer:")
        data_transfer = self.orchestrator.get_metrics()['data_transfer_metrics']
        print(f"  Centralized: {data_transfer['centralized_mb']:.2f} MB")
        print(f"  Federated:   {data_transfer['federated_mb']:.2f} MB")
        print(f"  Reduction:   {data_transfer['reduction_percent']:.1f}%")
        print(f"{'='*70}\n")
        
        return comparison
    
    def _evaluate_federated_model(self, X_test, y_test, scaler):
        """Evaluate the trained federated model"""
        self.global_model.eval()
        
        # Normalize test data
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        X_tensor = torch.FloatTensor(X_test_scaled)
        
        with torch.no_grad():
            predictions = self.global_model(X_tensor).numpy()
        
        # Calculate metrics (custom MAPE for compatibility)
        def mean_absolute_percentage_error(y_true, y_pred):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            mask = y_true != 0
            if not np.any(mask):
                return 0.0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        from sklearn.metrics import mean_squared_error
        mape = mean_absolute_percentage_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = np.mean(np.abs(y_test - predictions))
        
        return {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'targets': y_test
        }
    
    def get_metrics(self):
        """Get all training metrics"""
        return self.orchestrator.get_metrics()

if __name__ == "__main__":
    # Example usage
    trainer = FederatedTrainer(task='cash_flow_30d', num_rounds=10, local_epochs=5)
    history = trainer.train()
    comparison = trainer.compare_with_centralized()
    metrics = trainer.get_metrics()


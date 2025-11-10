"""
Federated Learning Client
Represents a financial institution participating in federated learning
"""

import torch
import numpy as np
from typing import Dict
import pandas as pd
from models.forecasting_model import FinancialForecastingModel, ModelTrainer, FinancialDataset
from torch.utils.data import DataLoader

# PySyft is optional - code works without it
try:
    import syft as sy
    PYSYFT_AVAILABLE = True
except ImportError:
    PYSYFT_AVAILABLE = False

class FederatedClient:
    """
    Client node representing a financial institution
    
    Each institution:
    - Runs in its own Docker container (isolated)
    - Fetches and stores its own data (no data sharing)
    - Trains model locally on private data
    - Shares ONLY model parameters (not raw data)
    - Maintains data sovereignty
    """
    
    def __init__(self, client_id: str, data_path: str, target_task: str, 
                 local_epochs: int = 5, batch_size: int = 32):
        self.client_id = client_id
        self.data_path = data_path
        self.target_task = target_task
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        
        # Load local data
        self.data = self._load_data()
        
        # Initialize local model (will be replaced by global model)
        self.model = None
        self.trainer = None
        
        # Local training history
        self.local_history = {
            'loss': [],
            'mape': []
        }
    
    def _load_data(self):
        """Load institution's local data"""
        try:
            # Handle CSV parsing errors (e.g., malformed rows)
            try:
                df = pd.read_csv(self.data_path, index_col=0, error_bad_lines=False, warn_bad_lines=False)
            except TypeError:
                # For newer pandas versions
                df = pd.read_csv(self.data_path, index_col=0, on_bad_lines='skip')
            print(f"  ✓ Loaded data for {self.client_id}: {len(df)} samples")
            return df
        except Exception as e:
            print(f"  ✗ Error loading data for {self.client_id}: {str(e)}")
            return None
    
    def _get_target_column(self):
        """Determine target column based on task"""
        task_mapping = {
            'cash_flow_30d': 'cash_flow',
            'cash_flow_60d': 'cash_flow',
            'cash_flow_90d': 'cash_flow',
            'default_risk': 'default_risk',
            'investment_return': 'investment_return'
        }
        return task_mapping.get(self.target_task, 'cash_flow')
    
    def train_local_model(self, global_model_state: Dict):
        """
        Train model locally on private data
        Returns: model update (state dict difference) and metrics
        """
        if self.data is None or len(self.data) == 0:
            return None, {'loss': float('inf'), 'mape': 100.0, 'sample_size': 0}
        
        target_col = self._get_target_column()
        
        # Prepare data
        exclude_cols = [target_col, f'{target_col}_target', 'institution', 'Date']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        X = self.data[feature_cols].values
        y = self.data[f'{target_col}_target'].values
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split train/val (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Initialize model if not done
        if self.model is None:
            input_size = X_train.shape[1]
            self.model = FinancialForecastingModel(input_size=input_size)
            self.trainer = ModelTrainer(self.model, device='cpu')
            self.trainer.scaler = scaler
        
        # Load global model state
        self.model.load_state_dict(global_model_state)
        
        # Local training
        train_dataset = FinancialDataset(X_train, y_train)
        val_dataset = FinancialDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training setup
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Local training loop (data stays local - privacy preserved)
        self.model.train()
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        # Return trained model state (not raw data - privacy preserved)
        trained_model_state = self.model.state_dict()
        
        # Evaluate on validation set
        self.model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
                # Handle 0-d arrays by flattening to ensure iterability
                output_np = outputs.cpu().numpy()
                target_np = batch_targets.cpu().numpy()
                if output_np.ndim == 0:
                    val_preds.append(float(output_np))
                else:
                    val_preds.extend(output_np.flatten().tolist())
                if target_np.ndim == 0:
                    val_targets.append(float(target_np))
                else:
                    val_targets.extend(target_np.flatten().tolist())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate MAPE (custom function for compatibility)
        def mean_absolute_percentage_error(y_true, y_pred):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            mask = y_true != 0
            if not np.any(mask):
                return 0.0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        val_mape = mean_absolute_percentage_error(val_targets, val_preds)
        
        # Update local history
        self.local_history['loss'].append(avg_val_loss)
        self.local_history['mape'].append(val_mape)
        
        metrics = {
            'loss': avg_val_loss,
            'mape': val_mape,
            'sample_size': len(X_train)
        }
        
        return trained_model_state, metrics
    
    def get_local_metrics(self):
        """Get local training metrics"""
        return {
            'client_id': self.client_id,
            'data_samples': len(self.data) if self.data is not None else 0,
            'local_history': self.local_history
        }


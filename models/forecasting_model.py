"""
Financial Forecasting Models
Implements neural network models for various financial forecasting tasks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd

# Custom MAPE function for compatibility with older scikit-learn versions
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

class FinancialDataset(Dataset):
    """PyTorch Dataset for financial time series data"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class FinancialForecastingModel(nn.Module):
    """
    Lightweight LSTM Neural Network for Financial Forecasting
    
    Architecture:
    - 2 LSTM layers (64 hidden units each) - captures temporal patterns
    - 1 Dense layer (32 units) - feature transformation
    - 1 Output layer (1 unit) - final prediction
    
    Model Size: ~100KB (lightweight, perfect for federated learning)
    Total Parameters: ~10,000-20,000
    
    Why LSTM (not Random Forest)?
    - Excellent for time series data (financial forecasting)
    - Gradient-based (works with Federated Averaging)
    - Lightweight model size (easy to transfer)
    - Can learn long-term dependencies in financial trends
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(FinancialForecastingModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers for temporal patterns
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dense layers for final prediction
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # Reshape if needed (batch_size, seq_len, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Dense layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out.squeeze()

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mape': [],
            'val_mape': []
        }
    
    def prepare_data(self, df, target_col, test_size=0.2):
        """
        Prepare data for training
        Returns: X_train, X_test, y_train, y_test
        """
        # Get feature columns (exclude target and metadata columns)
        exclude_cols = [target_col, f'{target_col}_target', 'institution', 'Date']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[f'{target_col}_target'].values
        
        # Normalize features
        X = self.scaler.fit_transform(X)
        
        # Split train/test
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, learning_rate=0.001):
        """Train the model"""
        
        # Create datasets
        train_dataset = FinancialDataset(X_train, y_train)
        val_dataset = FinancialDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        best_val_loss = float('inf')
        # Initialize best_model_state with current model state
        self.best_model_state = self.model.state_dict().copy()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                # Handle 0-d arrays by flattening to ensure iterability
                output_np = outputs.detach().cpu().numpy()
                target_np = batch_targets.cpu().numpy()
                if output_np.ndim == 0:
                    train_preds.append(float(output_np))
                else:
                    train_preds.extend(output_np.flatten().tolist())
                if target_np.ndim == 0:
                    train_targets.append(float(target_np))
                else:
                    train_targets.extend(target_np.flatten().tolist())
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
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
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_mape = mean_absolute_percentage_error(train_targets, train_preds) * 100
            val_mape = mean_absolute_percentage_error(val_targets, val_preds) * 100
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Save history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_mape'].append(train_mape)
            self.history['val_mape'].append(val_mape)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_model_state = self.model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                      f"Train MAPE: {train_mape:.2f}%, Val MAPE: {val_mape:.2f}%")
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        self.model.eval()
        
        test_dataset = FinancialDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_targets.numpy())
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(targets, predictions) * 100
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = np.mean(np.abs(np.array(targets) - np.array(predictions)))
        
        return {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'targets': targets
        }
    
    def predict(self, X):
        """Make predictions on new data"""
        self.model.eval()
        
        # Normalize if scaler is fitted
        if hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions


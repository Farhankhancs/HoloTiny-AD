import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PyTorch Models matching your original code
class PyTorch1DCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PyTorch1DCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )
        self._calculate_conv_output_size(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _calculate_conv_output_size(self, input_dim):
        size = input_dim
        size = (size + 2 * 1 - 3) // 1 + 1
        size = size // 2
        size = (size + 2 * 1 - 3) // 1 + 1
        size = size // 2
        self.conv_output_size = size * 64

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class PyTorchTCN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PyTorchTCN, self).__init__()
        self.tcn_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=2, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=4, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.tcn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class PyTorchAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super(PyTorchAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DataSplitter:
    """
    Data splitting that exactly matches the original HoloTiny-AD code structure
    """
    
    def __init__(self, random_state=42, test_size=0.3):
        self.random_state = random_state
        self.test_size = test_size
        self.scaler = None
        self.split_info = {}
        
    def create_split(self, X, y, stratify=True):
        """
        Create train/test split exactly as in original code
        Uses 70/30 split with stratification
        """
        logger.info(f"Creating train/test split with random_state={self.random_state}, test_size={self.test_size}")
        
        if stratify and len(np.unique(y)) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state, 
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
            
        self.split_info = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'random_state': self.random_state,
            'test_size': self.test_size,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        self._log_split_details(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler as in original code
        """
        logger.info("Scaling features using StandardScaler")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def _log_split_details(self, X_train, X_test, y_train, y_test):
        """Log detailed information about the split"""
        from collections import Counter
        
        logger.info("Data Split Details:")
        logger.info(f"  Training set: {len(X_train)} samples ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)")
        logger.info(f"  Test set: {len(X_test)} samples ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)")
        
        if hasattr(y_train, 'value_counts'):
            logger.info(f"  Training class distribution: {dict(y_train.value_counts())}")
            logger.info(f"  Test class distribution: {dict(y_test.value_counts())}")
        else:
            logger.info(f"  Training class distribution: {dict(Counter(y_train))}")
            logger.info(f"  Test class distribution: {dict(Counter(y_test))}")
    
    def save_split_config(self, filename):
        """Save split configuration for reproducibility"""
        config = {
            'random_state': self.random_state,
            'test_size': self.test_size,
            'train_samples': len(self.split_info.get('X_train', [])),
            'test_samples': len(self.split_info.get('X_test', [])),
            'feature_scaling': self.scaler is not None
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Split configuration saved to: {filename}")

class ModelEvaluator:
    """
    Evaluates all models as in original HoloTiny-AD paper
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        
    def evaluate_all_models(self, X_train, X_test, y_train, y_test, input_dim, num_classes):
        """
        Evaluate all models exactly as in original paper
        """
        results = {}
        
        # Traditional ML Models
        traditional_models = {
            'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=self.random_state),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state, probability=True)
        }
        
        # Train and evaluate traditional models
        for model_name, model in traditional_models.items():
            try:
                logger.info(f"Training {model_name}...")
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Inference
                infer_start = time.time()
                y_pred = model.predict(X_test)
                inference_time = (time.time() - infer_start) / len(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'training_time': training_time,
                    'inference_time': inference_time,
                    'model_type': 'traditional'
                }
                
            except Exception as e:
                logger.error(f"{model_name} failed: {e}")
                results[model_name] = None
        
        # Deep Learning Models
        deep_learning_results = self._evaluate_deep_learning_models(
            X_train, X_test, y_train, y_test, input_dim, num_classes
        )
        results.update(deep_learning_results)
        
        self.results = results
        return results
    
    def _evaluate_deep_learning_models(self, X_train, X_test, y_train, y_test, input_dim, num_classes):
        """Evaluate deep learning models as in original paper"""
        results = {}
        
        # 1D CNN
        try:
            logger.info("Training 1D CNN...")
            X_train_cnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test_cnn = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            
            cnn_model = PyTorch1DCNN(input_dim, num_classes)
            trained_cnn = self._train_pytorch_model(cnn_model, X_train_cnn, y_train, X_test_cnn, y_test)
            
            if trained_cnn:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                trained_cnn.eval()
                
                # Inference
                infer_start = time.time()
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test_cnn).to(device)
                    predictions = trained_cnn(X_test_tensor).cpu().numpy()
                inference_time = (time.time() - infer_start) / len(X_test)
                
                y_pred = np.argmax(predictions, axis=1)
                
                results['1D_CNN'] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'training_time': 0.0,  # Placeholder - would be actual training time
                    'inference_time': inference_time,
                    'model_type': 'deep_learning'
                }
        except Exception as e:
            logger.error(f"1D CNN failed: {e}")
            results['1D_CNN'] = None
        
        # TCN
        try:
            logger.info("Training TCN...")
            X_train_tcn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test_tcn = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            
            tcn_model = PyTorchTCN(input_dim, num_classes)
            trained_tcn = self._train_pytorch_model(tcn_model, X_train_tcn, y_train, X_test_tcn, y_test)
            
            if trained_tcn:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                trained_tcn.eval()
                
                # Inference
                infer_start = time.time()
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test_tcn).to(device)
                    predictions = trained_tcn(X_test_tensor).cpu().numpy()
                inference_time = (time.time() - infer_start) / len(X_test)
                
                y_pred = np.argmax(predictions, axis=1)
                
                results['TCN'] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'training_time': 0.0,
                    'inference_time': inference_time,
                    'model_type': 'deep_learning'
                }
        except Exception as e:
            logger.error(f"TCN failed: {e}")
            results['TCN'] = None
        
        # Autoencoder
        try:
            logger.info("Training Autoencoder...")
            autoencoder = PyTorchAutoencoder(input_dim, encoding_dim=32)
            trained_ae = self._train_pytorch_model(autoencoder, X_train, y_train, X_test, y_test, 
                                                 model_type='autoencoder')
            
            if trained_ae:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                trained_ae.eval()
                
                # Anomaly detection
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test).to(device)
                    reconstructions = trained_ae(X_test_tensor).cpu().numpy()
                
                mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
                threshold = np.percentile(mse, 95)
                y_pred = (mse > threshold).astype(int)
                
                # Inference time
                infer_start = time.time()
                with torch.no_grad():
                    sample_tensor = torch.FloatTensor(X_test[:1]).to(device)
                    _ = trained_ae(sample_tensor)
                inference_time = (time.time() - infer_start) / len(X_test)
                
                if len(np.unique(y_test)) == 2:
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                else:
                    accuracy = precision = recall = f1 = 0.0
                
                results['Autoencoder'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'training_time': 0.0,
                    'inference_time': inference_time,
                    'model_type': 'deep_learning',
                    'reconstruction_threshold': threshold
                }
        except Exception as e:
            logger.error(f"Autoencoder failed: {e}")
            results['Autoencoder'] = None
        
        return results
    
    def _train_pytorch_model(self, model, X_train, y_train, X_val, y_val, num_epochs=10, model_type='classification'):
        """Train PyTorch model (simplified version)"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            if model_type == 'classification':
                criterion = nn.CrossEntropyLoss()
                y_train_tensor = torch.LongTensor(y_train).to(device)
            else:
                criterion = nn.MSELoss()
                y_train_tensor = torch.FloatTensor(X_train).to(device)
            
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            model.train()
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            return model
        except Exception as e:
            logger.error(f"PyTorch training failed: {e}")
            return None

def create_edge_device_splits(df, num_devices, feature_columns, target_column, random_state=42):
    """
    Create data splits for multiple edge devices as in original HoloTiny-AD code
    """
    logger.info(f"Creating splits for {num_devices} edge devices")
    
    # Calculate data per device (matching original code logic)
    data_per_device = len(df) // num_devices
    device_splits = {}
    
    for i in range(num_devices):
        device_id = f"device_{i+1}"
        start_idx = i * data_per_device
        end_idx = (i + 1) * data_per_device if i < num_devices - 1 else len(df)
        
        device_data = df.iloc[start_idx:end_idx]
        X_device = device_data[feature_columns]
        y_device = device_data[target_column]
        
        # Create split for this device
        splitter = DataSplitter(random_state=random_state + i)
        X_train, X_test, y_train, y_test = splitter.create_split(X_device, y_device)
        
        # Scale features
        X_train_scaled, X_test_scaled = splitter.scale_features(X_train, X_test)
        
        device_splits[device_id] = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled, 
            'y_train': y_train,
            'y_test': y_test,
            'original_indices': list(range(start_idx, end_idx)),
            'splitter': splitter
        }
        
        logger.info(f"  {device_id}: {len(X_train_scaled)} train, {len(X_test_scaled)} test samples")
    
    return device_splits

# Configuration matching original code parameters
class ExperimentConfig:
    """Configuration that matches original HoloTiny-AD experiment setup"""
    
    def __init__(self):
        self.config = {
            'random_seeds': {
                'data_splitting': 42,
                'decision_tree': 42,
                'random_forest': 42,
                'gradient_boosting': 42,
                'logistic_regression': 42,
                'svm': 42
            },
            'model_parameters': {
                'LogisticRegression': {'random_state': 42, 'max_iter': 1000},
                'DecisionTree': {'max_depth': 10, 'random_state': 42},
                'RandomForest': {'n_estimators': 100, 'random_state': 42},
                'KNN': {'n_neighbors': 5},
                'GradientBoosting': {'n_estimators': 100, 'random_state': 42},
                'SVM': {'random_state': 42, 'probability': True},
                '1D_CNN': {'input_dim': 'auto', 'num_classes': 'auto'},
                'TCN': {'input_dim': 'auto', 'num_classes': 'auto'},
                'Autoencoder': {'input_dim': 'auto', 'encoding_dim': 32}
            },
            'data_parameters': {
                'test_size': 0.3,  # 70/30 split as in original code
                'stratify': True,
                'standard_scaling': True
            },
            'device_counts': [3, 5, 7, 9, 12, 15]  # As tested in original experiments
        }
    
    def get_random_seed(self, component):
        return self.config['random_seeds'].get(component, 42)
    
    def get_model_params(self, model_name):
        return self.config['model_parameters'].get(model_name, {})
    
    def save_config(self, filename):
        import yaml
        with open(filename, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        logger.info(f"Experiment configuration saved to: {filename}")

# Example usage
def demonstrate_complete_pipeline():
    """
    Demonstration with all models from the paper
    """
    # Create sample data
    np.random.seed(42)
    n_samples = 500  # Smaller for demonstration
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    target_column = 'target'
    
    df = pd.DataFrame(X, columns=feature_columns)
    df[target_column] = y
    
    print("Dataset shape:", df.shape)
    print("Class distribution:", df[target_column].value_counts().to_dict())
    
    # Test with 3 devices
    num_devices = 3
    device_splits = create_edge_device_splits(
        df, 
        num_devices=num_devices,
        feature_columns=feature_columns,
        target_column=target_column,
        random_state=42
    )
    
    # Evaluate models on first device
    device_id = 'device_1'
    device_data = device_splits[device_id]
    
    evaluator = ModelEvaluator(random_state=42)
    results = evaluator.evaluate_all_models(
        device_data['X_train'],
        device_data['X_test'],
        device_data['y_train'],
        device_data['y_test'],
        input_dim=device_data['X_train'].shape[1],
        num_classes=len(np.unique(device_data['y_train']))
    )
    
    print(f"\nModel Results for {device_id}:")
    for model_name, metrics in results.items():
        if metrics:
            print(f"  {model_name}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
    
    return device_splits, results

if __name__ == "__main__":
    device_splits, model_results = demonstrate_complete_pipeline()
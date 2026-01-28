"""
Model Training Script
=====================
Comprehensive training pipeline with evaluation and visualization.

Author: Solar Panel Efficiency Research Team
Version: 1.0.0
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_and_save_datasets
from src.preprocessing import preprocess_pipeline, DataPreprocessor
from src.model import create_model, SolarPanelEfficiencyModel

import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelTrainer:
    """
    Comprehensive model training pipeline with evaluation and visualization.
    """
    
    def __init__(self, 
                 data_dir: str = 'data',
                 model_dir: str = 'models',
                 log_dir: str = 'logs'):
        """
        Initialize the trainer.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing training data
        model_dir : str
            Directory to save trained models
        log_dir : str
            Directory for training logs
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
        
        self.model = None
        self.factory = None
        self.preprocessor = None
        self.history = None
        self.metrics = {}
        
    def prepare_data(self, 
                    engineer_features: bool = True,
                    scaler_type: str = 'standard') -> dict:
        """
        Prepare data for training.
        """
        print("\n" + "=" * 60)
        print("STEP 1: Data Preparation")
        print("=" * 60)
        
        # Check if data exists, if not generate it
        train_path = os.path.join(self.data_dir, 'train_data.csv')
        if not os.path.exists(train_path):
            print("Generating synthetic dataset...")
            generate_and_save_datasets(self.data_dir)
        
        # Preprocess data
        data = preprocess_pipeline(
            data_dir=self.data_dir,
            scaler_type=scaler_type,
            engineer_features=engineer_features
        )
        
        self.preprocessor = data['preprocessor']
        
        print(f"\nData shapes:")
        print(f"  Training: {data['X_train'].shape}")
        print(f"  Validation: {data['X_val'].shape}")
        print(f"  Test: {data['X_test'].shape}")
        print(f"  Features: {len(data['feature_columns'])}")
        
        return data
    
    def build_model(self, 
                   input_dim: int,
                   model_type: str = 'deep',
                   learning_rate: float = 0.001,
                   **model_kwargs) -> None:
        """
        Build and compile the model.
        """
        print("\n" + "=" * 60)
        print("STEP 2: Model Building")
        print("=" * 60)
        
        self.model, self.factory = create_model(
            input_dim=input_dim,
            model_type=model_type,
            **model_kwargs
        )
        
        # Recompile with custom learning rate if different
        if learning_rate != 0.001:
            self.factory.compile(learning_rate=learning_rate)
        
        print(f"\nModel Type: {model_type}")
        print(f"Parameters: {self.model.count_params():,}")
        
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = 100,
             batch_size: int = 32,
             patience: int = 15) -> dict:
        """
        Train the model.
        """
        print("\n" + "=" * 60)
        print("STEP 3: Model Training")
        print("=" * 60)
        
        # Get callbacks
        model_path = os.path.join(self.model_dir, 'best_model.keras')
        callbacks = self.factory.get_callbacks(
            model_path=model_path,
            patience=patience,
            log_dir=self.log_dir
        )
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Early Stopping Patience: {patience}")
        print(f"  Model will be saved to: {model_path}")
        
        # Train
        print("\nStarting training...\n")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def evaluate(self,
                X_test: np.ndarray,
                y_test: np.ndarray,
                preprocessor: DataPreprocessor = None) -> dict:
        """
        Evaluate the model on test data.
        """
        print("\n" + "=" * 60)
        print("STEP 4: Model Evaluation")
        print("=" * 60)
        
        # Predictions
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        
        # Inverse transform if preprocessor available
        if preprocessor is not None:
            y_test_original = preprocessor.inverse_transform_target(y_test)
            y_pred_original = preprocessor.inverse_transform_target(y_pred_scaled)
        else:
            y_test_original = y_test
            y_pred_original = y_pred_scaled
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_original, y_pred_original)
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, y_pred_original)
        mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + 1e-8))) * 100
        
        self.metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape)
        }
        
        print(f"\nTest Set Metrics:")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}%")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}%")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        return self.metrics, y_test_original, y_pred_original
    
    def plot_results(self,
                    history: dict,
                    y_test: np.ndarray,
                    y_pred: np.ndarray,
                    save_path: str = None) -> None:
        """
        Create visualization plots for training results.
        """
        print("\n" + "=" * 60)
        print("STEP 5: Results Visualization")
        print("=" * 60)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Solar Panel Efficiency Prediction - Training Results', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Training & Validation Loss
        ax1 = axes[0, 0]
        ax1.plot(history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training & Validation MAE
        ax2 = axes[0, 1]
        ax2.plot(history['mae'], label='Training MAE', linewidth=2)
        ax2.plot(history['val_mae'], label='Validation MAE', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Training & Validation MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Actual vs Predicted
        ax3 = axes[0, 2]
        ax3.scatter(y_test, y_pred, alpha=0.5, s=10)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax3.set_xlabel('Actual Efficiency (%)')
        ax3.set_ylabel('Predicted Efficiency (%)')
        ax3.set_title('Actual vs Predicted Efficiency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Prediction Error Distribution
        ax4 = axes[1, 0]
        errors = y_test.flatten() - y_pred.flatten()
        ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Prediction Error (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Prediction Error Distribution')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Residual Plot
        ax5 = axes[1, 1]
        ax5.scatter(y_pred, errors, alpha=0.5, s=10)
        ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax5.set_xlabel('Predicted Efficiency (%)')
        ax5.set_ylabel('Residual')
        ax5.set_title('Residual Plot')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Metrics Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        metrics_text = f"""
        MODEL PERFORMANCE METRICS
        ─────────────────────────
        
        MAE:  {self.metrics['mae']:.4f}%
        
        RMSE: {self.metrics['rmse']:.4f}%
        
        R²:   {self.metrics['r2']:.4f}
        
        MAPE: {self.metrics['mape']:.2f}%
        """
        ax6.text(0.1, 0.5, metrics_text, fontsize=14, fontfamily='monospace',
                verticalalignment='center', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, feature_columns: list) -> None:
        """
        Save training results and model artifacts.
        """
        print("\n" + "=" * 60)
        print("STEP 6: Saving Results")
        print("=" * 60)
        
        # Save model
        model_path = os.path.join(self.model_dir, 'final_model.keras')
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save training history
        history_path = os.path.join(self.model_dir, 'training_history.json')
        history_dict = {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"Training history saved to: {history_path}")
        
        # Save metrics
        metrics_path = os.path.join(self.model_dir, 'metrics.json')
        results = {
            'metrics': self.metrics,
            'feature_columns': feature_columns,
            'model_type': self.factory.model_type,
            'timestamp': datetime.now().isoformat()
        }
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")


def train_pipeline(model_type: str = 'deep',
                  epochs: int = 100,
                  batch_size: int = 32,
                  learning_rate: float = 0.001,
                  patience: int = 15,
                  engineer_features: bool = True,
                  data_dir: str = 'data',
                  model_dir: str = 'models') -> dict:
    """
    Complete training pipeline.
    
    Parameters:
    -----------
    model_type : str
        Type of model to train ('standard', 'deep', 'attention', 'ensemble')
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Training batch size
    learning_rate : float
        Initial learning rate
    patience : int
        Early stopping patience
    engineer_features : bool
        Whether to create engineered features
    data_dir : str
        Directory containing data
    model_dir : str
        Directory to save models
        
    Returns:
    --------
    dict
        Training results including metrics and paths
    """
    print("\n" + "=" * 60)
    print("SOLAR PANEL EFFICIENCY PREDICTION")
    print("Deep Learning Training Pipeline")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize trainer
    trainer = ModelTrainer(data_dir=data_dir, model_dir=model_dir)
    
    # Step 1: Prepare data
    data = trainer.prepare_data(engineer_features=engineer_features)
    
    # Step 2: Build model
    input_dim = data['X_train'].shape[1]
    trainer.build_model(
        input_dim=input_dim,
        model_type=model_type,
        learning_rate=learning_rate
    )
    
    # Step 3: Train model
    history = trainer.train(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        epochs=epochs,
        batch_size=batch_size,
        patience=patience
    )
    
    # Step 4: Evaluate model
    metrics, y_test, y_pred = trainer.evaluate(
        X_test=data['X_test'],
        y_test=data['y_test'],
        preprocessor=data['preprocessor']
    )
    
    # Step 5: Plot results
    plot_path = os.path.join(model_dir, 'training_results.png')
    trainer.plot_results(history, y_test, y_pred, save_path=plot_path)
    
    # Step 6: Save results
    trainer.save_results(feature_columns=data['feature_columns'])
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return {
        'metrics': metrics,
        'model_path': os.path.join(model_dir, 'final_model.keras'),
        'history': history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Solar Panel Efficiency Model')
    parser.add_argument('--model-type', type=str, default='deep',
                       choices=['standard', 'deep', 'attention', 'ensemble'],
                       help='Type of model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    results = train_pipeline(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience
    )

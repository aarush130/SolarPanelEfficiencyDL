"""
Data Preprocessing Module
=========================
Handles data loading, cleaning, feature engineering, and preparation
for deep learning model training.

Author: Solar Panel Efficiency Research Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Tuple, Dict, Optional


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for solar panel efficiency data.
    
    Features:
    - Data loading and validation
    - Missing value handling
    - Feature engineering
    - Multiple scaling options
    - Train/validation/test splitting
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        scaler_type : str
            Type of scaler to use: 'standard', 'minmax', or 'robust'
        """
        self.scaler_type = scaler_type
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_columns = None
        self.target_column = 'efficiency'
        
        # Initialize scaler based on type
        scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler
        }
        
        if scaler_type not in scalers:
            raise ValueError(f"Scaler type must be one of: {list(scalers.keys())}")
        
        self.feature_scaler = scalers[scaler_type]()
        self.target_scaler = scalers[scaler_type]()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples from {filepath}")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate the loaded data and return statistics.
        
        Returns:
        --------
        dict
            Dictionary containing validation results and statistics
        """
        stats = {
            'n_samples': len(df),
            'n_features': len(df.columns) - 1,  # Excluding target
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'dtypes': df.dtypes.to_dict()
        }
        
        # Check for required columns
        required_columns = [
            'solar_irradiance', 'ambient_temperature', 'panel_temperature',
            'humidity', 'wind_speed', 'dust_accumulation', 'panel_age',
            'tilt_angle', 'cloud_cover', 'hour_of_day', 'efficiency'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        stats['missing_columns'] = missing_columns
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
        
        return stats
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with additional engineered features
        """
        df = df.copy()
        
        # Temperature difference (important for heat transfer)
        df['temp_difference'] = df['panel_temperature'] - df['ambient_temperature']
        
        # Irradiance per unit temperature (normalized irradiance)
        df['irradiance_temp_ratio'] = df['solar_irradiance'] / (df['ambient_temperature'] + 273.15)
        
        # Effective irradiance (accounting for cloud cover and dust)
        df['effective_irradiance'] = df['solar_irradiance'] * (1 - df['cloud_cover']/100) * (1 - df['dust_accumulation']/100)
        
        # Time-based features
        df['is_peak_hours'] = ((df['hour_of_day'] >= 10) & (df['hour_of_day'] <= 15)).astype(int)
        
        # Optimal conditions indicator
        df['optimal_conditions'] = (
            (df['solar_irradiance'] > 700) & 
            (df['ambient_temperature'] > 15) & 
            (df['ambient_temperature'] < 35) &
            (df['cloud_cover'] < 30)
        ).astype(int)
        
        # Panel age category
        df['panel_age_category'] = pd.cut(
            df['panel_age'], 
            bins=[0, 5, 10, 15, 25], 
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Wind cooling factor
        df['wind_cooling_factor'] = df['wind_speed'] * (df['panel_temperature'] - df['ambient_temperature']) / 100
        
        print(f"Engineered {7} additional features")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        strategy : str
            Strategy for handling missing values: 'median', 'mean', 'drop'
        """
        df = df.copy()
        
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'median':
            df = df.fillna(df.median())
        elif strategy == 'mean':
            df = df.fillna(df.mean())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, 
                         exclude_columns: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for model training.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        exclude_columns : list, optional
            Columns to exclude from features
            
        Returns:
        --------
        tuple
            (features, target) as numpy arrays
        """
        exclude_columns = exclude_columns or ['timestamp', 'efficiency']
        
        # Identify feature columns
        self.feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        X = df[self.feature_columns].values
        y = df[self.target_column].values.reshape(-1, 1)
        
        return X, y
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scalers and transform the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature array
        y : np.ndarray
            Target array
            
        Returns:
        --------
        tuple
            (scaled_X, scaled_y)
        """
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        
        return X_scaled, y_scaled
    
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted scalers.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature array
        y : np.ndarray, optional
            Target array
            
        Returns:
        --------
        tuple
            (scaled_X, scaled_y) or (scaled_X, None)
        """
        X_scaled = self.feature_scaler.transform(X)
        y_scaled = self.target_scaler.transform(y) if y is not None else None
        
        return X_scaled, y_scaled
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled target values."""
        return self.target_scaler.inverse_transform(y_scaled)
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save the preprocessor state to disk."""
        state = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns,
            'scaler_type': self.scaler_type
        }
        joblib.dump(state, filepath)
        print(f"Preprocessor saved to: {filepath}")
    
    def load_preprocessor(self, filepath: str) -> None:
        """Load the preprocessor state from disk."""
        state = joblib.load(filepath)
        self.feature_scaler = state['feature_scaler']
        self.target_scaler = state['target_scaler']
        self.feature_columns = state['feature_columns']
        self.scaler_type = state['scaler_type']
        print(f"Preprocessor loaded from: {filepath}")


def preprocess_pipeline(data_dir: str = "data", 
                        scaler_type: str = 'standard',
                        engineer_features: bool = True) -> Dict:
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
    scaler_type : str
        Type of scaler to use
    engineer_features : bool
        Whether to create engineered features
        
    Returns:
    --------
    dict
        Dictionary containing preprocessed data and preprocessor
    """
    preprocessor = DataPreprocessor(scaler_type=scaler_type)
    
    # Load datasets
    train_df = preprocessor.load_data(os.path.join(data_dir, "train_data.csv"))
    val_df = preprocessor.load_data(os.path.join(data_dir, "val_data.csv"))
    test_df = preprocessor.load_data(os.path.join(data_dir, "test_data.csv"))
    
    # Validate data
    print("\nValidating training data...")
    stats = preprocessor.validate_data(train_df)
    print(f"Samples: {stats['n_samples']}, Features: {stats['n_features']}")
    
    # Engineer features if requested
    if engineer_features:
        print("\nEngineering features...")
        train_df = preprocessor.engineer_features(train_df)
        val_df = preprocessor.engineer_features(val_df)
        test_df = preprocessor.engineer_features(test_df)
    
    # Handle missing values
    train_df = preprocessor.handle_missing_values(train_df)
    val_df = preprocessor.handle_missing_values(val_df)
    test_df = preprocessor.handle_missing_values(test_df)
    
    # Prepare features
    X_train, y_train = preprocessor.prepare_features(train_df)
    X_val, y_val = preprocessor.prepare_features(val_df)
    X_test, y_test = preprocessor.prepare_features(test_df)
    
    # Fit and transform
    X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_val_scaled, y_val_scaled = preprocessor.transform(X_val, y_val)
    X_test_scaled, y_test_scaled = preprocessor.transform(X_test, y_test)
    
    # Save preprocessor
    preprocessor.save_preprocessor(os.path.join(data_dir, "preprocessor.joblib"))
    
    return {
        'X_train': X_train_scaled,
        'y_train': y_train_scaled,
        'X_val': X_val_scaled,
        'y_val': y_val_scaled,
        'X_test': X_test_scaled,
        'y_test': y_test_scaled,
        'preprocessor': preprocessor,
        'feature_columns': preprocessor.feature_columns
    }


if __name__ == "__main__":
    result = preprocess_pipeline()
    print(f"\nPreprocessing complete!")
    print(f"Training set shape: {result['X_train'].shape}")
    print(f"Validation set shape: {result['X_val'].shape}")
    print(f"Test set shape: {result['X_test'].shape}")

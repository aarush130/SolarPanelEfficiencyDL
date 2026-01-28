"""
Solar Panel Efficiency Dataset Generator
=========================================
Generates realistic synthetic data for training deep learning models
to predict solar panel efficiency based on environmental and panel parameters.

Author: Solar Panel Efficiency Research Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


class SolarPanelDataGenerator:
    """
    Generates realistic synthetic solar panel efficiency data.
    
    The efficiency is calculated based on multiple factors:
    - Solar Irradiance (W/m²)
    - Ambient Temperature (°C)
    - Panel Temperature (°C)
    - Humidity (%)
    - Wind Speed (m/s)
    - Dust Accumulation (%)
    - Panel Age (years)
    - Panel Tilt Angle (degrees)
    - Cloud Cover (%)
    - Time of Day (hour)
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed
        
    def _calculate_efficiency(self, irradiance, ambient_temp, panel_temp, 
                              humidity, wind_speed, dust, panel_age, 
                              tilt_angle, cloud_cover, hour):
        """
        Calculate solar panel efficiency based on multiple factors.
        
        This formula is based on simplified physics models and empirical data
        from solar panel performance studies.
        """
        # Base efficiency (modern silicon panels typically 15-22%)
        base_efficiency = 20.0
        
        # Temperature coefficient: efficiency drops ~0.4% per °C above 25°C
        temp_coefficient = -0.004
        temp_effect = temp_coefficient * (panel_temp - 25) * base_efficiency
        
        # Irradiance effect (normalized around 1000 W/m²)
        irradiance_factor = np.clip(irradiance / 1000, 0.1, 1.2)
        
        # Humidity effect (slight negative impact above 70%)
        humidity_effect = -0.02 * np.maximum(humidity - 70, 0)
        
        # Wind cooling effect (helps reduce panel temperature impact)
        wind_benefit = 0.1 * np.minimum(wind_speed, 10) / 10
        
        # Dust accumulation impact (reduces light transmission)
        dust_impact = -0.15 * dust / 100
        
        # Panel aging effect (degradation ~0.5% per year)
        age_degradation = -0.005 * panel_age
        
        # Tilt angle optimization (optimal around 30-35 degrees for most locations)
        optimal_tilt = 32
        tilt_penalty = -0.001 * (tilt_angle - optimal_tilt) ** 2 / 100
        
        # Cloud cover impact
        cloud_impact = -0.3 * cloud_cover / 100
        
        # Time of day effect (peak efficiency around solar noon)
        hour_factor = np.exp(-0.5 * ((hour - 12) / 4) ** 2)
        
        # Calculate final efficiency
        efficiency = (base_efficiency + temp_effect + humidity_effect + 
                     wind_benefit + dust_impact + age_degradation * base_efficiency + 
                     tilt_penalty + cloud_impact)
        
        efficiency = efficiency * irradiance_factor * hour_factor
        
        # Add realistic noise
        noise = np.random.normal(0, 0.5, size=efficiency.shape)
        efficiency += noise
        
        # Clip to realistic bounds
        efficiency = np.clip(efficiency, 0, 25)
        
        return efficiency
    
    def generate_dataset(self, n_samples: int = 10000, 
                         include_timestamp: bool = True) -> pd.DataFrame:
        """
        Generate a synthetic dataset for solar panel efficiency prediction.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        include_timestamp : bool
            Whether to include timestamp column
            
        Returns:
        --------
        pd.DataFrame
            Generated dataset with features and target variable
        """
        print(f"Generating {n_samples} samples...")
        
        # Generate features with realistic distributions
        data = {
            # Solar Irradiance: 0-1200 W/m² (varies with time and weather)
            'solar_irradiance': np.random.uniform(100, 1100, n_samples),
            
            # Ambient Temperature: -10 to 45°C
            'ambient_temperature': np.random.normal(25, 10, n_samples),
            
            # Humidity: 20-95%
            'humidity': np.random.uniform(20, 95, n_samples),
            
            # Wind Speed: 0-15 m/s
            'wind_speed': np.random.exponential(3, n_samples),
            
            # Dust Accumulation: 0-50%
            'dust_accumulation': np.random.exponential(10, n_samples),
            
            # Panel Age: 0-25 years
            'panel_age': np.random.uniform(0, 25, n_samples),
            
            # Tilt Angle: 0-60 degrees
            'tilt_angle': np.random.uniform(10, 50, n_samples),
            
            # Cloud Cover: 0-100%
            'cloud_cover': np.random.beta(2, 5, n_samples) * 100,
            
            # Hour of Day: 6-20 (daylight hours)
            'hour_of_day': np.random.uniform(6, 20, n_samples),
        }
        
        # Clip values to realistic ranges
        data['ambient_temperature'] = np.clip(data['ambient_temperature'], -10, 45)
        data['wind_speed'] = np.clip(data['wind_speed'], 0, 15)
        data['dust_accumulation'] = np.clip(data['dust_accumulation'], 0, 50)
        
        # Calculate panel temperature (affected by irradiance, ambient temp, and wind)
        data['panel_temperature'] = (
            data['ambient_temperature'] + 
            0.03 * data['solar_irradiance'] - 
            0.5 * data['wind_speed'] + 
            np.random.normal(0, 2, n_samples)
        )
        data['panel_temperature'] = np.clip(data['panel_temperature'], -5, 80)
        
        # Calculate efficiency (target variable)
        data['efficiency'] = self._calculate_efficiency(
            data['solar_irradiance'],
            data['ambient_temperature'],
            data['panel_temperature'],
            data['humidity'],
            data['wind_speed'],
            data['dust_accumulation'],
            data['panel_age'],
            data['tilt_angle'],
            data['cloud_cover'],
            data['hour_of_day']
        )
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add timestamp if requested
        if include_timestamp:
            base_date = datetime(2024, 1, 1)
            timestamps = [base_date + timedelta(hours=i) for i in range(n_samples)]
            df['timestamp'] = timestamps
        
        # Reorder columns
        columns = ['timestamp'] if include_timestamp else []
        columns += [
            'solar_irradiance', 'ambient_temperature', 'panel_temperature',
            'humidity', 'wind_speed', 'dust_accumulation', 'panel_age',
            'tilt_angle', 'cloud_cover', 'hour_of_day', 'efficiency'
        ]
        df = df[columns]
        
        print(f"Dataset generated successfully with shape: {df.shape}")
        return df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        """Save the dataset to a CSV file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to: {filepath}")


def generate_and_save_datasets(output_dir: str = "data"):
    """Generate train, validation, and test datasets."""
    generator = SolarPanelDataGenerator(seed=42)
    
    # Generate datasets
    train_data = generator.generate_dataset(n_samples=8000)
    val_data = generator.generate_dataset(n_samples=1000)
    test_data = generator.generate_dataset(n_samples=1000)
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    generator.save_dataset(train_data, os.path.join(output_dir, "train_data.csv"))
    generator.save_dataset(val_data, os.path.join(output_dir, "val_data.csv"))
    generator.save_dataset(test_data, os.path.join(output_dir, "test_data.csv"))
    
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"\nEfficiency Range: {train_data['efficiency'].min():.2f}% - {train_data['efficiency'].max():.2f}%")
    print(f"Mean Efficiency: {train_data['efficiency'].mean():.2f}%")
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    generate_and_save_datasets()

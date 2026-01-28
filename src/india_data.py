"""
India Solar Data Module
=======================
Real solar irradiance and climate data for major Indian cities.
Data based on MNRE (Ministry of New and Renewable Energy) and IMD sources.

Author: Solar Panel Efficiency Research Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np

# Comprehensive data for major Indian cities
# Solar irradiance values are annual average Global Horizontal Irradiance (GHI) in kWh/m²/day
# Temperature, humidity are annual averages

INDIA_CITIES_DATA = [
    # Rajasthan - High solar potential
    {"city": "Jaisalmer", "state": "Rajasthan", "latitude": 26.92, "longitude": 70.90, "solar_irradiance": 5.89, "avg_temperature": 27.5, "humidity": 35, "annual_sunshine_hours": 3200, "dust_factor": 25},
    {"city": "Jodhpur", "state": "Rajasthan", "latitude": 26.28, "longitude": 73.02, "solar_irradiance": 5.85, "avg_temperature": 27.0, "humidity": 38, "annual_sunshine_hours": 3150, "dust_factor": 22},
    {"city": "Bikaner", "state": "Rajasthan", "latitude": 28.02, "longitude": 73.31, "solar_irradiance": 5.80, "avg_temperature": 26.5, "humidity": 36, "annual_sunshine_hours": 3100, "dust_factor": 24},
    {"city": "Jaipur", "state": "Rajasthan", "latitude": 26.92, "longitude": 75.79, "solar_irradiance": 5.65, "avg_temperature": 25.8, "humidity": 42, "annual_sunshine_hours": 3000, "dust_factor": 18},
    {"city": "Udaipur", "state": "Rajasthan", "latitude": 24.58, "longitude": 73.71, "solar_irradiance": 5.55, "avg_temperature": 25.2, "humidity": 45, "annual_sunshine_hours": 2950, "dust_factor": 15},
    
    # Gujarat - High solar potential
    {"city": "Kutch", "state": "Gujarat", "latitude": 23.73, "longitude": 69.86, "solar_irradiance": 5.82, "avg_temperature": 27.2, "humidity": 40, "annual_sunshine_hours": 3100, "dust_factor": 20},
    {"city": "Ahmedabad", "state": "Gujarat", "latitude": 23.02, "longitude": 72.57, "solar_irradiance": 5.60, "avg_temperature": 27.5, "humidity": 48, "annual_sunshine_hours": 2950, "dust_factor": 18},
    {"city": "Rajkot", "state": "Gujarat", "latitude": 22.30, "longitude": 70.80, "solar_irradiance": 5.55, "avg_temperature": 27.0, "humidity": 50, "annual_sunshine_hours": 2900, "dust_factor": 16},
    {"city": "Surat", "state": "Gujarat", "latitude": 21.17, "longitude": 72.83, "solar_irradiance": 5.40, "avg_temperature": 28.0, "humidity": 65, "annual_sunshine_hours": 2800, "dust_factor": 14},
    {"city": "Vadodara", "state": "Gujarat", "latitude": 22.31, "longitude": 73.19, "solar_irradiance": 5.45, "avg_temperature": 27.5, "humidity": 55, "annual_sunshine_hours": 2850, "dust_factor": 15},
    
    # Maharashtra
    {"city": "Nagpur", "state": "Maharashtra", "latitude": 21.15, "longitude": 79.09, "solar_irradiance": 5.45, "avg_temperature": 27.0, "humidity": 52, "annual_sunshine_hours": 2850, "dust_factor": 15},
    {"city": "Pune", "state": "Maharashtra", "latitude": 18.52, "longitude": 73.86, "solar_irradiance": 5.35, "avg_temperature": 25.5, "humidity": 55, "annual_sunshine_hours": 2750, "dust_factor": 12},
    {"city": "Mumbai", "state": "Maharashtra", "latitude": 19.08, "longitude": 72.88, "solar_irradiance": 5.10, "avg_temperature": 27.5, "humidity": 72, "annual_sunshine_hours": 2600, "dust_factor": 18},
    {"city": "Nashik", "state": "Maharashtra", "latitude": 20.00, "longitude": 73.78, "solar_irradiance": 5.40, "avg_temperature": 25.0, "humidity": 50, "annual_sunshine_hours": 2800, "dust_factor": 13},
    {"city": "Aurangabad", "state": "Maharashtra", "latitude": 19.88, "longitude": 75.34, "solar_irradiance": 5.50, "avg_temperature": 26.5, "humidity": 48, "annual_sunshine_hours": 2900, "dust_factor": 14},
    
    # Karnataka
    {"city": "Bengaluru", "state": "Karnataka", "latitude": 12.97, "longitude": 77.59, "solar_irradiance": 5.40, "avg_temperature": 24.5, "humidity": 55, "annual_sunshine_hours": 2800, "dust_factor": 12},
    {"city": "Mysuru", "state": "Karnataka", "latitude": 12.30, "longitude": 76.64, "solar_irradiance": 5.35, "avg_temperature": 24.0, "humidity": 58, "annual_sunshine_hours": 2750, "dust_factor": 10},
    {"city": "Hubli", "state": "Karnataka", "latitude": 15.36, "longitude": 75.12, "solar_irradiance": 5.50, "avg_temperature": 25.5, "humidity": 52, "annual_sunshine_hours": 2850, "dust_factor": 13},
    {"city": "Belgaum", "state": "Karnataka", "latitude": 15.85, "longitude": 74.50, "solar_irradiance": 5.45, "avg_temperature": 24.5, "humidity": 55, "annual_sunshine_hours": 2800, "dust_factor": 12},
    {"city": "Mangaluru", "state": "Karnataka", "latitude": 12.91, "longitude": 74.86, "solar_irradiance": 5.20, "avg_temperature": 27.5, "humidity": 75, "annual_sunshine_hours": 2600, "dust_factor": 10},
    
    # Tamil Nadu
    {"city": "Chennai", "state": "Tamil Nadu", "latitude": 13.08, "longitude": 80.27, "solar_irradiance": 5.35, "avg_temperature": 29.0, "humidity": 70, "annual_sunshine_hours": 2800, "dust_factor": 15},
    {"city": "Coimbatore", "state": "Tamil Nadu", "latitude": 11.02, "longitude": 76.96, "solar_irradiance": 5.45, "avg_temperature": 26.5, "humidity": 60, "annual_sunshine_hours": 2850, "dust_factor": 12},
    {"city": "Madurai", "state": "Tamil Nadu", "latitude": 9.92, "longitude": 78.12, "solar_irradiance": 5.55, "avg_temperature": 28.5, "humidity": 62, "annual_sunshine_hours": 2900, "dust_factor": 14},
    {"city": "Tiruchirappalli", "state": "Tamil Nadu", "latitude": 10.79, "longitude": 78.69, "solar_irradiance": 5.50, "avg_temperature": 28.0, "humidity": 65, "annual_sunshine_hours": 2880, "dust_factor": 13},
    {"city": "Salem", "state": "Tamil Nadu", "latitude": 11.65, "longitude": 78.16, "solar_irradiance": 5.48, "avg_temperature": 27.5, "humidity": 58, "annual_sunshine_hours": 2860, "dust_factor": 12},
    
    # Andhra Pradesh & Telangana
    {"city": "Hyderabad", "state": "Telangana", "latitude": 17.39, "longitude": 78.49, "solar_irradiance": 5.50, "avg_temperature": 27.0, "humidity": 55, "annual_sunshine_hours": 2900, "dust_factor": 16},
    {"city": "Warangal", "state": "Telangana", "latitude": 17.98, "longitude": 79.60, "solar_irradiance": 5.48, "avg_temperature": 27.5, "humidity": 52, "annual_sunshine_hours": 2880, "dust_factor": 14},
    {"city": "Visakhapatnam", "state": "Andhra Pradesh", "latitude": 17.69, "longitude": 83.22, "solar_irradiance": 5.30, "avg_temperature": 28.0, "humidity": 72, "annual_sunshine_hours": 2700, "dust_factor": 12},
    {"city": "Vijayawada", "state": "Andhra Pradesh", "latitude": 16.51, "longitude": 80.65, "solar_irradiance": 5.45, "avg_temperature": 29.0, "humidity": 68, "annual_sunshine_hours": 2850, "dust_factor": 14},
    {"city": "Tirupati", "state": "Andhra Pradesh", "latitude": 13.63, "longitude": 79.42, "solar_irradiance": 5.48, "avg_temperature": 28.5, "humidity": 62, "annual_sunshine_hours": 2870, "dust_factor": 13},
    {"city": "Anantapur", "state": "Andhra Pradesh", "latitude": 14.68, "longitude": 77.60, "solar_irradiance": 5.72, "avg_temperature": 28.0, "humidity": 48, "annual_sunshine_hours": 3000, "dust_factor": 16},
    
    # Madhya Pradesh
    {"city": "Bhopal", "state": "Madhya Pradesh", "latitude": 23.26, "longitude": 77.41, "solar_irradiance": 5.50, "avg_temperature": 25.5, "humidity": 50, "annual_sunshine_hours": 2850, "dust_factor": 15},
    {"city": "Indore", "state": "Madhya Pradesh", "latitude": 22.72, "longitude": 75.86, "solar_irradiance": 5.55, "avg_temperature": 25.0, "humidity": 48, "annual_sunshine_hours": 2900, "dust_factor": 14},
    {"city": "Jabalpur", "state": "Madhya Pradesh", "latitude": 23.18, "longitude": 79.95, "solar_irradiance": 5.45, "avg_temperature": 26.0, "humidity": 52, "annual_sunshine_hours": 2800, "dust_factor": 13},
    {"city": "Gwalior", "state": "Madhya Pradesh", "latitude": 26.22, "longitude": 78.18, "solar_irradiance": 5.52, "avg_temperature": 26.5, "humidity": 45, "annual_sunshine_hours": 2880, "dust_factor": 16},
    {"city": "Ujjain", "state": "Madhya Pradesh", "latitude": 23.18, "longitude": 75.78, "solar_irradiance": 5.58, "avg_temperature": 25.5, "humidity": 46, "annual_sunshine_hours": 2920, "dust_factor": 15},
    
    # Uttar Pradesh
    {"city": "Lucknow", "state": "Uttar Pradesh", "latitude": 26.85, "longitude": 80.95, "solar_irradiance": 5.20, "avg_temperature": 26.0, "humidity": 58, "annual_sunshine_hours": 2650, "dust_factor": 18},
    {"city": "Kanpur", "state": "Uttar Pradesh", "latitude": 26.45, "longitude": 80.35, "solar_irradiance": 5.18, "avg_temperature": 26.5, "humidity": 56, "annual_sunshine_hours": 2630, "dust_factor": 20},
    {"city": "Varanasi", "state": "Uttar Pradesh", "latitude": 25.32, "longitude": 82.99, "solar_irradiance": 5.15, "avg_temperature": 26.8, "humidity": 60, "annual_sunshine_hours": 2600, "dust_factor": 18},
    {"city": "Agra", "state": "Uttar Pradesh", "latitude": 27.18, "longitude": 78.02, "solar_irradiance": 5.25, "avg_temperature": 26.0, "humidity": 52, "annual_sunshine_hours": 2700, "dust_factor": 20},
    {"city": "Noida", "state": "Uttar Pradesh", "latitude": 28.57, "longitude": 77.32, "solar_irradiance": 5.15, "avg_temperature": 25.5, "humidity": 55, "annual_sunshine_hours": 2600, "dust_factor": 22},
    
    # Delhi NCR
    {"city": "New Delhi", "state": "Delhi", "latitude": 28.61, "longitude": 77.21, "solar_irradiance": 5.20, "avg_temperature": 25.5, "humidity": 52, "annual_sunshine_hours": 2650, "dust_factor": 25},
    {"city": "Gurgaon", "state": "Haryana", "latitude": 28.46, "longitude": 77.03, "solar_irradiance": 5.22, "avg_temperature": 25.8, "humidity": 50, "annual_sunshine_hours": 2680, "dust_factor": 22},
    {"city": "Faridabad", "state": "Haryana", "latitude": 28.41, "longitude": 77.31, "solar_irradiance": 5.18, "avg_temperature": 25.6, "humidity": 52, "annual_sunshine_hours": 2640, "dust_factor": 23},
    
    # Punjab & Haryana
    {"city": "Chandigarh", "state": "Chandigarh", "latitude": 30.73, "longitude": 76.78, "solar_irradiance": 5.10, "avg_temperature": 24.0, "humidity": 55, "annual_sunshine_hours": 2550, "dust_factor": 15},
    {"city": "Ludhiana", "state": "Punjab", "latitude": 30.90, "longitude": 75.85, "solar_irradiance": 5.05, "avg_temperature": 24.5, "humidity": 58, "annual_sunshine_hours": 2500, "dust_factor": 18},
    {"city": "Amritsar", "state": "Punjab", "latitude": 31.63, "longitude": 74.87, "solar_irradiance": 5.00, "avg_temperature": 24.0, "humidity": 55, "annual_sunshine_hours": 2480, "dust_factor": 16},
    {"city": "Jalandhar", "state": "Punjab", "latitude": 31.33, "longitude": 75.58, "solar_irradiance": 5.02, "avg_temperature": 24.2, "humidity": 56, "annual_sunshine_hours": 2490, "dust_factor": 17},
    
    # West Bengal
    {"city": "Kolkata", "state": "West Bengal", "latitude": 22.57, "longitude": 88.36, "solar_irradiance": 4.85, "avg_temperature": 27.5, "humidity": 75, "annual_sunshine_hours": 2400, "dust_factor": 18},
    {"city": "Durgapur", "state": "West Bengal", "latitude": 23.55, "longitude": 87.32, "solar_irradiance": 4.90, "avg_temperature": 27.0, "humidity": 70, "annual_sunshine_hours": 2450, "dust_factor": 16},
    {"city": "Siliguri", "state": "West Bengal", "latitude": 26.71, "longitude": 88.43, "solar_irradiance": 4.70, "avg_temperature": 24.5, "humidity": 78, "annual_sunshine_hours": 2300, "dust_factor": 12},
    
    # Bihar & Jharkhand
    {"city": "Patna", "state": "Bihar", "latitude": 25.59, "longitude": 85.14, "solar_irradiance": 5.00, "avg_temperature": 27.0, "humidity": 65, "annual_sunshine_hours": 2500, "dust_factor": 18},
    {"city": "Gaya", "state": "Bihar", "latitude": 24.75, "longitude": 85.01, "solar_irradiance": 5.05, "avg_temperature": 27.5, "humidity": 62, "annual_sunshine_hours": 2550, "dust_factor": 16},
    {"city": "Ranchi", "state": "Jharkhand", "latitude": 23.36, "longitude": 85.33, "solar_irradiance": 4.95, "avg_temperature": 24.5, "humidity": 60, "annual_sunshine_hours": 2480, "dust_factor": 14},
    {"city": "Jamshedpur", "state": "Jharkhand", "latitude": 22.80, "longitude": 86.18, "solar_irradiance": 4.90, "avg_temperature": 26.0, "humidity": 62, "annual_sunshine_hours": 2450, "dust_factor": 16},
    
    # Odisha
    {"city": "Bhubaneswar", "state": "Odisha", "latitude": 20.30, "longitude": 85.82, "solar_irradiance": 5.15, "avg_temperature": 27.5, "humidity": 70, "annual_sunshine_hours": 2600, "dust_factor": 14},
    {"city": "Cuttack", "state": "Odisha", "latitude": 20.46, "longitude": 85.88, "solar_irradiance": 5.12, "avg_temperature": 27.5, "humidity": 72, "annual_sunshine_hours": 2580, "dust_factor": 13},
    {"city": "Rourkela", "state": "Odisha", "latitude": 22.26, "longitude": 84.85, "solar_irradiance": 5.00, "avg_temperature": 26.5, "humidity": 65, "annual_sunshine_hours": 2500, "dust_factor": 15},
    
    # Kerala
    {"city": "Thiruvananthapuram", "state": "Kerala", "latitude": 8.52, "longitude": 76.94, "solar_irradiance": 5.10, "avg_temperature": 27.5, "humidity": 78, "annual_sunshine_hours": 2550, "dust_factor": 8},
    {"city": "Kochi", "state": "Kerala", "latitude": 9.93, "longitude": 76.27, "solar_irradiance": 5.00, "avg_temperature": 28.0, "humidity": 80, "annual_sunshine_hours": 2480, "dust_factor": 8},
    {"city": "Kozhikode", "state": "Kerala", "latitude": 11.25, "longitude": 75.77, "solar_irradiance": 5.05, "avg_temperature": 27.5, "humidity": 78, "annual_sunshine_hours": 2500, "dust_factor": 8},
    
    # Northeast
    {"city": "Guwahati", "state": "Assam", "latitude": 26.14, "longitude": 91.74, "solar_irradiance": 4.60, "avg_temperature": 25.0, "humidity": 80, "annual_sunshine_hours": 2200, "dust_factor": 10},
    {"city": "Imphal", "state": "Manipur", "latitude": 24.82, "longitude": 93.95, "solar_irradiance": 4.50, "avg_temperature": 22.5, "humidity": 75, "annual_sunshine_hours": 2150, "dust_factor": 8},
    {"city": "Shillong", "state": "Meghalaya", "latitude": 25.57, "longitude": 91.88, "solar_irradiance": 4.40, "avg_temperature": 18.5, "humidity": 82, "annual_sunshine_hours": 2100, "dust_factor": 6},
    {"city": "Agartala", "state": "Tripura", "latitude": 23.83, "longitude": 91.28, "solar_irradiance": 4.55, "avg_temperature": 26.0, "humidity": 78, "annual_sunshine_hours": 2180, "dust_factor": 10},
    
    # Himalayan States
    {"city": "Dehradun", "state": "Uttarakhand", "latitude": 30.32, "longitude": 78.03, "solar_irradiance": 4.90, "avg_temperature": 21.5, "humidity": 60, "annual_sunshine_hours": 2400, "dust_factor": 10},
    {"city": "Shimla", "state": "Himachal Pradesh", "latitude": 31.10, "longitude": 77.17, "solar_irradiance": 4.70, "avg_temperature": 15.5, "humidity": 65, "annual_sunshine_hours": 2300, "dust_factor": 8},
    {"city": "Leh", "state": "Ladakh", "latitude": 34.16, "longitude": 77.58, "solar_irradiance": 5.90, "avg_temperature": 8.0, "humidity": 30, "annual_sunshine_hours": 3250, "dust_factor": 12},
    {"city": "Srinagar", "state": "Jammu & Kashmir", "latitude": 34.08, "longitude": 74.79, "solar_irradiance": 4.80, "avg_temperature": 14.0, "humidity": 55, "annual_sunshine_hours": 2350, "dust_factor": 10},
    {"city": "Jammu", "state": "Jammu & Kashmir", "latitude": 32.73, "longitude": 74.87, "solar_irradiance": 5.00, "avg_temperature": 22.0, "humidity": 52, "annual_sunshine_hours": 2500, "dust_factor": 14},
    
    # Chhattisgarh
    {"city": "Raipur", "state": "Chhattisgarh", "latitude": 21.25, "longitude": 81.63, "solar_irradiance": 5.35, "avg_temperature": 27.0, "humidity": 55, "annual_sunshine_hours": 2750, "dust_factor": 14},
    {"city": "Bilaspur", "state": "Chhattisgarh", "latitude": 22.08, "longitude": 82.15, "solar_irradiance": 5.30, "avg_temperature": 26.5, "humidity": 58, "annual_sunshine_hours": 2700, "dust_factor": 13},
    
    # Goa
    {"city": "Panaji", "state": "Goa", "latitude": 15.50, "longitude": 73.83, "solar_irradiance": 5.25, "avg_temperature": 27.5, "humidity": 72, "annual_sunshine_hours": 2700, "dust_factor": 10},
]


def get_india_cities_dataframe():
    """Get the India cities data as a DataFrame."""
    df = pd.DataFrame(INDIA_CITIES_DATA)
    
    # Calculate estimated panel temperature (higher than ambient due to solar heating)
    df['panel_temperature'] = df['avg_temperature'] + 15 + (df['solar_irradiance'] - 5) * 5
    
    # Convert GHI from kWh/m²/day to W/m² (average during daylight hours, ~6 hours peak)
    df['solar_irradiance_wm2'] = df['solar_irradiance'] * 1000 / 6
    
    return df


def calculate_city_efficiency(row, panel_age=5, tilt_angle=30):
    """
    Calculate estimated solar panel efficiency for a city.
    
    Based on real-world factors affecting solar panel performance.
    """
    # Base efficiency for modern monocrystalline panels
    base_efficiency = 20.0
    
    # Temperature coefficient (-0.4% per °C above 25°C)
    temp_effect = -0.4 * max(0, row['panel_temperature'] - 25)
    
    # Irradiance factor (normalized to 5.5 kWh/m²/day as reference)
    irradiance_factor = row['solar_irradiance'] / 5.5
    
    # Humidity effect (reduces efficiency above 60%)
    humidity_effect = -0.05 * max(0, row['humidity'] - 60)
    
    # Dust effect (significant in India)
    dust_effect = -0.2 * row['dust_factor'] / 100 * base_efficiency
    
    # Panel age degradation (~0.5% per year)
    age_effect = -0.5 * panel_age
    
    # Calculate final efficiency
    efficiency = (base_efficiency + temp_effect + humidity_effect + dust_effect + age_effect) * irradiance_factor
    
    # Ensure reasonable bounds
    efficiency = max(8, min(22, efficiency))
    
    return efficiency


def calculate_annual_generation(row, panel_capacity_kw=1, efficiency=None):
    """
    Calculate estimated annual electricity generation.
    
    Parameters:
    -----------
    row : pd.Series
        City data row
    panel_capacity_kw : float
        Installed panel capacity in kW
    efficiency : float, optional
        Panel efficiency (calculated if not provided)
        
    Returns:
    --------
    float
        Annual generation in kWh
    """
    if efficiency is None:
        efficiency = calculate_city_efficiency(row)
    
    # Annual generation = GHI × 365 × Panel Area × Efficiency × Performance Ratio
    # For 1 kW system, area ≈ 5 m² (modern panels ~200W/m²)
    panel_area = panel_capacity_kw * 5
    performance_ratio = 0.75  # Typical system losses
    
    annual_kwh = row['solar_irradiance'] * 365 * panel_area * (efficiency / 100) * performance_ratio
    
    return annual_kwh


def get_state_analysis(state_name=None):
    """
    Get analysis for a specific state or all states.
    
    Returns DataFrame with efficiency and generation estimates.
    """
    df = get_india_cities_dataframe()
    
    if state_name:
        df = df[df['state'] == state_name]
    
    # Calculate efficiency for each city
    df['estimated_efficiency'] = df.apply(calculate_city_efficiency, axis=1)
    
    # Calculate annual generation for 1kW system
    df['annual_generation_kwh'] = df.apply(
        lambda row: calculate_annual_generation(row, efficiency=row['estimated_efficiency']), 
        axis=1
    )
    
    # Calculate solar potential score (0-100)
    df['solar_score'] = (
        (df['solar_irradiance'] / df['solar_irradiance'].max()) * 40 +
        (df['annual_sunshine_hours'] / df['annual_sunshine_hours'].max()) * 30 +
        (1 - df['humidity'] / 100) * 15 +
        (1 - df['dust_factor'] / df['dust_factor'].max()) * 15
    ).round(1)
    
    # Rank cities
    df['rank'] = df['solar_score'].rank(ascending=False).astype(int)
    
    return df.sort_values('solar_score', ascending=False)


def get_all_states():
    """Get list of all states in the dataset."""
    df = get_india_cities_dataframe()
    return sorted(df['state'].unique().tolist())


def get_top_cities(n=10, state=None):
    """Get top N cities for solar installation."""
    df = get_state_analysis(state)
    return df.head(n)


def get_state_summary():
    """Get summary statistics for each state."""
    df = get_state_analysis()
    
    summary = df.groupby('state').agg({
        'solar_irradiance': 'mean',
        'estimated_efficiency': 'mean',
        'annual_generation_kwh': 'mean',
        'solar_score': 'mean',
        'city': 'count'
    }).round(2)
    
    summary.columns = ['Avg GHI (kWh/m²/day)', 'Avg Efficiency (%)', 
                       'Avg Annual Gen (kWh/kW)', 'Avg Solar Score', 'Cities Count']
    
    return summary.sort_values('Avg Solar Score', ascending=False)


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("INDIA SOLAR DATA MODULE TEST")
    print("=" * 60)
    
    print("\nAll States:")
    print(get_all_states())
    
    print("\n\nTop 10 Cities for Solar Installation:")
    print(get_top_cities(10)[['city', 'state', 'solar_irradiance', 'estimated_efficiency', 'solar_score']])
    
    print("\n\nState-wise Summary:")
    print(get_state_summary())

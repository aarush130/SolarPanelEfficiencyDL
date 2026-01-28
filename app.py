"""
Solar Panel Efficiency Prediction - Web Application
===================================================
A beautiful Streamlit web application for predicting solar panel efficiency
using deep learning models.

Author: Solar Panel Efficiency Research Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Solar Panel Efficiency Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, classy look
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #ff8c00 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #ff8c00;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3c72;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(17,153,142,0.3);
        margin: 1rem 0;
    }
    
    .prediction-value {
        font-size: 4rem;
        font-weight: 800;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .prediction-label {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 10px 10px 0 0;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #ff8c00;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff8c00 0%, #ff6b00 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(255,140,0,0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255,140,0,0.5);
    }
</style>
""", unsafe_allow_html=True)


def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    import tensorflow as tf
    import joblib
    
    model_path = 'models/final_model.keras'
    preprocessor_path = 'data/preprocessor.joblib'
    
    model = None
    preprocessor = None
    
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            st.warning(f"Could not load model: {e}")
    
    if os.path.exists(preprocessor_path):
        try:
            preprocessor = joblib.load(preprocessor_path)
        except Exception as e:
            st.warning(f"Could not load preprocessor: {e}")
    
    return model, preprocessor


def load_metrics():
    """Load training metrics."""
    metrics_path = 'models/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def load_sample_data():
    """Load sample data for visualization."""
    data_path = 'data/test_data.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


def predict_efficiency(model, preprocessor, input_features):
    """Make a prediction using the trained model."""
    if model is None or preprocessor is None:
        return None
    
    try:
        # Create feature array in correct order
        feature_columns = preprocessor['feature_columns']
        X = np.array([[input_features.get(col, 0) for col in feature_columns]])
        
        # Scale features
        X_scaled = preprocessor['feature_scaler'].transform(X)
        
        # Predict
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        
        # Inverse transform
        y_pred = preprocessor['target_scaler'].inverse_transform(y_pred_scaled)
        
        return float(y_pred[0][0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def create_gauge_chart(value, title="Efficiency"):
    """Create a beautiful gauge chart for efficiency display."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': '#1e3c72'}},
        delta={'reference': 15, 'increasing': {'color': "#11998e"}, 'decreasing': {'color': "#e74c3c"}},
        gauge={
            'axis': {'range': [0, 25], 'tickwidth': 1, 'tickcolor': "#1e3c72"},
            'bar': {'color': "#ff8c00"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#1e3c72",
            'steps': [
                {'range': [0, 8], 'color': '#ffcccc'},
                {'range': [8, 15], 'color': '#ffffcc'},
                {'range': [15, 25], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "#11998e", 'width': 4},
                'thickness': 0.75,
                'value': 18
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1e3c72", 'family': "Arial"},
        height=300
    )
    
    return fig


def create_feature_importance_chart():
    """Create feature importance visualization."""
    features = [
        'Solar Irradiance', 'Panel Temperature', 'Cloud Cover', 
        'Dust Accumulation', 'Panel Age', 'Ambient Temp',
        'Humidity', 'Tilt Angle', 'Hour of Day', 'Wind Speed'
    ]
    importance = [0.25, 0.18, 0.15, 0.12, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Oranges',
            showscale=False
        ),
        text=[f'{v*100:.1f}%' for v in importance],
        textposition='outside'
    ))
    
    fig.update_layout(
        title={'text': 'Feature Importance for Efficiency Prediction', 'font': {'size': 18}},
        xaxis_title='Relative Importance',
        yaxis_title='',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150)
    )
    
    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp' in df.columns:
        numeric_cols = [c for c in numeric_cols if c != 'timestamp']
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': 'Feature Correlation Matrix', 'font': {'size': 18}},
        height=500,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_distribution_plot(df, column):
    """Create distribution plot for a feature."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[column],
        nbinsx=50,
        marker_color='#ff8c00',
        opacity=0.7,
        name=column
    ))
    
    fig.update_layout(
        title={'text': f'Distribution of {column.replace("_", " ").title()}', 'font': {'size': 16}},
        xaxis_title=column.replace("_", " ").title(),
        yaxis_title='Frequency',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    
    return fig


def create_scatter_plot(df, x_col, y_col):
    """Create scatter plot."""
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color='efficiency',
        color_continuous_scale='Viridis',
        opacity=0.6,
        title=f'{x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    return fig


def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>☀️ Solar Panel Efficiency Predictor</h1>
        <p>Deep Learning-Powered Predictions for Optimal Solar Energy Harvesting</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load resources
    model, preprocessor = load_model_and_preprocessor()
    metrics = load_metrics()
    sample_data = load_sample_data()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Prediction", 
        "📊 Data Analysis", 
        "📈 Model Performance",
        "📚 About"
    ])
    
    # ==================== TAB 1: PREDICTION ====================
    with tab1:
        st.markdown("### Enter Solar Panel Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**☀️ Environmental Conditions**")
            solar_irradiance = st.slider(
                "Solar Irradiance (W/m²)",
                min_value=100, max_value=1200, value=800,
                help="Amount of solar radiation reaching the panel"
            )
            ambient_temp = st.slider(
                "Ambient Temperature (°C)",
                min_value=-10, max_value=45, value=25,
                help="Surrounding air temperature"
            )
            humidity = st.slider(
                "Humidity (%)",
                min_value=20, max_value=95, value=50,
                help="Relative humidity in the air"
            )
            cloud_cover = st.slider(
                "Cloud Cover (%)",
                min_value=0, max_value=100, value=20,
                help="Percentage of sky covered by clouds"
            )
        
        with col2:
            st.markdown("**🔧 Panel Parameters**")
            panel_temp = st.slider(
                "Panel Temperature (°C)",
                min_value=-5, max_value=80, value=45,
                help="Temperature of the solar panel surface"
            )
            panel_age = st.slider(
                "Panel Age (years)",
                min_value=0, max_value=25, value=5,
                help="Age of the solar panel installation"
            )
            tilt_angle = st.slider(
                "Tilt Angle (degrees)",
                min_value=10, max_value=50, value=30,
                help="Angle of panel tilt from horizontal"
            )
            dust_accumulation = st.slider(
                "Dust Accumulation (%)",
                min_value=0, max_value=50, value=10,
                help="Dust coverage on panel surface"
            )
        
        with col3:
            st.markdown("**🌡️ Additional Factors**")
            wind_speed = st.slider(
                "Wind Speed (m/s)",
                min_value=0.0, max_value=15.0, value=3.0, step=0.5,
                help="Wind speed for cooling effect"
            )
            hour_of_day = st.slider(
                "Hour of Day",
                min_value=6, max_value=20, value=12,
                help="Time of day (6 AM - 8 PM)"
            )
        
        # Calculate derived features
        temp_difference = panel_temp - ambient_temp
        irradiance_temp_ratio = solar_irradiance / (ambient_temp + 273.15)
        effective_irradiance = solar_irradiance * (1 - cloud_cover/100) * (1 - dust_accumulation/100)
        is_peak_hours = 1 if 10 <= hour_of_day <= 15 else 0
        optimal_conditions = 1 if (solar_irradiance > 700 and 15 < ambient_temp < 35 and cloud_cover < 30) else 0
        panel_age_category = min(3, int(panel_age / 5))
        wind_cooling_factor = wind_speed * temp_difference / 100
        
        # Predict button
        st.markdown("---")
        
        if st.button("🔮 Predict Efficiency", use_container_width=True):
            if model is not None and preprocessor is not None:
                # Prepare input features
                input_features = {
                    'solar_irradiance': solar_irradiance,
                    'ambient_temperature': ambient_temp,
                    'panel_temperature': panel_temp,
                    'humidity': humidity,
                    'wind_speed': wind_speed,
                    'dust_accumulation': dust_accumulation,
                    'panel_age': panel_age,
                    'tilt_angle': tilt_angle,
                    'cloud_cover': cloud_cover,
                    'hour_of_day': hour_of_day,
                    'temp_difference': temp_difference,
                    'irradiance_temp_ratio': irradiance_temp_ratio,
                    'effective_irradiance': effective_irradiance,
                    'is_peak_hours': is_peak_hours,
                    'optimal_conditions': optimal_conditions,
                    'panel_age_category': panel_age_category,
                    'wind_cooling_factor': wind_cooling_factor
                }
                
                # Make prediction
                efficiency = predict_efficiency(model, preprocessor, input_features)
                
                if efficiency is not None:
                    # Display results
                    col_a, col_b = st.columns([1, 1])
                    
                    with col_a:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-value">{efficiency:.2f}%</div>
                            <div class="prediction-label">Predicted Solar Panel Efficiency</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Efficiency rating
                        if efficiency >= 18:
                            rating = "🌟 Excellent"
                            color = "#11998e"
                        elif efficiency >= 15:
                            rating = "✅ Good"
                            color = "#3498db"
                        elif efficiency >= 12:
                            rating = "⚠️ Fair"
                            color = "#f39c12"
                        else:
                            rating = "❌ Poor"
                            color = "#e74c3c"
                        
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 1rem;">
                            <span style="font-size: 1.5rem; color: {color}; font-weight: 600;">{rating}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        gauge_fig = create_gauge_chart(efficiency, "Efficiency (%)")
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Optimization Recommendations")
                    
                    recommendations = []
                    if dust_accumulation > 20:
                        recommendations.append("🧹 **Clean the panels** - High dust accumulation is reducing efficiency")
                    if panel_temp > 50:
                        recommendations.append("❄️ **Improve cooling** - Panel temperature is high, consider better ventilation")
                    if tilt_angle < 20 or tilt_angle > 40:
                        recommendations.append("📐 **Adjust tilt angle** - Optimal tilt is typically 25-35 degrees")
                    if cloud_cover > 50:
                        recommendations.append("☁️ **Weather impact** - High cloud cover is limiting irradiance")
                    if panel_age > 15:
                        recommendations.append("🔄 **Consider upgrade** - Panel age may be affecting performance")
                    
                    if recommendations:
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                    else:
                        st.success("✅ Your panel configuration is well-optimized!")
            else:
                st.warning("⚠️ Model not loaded. Please train the model first using `python src/train.py`")
                
                # Show simulated prediction
                st.info("Showing simulated prediction based on physics model...")
                
                # Simple physics-based estimation
                base_eff = 20
                temp_effect = -0.004 * (panel_temp - 25) * base_eff
                irr_factor = min(1.2, solar_irradiance / 1000)
                cloud_effect = -0.3 * cloud_cover / 100
                dust_effect = -0.15 * dust_accumulation / 100
                age_effect = -0.005 * panel_age * base_eff
                
                estimated = (base_eff + temp_effect + cloud_effect * base_eff + 
                           dust_effect * base_eff + age_effect) * irr_factor
                estimated = max(0, min(25, estimated))
                
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-value">{estimated:.2f}%</div>
                    <div class="prediction-label">Estimated Efficiency (Physics Model)</div>
                </div>
                """, unsafe_allow_html=True)
    
    # ==================== TAB 2: DATA ANALYSIS ====================
    with tab2:
        st.markdown("### 📊 Dataset Analysis")
        
        if sample_data is not None:
            # Dataset overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", f"{len(sample_data):,}")
            with col2:
                st.metric("Features", f"{len(sample_data.columns) - 1}")
            with col3:
                st.metric("Avg Efficiency", f"{sample_data['efficiency'].mean():.2f}%")
            with col4:
                st.metric("Efficiency Std", f"{sample_data['efficiency'].std():.2f}%")
            
            st.markdown("---")
            
            # Visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.plotly_chart(create_feature_importance_chart(), use_container_width=True)
            
            with viz_col2:
                st.plotly_chart(create_correlation_heatmap(sample_data), use_container_width=True)
            
            st.markdown("---")
            
            # Distribution plots
            st.markdown("### Feature Distributions")
            dist_col1, dist_col2, dist_col3 = st.columns(3)
            
            with dist_col1:
                st.plotly_chart(create_distribution_plot(sample_data, 'efficiency'), use_container_width=True)
            with dist_col2:
                st.plotly_chart(create_distribution_plot(sample_data, 'solar_irradiance'), use_container_width=True)
            with dist_col3:
                st.plotly_chart(create_distribution_plot(sample_data, 'panel_temperature'), use_container_width=True)
            
            # Scatter plots
            st.markdown("### Relationship Analysis")
            scatter_col1, scatter_col2 = st.columns(2)
            
            with scatter_col1:
                st.plotly_chart(
                    create_scatter_plot(sample_data, 'solar_irradiance', 'efficiency'),
                    use_container_width=True
                )
            
            with scatter_col2:
                st.plotly_chart(
                    create_scatter_plot(sample_data, 'panel_temperature', 'efficiency'),
                    use_container_width=True
                )
            
            # Raw data preview
            st.markdown("### Raw Data Preview")
            st.dataframe(sample_data.head(100), use_container_width=True)
        else:
            st.info("📁 No data found. Generate data using `python src/data_generator.py`")
    
    # ==================== TAB 3: MODEL PERFORMANCE ====================
    with tab3:
        st.markdown("### 📈 Model Performance Metrics")
        
        if metrics:
            # Metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Mean Absolute Error</div>
                    <div class="metric-value">{:.4f}%</div>
                </div>
                """.format(metrics['metrics']['mae']), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value">{:.4f}%</div>
                </div>
                """.format(metrics['metrics']['rmse']), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">R² Score</div>
                    <div class="metric-value">{:.4f}</div>
                </div>
                """.format(metrics['metrics']['r2']), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">MAPE</div>
                    <div class="metric-value">{:.2f}%</div>
                </div>
                """.format(metrics['metrics']['mape']), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Model info
            st.markdown("### Model Information")
            st.json({
                "Model Type": metrics.get('model_type', 'Deep Residual Network'),
                "Features Used": len(metrics.get('feature_columns', [])),
                "Training Timestamp": metrics.get('timestamp', 'N/A')
            })
            
            # Training history plot
            history_path = 'models/training_history.json'
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                st.markdown("### Training History")
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Mean Absolute Error'))
                
                fig.add_trace(
                    go.Scatter(y=history['loss'], name='Training Loss', line=dict(color='#1e3c72')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(y=history['val_loss'], name='Validation Loss', line=dict(color='#ff8c00')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(y=history['mae'], name='Training MAE', line=dict(color='#1e3c72')),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(y=history['val_mae'], name='Validation MAE', line=dict(color='#ff8c00')),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=True, paper_bgcolor='rgba(0,0,0,0)')
                fig.update_xaxes(title_text="Epoch")
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Results image
            results_path = 'models/training_results.png'
            if os.path.exists(results_path):
                st.markdown("### Detailed Results")
                st.image(results_path, caption='Training Results Visualization', use_container_width=True)
        else:
            st.info("📊 No metrics found. Train the model first using `python src/train.py`")
    
    # ==================== TAB 4: ABOUT ====================
    with tab4:
        st.markdown("""
        ### 📚 About This Project
        
        This **Solar Panel Efficiency Prediction** system uses **Deep Learning** to predict the efficiency
        of solar panels based on various environmental and panel parameters.
        
        ---
        
        ### 🎯 Features
        
        - **Deep Residual Networks** with skip connections for better gradient flow
        - **Attention Mechanisms** for feature importance learning
        - **Ensemble Models** combining multiple architectures
        - **Real-time Predictions** through an intuitive web interface
        - **Comprehensive Visualizations** for data analysis
        
        ---
        
        ### 📊 Input Parameters
        
        | Parameter | Description | Range |
        |-----------|-------------|-------|
        | Solar Irradiance | Solar radiation intensity | 100-1200 W/m² |
        | Ambient Temperature | Surrounding air temperature | -10 to 45°C |
        | Panel Temperature | Panel surface temperature | -5 to 80°C |
        | Humidity | Relative humidity | 20-95% |
        | Wind Speed | Wind speed for cooling | 0-15 m/s |
        | Dust Accumulation | Dust on panel surface | 0-50% |
        | Panel Age | Installation age | 0-25 years |
        | Tilt Angle | Panel tilt from horizontal | 10-50° |
        | Cloud Cover | Sky coverage by clouds | 0-100% |
        | Hour of Day | Time of measurement | 6-20 hours |
        
        ---
        
        ### 🧠 Model Architecture
        
        The deep learning model uses:
        - **Input Layer**: 17 features (10 primary + 7 engineered)
        - **Residual Blocks**: 4 blocks with skip connections
        - **Batch Normalization**: For stable training
        - **Dropout**: 0.2 for regularization
        - **Output Layer**: Single neuron for efficiency prediction
        
        ---
        
        ### 📁 Project Structure
        
        ```
        SolarPanelEfficiencyDL/
        ├── app.py                 # Streamlit web application
        ├── requirements.txt       # Python dependencies
        ├── src/
        │   ├── data_generator.py  # Synthetic data generation
        │   ├── preprocessing.py   # Data preprocessing
        │   ├── model.py          # Neural network architectures
        │   └── train.py          # Training pipeline
        ├── data/                  # Dataset files
        ├── models/                # Trained models
        └── notebooks/             # Jupyter notebooks
        ```
        
        ---
        
        ### 🚀 Quick Start
        
        ```bash
        # Install dependencies
        pip install -r requirements.txt
        
        # Generate data
        python src/data_generator.py
        
        # Train model
        python src/train.py --model-type deep --epochs 100
        
        # Run web application
        streamlit run app.py
        ```
        
        ---
        
        ### 👨‍💻 Final Semester Project
        
        This project demonstrates the application of deep learning techniques
        for renewable energy optimization, combining:
        
        - Data Science & Feature Engineering
        - Deep Learning & Neural Networks
        - Web Development with Streamlit
        - Scientific Visualization
        
        ---
        
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>Built with ❤️ using TensorFlow & Streamlit</p>
            <p>© 2024 Solar Panel Efficiency Research Team</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🌞 Solar Panel Efficiency Prediction System | Deep Learning Project | Final Semester 2024</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

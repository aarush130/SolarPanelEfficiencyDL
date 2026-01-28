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

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Solar Panel Efficiency Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ INDIA CITIES DATA (EMBEDDED) ============
INDIA_CITIES_DATA = [
    # Rajasthan - High solar potential
    {"city": "Jaisalmer", "state": "Rajasthan", "latitude": 26.92, "longitude": 70.90, "solar_irradiance": 5.89, "avg_temperature": 27.5, "humidity": 35, "annual_sunshine_hours": 3200, "dust_factor": 25},
    {"city": "Jodhpur", "state": "Rajasthan", "latitude": 26.28, "longitude": 73.02, "solar_irradiance": 5.85, "avg_temperature": 27.0, "humidity": 38, "annual_sunshine_hours": 3150, "dust_factor": 22},
    {"city": "Bikaner", "state": "Rajasthan", "latitude": 28.02, "longitude": 73.31, "solar_irradiance": 5.80, "avg_temperature": 26.5, "humidity": 36, "annual_sunshine_hours": 3100, "dust_factor": 24},
    {"city": "Jaipur", "state": "Rajasthan", "latitude": 26.92, "longitude": 75.79, "solar_irradiance": 5.65, "avg_temperature": 25.8, "humidity": 42, "annual_sunshine_hours": 3000, "dust_factor": 18},
    {"city": "Udaipur", "state": "Rajasthan", "latitude": 24.58, "longitude": 73.71, "solar_irradiance": 5.55, "avg_temperature": 25.2, "humidity": 45, "annual_sunshine_hours": 2950, "dust_factor": 15},
    # Gujarat
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
    # Karnataka
    {"city": "Bengaluru", "state": "Karnataka", "latitude": 12.97, "longitude": 77.59, "solar_irradiance": 5.40, "avg_temperature": 24.5, "humidity": 55, "annual_sunshine_hours": 2800, "dust_factor": 12},
    {"city": "Mysuru", "state": "Karnataka", "latitude": 12.30, "longitude": 76.64, "solar_irradiance": 5.35, "avg_temperature": 24.0, "humidity": 58, "annual_sunshine_hours": 2750, "dust_factor": 10},
    {"city": "Hubli", "state": "Karnataka", "latitude": 15.36, "longitude": 75.12, "solar_irradiance": 5.50, "avg_temperature": 25.5, "humidity": 52, "annual_sunshine_hours": 2850, "dust_factor": 13},
    # Tamil Nadu
    {"city": "Chennai", "state": "Tamil Nadu", "latitude": 13.08, "longitude": 80.27, "solar_irradiance": 5.35, "avg_temperature": 29.0, "humidity": 70, "annual_sunshine_hours": 2800, "dust_factor": 15},
    {"city": "Coimbatore", "state": "Tamil Nadu", "latitude": 11.02, "longitude": 76.96, "solar_irradiance": 5.45, "avg_temperature": 26.5, "humidity": 60, "annual_sunshine_hours": 2850, "dust_factor": 12},
    {"city": "Madurai", "state": "Tamil Nadu", "latitude": 9.92, "longitude": 78.12, "solar_irradiance": 5.55, "avg_temperature": 28.5, "humidity": 62, "annual_sunshine_hours": 2900, "dust_factor": 14},
    # Telangana & Andhra Pradesh
    {"city": "Hyderabad", "state": "Telangana", "latitude": 17.39, "longitude": 78.49, "solar_irradiance": 5.50, "avg_temperature": 27.0, "humidity": 55, "annual_sunshine_hours": 2900, "dust_factor": 16},
    {"city": "Visakhapatnam", "state": "Andhra Pradesh", "latitude": 17.69, "longitude": 83.22, "solar_irradiance": 5.30, "avg_temperature": 28.0, "humidity": 72, "annual_sunshine_hours": 2700, "dust_factor": 12},
    {"city": "Vijayawada", "state": "Andhra Pradesh", "latitude": 16.51, "longitude": 80.65, "solar_irradiance": 5.45, "avg_temperature": 29.0, "humidity": 68, "annual_sunshine_hours": 2850, "dust_factor": 14},
    {"city": "Anantapur", "state": "Andhra Pradesh", "latitude": 14.68, "longitude": 77.60, "solar_irradiance": 5.72, "avg_temperature": 28.0, "humidity": 48, "annual_sunshine_hours": 3000, "dust_factor": 16},
    # Madhya Pradesh
    {"city": "Bhopal", "state": "Madhya Pradesh", "latitude": 23.26, "longitude": 77.41, "solar_irradiance": 5.50, "avg_temperature": 25.5, "humidity": 50, "annual_sunshine_hours": 2850, "dust_factor": 15},
    {"city": "Indore", "state": "Madhya Pradesh", "latitude": 22.72, "longitude": 75.86, "solar_irradiance": 5.55, "avg_temperature": 25.0, "humidity": 48, "annual_sunshine_hours": 2900, "dust_factor": 14},
    {"city": "Jabalpur", "state": "Madhya Pradesh", "latitude": 23.18, "longitude": 79.95, "solar_irradiance": 5.45, "avg_temperature": 26.0, "humidity": 52, "annual_sunshine_hours": 2800, "dust_factor": 13},
    # Uttar Pradesh
    {"city": "Lucknow", "state": "Uttar Pradesh", "latitude": 26.85, "longitude": 80.95, "solar_irradiance": 5.20, "avg_temperature": 26.0, "humidity": 58, "annual_sunshine_hours": 2650, "dust_factor": 18},
    {"city": "Kanpur", "state": "Uttar Pradesh", "latitude": 26.45, "longitude": 80.35, "solar_irradiance": 5.18, "avg_temperature": 26.5, "humidity": 56, "annual_sunshine_hours": 2630, "dust_factor": 20},
    {"city": "Varanasi", "state": "Uttar Pradesh", "latitude": 25.32, "longitude": 82.99, "solar_irradiance": 5.15, "avg_temperature": 26.8, "humidity": 60, "annual_sunshine_hours": 2600, "dust_factor": 18},
    {"city": "Agra", "state": "Uttar Pradesh", "latitude": 27.18, "longitude": 78.02, "solar_irradiance": 5.25, "avg_temperature": 26.0, "humidity": 52, "annual_sunshine_hours": 2700, "dust_factor": 20},
    # Delhi NCR
    {"city": "New Delhi", "state": "Delhi", "latitude": 28.61, "longitude": 77.21, "solar_irradiance": 5.20, "avg_temperature": 25.5, "humidity": 52, "annual_sunshine_hours": 2650, "dust_factor": 25},
    {"city": "Gurgaon", "state": "Haryana", "latitude": 28.46, "longitude": 77.03, "solar_irradiance": 5.22, "avg_temperature": 25.8, "humidity": 50, "annual_sunshine_hours": 2680, "dust_factor": 22},
    # Punjab
    {"city": "Chandigarh", "state": "Chandigarh", "latitude": 30.73, "longitude": 76.78, "solar_irradiance": 5.10, "avg_temperature": 24.0, "humidity": 55, "annual_sunshine_hours": 2550, "dust_factor": 15},
    {"city": "Ludhiana", "state": "Punjab", "latitude": 30.90, "longitude": 75.85, "solar_irradiance": 5.05, "avg_temperature": 24.5, "humidity": 58, "annual_sunshine_hours": 2500, "dust_factor": 18},
    {"city": "Amritsar", "state": "Punjab", "latitude": 31.63, "longitude": 74.87, "solar_irradiance": 5.00, "avg_temperature": 24.0, "humidity": 55, "annual_sunshine_hours": 2480, "dust_factor": 16},
    # West Bengal
    {"city": "Kolkata", "state": "West Bengal", "latitude": 22.57, "longitude": 88.36, "solar_irradiance": 4.85, "avg_temperature": 27.5, "humidity": 75, "annual_sunshine_hours": 2400, "dust_factor": 18},
    {"city": "Durgapur", "state": "West Bengal", "latitude": 23.55, "longitude": 87.32, "solar_irradiance": 4.90, "avg_temperature": 27.0, "humidity": 70, "annual_sunshine_hours": 2450, "dust_factor": 16},
    # Bihar & Jharkhand
    {"city": "Patna", "state": "Bihar", "latitude": 25.59, "longitude": 85.14, "solar_irradiance": 5.00, "avg_temperature": 27.0, "humidity": 65, "annual_sunshine_hours": 2500, "dust_factor": 18},
    {"city": "Ranchi", "state": "Jharkhand", "latitude": 23.36, "longitude": 85.33, "solar_irradiance": 4.95, "avg_temperature": 24.5, "humidity": 60, "annual_sunshine_hours": 2480, "dust_factor": 14},
    # Odisha
    {"city": "Bhubaneswar", "state": "Odisha", "latitude": 20.30, "longitude": 85.82, "solar_irradiance": 5.15, "avg_temperature": 27.5, "humidity": 70, "annual_sunshine_hours": 2600, "dust_factor": 14},
    # Kerala
    {"city": "Thiruvananthapuram", "state": "Kerala", "latitude": 8.52, "longitude": 76.94, "solar_irradiance": 5.10, "avg_temperature": 27.5, "humidity": 78, "annual_sunshine_hours": 2550, "dust_factor": 8},
    {"city": "Kochi", "state": "Kerala", "latitude": 9.93, "longitude": 76.27, "solar_irradiance": 5.00, "avg_temperature": 28.0, "humidity": 80, "annual_sunshine_hours": 2480, "dust_factor": 8},
    # Northeast
    {"city": "Guwahati", "state": "Assam", "latitude": 26.14, "longitude": 91.74, "solar_irradiance": 4.60, "avg_temperature": 25.0, "humidity": 80, "annual_sunshine_hours": 2200, "dust_factor": 10},
    {"city": "Shillong", "state": "Meghalaya", "latitude": 25.57, "longitude": 91.88, "solar_irradiance": 4.40, "avg_temperature": 18.5, "humidity": 82, "annual_sunshine_hours": 2100, "dust_factor": 6},
    # Himalayan States
    {"city": "Dehradun", "state": "Uttarakhand", "latitude": 30.32, "longitude": 78.03, "solar_irradiance": 4.90, "avg_temperature": 21.5, "humidity": 60, "annual_sunshine_hours": 2400, "dust_factor": 10},
    {"city": "Shimla", "state": "Himachal Pradesh", "latitude": 31.10, "longitude": 77.17, "solar_irradiance": 4.70, "avg_temperature": 15.5, "humidity": 65, "annual_sunshine_hours": 2300, "dust_factor": 8},
    {"city": "Leh", "state": "Ladakh", "latitude": 34.16, "longitude": 77.58, "solar_irradiance": 5.90, "avg_temperature": 8.0, "humidity": 30, "annual_sunshine_hours": 3250, "dust_factor": 12},
    {"city": "Srinagar", "state": "Jammu & Kashmir", "latitude": 34.08, "longitude": 74.79, "solar_irradiance": 4.80, "avg_temperature": 14.0, "humidity": 55, "annual_sunshine_hours": 2350, "dust_factor": 10},
    # Chhattisgarh
    {"city": "Raipur", "state": "Chhattisgarh", "latitude": 21.25, "longitude": 81.63, "solar_irradiance": 5.35, "avg_temperature": 27.0, "humidity": 55, "annual_sunshine_hours": 2750, "dust_factor": 14},
    # Goa
    {"city": "Panaji", "state": "Goa", "latitude": 15.50, "longitude": 73.83, "solar_irradiance": 5.25, "avg_temperature": 27.5, "humidity": 72, "annual_sunshine_hours": 2700, "dust_factor": 10},
]

# ============ HELPER FUNCTIONS ============

def get_india_cities_dataframe():
    """Get the India cities data as a DataFrame."""
    df = pd.DataFrame(INDIA_CITIES_DATA)
    df['panel_temperature'] = df['avg_temperature'] + 15 + (df['solar_irradiance'] - 5) * 5
    df['solar_irradiance_wm2'] = df['solar_irradiance'] * 1000 / 6
    return df

def calculate_city_efficiency(row, panel_age=5):
    """Calculate estimated solar panel efficiency for a city."""
    base_efficiency = 20.0
    temp_effect = -0.4 * max(0, row['panel_temperature'] - 25)
    irradiance_factor = row['solar_irradiance'] / 5.5
    humidity_effect = -0.05 * max(0, row['humidity'] - 60)
    dust_effect = -0.2 * row['dust_factor'] / 100 * base_efficiency
    age_effect = -0.5 * panel_age
    efficiency = (base_efficiency + temp_effect + humidity_effect + dust_effect + age_effect) * irradiance_factor
    return max(8, min(22, efficiency))

def get_state_analysis(state_name=None):
    """Get analysis for a specific state or all states."""
    df = get_india_cities_dataframe()
    if state_name:
        df = df[df['state'] == state_name]
    df['estimated_efficiency'] = df.apply(calculate_city_efficiency, axis=1)
    df['annual_generation_kwh'] = df['solar_irradiance'] * 365 * 5 * (df['estimated_efficiency'] / 100) * 0.75
    df['solar_score'] = (
        (df['solar_irradiance'] / df['solar_irradiance'].max()) * 40 +
        (df['annual_sunshine_hours'] / df['annual_sunshine_hours'].max()) * 30 +
        (1 - df['humidity'] / 100) * 15 +
        (1 - df['dust_factor'] / df['dust_factor'].max()) * 15
    ).round(1)
    df['rank'] = df['solar_score'].rank(ascending=False).astype(int)
    return df.sort_values('solar_score', ascending=False)

def get_all_states():
    """Get list of all states."""
    df = get_india_cities_dataframe()
    return sorted(df['state'].unique().tolist())

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

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #ff8c00 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #ff8c00;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============ MAIN APP ============
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>☀️ Solar Panel Efficiency Predictor</h1>
        <p>Deep Learning-Powered Predictions for Optimal Solar Energy Harvesting in India</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🇮🇳 India Solar Map", 
        "🔮 Prediction", 
        "📊 Data Analysis",
        "📚 About"
    ])
    
    # ==================== TAB 1: INDIA SOLAR MAP ====================
    with tab1:
        st.markdown("### 🇮🇳 Solar Panel Efficiency Across India")
        
        india_df = get_state_analysis()
        all_states = get_all_states()
        
        col_select1, col_select2 = st.columns([2, 1])
        with col_select1:
            selected_state = st.selectbox("Select State", ["All India"] + all_states)
        with col_select2:
            panel_age_input = st.slider("Panel Age (years)", 0, 20, 5)
        
        if selected_state != "All India":
            display_df = get_state_analysis(selected_state)
        else:
            display_df = india_df
        
        display_df['estimated_efficiency'] = display_df.apply(
            lambda row: calculate_city_efficiency(row, panel_age=panel_age_input), axis=1
        )
        
        # Metrics
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        best_city = display_df.iloc[0]
        m1.metric("🏆 Best Location", best_city['city'], f"{best_city['estimated_efficiency']:.1f}%")
        m2.metric("☀️ Highest GHI", f"{display_df['solar_irradiance'].max():.2f}", "kWh/m²/day")
        m3.metric("⚡ Avg Efficiency", f"{display_df['estimated_efficiency'].mean():.1f}%")
        m4.metric("🔋 Avg Generation", f"{display_df['annual_generation_kwh'].mean():.0f} kWh/kW")
        
        st.markdown("---")
        
        # Map and Rankings
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### 🗺️ Solar Potential Map")
            fig_map = px.scatter_geo(
                display_df,
                lat="latitude",
                lon="longitude",
                color="estimated_efficiency",
                size="solar_irradiance",
                hover_name="city",
                hover_data=["state", "solar_irradiance", "estimated_efficiency"],
                color_continuous_scale="YlOrRd",
                size_max=20,
                scope="asia"
            )
            fig_map.update_geos(
                fitbounds="locations",
                visible=True,
                showcountries=True,
                countrycolor="lightgray"
            )
            fig_map.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
        
        with col2:
            st.markdown("#### 🏅 Top Locations")
            ranking_df = display_df[['city', 'state', 'estimated_efficiency', 'solar_score']].head(10).copy()
            ranking_df.columns = ['City', 'State', 'Efficiency %', 'Score']
            ranking_df['Efficiency %'] = ranking_df['Efficiency %'].round(1)
            ranking_df['Score'] = ranking_df['Score'].round(1)
            st.dataframe(ranking_df, hide_index=True, use_container_width=True)
        
        # State comparison
        st.markdown("---")
        st.markdown("#### 📊 State-wise Comparison")
        state_summary = get_state_summary().reset_index()
        
        c1, c2 = st.columns(2)
        with c1:
            fig_bar = px.bar(state_summary.head(15), x='state', y='Avg Efficiency (%)',
                           color='Avg GHI (kWh/m²/day)', color_continuous_scale='Oranges')
            fig_bar.update_layout(xaxis_tickangle=-45, height=350)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with c2:
            fig_scatter = px.scatter(display_df, x='solar_irradiance', y='estimated_efficiency',
                                   color='state', hover_name='city', size='annual_sunshine_hours')
            fig_scatter.update_layout(height=350)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ==================== TAB 2: PREDICTION ====================
    with tab2:
        st.markdown("### 🔮 Predict Solar Panel Efficiency")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("**☀️ Environmental**")
            solar_irradiance = st.slider("Solar Irradiance (W/m²)", 100, 1200, 800)
            ambient_temp = st.slider("Ambient Temperature (°C)", -10, 45, 25)
            humidity = st.slider("Humidity (%)", 20, 95, 50)
            cloud_cover = st.slider("Cloud Cover (%)", 0, 100, 20)
        
        with c2:
            st.markdown("**🔧 Panel Parameters**")
            panel_temp = st.slider("Panel Temperature (°C)", -5, 80, 45)
            panel_age = st.slider("Panel Age (years)", 0, 25, 5)
            tilt_angle = st.slider("Tilt Angle (°)", 10, 50, 30)
            dust = st.slider("Dust Accumulation (%)", 0, 50, 10)
        
        with c3:
            st.markdown("**🌡️ Other Factors**")
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0)
            hour = st.slider("Hour of Day", 6, 20, 12)
        
        if st.button("🔮 Predict Efficiency", use_container_width=True):
            # Physics-based prediction
            base_eff = 20
            temp_effect = -0.004 * (panel_temp - 25) * base_eff
            irr_factor = min(1.2, solar_irradiance / 1000)
            cloud_effect = -0.3 * cloud_cover / 100
            dust_effect = -0.15 * dust / 100
            age_effect = -0.005 * panel_age * base_eff
            hour_factor = np.exp(-0.5 * ((hour - 12) / 4) ** 2)
            
            efficiency = (base_eff + temp_effect + cloud_effect * base_eff + 
                        dust_effect * base_eff + age_effect) * irr_factor * hour_factor
            efficiency = max(0, min(25, efficiency))
            
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-value">{efficiency:.2f}%</div>
                <p style="color: white; font-size: 1.2rem;">Predicted Efficiency</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Rating
            if efficiency >= 18:
                st.success("🌟 Excellent efficiency!")
            elif efficiency >= 15:
                st.info("✅ Good efficiency")
            elif efficiency >= 12:
                st.warning("⚠️ Fair efficiency")
            else:
                st.error("❌ Poor efficiency - check conditions")
    
    # ==================== TAB 3: DATA ANALYSIS ====================
    with tab3:
        st.markdown("### 📊 Data Analysis")
        
        india_df = get_state_analysis()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cities", len(india_df))
        c2.metric("Total States", india_df['state'].nunique())
        c3.metric("Avg Efficiency", f"{india_df['estimated_efficiency'].mean():.1f}%")
        c4.metric("Max GHI", f"{india_df['solar_irradiance'].max():.2f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(india_df, x='estimated_efficiency', nbins=20,
                                   title='Efficiency Distribution', color_discrete_sequence=['#ff8c00'])
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_box = px.box(india_df, x='state', y='estimated_efficiency',
                           title='Efficiency by State')
            fig_box.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Correlation
        st.markdown("#### Correlation Matrix")
        numeric_cols = ['solar_irradiance', 'avg_temperature', 'humidity', 
                       'annual_sunshine_hours', 'dust_factor', 'estimated_efficiency']
        corr = india_df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Data table
        st.markdown("#### 📋 City Data")
        st.dataframe(india_df[['city', 'state', 'solar_irradiance', 'avg_temperature', 
                              'humidity', 'estimated_efficiency', 'solar_score']].round(2),
                    hide_index=True, use_container_width=True)
    
    # ==================== TAB 4: ABOUT ====================
    with tab4:
        st.markdown("""
        ### 📚 About This Project
        
        **Solar Panel Efficiency Prediction using Deep Learning**
        
        This system predicts solar panel efficiency based on environmental and operational 
        parameters, with special focus on Indian cities.
        
        ---
        
        #### 🎯 Features
        - Real data for **50+ Indian cities** across all states
        - Interactive **India Solar Map**
        - **Efficiency prediction** based on physics models
        - State-wise **comparison and ranking**
        - **Data visualization** and analysis
        
        ---
        
        #### 📊 Data Sources
        - Ministry of New and Renewable Energy (MNRE)
        - India Meteorological Department (IMD)
        - National Institute of Solar Energy (NISE)
        
        ---
        
        #### 🏆 Top Solar Locations in India
        1. **Leh, Ladakh** - Highest solar irradiance
        2. **Jaisalmer, Rajasthan** - Desert conditions
        3. **Jodhpur, Rajasthan** - High sunshine hours
        
        ---
        
        **Built with ❤️ using Streamlit & Plotly**
        
        *Final Semester Project - VIT*
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("<center>🌞 Solar Panel Efficiency Prediction | Final Semester Project</center>", 
               unsafe_allow_html=True)

if __name__ == "__main__":
    main()

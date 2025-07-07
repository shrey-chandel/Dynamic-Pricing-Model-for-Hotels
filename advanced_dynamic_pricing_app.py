import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
from datetime import datetime
import io

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    try:
        data = pd.read_csv('hotel_dynamic_pricing.csv')  # Relative path to repo root
    except FileNotFoundError:
        st.error("Error: 'hotel_dynamic_pricing.csv' not found. Please ensure it’s in the repository root.")
        return None, None, None
    
    features = ['city', 'hotel_type', 'room_type', 'customer_segment', 'length_of_stay', 
                'adults', 'children', 'lead_time_days', 'occupancy_rate', 
                'base_adr', 'competitor_adr', 'is_peak_season', 'is_special_event', 'is_weekend_checkin']
    target = 'dynamic_adr'
    
    if data.empty or not all(col in data.columns for col in features + [target]):
        st.error("Error: Dataset is empty or missing required columns.")
        return None, None, None
    
    categorical_cols = ['city', 'hotel_type', 'room_type', 'customer_segment']
    numerical_cols = ['length_of_stay', 'adults', 'children', 'lead_time_days', 'occupancy_rate', 
                      'base_adr', 'competitor_adr', 'is_peak_season', 'is_special_event', 'is_weekend_checkin']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ])
    
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_processed, y_train)
    
    return data, preprocessor, model

# Streamlit app configuration (first command)
st.set_page_config(page_title="Advanced Pricing Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for advanced theme
st.markdown("""
    <style>
    .stApp { background-color: #e9ecef; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .sidebar .sidebar-content { background-color: #ffffff; border-right: 2px solid #dee2e6; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 5px; }
    .stButton>button:hover { background-color: #0056b3; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# App title and branding
st.image("https://via.placeholder.com/150", width=150)  # Replace with your logo URL or local file
st.title("Advanced Dynamic Pricing Dashboard")
st.caption("Optimize hotel pricing with real-time predictions and market insights.")

# Help section
with st.expander("Dashboard Guide"):
    st.write("""
    - **Configure Scenario**: Adjust inputs in the sidebar for real-time pricing.
    - **Compare Scenarios**: Use sliders to see multiple predictions side-by-side.
    - **Trends**: Explore the line chart for real-time ADR trends.
    - **Export**: Download a report of your analysis.
    """)

# Load data and model
data, preprocessor, model = load_and_preprocess_data()
if data is None or preprocessor is None or model is None:
    st.stop()

# Sidebar for input features
st.sidebar.header("Scenario Configuration")
city = st.sidebar.selectbox("Select City", data['city'].unique(), index=0)
hotel_type = st.sidebar.selectbox("Select Hotel Type", data['hotel_type'].unique(), index=0)
base_occupancy = st.sidebar.slider("Base Occupancy Rate (%)", 50, 100, 95, 1) / 100
alt_occupancy = st.sidebar.slider("Alternate Occupancy Rate (%)", 50, 100, 90, 1) / 100
competitor_adr = st.sidebar.slider("Competitor ADR (INR)", int(data['competitor_adr'].min()), int(data['competitor_adr'].max()), 14459, 100)
is_peak_season = st.sidebar.checkbox("Peak Season", value=True)

# Prepare input data for base and alternate scenarios
base_input = pd.DataFrame({
    'city': [city], 'hotel_type': [hotel_type], 'room_type': ['Suite'],
    'customer_segment': ['Leisure'], 'length_of_stay': [7], 'adults': [2],
    'children': [2], 'lead_time_days': [19], 'occupancy_rate': [base_occupancy],
    'base_adr': [8500], 'competitor_adr': [competitor_adr],
    'is_peak_season': [1 if is_peak_season else 0], 'is_special_event': [0],
    'is_weekend_checkin': [0]
})
alt_input = base_input.copy()
alt_input['occupancy_rate'] = alt_occupancy

# Predict prices
try:
    base_processed = preprocessor.transform(base_input)
    base_price = model.predict(base_processed)[0]
    alt_processed = preprocessor.transform(alt_input)
    alt_price = model.predict(alt_processed)[0]
except Exception as e:
    st.error(f"Prediction error: {str(e)}")
    st.stop()

# Display predictions
st.header("Price Scenarios")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Base Scenario ADR", value=f"{base_price:.2f} INR", delta=f"{base_price - alt_price:.2f} vs Alternate")
with col2:
    st.metric(label="Alternate Scenario ADR", value=f"{alt_price:.2f} INR", delta=f"{alt_price - base_price:.2f} vs Base")
if st.button("Refresh Scenarios"):
    st.rerun()

# Trends section with line chart
st.header("Real-Time Pricing Trends")
occupancy_range = np.linspace(0.5, 1.0, 50)
trend_data = pd.DataFrame({
    'occupancy_rate': occupancy_range,
    'dynamic_adr': [model.predict(preprocessor.transform(base_input.assign(occupancy_rate=rate)))[0] for rate in occupancy_range]
})
fig = px.line(trend_data, x='occupancy_rate', y='dynamic_adr',
              title="ADR vs Occupancy Rate",
              labels={'occupancy_rate': 'Occupancy Rate', 'dynamic_adr': 'Predicted ADR (INR)'},
              color_discrete_sequence=['#007bff'])
fig.update_traces(mode='lines+markers')
st.plotly_chart(fig)

# Summary and export
st.header("Analysis Summary")
summary = pd.DataFrame({
    'Scenario': ['Base', 'Alternate'],
    'Occupancy Rate': [base_occupancy, alt_occupancy],
    'Competitor ADR': [competitor_adr, competitor_adr],
    'Predicted ADR': [base_price, alt_price]
})
st.table(summary.style.format({'Occupancy Rate': '{:.2f}', 'Competitor ADR': '{:.0f}', 'Predicted ADR': '{:.2f}'}))

# Export report
report = summary.to_string(index=False)
report_buffer = io.StringIO()
report_buffer.write(f"Dynamic Pricing Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
report_buffer.write("-" * 50 + "\n")
report_buffer.write(report)
st.download_button(label="Download Report", data=report_buffer.getvalue(), file_name=f"pricing_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")

# Footer
st.markdown("---")
st.caption("Developed by Shrey Chandel | 2025")
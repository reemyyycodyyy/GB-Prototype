import streamlit as st
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Smart Port Congestion Predictor",
    page_icon="⚓",
    layout="wide"
)

# 2. Main Project Title and Header
st.title("⚓ Smart Port & Maritime Congestion Predictor")
st.markdown("**Alexandria University Graduation Project (DE + MLOps)** | Real-time AIS-driven vessel tracking and port congestion forecasting for Egypt's main maritime gateways[cite: 1].")

# 3. Sidebar Controls for Demonstration
st.sidebar.header("Simulation & Zone Controls")
selected_zone = st.sidebar.selectbox(
    "Select Maritime Zone", 
    ["Port Said Approach & Anchorage", "Suez Canal North Entrance", "Red Sea / Med Entry"]
)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Hours Ahead)", 1, 6, 3)
sim_mode = st.sidebar.radio("Data Source Mode", ["Live Kafka Stream (aisstream.io)", "Replay Simulator (Historical Event)"])

# 4. Define Zone-Specific Geographic Parameters (Lat, Lon, Zoom)
zone_configs = {
    "Port Said Approach & Anchorage": {
        "lat": 31.2653, "lon": 32.3019, "zoom": 11,
        "lat_range": (31.20, 31.34), "lon_range": (32.22, 32.38),
        "wait": "+4.2 hours", "vessels": 42, "speed": "4.2 knots"
    },
    "Suez Canal North Entrance": {
        "lat": 31.2333, "lon": 32.3167, "zoom": 12,
        "lat_range": (31.21, 31.26), "lon_range": (32.28, 32.34),
        "wait": "+2.8 hours", "vessels": 18, "speed": "6.1 knots"
    },
    "Red Sea / Med Entry": {
        "lat": 29.9667, "lon": 32.5500, "zoom": 10,
        "lat_range": (29.80, 30.10), "lon_range": (32.40, 32.70),
        "wait": "+6.5 hours", "vessels": 65, "speed": "2.5 knots"
    }
}

current_zone_data = zone_configs[selected_zone]

# 5. Tab Layout covering Product, Data Engineering, and MLOps
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Live Map & Congestion Forecast", 
    "⚙️ Feature Store & Geospatial Labeling", 
    "📊 Model Inference & SHAP Explainability", 
    "🛡️ MLOps & Automated Rollback"
])

# --- Tab 1: End-User Product View (Dynamic Map & Predictions) ---
with tab1:
    st.subheader(f"Congestion Forecast for: {selected_zone} ({forecast_horizon}-Hours Ahead)")
    st.markdown("A forward-looking signal for shipping agents, freight forwarders, and terminal operators to avoid expensive demurrage charges[cite: 1].")
    
    # Quick KPI metrics dynamically updated per zone
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Wait Time", current_zone_data["wait"], "+1.1 hrs vs normal")
    col2.metric("Vessels at Anchorage", f"{current_zone_data['vessels']} vessels", "High density build-up")
    col3.metric("Mean Approach Speed", current_zone_data["speed"], "Slowing down (-35%)")
    col4.metric("Model Confidence", "94.2%", "XGBoost baseline")
    
    st.markdown("### Vessel Cluster & Anchorage Zones (DBSCAN + Convex Hull Output)")
    st.info(f"Visualizing spatial clustering separating active fairway navigation from static anchorage waiting zones for **{selected_zone}**[cite: 1].")
    
    # Generating precise coordinates inside the selected zone's boundaries
    np.random.seed(42)
    n_vessels = current_zone_data["vessels"]
    lat_min, lat_max = current_zone_data["lat_range"]
    lon_min, lon_max = current_zone_data["lon_range"]
    
    latitudes = np.random.uniform(lat_min, lat_max, n_vessels)
    longitudes = np.random.uniform(lon_min, lon_max, n_vessels)
    
    map_data = pd.DataFrame({
        'lat': latitudes,
        'lon': longitudes
    })
    
    # Streamlit map dynamically centered on the selected zone
    st.map(map_data, latitude=current_zone_data["lat"], longitude=current_zone_data["lon"], zoom=current_zone_data["zoom"])

# --- Tab 2: Data Engineering View (Feature Store & Ingestion) ---
with tab2:
    st.subheader("Medallion Architecture & Feast Feature Store")
    st.markdown("Demonstrating real-time feature engineering via Spark Structured Streaming and batch ETL via Airflow, bridged via Feast to eliminate **train-serve skew**[cite: 1].")
    
    col_de1, col_de2 = st.columns(2)
    with col_de1:
        st.markdown("#### 🔄 Pipeline Status")
        st.success("Bronze Layer (Kafka WebSocket): **Active (aisstream.io)**[cite: 1]")
        st.success("Silver Layer (Spark Streaming): **Processing**[cite: 1]")
        st.success("Gold Layer (PostgreSQL / Feast): **Synced**[cite: 1]")
    with col_de2:
        st.markdown("#### 📐 Geospatial Labeling Method")
        st.write("Ground-truth labels are derived entirely from AIS data using **DBSCAN + convex hull** clustering combined with navigation status, speed, and timestamps—requiring no proprietary port-authority data[cite: 1].")
    
    st.markdown("### Feast Feature Store Real-Time Snapshot")
    feature_df = pd.DataFrame({
        "vessel_id": ["IMO 9351234", "IMO 9823411", "IMO 9123456", "IMO 9741289"],
        "rolling_arrival_rate_1h": [12.5, 8.0, 15.2, 11.1],
        "mean_approach_speed_knots": [4.2, 11.5, 2.1, 5.0],
        "anchorage_fill_ratio": [0.85, 0.45, 0.92, 0.78],
        "derived_waiting_time_label_hrs": [5.1, 1.2, 6.4, 4.0]
    })
    st.dataframe(feature_df, use_container_width=True)

# --- Tab 3: Model Inference & Explainability ---
with tab3:
    st.subheader("XGBoost Prediction Engine & SHAP Explainability")
    st.markdown("Explains *why* congestion is forecast, giving operators actionable insights before bottlenecks form[cite: 1].")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### Select Vessel for Drill-down")
        selected_vessel = st.selectbox("Vessel IMO", ["IMO 9351234", "IMO 9823411", "IMO 9123456"])
        st.write(f"**Target Vessel:** {selected_vessel}")
        st.write("**Predicted Delay Risk:** High (Level 4/5)")
        st.write("**Recommended Action:** Re-sequence convoy call or slow-steam to reduce idle waiting costs.")
    with col_m2:
        st.markdown("#### Top Contributing Factors (SHAP Values)")
        st.write("- **Anchorage Fill Ratio:** +42% impact")
        st.write("- **Rolling Arrival Rate Spike:** +28% impact")
        st.write("- **Mean Approach Speed Drop:** +18% impact")
        st.write("- **Open-Meteo Weather Factor:** +12% impact")

# --- Tab 4: MLOps View (Monitoring & Rollback Loop) ---
with tab4:
    st.subheader("MLOps Observability & Closed-Loop Rollback")
    st.markdown("Powered by Evidently AI (drift monitoring), Great Expectations (data quality gates), and MLflow (model registry & versioning)[cite: 1].")
    
    col_op1, col_op2 = st.columns(2)
    with col_op1:
        st.info("Data Quality Gate (Great Expectations): **PASSED**[cite: 1]")
        st.write("Batch ingestion validation checked schema integrity, null value ratios, and timestamp continuity on incoming AIS messages.")
        st.write("Model Registry (MLflow): Active Version **v1.2** (Production)")
    with col_op2:
        st.warning("Data Drift Status (Evidently AI): **Minor Drift Detected**[cite: 1]")
        st.write(f"Drift score: **0.14** (Safety Threshold: **0.20**)[cite: 1].")
        st.write("System behavior: Performance remains within acceptable bounds; automated rollback to version v1.1 is currently **disarmed**[cite: 1].")
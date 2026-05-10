import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Page Config ---
st.set_page_config(page_title="Labor Market System Dashboard", layout="wide")

# --- 1. Data Loading ---
@st.cache_data
def load_data():
    # 1. Update the paths to point to the 'data' folder
    openings = pd.read_csv('data/job_openings.csv', index_col='date')
    hires = pd.read_csv('data/hires.csv', index_col='date')
    quits = pd.read_csv('data/quits.csv', index_col='date')

    # 2. Convert index to datetime objects (crucial for time series analysis)
    openings.index = pd.to_datetime(openings.index)
    hires.index = pd.to_datetime(hires.index)
    quits.index = pd.to_datetime(quits.index)

    # 3. Set the frequency to 'MS' (Monthly Start) 
    # This is a requirement for statsmodels (Holt-Winters and ARIMA)
    openings = openings.asfreq('MS')
    hires = hires.asfreq('MS')
    quits = quits.asfreq('MS')

    df = pd.concat([openings, hires, quits], axis=1)
    df.columns = ['Openings', 'Hires', 'Quits']
    df = df.asfreq('MS').ffill()
    return df

df = load_data()

# --- 2. Sidebar Controls ---
st.sidebar.header("Forecast Settings")
target_series = st.sidebar.selectbox("Select Target Series", df.columns)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 24, 12)
model_family = st.sidebar.radio("Select Model Family", 
                                ["Holt-Winters (Smoothing)", "ARIMA (Box-Jenkins)", "XGBoost (ML)"])

# --- 3. Main Dashboard Layout ---
st.title("Applied Time Series: Labor Market Churn")
st.markdown(f"Currently analyzing: **{target_series}** as part of a three-series systemic study.")

col1, col2 = st.columns([3, 1])

# --- 4. Modeling Logic ---
def get_forecast(series, model_type, horizon):
    train = series.iloc[:-12] # Keep last year for validation
    test = series.iloc[-12:]
    
    if model_type == "Holt-Winters (Smoothing)":
        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
        forecast = model.forecast(horizon)
        name = "Holt-Winters"
        
    elif model_type == "ARIMA (Box-Jenkins)":
        model = pm.auto_arima(train, seasonal=True, m=12)
        forecast = model.predict(n_periods=horizon)
        name = f"SARIMA {model.order}"
        
    else: # XGBoost logic simplified for dashboard
        # (Insert your lag creation logic here)
        forecast = np.tile(train.mean(), horizon) # Placeholder
        name = "XGBoost"
        
    return forecast, test, name

forecast, test_actual, model_name = get_forecast(df[target_series], model_family, forecast_horizon)

# --- 5. Visualizations ---
with col1:
    fig = go.Figure()
    # Historical
    fig.add_trace(go.Scatter(x=df.index, y=df[target_series], name="Historical Data"))
    # Forecast
    forecast_index = pd.date_range(df.index[-1], periods=forecast_horizon + 1, freq='MS')[1:]
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, name=f"Forecast ({model_name})", line=dict(dash='dash')))
    
    fig.update_layout(title=f"Forecast for {target_series}", xaxis_title="Date", yaxis_title="Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Model Performance")
    # In a real app, calculate metrics against the 'test' set
    mae = mean_absolute_error(test_actual, forecast[:len(test_actual)])
    st.metric("MAE (Validation)", f"{mae:.3f}")
    st.write("A lower MAE indicates a more reliable forecast for this specific series.")

# --- 6. Diagnostics Panel ---
with st.expander("Inspect Model Diagnostics (Residuals)"):
    diag_col1, diag_col2 = st.columns(2)
    # Calculate residuals
    resids = test_actual - forecast[:len(test_actual)]
    
    with diag_col1:
        st.write("Residuals Over Time")
        st.line_chart(resids)
    with diag_col2:
        st.write("Error Distribution")
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(resids, bins=10)
        st.pyplot(fig_hist)
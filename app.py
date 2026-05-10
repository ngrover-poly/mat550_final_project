import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

# --- 1. Page Setup & Data Loading ---
st.set_page_config(page_title="Labor Market System", layout="wide")

@st.cache_data 
def load_data():
    # Loading from your data folder structure
    o = pd.read_csv('data/job_openings.csv', index_col='date', parse_dates=True)
    h = pd.read_csv('data/hires.csv', index_col='date', parse_dates=True)
    q = pd.read_csv('data/quits.csv', index_col='date', parse_dates=True)
    df = pd.concat([o, h, q], axis=1)
    df.columns = ['Openings', 'Hires', 'Quits']
    return df.asfreq('MS').ffill()

df = load_data()

# --- 2. Sidebar Controls ---
st.sidebar.header("Dashboard Controls")
target = st.sidebar.selectbox("Select Target Series", df.columns)
horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 24, 12)
model_family = st.sidebar.radio("Select Model Family", 
                                ["Holt-Winters (Smoothing)", "ARIMA (Box-Jenkins)", "Machine Learning (XGBoost)"])

# --- 3. Modeling Logic ---
# Keep the last 24 months as the test set to match your notebook
train = df[target].iloc[:-24]
test = df[target].iloc[-24:]

if model_family == "Holt-Winters (Smoothing)":
    # Holt-Winters trains fast enough to do it live
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit(optimized=True)
    forecast = model.forecast(horizon)
    forecast_name = "Holt-Winters Forecast"

elif model_family == "ARIMA (Box-Jenkins)":
    # --- THIS IS FIX 2: Loading the pre-trained model ---
    model_path = f'models/arima_{target}.pkl'
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        forecast = model.predict(n_periods=horizon)
        forecast_name = "SARIMA Forecast"
    else:
        st.error(f"Model file not found at {model_path}. Did you upload the models folder to GitHub?")
        forecast = np.full(horizon, np.nan) # Dummy data so the app doesn't crash
        forecast_name = "Error"

elif model_family == "Machine Learning (XGBoost)":
    model_path = f'models/xgb_{target}.pkl'
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        # XGBoost requires a recursive forecasting loop.
        # We start with the training dataset (everything up to the last 24 months)
        system_data = df.iloc[:-24].copy()
        n_lags = 6 # Must match the notebook
        
        forecast_list = []
        
        # Predict one month at a time
        for step in range(horizon):
            # 1. Get the last 6 months of the entire system (Openings, Hires, Quits)
            last_6_months = system_data.iloc[-n_lags:].values
            
            # 2. Flatten it into a single row to match how XGBoost was trained
            X_input = last_6_months.flatten().reshape(1, -1)
            
            # 3. Make the prediction for the target variable
            pred = model.predict(X_input)[0]
            forecast_list.append(pred)
            
            # 4. Create a new row for the next step's calculations
            # We use the prediction for the target, and carry-forward the last known value for the other two
            new_row = system_data.iloc[-1].copy()
            new_row[target] = pred
            
            # Append the new row using pd.concat
            system_data = pd.concat([system_data, new_row.to_frame().T], ignore_index=True)
            
        forecast = np.array(forecast_list)
        forecast_name = "XGBoost System Forecast"
        
    else:
        st.error(f"Model file not found at {model_path}. Did you upload it to GitHub?")
        forecast = np.full(horizon, np.nan)
        forecast_name = "Error"

# --- 4. Main Display ---
st.title("US Labor Market Forecast System")
st.markdown(f"Currently analyzing: **{target}**")

fig, ax = plt.subplots(figsize=(12, 5))
# Plot the historical data (just the last 5 years for better visibility)
ax.plot(df.index[-80:], df[target].iloc[-80:], label="Historical Data", color="black")

# Plot the forecast
if not np.isnan(forecast).all():
    forecast_index = pd.date_range(train.index[-1], periods=horizon+1, freq='MS')[1:]
    ax.plot(forecast_index, forecast, label=forecast_name, linestyle="--", color="red", linewidth=2)

ax.set_ylabel("Rate")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# --- 5. Diagnostics & Metrics Panel ---
st.markdown("---")
col1, col2 = st.columns([1, 2])

# Calculate how much of the forecast overlaps with our known test set
overlap = min(horizon, len(test))

with col1:
    st.subheader("Model Accuracy")
    if overlap > 0 and not np.isnan(forecast).all():
        mae = mean_absolute_error(test.iloc[:overlap], forecast[:overlap])
        st.metric(label=f"MAE (First {overlap} months)", value=f"{mae:.4f}")
        st.write("*Lower MAE indicates better predictive accuracy on the holdout set.*")
    else:
        st.write("Not enough overlap to calculate test metrics.")

with col2:
    st.subheader("Residual Diagnostics")
    if overlap > 0 and not np.isnan(forecast).all():
        residuals = test.iloc[:overlap].values - forecast[:overlap]
        
        fig_res, ax_res = plt.subplots(1, 2, figsize=(10, 3))
        
        # Plot 1: Residuals over time
        ax_res[0].plot(residuals, marker='o')
        ax_res[0].axhline(0, color='black', linestyle='--')
        ax_res[0].set_title("Errors Over Time")
        
        # Plot 2: Histogram
        ax_res[1].hist(residuals, bins=10, edgecolor='black')
        ax_res[1].set_title("Error Distribution")
        
        plt.tight_layout()
        st.pyplot(fig_res)
    else:
        st.write("Run a forecast within the test horizon to see diagnostics.")
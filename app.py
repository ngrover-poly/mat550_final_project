import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import joblib
import os

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox

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

st.sidebar.info(
    "**About this System:**\n\n"
    "This dashboard forecasts monthly US Labor Market dynamics (Job openings, Hires, and Quits) "
    "using data from the Bureau of Labor Statistics JOLTS report. Select parameters below "
    "to explore future trends and evaluate statistical diagnostics."
)

target = st.sidebar.selectbox("Select Target Series", df.columns)
horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 24, 12)
model_family = st.sidebar.radio("Select Model Family", 
                                ["Holt-Winters (Smoothing)", "ARIMA (Box-Jenkins)", "Machine Learning (XGBoost)"])

# Keep the last 24 months as the test set
train = df[target].iloc[:-24]
test = df[target].iloc[-24:]

# Predict the 24 historical test months PLUS the actual future horizon
total_steps = 24 + horizon 

if model_family == "Holt-Winters (Smoothing)":
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit(optimized=True)
    full_forecast = model.forecast(total_steps)
    
    # HW Approximation for Prediction Intervals (using residual standard error)
    se = np.std(train - model.fittedvalues)
    margin = 1.96 * se * np.sqrt(np.arange(1, total_steps + 1))
    lower_bound = full_forecast - margin
    upper_bound = full_forecast + margin
    forecast_name = "Holt-Winters Forecast"

elif model_family == "ARIMA (Box-Jenkins)":
    model_path = f'models/arima_{target}.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # ARIMA natively provides confidence intervals!
        full_forecast, conf_int = model.predict(n_periods=total_steps, return_conf_int=True)
        lower_bound = conf_int[:, 0]
        upper_bound = conf_int[:, 1]
        forecast_name = "SARIMA Forecast"
    else:
        st.error(f"Model file not found at {model_path}.")
        full_forecast = np.full(total_steps, np.nan)

elif model_family == "Machine Learning (XGBoost)":
    model_path = f'models/xgb_{target}.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        system_data = df.iloc[:-24].copy()
        n_lags = 6 
        forecast_list = []
        
        # Recursive loop for 24 test months + future horizon
        for step in range(total_steps):
            last_6_months = system_data.iloc[-n_lags:].values
            X_input = last_6_months.flatten().reshape(1, -1)
            pred = model.predict(X_input)[0]
            forecast_list.append(pred)
            
            new_row = system_data.iloc[-1].copy()
            new_row[target] = pred
            system_data = pd.concat([system_data, new_row.to_frame().T], ignore_index=True)
            
        full_forecast = np.array(forecast_list)
        
        # XGBoost Approximation for Intervals
        se = np.std(train) * 0.15 # Using 15% of variance as proxy for ML error
        margin = 1.96 * se * np.sqrt(np.arange(1, total_steps + 1) / 6)
        lower_bound = full_forecast - margin
        upper_bound = full_forecast + margin
        forecast_name = "XGBoost System Forecast"
    else:
        st.error(f"XGBoost model file not found at {model_path}.")
        full_forecast = np.full(total_steps, np.nan)

# Split the forecast into "Test Coverage" and "Actual Future"
test_forecast = full_forecast[:24]
future_forecast = full_forecast[24:]
future_lower = lower_bound[24:]
future_upper = upper_bound[24:]


# --- 4. Main Display ---
st.title("US Labor Market Forecast System")
st.markdown(f"Currently analyzing: **{target}**")

fig, ax = plt.subplots(figsize=(12, 5))

# Plot 1: The Historical Data
ax.plot(df.index[-80:], df[target].iloc[-80:], label="Historical Data", color="black", linewidth=2)

if not np.isnan(full_forecast).all():
    # Plot 2: The Holdout Set Prediction (Shows how well the model learned)
    ax.plot(test.index, test_forecast, label="Holdout Set Prediction", color="orange", alpha=0.8)
    
    # Plot 3: The Actual Future Forecast
    future_index = pd.date_range(df.index[-1], periods=horizon+1, freq='MS')[1:]
    ax.plot(future_index, future_forecast, label=f"Future {forecast_name}", linestyle="--", color="red", linewidth=2)
    
    # Plot 4: The Prediction Intervals (Confidence Bounds)
    ax.fill_between(future_index, future_lower, future_upper, color="red", alpha=0.15, label="95% Prediction Interval")

ax.set_ylabel("Rate (%)")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
st.pyplot(fig)

# --- 5. Diagnostics & Metrics Panel ---
st.markdown("---")
st.subheader("Model Diagnostics & Validation")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Error Metrics (Holdout Set)")
    
    if len(test) == len(test_forecast) and not np.isnan(test_forecast).all():
        # Calculate Metrics
        mae = mean_absolute_error(test, test_forecast)
        rmse = np.sqrt(mean_squared_error(test, test_forecast))
        
        # Requirement: Summary table comparing accuracy
        metrics_df = pd.DataFrame({
            "Metric": ["Mean Absolute Error (MAE)", "Root Mean Sq. Error (RMSE)"],
            "Value": [f"{mae:.4f}", f"{rmse:.4f}"]
        })
        st.table(metrics_df)
        
        # Calculate Residuals for the tests
        # We flatten test.values in case of dimensionality issues
        residuals = test.values.flatten() - test_forecast
        
        # Requirement: Ljung-Box Formal Test from Notebook
        st.markdown("### Residual White Noise Test")
        lb_test = acorr_ljungbox(residuals, lags=[10])
        p_value = lb_test.lb_pvalue.values[0]
        
        if p_value > 0.05:
            st.success(f"**Pass (p={p_value:.3f}):** Residuals are White Noise.")
            st.write("*The model has captured all available signals. Forecasts are statistically sound.*")
        else:
            st.error(f"**Fail (p={p_value:.3f}):** Residuals have remaining structure.")
            st.write("*The model missed some patterns (e.g., missed seasonality).*")
            
    else:
        st.write("Run a valid forecast to see metrics.")

with col2:
    st.markdown("### Diagnostic Plots")
    if len(test) == len(test_forecast) and not np.isnan(test_forecast).all():
        # Requirement: The 4-plot diagnostic panel from your notebook
        fig_diag = plt.figure(figsize=(10, 8))
        gs = fig_diag.add_gridspec(2, 2)

        # 1. Residuals Over Time
        ax1 = fig_diag.add_subplot(gs[0, 0])
        ax1.plot(test.index, residuals, marker='o')
        ax1.axhline(0, color='red', linestyle='--')
        ax1.set_title('Residuals Over Time')

        # 2. Histogram / Normality
        ax2 = fig_diag.add_subplot(gs[0, 1])
        sns.histplot(residuals, kde=True, ax=ax2, edgecolor='black')
        ax2.set_title('Distribution of Errors')

        # 3. Autocorrelation (ACF)
        ax3 = fig_diag.add_subplot(gs[1, 0])
        # Using lags=10 since our holdout set is only 24 months long
        sm.graphics.tsa.plot_acf(residuals, lags=10, ax=ax3)
        ax3.set_title('Residual Autocorrelation (ACF)')

        # 4. Q-Q Plot
        ax4 = fig_diag.add_subplot(gs[1, 1])
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Normal Q-Q Plot')

        plt.tight_layout()
        st.pyplot(fig_diag)

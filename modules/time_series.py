import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def run():
    st.header("📈 Time Series Analysis")
    
    if st.session_state.data is not None:
        df = st.session_state.data.copy()
        
        # Check if any datetime columns exist
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        if not datetime_cols:
            st.warning("No datetime columns detected in your dataset! Please go to **Data Types** and convert a column to 'DateTime' first.")
            return
            
        st.markdown("Automate chronological analysis. Select your dataset's time index and a numeric variable to uncover long-term trends, rolling averages, and ARIMA forecasts.")
        
        col1, col2 = st.columns(2)
        with col1:
            time_col = st.selectbox("Select Time / Date Column:", datetime_cols)
        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_col = st.selectbox("Select Numeric Variable to Analyze:", numeric_cols)
            
        if not value_col:
            st.error("No numeric columns available.")
            return

        # Prepare TS dataframe
        ts_df = df.copy()
        ts_df = ts_df.sort_values(by=time_col)
        # Drop rows where time or value is missing
        ts_df = ts_df.dropna(subset=[time_col, value_col])
        
        freq = st.selectbox("Resampling Frequency", ["None (Daily/Raw)", "Weekly (Mean)", "Monthly (Mean)", "Yearly (Mean)"])
        
        if freq != "None (Daily/Raw)":
            ts_df = ts_df.set_index(time_col)
            if "Weekly" in freq:
                ts_df = ts_df.resample('W').mean().reset_index()
            elif "Monthly" in freq:
                ts_df = ts_df.resample('M').mean().reset_index()
            elif "Yearly" in freq:
                ts_df = ts_df.resample('Y').mean().reset_index()
        
        tab1, tab2 = st.tabs(["Trends & Rolling Averages", "ARIMA Forecasting"])
        
        with tab1:
            st.subheader("Time Series Components")
            
            window = st.slider("Select Rolling Average Window (e.g., 7-day moving average)", min_value=1, max_value=60, value=7)
            
            ts_df['Rolling Mean'] = ts_df[value_col].rolling(window=window).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_df[time_col], y=ts_df[value_col], mode='lines', name='Original Value', line=dict(color='gray', width=1)))
            fig.add_trace(go.Scatter(x=ts_df[time_col], y=ts_df['Rolling Mean'], mode='lines', name=f'{window}-Period Rolling Avg', line=dict(color='#00d4aa', width=3)))
            
            fig.update_layout(title="Trend Analysis", xaxis_title="Date", yaxis_title=value_col, template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("ARIMA Forecasting")
            st.markdown("Predict future values based on historical trends using the Auto-Regressive Integrated Moving Average (ARIMA) model.")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                p = st.number_input("AR (p) Lags", min_value=0, max_value=5, value=1)
            with c2:
                d = st.number_input("Integration (d)", min_value=0, max_value=2, value=1)
            with c3:
                q = st.number_input("MA (q) Lags", min_value=0, max_value=5, value=1)
                
            steps = st.number_input("Forecast Steps (Periods ahead)", min_value=1, max_value=100, value=10)
            
            if st.button("Run ARIMA Forecast"):
                with st.spinner("Training ARIMA model (this might take a moment...)"):
                    try:
                        # Prepare Data for Statsmodels
                        train_data = ts_df[value_col].values
                        model = ARIMA(train_data, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        forecast = fitted_model.forecast(steps=steps)
                        
                        # Generate future dates
                        last_date = ts_df[time_col].iloc[-1]
                        
                        if freq == "None (Daily/Raw)":
                            future_dates = pd.date_range(start=last_date, periods=steps+1, freq='D')[1:]
                        elif "Weekly" in freq:
                            future_dates = pd.date_range(start=last_date, periods=steps+1, freq='W')[1:]
                        elif "Monthly" in freq:
                            future_dates = pd.date_range(start=last_date, periods=steps+1, freq='M')[1:]
                        else:
                            future_dates = pd.date_range(start=last_date, periods=steps+1, freq='Y')[1:]
                            
                        # Plot
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=ts_df[time_col], y=ts_df[value_col], mode='lines', name='Historical Data', line=dict(color='#00d4aa')))
                        fig2.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='ARIMA Forecast', line=dict(color='#ff5733', dash='dash', width=3)))
                        fig2.update_layout(title=f"ARIMA({p},{d},{q}) Forecast for {steps} periods", xaxis_title="Date", yaxis_title=value_col, template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly")
                        
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Show summary
                        st.subheader("Model Summary")
                        st.text(fitted_model.summary().as_text())
                        
                    except Exception as e:
                        st.error(f"ARIMA Modeling Error: {e}")
                        st.info("Try adjusting your p, d, q parameters or resampling the data if it isn't strictly stationary.")

    else:
        st.warning("Please upload data first in the Data Import module.")

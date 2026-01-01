# =========================================
# 📊 SALES ANALYTICS DASHBOARD + ML FORECAST
# =========================================

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# ---- Add src folder to path ----
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from data_processing import SalesDataProcessor

import streamlit as st
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📈 Sales Analytics Dashboard")

# =========================================
# SIDEBAR – FILE UPLOAD
# =========================================
st.sidebar.header("📁 Upload Sales Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV / Excel",
    type=["csv", "xlsx", "xls"]
)

processor = SalesDataProcessor()

# =========================================
# MAIN APP
# =========================================
if uploaded_file:

    # ---- Save uploaded file temporarily ----
    ext = uploaded_file.name.split(".")[-1]
    temp_path = f"temp_upload.{ext}"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔄 Processing data..."):
        df = processor.load_data(temp_path)
        df = processor.clean_data()

    os.remove(temp_path)

    if df is None or df.empty:
        st.error("❌ No usable data found.")
        st.stop()

    # =========================================
    # DATE FILTER
    # =========================================
    if "Date" in df.columns:
        st.sidebar.header("📅 Date Filter")
        start_date = st.sidebar.date_input(
            "Start Date", df["Date"].min().date()
        )
        end_date = st.sidebar.date_input(
            "End Date", df["Date"].max().date()
        )

        df = df[
            (df["Date"] >= pd.to_datetime(start_date)) &
            (df["Date"] <= pd.to_datetime(end_date))
        ]

    if df.empty:
        st.warning("⚠️ No data for selected date range.")
        st.stop()

    # =========================================
    # 🔑 KPIs (FIXED PROFIT MARGIN)
    # =========================================
    total_sales = df["Sales"].sum() if "Sales" in df.columns else 0
    total_profit = df["Profit"].sum() if "Profit" in df.columns else 0

    # ✅ CORRECT PROFIT MARGIN CALCULATION
    avg_profit_margin = (
        (total_profit / total_sales) * 100
        if total_sales > 0 else 0
    )

    st.subheader("📊 Key Metrics")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sales", f"${total_sales:,.0f}")
    c2.metric("Total Profit", f"${total_profit:,.0f}")
    c3.metric("Avg Profit Margin", f"{avg_profit_margin:.2f}%")
    c4.metric("Total Orders", len(df))

    st.markdown("---")

    # =========================================
    # MONTHLY SALES AGGREGATION
    # =========================================
    monthly = (
        df.groupby(df["Date"].dt.to_period("M"))["Sales"]
        .sum()
        .reset_index()
    )
    monthly["Date"] = monthly["Date"].dt.to_timestamp()

    st.subheader("📈 Monthly Sales Trend")
    st.plotly_chart(
        px.line(monthly, x="Date", y="Sales", markers=True),
        use_container_width=True
    )

    # =========================================
    # DATA SUFFICIENCY CHECK
    # =========================================
    st.markdown("---")
    st.subheader("🔍 Forecast Readiness")

    if len(monthly) < 6:
        st.error(
            f"❌ Only {len(monthly)} months available. "
            "At least **6 months** required."
        )
        st.stop()
    else:
        st.success(f"✅ {len(monthly)} months of data available.")

    # =========================================
    # TIME-SERIES REGRESSION (LAG FEATURES)
    # =========================================
    ts = monthly.copy()
    ts["lag_1"] = ts["Sales"].shift(1)
    ts["lag_2"] = ts["Sales"].shift(2)
    ts["lag_3"] = ts["Sales"].shift(3)
    ts.dropna(inplace=True)

    X = ts[["lag_1", "lag_2", "lag_3"]]
    y = ts["Sales"]

    # ---- Time-aware train/test split ----
    split_idx = int(len(ts) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # =========================================
    # FORECAST ACCURACY
    # =========================================
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = mean_squared_error(y_test, y_test_pred, squared=False)

    st.markdown("---")
    st.subheader("📐 Forecast Accuracy (Test Set)")

    m1, m2 = st.columns(2)
    m1.metric("MAE", f"{mae:,.2f}")
    m2.metric("RMSE", f"{rmse:,.2f}")

    # =========================================
    # ACTUAL vs PREDICTED
    # =========================================
    ts["Predicted"] = np.nan
    ts.iloc[:split_idx, ts.columns.get_loc("Predicted")] = y_train_pred
    ts.iloc[split_idx:, ts.columns.get_loc("Predicted")] = y_test_pred

    st.plotly_chart(
        px.line(
            ts,
            x="Date",
            y=["Sales", "Predicted"],
            title="Actual vs Predicted Sales"
        ),
        use_container_width=True
    )

    # =========================================
    # CONFIDENCE INTERVALS (95%)
    # =========================================
    residuals = y_train - y_train_pred
    std_error = residuals.std()
    z = norm.ppf(0.975)

    ts["Upper"] = ts["Predicted"] + z * std_error
    ts["Lower"] = ts["Predicted"] - z * std_error

    st.plotly_chart(
        px.line(
            ts,
            x="Date",
            y=["Sales", "Predicted", "Upper", "Lower"],
            title="Predictions with 95% Confidence Interval"
        ),
        use_container_width=True
    )

    # =========================================
    # FUTURE FORECAST
    # =========================================
    st.markdown("### 📅 Future Sales Forecast")

    months_ahead = st.slider("Months to predict", 1, 12, 3)

    last_known = ts.iloc[-1][["lag_1", "lag_2", "lag_3"]].values.tolist()
    future_preds = []

    for _ in range(months_ahead):
        pred = model.predict([last_known])[0]
        future_preds.append(pred)
        last_known = [pred] + last_known[:2]

    future_dates = pd.date_range(
        start=monthly["Date"].max(),
        periods=months_ahead + 1,
        freq="M"
    )[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Sales": future_preds
    })

    st.plotly_chart(
        px.line(
            forecast_df,
            x="Date",
            y="Predicted_Sales",
            markers=True,
            title="Future Sales Forecast"
        ),
        use_container_width=True
    )

    # =========================================
    # DATA PREVIEW
    # =========================================
    st.markdown("---")
    st.subheader("📋 Data Preview")
    st.dataframe(df, use_container_width=True)

else:
    st.info("👈 Upload a sales dataset to begin.")

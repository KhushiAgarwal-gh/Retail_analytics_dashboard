import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Retail Analytics Dashboard",
    layout="wide"
)

st.title("📊 Retail Analytics Dashboard")

# df = pd.read_csv("retail_cleaned_data.csv.zip", compression="zip")
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "retail_cleaned_data.csv.zip",
    compression="zip"
)
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

df["TransactionType"] = df["Quantity"].apply(
    lambda x: "Return" if x < 0 else "Sale"
)

st.write("Dataset Preview")

st.dataframe(df.head())

# KPI Metrics

total_revenue = df["Revenue"].sum()
total_orders = df["InvoiceNo"].nunique()
total_customers = df["CustomerID"].nunique()

return_rate = (
    (df["TransactionType"] == "Return").sum()
    / len(df)
) * 100


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Revenue", f"£{total_revenue:,.0f}")

with col2:
    st.metric("Total Orders", total_orders)

with col3:
    st.metric("Total Customers", total_customers)

with col4:
    st.metric("Return Rate", f"{return_rate:.2f}%")

# Monthly Revenue Trend

df["Month"] = pd.to_datetime(df["InvoiceDate"]).dt.to_period("M").astype(str)

monthly_revenue = (
    df.groupby("Month")["Revenue"]
    .sum()
    .reset_index()
)

st.subheader("📈 Monthly Revenue Trend")

st.line_chart(
    monthly_revenue,
    x="Month",
    y="Revenue"
)

# Return Analysis

return_data = (
    df.groupby("TransactionType")["Revenue"]
    .sum()
    .reset_index()
)

st.subheader("↩️ Sales vs Return Revenue")

st.bar_chart(
    return_data,
    x="TransactionType",
    y="Revenue"
)

# Customer Segmentation

rfm = pd.read_csv("customer_segments.csv")

st.subheader("👥 Customer Segmentation")

segment_count = (
    rfm["Segment"]
    .value_counts()
    .reset_index()
)

segment_count.columns = ["Segment", "Customers"]

st.bar_chart(
    segment_count,
    x="Segment",
    y="Customers"
)
# Segment Filter

selected_segment = st.selectbox(
    "Select Customer Segment",
    rfm["Segment"].unique()
)

filtered_customers = rfm[
    rfm["Segment"] == selected_segment
]

st.subheader(f"{selected_segment} Details")

st.dataframe(
    filtered_customers.head(20)
)

# Country Revenue Analysis

country_revenue = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

st.subheader("🌍 Top 10 Countries by Revenue")

st.bar_chart(
    country_revenue,
    x="Country",
    y="Revenue"
)
# Product Revenue Analysis

product_revenue = (
    df.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

st.subheader("🛒 Top 10 Products by Revenue")

st.dataframe(product_revenue)

# Download Customer Segment Report

csv = filtered_customers.to_csv(index=False)

st.download_button(
    label="📥 Download Customer Report",
    data=csv,
    file_name=f"{selected_segment}_customers.csv",
    mime="text/csv"
)
# Sidebar

st.sidebar.title("Retail Analytics")

st.sidebar.info(
    """
    Dashboard Features:
    ✅ Revenue Analysis
    ✅ Return Analysis
    ✅ Customer Segmentation
    ✅ Product Insights
    """
)

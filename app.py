import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Retail Revenue Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f9fc;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    h1 {
        color: #111827;
        font-weight: 700;
    }

    h2, h3 {
        color: #1f2937;
    }

    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    div[data-testid="metric-container"] label {
        color: #6b7280 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data_path = BASE_DIR / "retail_cleaned_data.csv.zip"
    customer_path = BASE_DIR / "customer_segments.csv"

    df = pd.read_csv(
        data_path,
        compression="zip"
    )

    rfm = pd.read_csv(customer_path)

    return df, rfm


df, rfm = load_data()


# ============================================================
# DATA PREPARATION
# ============================================================

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

df["Revenue"] = (
    df["Quantity"] *
    df["UnitPrice"]
)

df["TransactionType"] = np.where(
    df["Quantity"] < 0,
    "Return",
    "Sale"
)


# ============================================================
# DUPLICATE TRANSACTION SIGNAL
# ============================================================

df["Is_Duplicate"] = df.duplicated(
    keep=False
)


# ============================================================
# FAST ANOMALY DETECTION
# ============================================================

quantity_threshold = df["Quantity"].abs().quantile(0.99)

price_threshold = df["UnitPrice"].quantile(0.99)

revenue_threshold = df["Revenue"].abs().quantile(0.99)


df["Is_Anomaly"] = (
    (df["Quantity"].abs() >= quantity_threshold)
    |
    (df["UnitPrice"] >= price_threshold)
    |
    (df["Revenue"].abs() >= revenue_threshold)
)


# ============================================================
# RISK SCORE
# ============================================================

df["Risk_Score"] = (
    df["Is_Duplicate"].astype(int) * 40
    +
    df["Is_Anomaly"].astype(int) * 40
    +
    (df["TransactionType"] == "Return").astype(int) * 20
)


# ============================================================
# RISK LEVEL
# ============================================================

df["Risk_Level"] = pd.cut(
    df["Risk_Score"],
    bins=[-1, 39, 69, 100],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)


# ============================================================
# POTENTIAL LEAKAGE
# ============================================================

df["Potential_Leakage"] = np.where(
    (
        (df["Risk_Level"] == "High Risk")
        &
        (df["Revenue"] > 0)
    ),
    df["Revenue"],
    0
)


# ============================================================
# MONTH
# ============================================================

df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Retail Intelligence")

st.sidebar.markdown(
    "Business analytics and revenue risk monitoring"
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Customer Intelligence",
        "Product Analysis",
        "Country Analysis",
        "Anomaly Monitoring",
        "Risk Monitoring"
    ]
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")


countries = sorted(
    df["Country"]
    .dropna()
    .unique()
)

selected_country = st.sidebar.selectbox(
    "Country",
    ["All Countries"] + countries
)


transaction_types = [
    "All Transactions",
    "Sale",
    "Return"
]

selected_transaction = st.sidebar.selectbox(
    "Transaction Type",
    transaction_types
)


risk_levels = [
    "All Risk Levels",
    "Low Risk",
    "Medium Risk",
    "High Risk"
]

selected_risk = st.sidebar.selectbox(
    "Risk Level",
    risk_levels
)


months = sorted(
    df["Month"].dropna().unique()
)

selected_month = st.sidebar.selectbox(
    "Month",
    ["All Months"] + list(months)
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df


if selected_country != "All Countries":

    filtered_df = filtered_df[
        filtered_df["Country"] == selected_country
    ]


if selected_transaction != "All Transactions":

    filtered_df = filtered_df[
        filtered_df["TransactionType"] == selected_transaction
    ]


if selected_risk != "All Risk Levels":

    filtered_df = filtered_df[
        filtered_df["Risk_Level"].astype(str)
        == selected_risk
    ]


if selected_month != "All Months":

    filtered_df = filtered_df[
        filtered_df["Month"] == selected_month
    ]


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.title("Retail Revenue Intelligence")

st.caption(
    "Executive analytics, customer intelligence and revenue "
    "risk monitoring for retail operations"
)

st.markdown(
    "📊 Business Analytics  |  "
    "⚠️ Risk Intelligence  |  "
    "👥 Customer Insights  |  "
    "🌍 Global Retail"
)

st.markdown("---")


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.header("Executive Overview")

    total_revenue = filtered_df["Revenue"].sum()
    total_orders = filtered_df["InvoiceNo"].nunique()
    total_customers = filtered_df["CustomerID"].nunique()
    total_transactions = len(filtered_df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Revenue",
            f"£{total_revenue:,.2f}"
        )

    with col2:
        st.metric(
            "Total Orders",
            f"{total_orders:,}"
        )

    with col3:
        st.metric(
            "Customers",
            f"{total_customers:,}"
        )

    with col4:
        st.metric(
            "Transactions",
            f"{total_transactions:,}"
        )


    st.subheader("Revenue Performance")

    monthly_revenue = (
        filtered_df
        .groupby("Month")["Revenue"]
        .sum()
        .reset_index()
    )

    if not monthly_revenue.empty:

        st.line_chart(
            monthly_revenue.set_index("Month")
        )

    else:

        st.info(
            "No data available for the selected filters."
        )


    st.subheader("Sales vs Returns")

    transaction_summary = (
        filtered_df
        .groupby("TransactionType")["Revenue"]
        .sum()
        .reset_index()
    )

    if not transaction_summary.empty:

        st.bar_chart(
            transaction_summary.set_index(
                "TransactionType"
            )
        )

    else:

        st.info(
            "No transaction data available."
        )


    st.subheader("Duplicate Transaction Monitoring")

    duplicate_count = int(
        filtered_df["Is_Duplicate"].sum()
    )

    duplicate_revenue = filtered_df.loc[
        filtered_df["Is_Duplicate"],
        "Revenue"
    ].sum()


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Duplicate Transactions",
            f"{duplicate_count:,}"
        )

    with col2:

        st.metric(
            "Duplicate Transaction Revenue",
            f"£{duplicate_revenue:,.2f}"
        )


# ============================================================
# CUSTOMER INTELLIGENCE
# ============================================================

elif page == "Customer Intelligence":

    st.header("Customer Intelligence")

    if rfm.empty:

        st.warning(
            "Customer segmentation data is not available."
        )

    else:

        segment_summary = (
            rfm["Segment"]
            .value_counts()
            .reset_index()
        )

        segment_summary.columns = [
            "Segment",
            "Customers"
        ]


        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Customer Segment Distribution"
            )

            st.bar_chart(
                segment_summary.set_index("Segment")
            )


        with col2:

            monetary_summary = (
                rfm
                .groupby("Segment")["Monetary"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.subheader(
                "Monetary Value by Segment"
            )

            st.bar_chart(
                monetary_summary
            )


        st.subheader("RFM Segment Summary")


        segment_table = (
            rfm
            .groupby("Segment")
            .agg(
                Customers=("CustomerID", "count"),
                Avg_Recency=("Recency", "mean"),
                Avg_Frequency=("Frequency", "mean"),
                Avg_Monetary=("Monetary", "mean")
            )
            .reset_index()
        )


        segment_table["Avg_Recency"] = (
            segment_table["Avg_Recency"].round(2)
        )

        segment_table["Avg_Frequency"] = (
            segment_table["Avg_Frequency"].round(2)
        )

        segment_table["Avg_Monetary"] = (
            segment_table["Avg_Monetary"].round(2)
        )


        st.dataframe(
            segment_table,
            use_container_width=True,
            hide_index=True
        )


        st.subheader("Customer Segmentation Data")

        st.dataframe(
            rfm,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PRODUCT ANALYSIS
# ============================================================

elif page == "Product Analysis":

    st.header("Product Analysis")


    product_revenue = (
        filtered_df
        .groupby("Description")["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )


    product_quantity = (
        filtered_df
        .groupby("Description")["Quantity"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Top Products by Revenue"
        )

        st.bar_chart(
            product_revenue
        )


    with col2:

        st.subheader(
            "Top Products by Quantity"
        )

        st.bar_chart(
            product_quantity
        )


    st.subheader("Product Performance Table")


    product_table = (
        filtered_df
        .groupby("Description")
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("InvoiceNo", "nunique")
        )
        .sort_values(
            "Revenue",
            ascending=False
        )
        .head(50)
        .reset_index()
    )


    product_table["Revenue"] = (
        product_table["Revenue"].round(2)
    )


    st.dataframe(
        product_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# COUNTRY ANALYSIS
# ============================================================

elif page == "Country Analysis":

    st.header("Country Analysis")


    country_revenue = (
        filtered_df
        .groupby("Country")["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(15)
    )


    country_orders = (
        filtered_df
        .groupby("Country")["InvoiceNo"]
        .nunique()
        .sort_values(
            ascending=False
        )
        .head(15)
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Revenue by Country")

        st.bar_chart(
            country_revenue
        )


    with col2:

        st.subheader("Orders by Country")

        st.bar_chart(
            country_orders
        )


    st.subheader("Country Performance Table")


    country_table = (
        filtered_df
        .groupby("Country")
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Transactions=("InvoiceNo", "count"),
            Customers=("CustomerID", "nunique")
        )
        .sort_values(
            "Revenue",
            ascending=False
        )
        .reset_index()
    )


    country_table["Revenue"] = (
        country_table["Revenue"].round(2)
    )


    st.dataframe(
        country_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ANOMALY MONITORING
# ============================================================

elif page == "Anomaly Monitoring":

    st.header("Anomaly Monitoring")


    anomaly_df = filtered_df[
        filtered_df["Is_Anomaly"]
    ]


    anomaly_count = len(anomaly_df)

    anomaly_revenue = anomaly_df[
        "Revenue"
    ].sum()


    duplicate_count = int(
        filtered_df["Is_Duplicate"].sum()
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Anomalous Transactions",
            f"{anomaly_count:,}"
        )


    with col2:

        st.metric(
            "Anomaly Revenue",
            f"£{anomaly_revenue:,.2f}"
        )


    with col3:

        st.metric(
            "Duplicate Transactions",
            f"{duplicate_count:,}"
        )


    st.subheader("Anomalous Transaction Details")


    if anomaly_df.empty:

        st.info(
            "No anomalous transactions found "
            "for the selected filters."
        )

    else:

        anomaly_columns = [
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "UnitPrice",
            "Revenue",
            "Country",
            "InvoiceDate",
            "Is_Duplicate",
            "Risk_Score",
            "Risk_Level"
        ]


        available_columns = [
            col for col in anomaly_columns
            if col in anomaly_df.columns
        ]


        st.dataframe(
            anomaly_df[
                available_columns
            ]
            .sort_values(
                "Risk_Score",
                ascending=False
            )
            .head(500),
            use_container_width=True,
            hide_index=True
        )


        report = (
            anomaly_df[available_columns]
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            "⬇️ Download Anomaly Report",
            data=report,
            file_name="anomaly_report.csv",
            mime="text/csv"
        )


# ============================================================
# RISK MONITORING
# ============================================================

elif page == "Risk Monitoring":

    st.header("Risk Monitoring")


    potential_leakage = (
        filtered_df["Potential_Leakage"].sum()
    )


    total_revenue = (
        filtered_df["Revenue"].sum()
    )


    high_risk_revenue = filtered_df.loc[
        filtered_df["Risk_Level"] == "High Risk",
        "Revenue"
    ].sum()


    recovery_opportunity = filtered_df.loc[
        filtered_df["Risk_Level"] != "Low Risk",
        "Revenue"
    ].clip(lower=0).sum()


    if total_revenue != 0:

        leakage_percentage = (
            potential_leakage /
            abs(total_revenue)
        ) * 100

    else:

        leakage_percentage = 0


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Potential Leakage",
            f"£{potential_leakage:,.2f}"
        )


    with col2:

        st.metric(
            "Potential Leakage %",
            f"{leakage_percentage:.4f}%"
        )


    with col3:

        st.metric(
            "High-Risk Revenue",
            f"£{high_risk_revenue:,.2f}"
        )


    with col4:

        st.metric(
            "Recovery Opportunity",
            f"£{recovery_opportunity:,.2f}"
        )


    st.subheader(
        "Revenue Leakage & Risk Monitoring"
    )


    risk_distribution = (
        filtered_df["Risk_Level"]
        .astype(str)
        .value_counts()
        .reindex(
            [
                "Low Risk",
                "Medium Risk",
                "High Risk"
            ],
            fill_value=0
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Risk Distribution")

        st.bar_chart(
            risk_distribution
        )


    with col2:

        leakage_country = (
            filtered_df
            .groupby("Country")["Potential_Leakage"]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(10)
        )

        st.subheader(
            "Potential Leakage by Country"
        )

        st.bar_chart(
            leakage_country
        )


    st.subheader(
        "Potential Leakage by Product"
    )


    leakage_product = (
        filtered_df
        .groupby("Description")["Potential_Leakage"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )


    if leakage_product.sum() > 0:

        st.bar_chart(
            leakage_product
        )

    else:

        st.info(
            "No positive potential leakage identified "
            "for the selected filters."
        )


    st.subheader(
        "Monthly Potential Leakage Trend"
    )


    monthly_leakage = (
        filtered_df
        .groupby("Month")["Potential_Leakage"]
        .sum()
    )


    if not monthly_leakage.empty:

        st.line_chart(
            monthly_leakage
        )

    else:

        st.info(
            "No monthly leakage data available."
        )


    st.subheader("Risk Scoring Logic")


    st.markdown(
        """
        **Risk Score Framework**

        - Duplicate transaction → **+40 points**
        - Anomalous transaction → **+40 points**
        - Return transaction → **+20 points**

        **Risk Classification**

        - 0–39 → Low Risk
        - 40–69 → Medium Risk
        - 70–100 → High Risk

        High-risk transactions are treated as **investigation
        opportunities**, not confirmed fraud or confirmed
        financial loss.
        """
    )


    st.subheader("High-Risk Transactions")


    high_risk_df = filtered_df[
        filtered_df["Risk_Level"] == "High Risk"
    ].copy()


    if high_risk_df.empty:

        st.info(
            "No high-risk transactions found "
            "for the selected filters."
        )

    else:

        risk_columns = [
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "UnitPrice",
            "Revenue",
            "Country",
            "InvoiceDate",
            "Is_Duplicate",
            "Is_Anomaly",
            "Risk_Score",
            "Risk_Level",
            "Potential_Leakage"
        ]


        available_risk_columns = [
            col for col in risk_columns
            if col in high_risk_df.columns
        ]


        st.dataframe(
            high_risk_df[
                available_risk_columns
            ]
            .sort_values(
                "Risk_Score",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


        risk_report = (
            high_risk_df[available_risk_columns]
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            "⬇️ Download High-Risk Report",
            data=risk_report,
            file_name="high_risk_transactions.csv",
            mime="text/csv"
        )


    st.info(
        "Business Note: Potential leakage represents "
        "transactions that meet the defined risk criteria. "
        "These transactions should be reviewed by the business "
        "team before treating them as actual financial losses."
    )
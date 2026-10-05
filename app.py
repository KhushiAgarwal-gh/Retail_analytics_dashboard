import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Retail Revenue Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Main content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Sidebar radio */
    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {
        padding: 8px 10px;
        border-radius: 6px;
    }

    /* Main title */
    h1 {
        font-weight: 700;
        color: #111827;
    }

    h2, h3 {
        color: #1f2937;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    /* Metric label */
    div[data-testid="stMetricLabel"] {
        color: #6b7280;
    }

    /* Metric value */
    div[data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 700;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Buttons */
    .stDownloadButton button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Divider */
    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Info boxes */
    .business-box {
        background-color: white;
        border-left: 5px solid #2563eb;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }

    .risk-box {
        background-color: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .success-box {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# FILE PATH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(
    BASE_DIR / "retail_cleaned_data.csv.zip",
    compression="zip"
)

rfm = pd.read_csv(
    BASE_DIR / "customer_segments.csv"
)


# ==================================================
# DATA PREPARATION
# ==================================================

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"]
)

df["Revenue"] = (
    df["Quantity"] * df["UnitPrice"]
)

df["TransactionType"] = df["Quantity"].apply(
    lambda x: "Return" if x < 0 else "Sale"
)

df["Is_Duplicate"] = df.duplicated(
    keep=False
)

df["Month"] = (
    df["InvoiceDate"]
    .dt.to_period("M")
    .astype(str)
)


# ==================================================
# MACHINE LEARNING - ANOMALY DETECTION
# ==================================================

anomaly_features = df[
    [
        "Quantity",
        "UnitPrice",
        "Revenue"
    ]
].copy()

anomaly_features = anomaly_features.replace(
    [float("inf"), float("-inf")],
    0
).fillna(0)


isolation_forest = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42
)

df["Anomaly"] = isolation_forest.fit_predict(
    anomaly_features
)

df["Is_Anomaly"] = df["Anomaly"].apply(
    lambda x: True if x == -1 else False
)


# ==================================================
# REVENUE LEAKAGE RISK SCORE
# ==================================================

df["Risk_Score"] = (
    df["Is_Duplicate"].astype(int) * 40
    + df["Is_Anomaly"].astype(int) * 40
    + (df["TransactionType"] == "Return").astype(int) * 20
)


# ==================================================
# RISK LEVEL
# ==================================================

df["Risk_Level"] = pd.cut(
    df["Risk_Score"],
    bins=[-1, 39, 69, 100],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)


# ==================================================
# POTENTIAL REVENUE LEAKAGE
# ==================================================

df["Potential_Leakage"] = df.apply(
    lambda row: row["Revenue"]
    if row["Risk_Level"] == "High Risk"
    and row["Revenue"] > 0
    else 0,
    axis=1
)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.markdown(
    """
    <h2 style="color:white; margin-bottom:0;">
    📊 Retail Intelligence
    </h2>

    <p style="color:#9ca3af;">
    Revenue & Risk Analytics
    </p>
    """,
    unsafe_allow_html=True
)


st.sidebar.divider()


page = st.sidebar.radio(
    "Dashboard",
    [
        "Executive Overview",
        "Customer Intelligence",
        "Product Analysis",
        "Country Analysis",
        "Anomaly Monitoring",
        "Risk Monitoring"
    ]
)


# ==================================================
# FILTERS
# ==================================================

st.sidebar.divider()

st.sidebar.markdown(
    "### 🔎 Filters"
)


country_options = sorted(
    df["Country"].dropna().unique()
)

selected_countries = st.sidebar.multiselect(
    "🌍 Country",
    options=country_options,
    default=country_options
)


transaction_options = sorted(
    df["TransactionType"].unique()
)

selected_transactions = st.sidebar.multiselect(
    "🔄 Transaction Type",
    options=transaction_options,
    default=transaction_options
)


risk_options = [
    "Low Risk",
    "Medium Risk",
    "High Risk"
]

selected_risks = st.sidebar.multiselect(
    "⚠️ Risk Level",
    options=risk_options,
    default=risk_options
)


month_options = sorted(
    df["Month"].unique()
)

selected_months = st.sidebar.multiselect(
    "📅 Month",
    options=month_options,
    default=month_options
)


# ==================================================
# APPLY FILTERS
# ==================================================

filtered_df = df[
    df["Country"].isin(selected_countries)
    & df["TransactionType"].isin(selected_transactions)
    & df["Risk_Level"].isin(selected_risks)
    & df["Month"].isin(selected_months)
].copy()


# ==================================================
# SIDEBAR SUMMARY
# ==================================================

st.sidebar.divider()

st.sidebar.metric(
    "Filtered Transactions",
    f"{len(filtered_df):,}"
)

st.sidebar.metric(
    "Filtered Revenue",
    f"£{filtered_df['Revenue'].sum():,.0f}"
)


st.sidebar.caption(
    "Use the filters to investigate specific markets, periods and risk categories."
)


# ==================================================
# EXECUTIVE OVERVIEW
# ==================================================

if page == "Executive Overview":

    st.title(
        "Retail Revenue Intelligence"
    )

    st.caption(
        "Executive view of revenue performance, customer activity and transaction risk."
    )


    st.markdown(
        """
        <div class="business-box">
        <b>Business Objective</b><br>
        Monitor retail revenue performance and identify transactions that may
        require investigation due to unusual or high-risk patterns.
        </div>
        """,
        unsafe_allow_html=True
    )


    total_revenue = filtered_df[
        "Revenue"
    ].sum()

    total_orders = filtered_df[
        "InvoiceNo"
    ].nunique()

    total_customers = filtered_df[
        "CustomerID"
    ].nunique()

    return_count = (
        filtered_df["TransactionType"]
        == "Return"
    ).sum()

    return_rate = (
        return_count
        / len(filtered_df)
        * 100
        if len(filtered_df) > 0
        else 0
    )

    potential_leakage = filtered_df[
        "Potential_Leakage"
    ].sum()


    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:
        st.metric(
            "Total Revenue",
            f"£{total_revenue:,.0f}"
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
            "Return Rate",
            f"{return_rate:.2f}%"
        )

    with col5:
        st.metric(
            "Potential Leakage",
            f"£{potential_leakage:,.0f}"
        )


    st.divider()


    st.subheader(
        "📈 Revenue Performance"
    )


    monthly_revenue = (
        filtered_df
        .groupby("Month")["Revenue"]
        .sum()
        .reset_index()
    )


    st.line_chart(
        monthly_revenue,
        x="Month",
        y="Revenue"
    )


    st.subheader(
        "↩️ Sales vs Returns"
    )


    return_data = (
        filtered_df
        .groupby("TransactionType")["Revenue"]
        .sum()
        .reset_index()
    )


    st.bar_chart(
        return_data,
        x="TransactionType",
        y="Revenue"
    )


    st.subheader(
        "⚠️ Duplicate Transaction Signals"
    )


    duplicate_data = filtered_df[
        filtered_df["Is_Duplicate"]
    ]


    st.write(
        f"**{len(duplicate_data):,}** filtered rows "
        "are exact duplicate records."
    )


    st.caption(
        "Duplicate records are investigation signals and are not automatically fraudulent."
    )


    st.dataframe(
        duplicate_data[
            [
                "InvoiceNo",
                "StockCode",
                "Description",
                "Quantity",
                "UnitPrice",
                "Revenue",
                "InvoiceDate",
                "CustomerID",
                "Country"
            ]
        ].head(20),
        use_container_width=True
    )


# ==================================================
# CUSTOMER INTELLIGENCE
# ==================================================

elif page == "Customer Intelligence":

    st.title(
        "👥 Customer Intelligence"
    )

    st.caption(
        "Understand customer value and engagement using RFM-based segmentation."
    )


    st.markdown(
        """
        <div class="business-box">
        <b>RFM Analysis</b><br>
        Customers are segmented using Recency, Frequency and Monetary value
        to identify valuable, regular and inactive customer groups.
        </div>
        """,
        unsafe_allow_html=True
    )


    segment_count = (
        rfm["Segment"]
        .value_counts()
        .reset_index()
    )

    segment_count.columns = [
        "Segment",
        "Customers"
    ]


    st.subheader(
        "Customer Segment Distribution"
    )


    st.bar_chart(
        segment_count,
        x="Segment",
        y="Customers"
    )


    st.divider()


    selected_segment = st.selectbox(
        "Select Customer Segment",
        sorted(
            rfm["Segment"].unique()
        )
    )


    filtered_customers = rfm[
        rfm["Segment"]
        == selected_segment
    ]


    st.subheader(
        f"{selected_segment} Customers"
    )


    col1, col2, col3 = st.columns(3)


    with col1:
        st.metric(
            "Customers",
            f"{len(filtered_customers):,}"
        )


    with col2:
        st.metric(
            "Average Frequency",
            f"{filtered_customers['Frequency'].mean():.1f}"
        )


    with col3:
        st.metric(
            "Average Monetary Value",
            f"£{filtered_customers['Monetary'].mean():,.0f}"
        )


    st.dataframe(
        filtered_customers,
        use_container_width=True
    )


    csv = filtered_customers.to_csv(
        index=False
    )


    st.download_button(
        label="📥 Download Customer Report",
        data=csv,
        file_name=f"{selected_segment}_customers.csv",
        mime="text/csv"
    )


# ==================================================
# PRODUCT ANALYSIS
# ==================================================

elif page == "Product Analysis":

    st.title(
        "🛒 Product Performance"
    )

    st.caption(
        "Identify products contributing most to recorded revenue."
    )


    product_revenue = (
        filtered_df
        .groupby("Description")["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )


    product_revenue.columns = [
        "Product",
        "Revenue"
    ]


    st.subheader(
        "Top 10 Products by Revenue"
    )


    st.bar_chart(
        product_revenue,
        x="Product",
        y="Revenue"
    )


    st.subheader(
        "Product Revenue Details"
    )


    st.dataframe(
        product_revenue,
        use_container_width=True
    )


# ==================================================
# COUNTRY ANALYSIS
# ==================================================

elif page == "Country Analysis":

    st.title(
        "🌍 Geographic Performance"
    )

    st.caption(
        "Compare revenue contribution across markets."
    )


    country_revenue = (
        filtered_df
        .groupby("Country")["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )


    country_revenue.columns = [
        "Country",
        "Revenue"
    ]


    st.subheader(
        "Top 10 Countries by Revenue"
    )


    st.bar_chart(
        country_revenue,
        x="Country",
        y="Revenue"
    )


    st.subheader(
        "Country Revenue Details"
    )


    st.dataframe(
        country_revenue,
        use_container_width=True
    )


# ==================================================
# ANOMALY MONITORING
# ==================================================

elif page == "Anomaly Monitoring":

    st.title(
        "🚨 Anomaly Monitoring"
    )

    st.caption(
        "Machine-learning based identification of unusual transaction patterns."
    )


    st.markdown(
        """
        <div class="risk-box">
        <b>Investigation Signal</b><br>
        Isolation Forest identifies transactions that differ significantly
        from normal transaction patterns. An anomaly is not automatically fraud.
        </div>
        """,
        unsafe_allow_html=True
    )


    anomaly_data = filtered_df[
        filtered_df["Is_Anomaly"]
    ].copy()


    total_transactions = len(
        filtered_df
    )

    total_anomalies = len(
        anomaly_data
    )


    anomaly_rate = (
        total_anomalies
        / total_transactions
        * 100
        if total_transactions > 0
        else 0
    )


    anomaly_revenue = (
        anomaly_data["Revenue"].sum()
    )


    col1, col2, col3 = st.columns(3)


    with col1:
        st.metric(
            "Potential Anomalies",
            f"{total_anomalies:,}"
        )


    with col2:
        st.metric(
            "Anomaly Rate",
            f"{anomaly_rate:.2f}%"
        )


    with col3:
        st.metric(
            "Anomalous Revenue",
            f"£{anomaly_revenue:,.0f}"
        )


    st.divider()


    st.subheader(
        "🔎 Transactions Requiring Investigation"
    )


    display_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "UnitPrice",
        "Revenue",
        "InvoiceDate",
        "CustomerID",
        "Country",
        "TransactionType"
    ]


    st.dataframe(
        anomaly_data[
            display_columns
        ]
        .sort_values(
            "Revenue",
            ascending=False
        )
        .head(100),
        use_container_width=True
    )


    anomaly_csv = anomaly_data.to_csv(
        index=False
    )


    st.download_button(
        label="📥 Download Anomaly Report",
        data=anomaly_csv,
        file_name="potential_anomalies.csv",
        mime="text/csv"
    )


# ==================================================
# RISK MONITORING
# ==================================================

elif page == "Risk Monitoring":

    st.title(
        "🚨 Revenue Leakage Risk Monitoring"
    )

    st.caption(
        "Prioritize transactions for investigation using business rules and machine-learning signals."
    )


    st.markdown(
        """
        <div class="risk-box">
        <b>How the risk engine works</b><br>
        Risk is prioritized using duplicate transaction signals,
        machine-learning anomalies and return activity.
        Multiple signals increase investigation priority.
        </div>
        """,
        unsafe_allow_html=True
    )


    high_risk_data = filtered_df[
        filtered_df["Risk_Level"]
        == "High Risk"
    ].copy()


    total_revenue = filtered_df[
        "Revenue"
    ].sum()


    potential_leakage = filtered_df[
        "Potential_Leakage"
    ].sum()


    potential_leakage_percentage = (
        potential_leakage
        / total_revenue
        * 100
        if total_revenue != 0
        else 0
    )


    high_risk_revenue = (
        high_risk_data["Revenue"]
        .clip(lower=0)
        .sum()
    )


    recovery_opportunity = (
        high_risk_data[
            "Potential_Leakage"
        ].sum()
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:
        st.metric(
            "Potential Leakage",
            f"£{potential_leakage:,.0f}"
        )


    with col2:
        st.metric(
            "Leakage %",
            f"{potential_leakage_percentage:.2f}%"
        )


    with col3:
        st.metric(
            "High-Risk Revenue",
            f"£{high_risk_revenue:,.0f}"
        )


    with col4:
        st.metric(
            "Recovery Opportunity",
            f"£{recovery_opportunity:,.0f}"
        )


    st.divider()


    # ==================================================
    # RISK COUNTS
    # ==================================================

    low_risk = (
        filtered_df["Risk_Level"]
        == "Low Risk"
    ).sum()


    medium_risk = (
        filtered_df["Risk_Level"]
        == "Medium Risk"
    ).sum()


    high_risk = (
        filtered_df["Risk_Level"]
        == "High Risk"
    ).sum()


    col1, col2, col3 = st.columns(3)


    with col1:
        st.metric(
            "Low Risk",
            f"{low_risk:,}"
        )


    with col2:
        st.metric(
            "Medium Risk",
            f"{medium_risk:,}"
        )


    with col3:
        st.metric(
            "High Risk",
            f"{high_risk:,}"
        )


    st.subheader(
        "📊 Risk Distribution"
    )


    risk_distribution = (
        filtered_df["Risk_Level"]
        .value_counts()
        .reindex(
            [
                "Low Risk",
                "Medium Risk",
                "High Risk"
            ],
            fill_value=0
        )
        .reset_index()
    )


    risk_distribution.columns = [
        "Risk Level",
        "Transactions"
    ]


    st.bar_chart(
        risk_distribution,
        x="Risk Level",
        y="Transactions"
    )


    st.divider()


    # ==================================================
    # LEAKAGE BY COUNTRY
    # ==================================================

    st.subheader(
        "🌍 Potential Leakage by Country"
    )


    country_leakage = (
        filtered_df
        .groupby("Country")[
            "Potential_Leakage"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )


    country_leakage.columns = [
        "Country",
        "Potential Leakage"
    ]


    st.bar_chart(
        country_leakage,
        x="Country",
        y="Potential Leakage"
    )


    st.dataframe(
        country_leakage,
        use_container_width=True
    )


    st.divider()


    # ==================================================
    # LEAKAGE BY PRODUCT
    # ==================================================

    st.subheader(
        "🛒 Potential Leakage by Product"
    )


    product_leakage = (
        filtered_df
        .groupby("Description")[
            "Potential_Leakage"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )


    product_leakage.columns = [
        "Product",
        "Potential Leakage"
    ]


    st.bar_chart(
        product_leakage,
        x="Product",
        y="Potential Leakage"
    )


    st.dataframe(
        product_leakage,
        use_container_width=True
    )


    st.divider()


    # ==================================================
    # MONTHLY LEAKAGE TREND
    # ==================================================

    st.subheader(
        "📈 Monthly Potential Leakage Trend"
    )


    monthly_leakage = (
        filtered_df
        .groupby("Month")[
            "Potential_Leakage"
        ]
        .sum()
        .reset_index()
    )


    st.line_chart(
        monthly_leakage,
        x="Month",
        y="Potential_Leakage"
    )


    st.divider()


    # ==================================================
    # POTENTIAL LEAKAGE EXPLANATION
    # ==================================================

    st.subheader(
        "💰 Potential Revenue Leakage"
    )


    st.write(
        f"""
        Based on the current investigation rules,
        **£{potential_leakage:,.0f}**
        is identified as potential revenue leakage.

        This represents
        **{potential_leakage_percentage:.2f}%**
        of filtered recorded revenue.
        """
    )


    st.caption(
        "This is a potential leakage estimate based on transaction risk signals. "
        "It is not confirmed financial leakage because contractual billing prices "
        "and invoice reconciliation data are not available in the dataset."
    )


    # ==================================================
    # HIGH RISK TRANSACTIONS
    # ==================================================

    st.subheader(
        "🔍 High-Risk Transactions"
    )


    risk_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "UnitPrice",
        "Revenue",
        "Potential_Leakage",
        "InvoiceDate",
        "CustomerID",
        "Country",
        "TransactionType",
        "Is_Duplicate",
        "Is_Anomaly",
        "Risk_Score",
        "Risk_Level"
    ]


    st.dataframe(
        high_risk_data[
            risk_columns
        ]
        .sort_values(
            "Risk_Score",
            ascending=False
        )
        .head(100),
        use_container_width=True
    )


    risk_csv = high_risk_data.to_csv(
        index=False
    )


    st.download_button(
        label="📥 Download High-Risk Report",
        data=risk_csv,
        file_name="high_risk_transactions.csv",
        mime="text/csv"
    )

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# 1. Dashboard configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation Dashboard")
st.write(
    "Explore customer behavior using Recency, Frequency, "
    "and Monetary (RFM) metrics."
)

# --------------------------------------------------
# 2. Locate and load data
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_segments.csv"


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


try:
    customers = load_data(str(DATA_PATH))

    required_columns = [
        "CustomerID", "Recency", "Frequency", "Monetary", "Cluster"
    ]
    missing_columns = [
        column for column in required_columns
        if column not in customers.columns
    ]

    if missing_columns:
        st.error(f"Required columns are missing: {missing_columns}")
        st.stop()

    # Add readable names to the existing cluster labels
    segment_names = {
        0: "Occasional Low-Spending Customers",
        1: "Frequent High-Value Customers"
    }

    if "Segment_Name" not in customers.columns:
        customers["Segment_Name"] = customers["Cluster"].map(segment_names)

    # --------------------------------------------------
    # 3. Interactive segment filter
    # --------------------------------------------------
    st.sidebar.header("Dashboard Filters")

    all_segments = sorted(customers["Segment_Name"].dropna().unique())

    selected_segments = st.sidebar.multiselect(
        "Select customer segments",
        options=all_segments,
        default=all_segments
    )

    filtered_customers = customers[
        customers["Segment_Name"].isin(selected_segments)
    ].copy()

    st.sidebar.caption(
        f"{len(filtered_customers):,} of {len(customers):,} customers selected"
    )

    if filtered_customers.empty:
        st.warning("Select at least one customer segment to view analytics.")
        st.stop()

    # --------------------------------------------------
    # 4. Overview metrics that respond to filters
    # --------------------------------------------------
    st.subheader("Customer Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Filtered Customers", f"{len(filtered_customers):,}")

    with col2:
        st.metric(
            "Selected Segments",
            filtered_customers["Cluster"].nunique()
        )

    with col3:
        st.metric(
            "Average Recency",
            f"{filtered_customers['Recency'].mean():,.1f} days"
        )

    with col4:
        st.metric(
            "Average Customer Spend",
            f"{filtered_customers['Monetary'].mean():,.2f}"
        )

    # --------------------------------------------------
    # 5. Customer distribution
    # --------------------------------------------------
    st.subheader("Customer Distribution by Segment")

    segment_counts = (
        filtered_customers.groupby("Segment_Name")
        .size()
        .reset_index(name="Customer Count")
        .sort_values("Customer Count", ascending=True)
    )

    fig_distribution = px.bar(
        segment_counts,
        x="Customer Count",
        y="Segment_Name",
        orientation="h",
        text="Customer Count",
        title="Customers in Each Selected Segment",
        labels={
            "Segment_Name": "Customer Segment",
            "Customer Count": "Number of Customers"
        }
    )
    fig_distribution.update_traces(textposition="outside", cliponaxis=False)
    fig_distribution.update_layout(
        yaxis_title="",
        height=350,
        showlegend=False,
        margin=dict(l=20, r=50, t=60, b=30)
    )
    st.plotly_chart(fig_distribution, use_container_width=True)

    # --------------------------------------------------
    # 6. RFM analytics
    # --------------------------------------------------
    st.subheader("RFM Analytics")

    metric_choice = st.selectbox(
        "Choose a metric to compare",
        options=["Recency", "Frequency", "Monetary"],
        help=(
            "Recency is days since last purchase; Frequency is transaction "
            "count; Monetary is total customer spending."
        )
    )

    metric_labels = {
        "Recency": "Recency (days)",
        "Frequency": "Number of Transactions",
        "Monetary": "Total Spending"
    }

    fig_rfm = px.box(
        filtered_customers,
        x="Segment_Name",
        y=metric_choice,
        color="Segment_Name",
        points="outliers",
        title=f"{metric_choice} Comparison by Customer Segment",
        labels={
            "Segment_Name": "Customer Segment",
            metric_choice: metric_labels[metric_choice]
        }
    )
    fig_rfm.update_layout(
        xaxis_title="Customer Segment",
        yaxis_title=metric_labels[metric_choice],
        showlegend=False
    )
    st.plotly_chart(fig_rfm, use_container_width=True)

    # --------------------------------------------------
    # 7. Segment summary table
    # --------------------------------------------------
    st.subheader("Segment Summary")

    segment_summary = (
        filtered_customers.groupby("Segment_Name")
        .agg(
            Customer_Count=("CustomerID", "count"),
            Average_Recency=("Recency", "mean"),
            Average_Frequency=("Frequency", "mean"),
            Average_Monetary=("Monetary", "mean")
        )
        .reset_index()
        .round(2)
        .rename(columns={
            "Segment_Name": "Customer Segment",
            "Customer_Count": "Number of Customers",
            "Average_Recency": "Average Recency (days)",
            "Average_Frequency": "Average Frequency",
            "Average_Monetary": "Average Monetary"
        })
    )
    st.dataframe(segment_summary, use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # 8. Filterable customer table
    # --------------------------------------------------
    st.subheader("Customer Explorer")

    search_id = st.text_input(
        "Search by Customer ID (optional)",
        placeholder="Enter a customer ID"
    )

    table_customers = filtered_customers.copy()

    if search_id.strip():
        table_customers = table_customers[
            table_customers["CustomerID"].astype(str).str.contains(
                search_id.strip(), case=False, na=False
            )
        ]

    st.caption(f"{len(table_customers):,} matching customer records")

    st.dataframe(
        table_customers[
            ["CustomerID", "Segment_Name", "Recency",
             "Frequency", "Monetary", "Cluster"]
        ],
        use_container_width=True,
        hide_index=True
    )

except FileNotFoundError:
    st.error(f"Customer data file not found: {DATA_PATH}")

except pd.errors.EmptyDataError:
    st.error("The customer data CSV file is empty.")

except pd.errors.ParserError:
    st.error("The customer data CSV file could not be read.")

except Exception as error:
    st.error(f"An unexpected error occurred: {error}")

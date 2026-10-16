
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# 1. Configure the dashboard
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# 2. Dashboard title and description
# --------------------------------------------------
st.title("📊 Customer Segmentation Dashboard")

st.write(
    "Analyze customer behavior using Recency, Frequency, "
    "and Monetary (RFM) metrics."
)

# --------------------------------------------------
# 3. Locate and load customer data
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_segments.csv"
)


@st.cache_data
def load_data(file_path):
    """Load customer segment data from a CSV file."""
    return pd.read_csv(file_path)


try:
    customers = load_data(str(DATA_PATH))

    # Check that the required columns exist
    required_columns = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "Cluster"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in customers.columns
    ]

    if missing_columns:
        st.error(
            f"Required columns are missing: {missing_columns}"
        )
        st.stop()

    # --------------------------------------------------
    # 4. Add readable customer segment names
    # --------------------------------------------------
    segment_names = {
        0: "Occasional Low-Spending Customers",
        1: "Frequent High-Value Customers"
    }

    if "Segment_Name" not in customers.columns:
        customers["Segment_Name"] = (
            customers["Cluster"].map(segment_names)
        )

    # --------------------------------------------------
    # 5. Customer overview metrics
    # --------------------------------------------------
    st.subheader("Customer Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(customers):,}"
        )

    with col2:
        st.metric(
            label="Customer Segments",
            value=customers["Cluster"].nunique()
        )

    with col3:
        st.metric(
            label="Average Customer Spend",
            value=f"{customers['Monetary'].mean():,.2f}"
        )

    # --------------------------------------------------
    # 6. Customer distribution chart
    # --------------------------------------------------
    st.subheader("Customer Distribution by Segment")

    segment_counts = (
        customers.groupby("Segment_Name")
        .size()
        .reset_index(name="Customer Count")
        .sort_values("Customer Count", ascending=True)
    )

    fig = px.bar(
        segment_counts,
        x="Customer Count",
        y="Segment_Name",
        orientation="h",
        text="Customer Count",
        title="Number of Customers in Each Segment",
        labels={
            "Segment_Name": "Customer Segment",
            "Customer Count": "Number of Customers"
        }
    )

    fig.update_layout(
        yaxis_title="",
        xaxis_title="Number of Customers",
        showlegend=False,
        height=400,
        margin=dict(l=20, r=50, t=60, b=30)
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------
    # 7. Customer data preview
    # --------------------------------------------------
    st.subheader("Customer Data Preview")

    st.caption(
        "Showing the first 20 customers from the segmentation dataset."
    )

    st.dataframe(
        customers.head(20),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # 8. Basic segment summary
    # --------------------------------------------------
    st.subheader("Segment Summary")

    segment_summary = (
        customers.groupby("Segment_Name")
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

    st.dataframe(
        segment_summary,
        use_container_width=True,
        hide_index=True
    )

except FileNotFoundError:
    st.error(
        "Customer data file was not found. "
        f"Expected location: {DATA_PATH}"
    )

except pd.errors.EmptyDataError:
    st.error("The customer data CSV file is empty.")

except pd.errors.ParserError:
    st.error("The customer data CSV file could not be read.")

except Exception as error:
    st.error(f"An unexpected error occurred: {error}")

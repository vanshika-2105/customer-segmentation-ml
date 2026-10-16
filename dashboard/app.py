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

    if customers.empty:
        st.error("The customer data file contains no customer records.")
        st.stop()

    # Ensure RFM columns contain numeric values
    for column in ["Recency", "Frequency", "Monetary", "Cluster"]:
        customers[column] = pd.to_numeric(
            customers[column], errors="coerce"
        )

    # Remove rows with missing values in required fields
    customers = customers.dropna(subset=required_columns).copy()

    if customers.empty:
        st.error("No valid customer records were found in the data file.")
        st.stop()

    # Use readable segment names
    segment_names = {
        0: "Occasional Low-Spending Customers",
        1: "Frequent High-Value Customers"
    }

    if "Segment_Name" not in customers.columns:
        customers["Segment_Name"] = customers["Cluster"].map(segment_names)

    # Fill any missing segment names safely
    customers["Segment_Name"] = customers["Segment_Name"].fillna(
        customers["Cluster"].apply(lambda cluster: f"Segment {int(cluster)}")
    )

    # --------------------------------------------------
    # 3. Segment filters
    # --------------------------------------------------
    st.sidebar.header("Dashboard Filters")

    all_segments = sorted(customers["Segment_Name"].unique().tolist())

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
    # 4. Customer overview
    # --------------------------------------------------
    st.subheader("Customer Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Filtered Customers", f"{len(filtered_customers):,}")
    col2.metric(
        "Selected Segments",
        filtered_customers["Cluster"].nunique()
    )
    col3.metric(
        "Average Recency",
        f"{filtered_customers['Recency'].mean():,.1f} days"
    )
    col4.metric(
        "Average Customer Spend",
        f"{filtered_customers['Monetary'].mean():,.2f}"
    )

    # --------------------------------------------------
    # 5. Customer distribution chart
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

    fig_distribution.update_traces(
        textposition="outside",
        cliponaxis=False
    )
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
        ["Recency", "Frequency", "Monetary"],
        help=(
            "Recency means days since the last purchase. "
            "Frequency means number of transactions. "
            "Monetary means total customer spending."
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
    # 7. Segment summary
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

    st.dataframe(
        segment_summary,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # 8. Day 17: Interactive Customer Explorer
    # --------------------------------------------------
    st.subheader("🔎 Interactive Customer Explorer")
    st.write("Filter customers using RFM values and inspect individual profiles.")

    st.sidebar.subheader("Customer-Level Filters")

    # Set slider limits from the selected customer segments
    recency_min = int(filtered_customers["Recency"].min())
    recency_max = int(filtered_customers["Recency"].max())

    frequency_min = int(filtered_customers["Frequency"].min())
    frequency_max = int(filtered_customers["Frequency"].max())

    monetary_min = float(filtered_customers["Monetary"].min())
    monetary_max = float(filtered_customers["Monetary"].max())

    recency_range = st.sidebar.slider(
        "Recency range (days)",
        min_value=recency_min,
        max_value=recency_max,
        value=(recency_min, recency_max)
    )

    frequency_range = st.sidebar.slider(
        "Frequency range",
        min_value=frequency_min,
        max_value=frequency_max,
        value=(frequency_min, frequency_max)
    )

    if monetary_min < monetary_max:
        monetary_range = st.sidebar.slider(
            "Monetary range",
            min_value=monetary_min,
            max_value=monetary_max,
            value=(monetary_min, monetary_max),
            step=max((monetary_max - monetary_min) / 100, 0.01)
        )
    else:
        monetary_range = (monetary_min, monetary_max)
        st.sidebar.caption(
            f"Monetary value is fixed at {monetary_min:,.2f} for these customers."
        )

    # Apply RFM filters
    table_customers = filtered_customers[
        filtered_customers["Recency"].between(*recency_range)
        & filtered_customers["Frequency"].between(*frequency_range)
        & filtered_customers["Monetary"].between(*monetary_range)
    ].copy()

    # Search for a customer
    search_id = st.text_input(
        "Search by Customer ID (optional)",
        placeholder="Enter a customer ID"
    )

    if search_id.strip():
        table_customers = table_customers[
            table_customers["CustomerID"].astype(str).str.contains(
                search_id.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    st.caption(f"{len(table_customers):,} customers match your filters.")

    display_columns = [
        "CustomerID", "Segment_Name", "Recency",
        "Frequency", "Monetary", "Cluster"
    ]

    st.dataframe(
        table_customers[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # 9. Individual customer profile
    # --------------------------------------------------
    st.subheader("👤 Individual Customer Profile")

    if table_customers.empty:
        st.info(
            "No customers match your current filters. "
            "Adjust the sliders or clear your customer search."
        )
    else:
        # Use a unique list of customer IDs for the dropdown
        customer_options = (
            table_customers["CustomerID"]
            .drop_duplicates()
            .tolist()
        )

        selected_customer_id = st.selectbox(
            "Choose a customer to inspect",
            options=customer_options,
            format_func=lambda customer_id: (
                str(int(customer_id))
                if isinstance(customer_id, (int, float))
                and float(customer_id).is_integer()
                else str(customer_id)
            )
        )

        customer = table_customers[
            table_customers["CustomerID"] == selected_customer_id
        ].iloc[0]

        customer_id_display = str(selected_customer_id)

        if customer_id_display.endswith(".0"):
            customer_id_display = customer_id_display[:-2]

        st.markdown(f"### Customer ID: {customer_id_display}")
        st.write(f"**Customer Segment:** {customer['Segment_Name']}")

        profile_col1, profile_col2, profile_col3 = st.columns(3)

        profile_col1.metric(
            "Recency",
            f"{customer['Recency']:.0f} days"
        )
        profile_col2.metric(
            "Frequency",
            f"{customer['Frequency']:.0f} transactions"
        )
        profile_col3.metric(
            "Monetary",
            f"{customer['Monetary']:,.2f}"
        )

        if customer["Segment_Name"] == "Frequent High-Value Customers":
            st.success(
                "Suggested action: Consider loyalty rewards, early access, "
                "and personalized retention offers for this segment."
            )
        else:
            st.info(
                "Suggested action: Consider relevant product suggestions "
                "and re-engagement campaigns to encourage repeat purchases."
            )

except FileNotFoundError:
    st.error(
        f"Customer data file not found: {DATA_PATH}. "
        "Check that the processed customer_segments.csv file exists."
    )

except pd.errors.EmptyDataError:
    st.error("The customer data CSV file is empty.")

except pd.errors.ParserError:
    st.error("The customer data CSV file could not be read as a valid CSV.")

except Exception as error:
    st.error(f"An unexpected error occurred: {error}")
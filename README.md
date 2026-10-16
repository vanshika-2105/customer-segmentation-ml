# Customer Segmentation Using Machine Learning

## Project Overview

This project uses unsupervised machine learning to segment customers based on their purchasing behavior. It analyzes customer transactions using Recency, Frequency, and Monetary (RFM) metrics, applies K-Means clustering, and generates business recommendations for different customer groups.

An interactive Streamlit dashboard allows users to explore customer segments, compare RFM metrics, filter customers, and inspect individual customer profiles.

## Objectives

- Analyze customer purchasing behavior.
- Clean and preprocess transaction data.
- Perform exploratory data analysis (EDA).
- Engineer customer-level features.
- Calculate RFM metrics.
- Scale features for clustering.
- Determine the number of customer segments.
- Build and evaluate a K-Means clustering model.
- Profile customer segments.
- Generate business recommendations.
- Build an interactive dashboard for customer analysis.

## Dataset

The project uses the **Online Retail** transaction dataset.

The processed data is used to calculate customer-level metrics:

- **Recency:** Number of days since the customer's most recent purchase.
- **Frequency:** Number of transactions made by the customer.
- **Monetary:** Total spending by the customer.

## Machine Learning Approach

The project uses an unsupervised clustering approach implemented with NumPy.

The K-Means model groups customers based on their scaled RFM values. The saved model contains the cluster centroids used to assign customers to the nearest cluster.

The selected model uses two customer segments.

## Customer Segments

### 1. Occasional Low-Spending Customers

Customers in this group have lower average transaction frequency and spending in the analyzed dataset.

**Suggested strategies:**
- Personalized re-engagement campaigns.
- Relevant product recommendations.
- Offers designed to encourage repeat purchases.

### 2. Frequent High-Value Customers

Customers in this group have higher average purchase frequency and spending in the analyzed dataset.

**Suggested strategies:**
- Loyalty rewards.
- Early access to new products.
- Personalized retention offers.

These descriptions reflect group-level averages and do not necessarily describe every customer in a segment.

## Interactive Dashboard

The Streamlit dashboard provides:

- Customer overview metrics.
- Customer distribution by segment.
- Interactive RFM comparison charts.
- Segment summary tables.
- Segment selection filters.
- Recency, Frequency, and Monetary range filters.
- Customer ID search.
- Individual customer profiles.
- Segment-based business recommendations.

## Tech Stack

- **Python** — Core programming language
- **Pandas** — Data processing and analysis
- **NumPy** — Numerical operations and K-Means implementation
- **Matplotlib and Seaborn** — Exploratory data visualization
- **Plotly** — Interactive charts
- **Streamlit** — Interactive web dashboard
- **Jupyter Notebook** — Data exploration and experimentation
- **Git and GitHub** — Version control

## Project Structure

```text
customer-segmentation-ml/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── customer_segmentation_model.py
├── models/
│   └── customer_segmentation_model.json
├── reports/
├── dashboard/
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vanshika-2105/customer-segmentation-ml.git
cd customer-segmentation-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`.

Ensure the required processed data file, `data/processed/customer_segments.csv`, is available before launching the dashboard.

## Model Files

The trained model configuration is stored in:

```text
models/customer_segmentation_model.json
```

The model implementation is located in:

```text
src/customer_segmentation_model.py
```

The prediction method expects RFM features that have been scaled consistently with the training data.

## Future Improvements

- Deploy the dashboard publicly.
- Improve automated testing.
- Add additional clustering evaluation and comparison.
- Enhance dashboard presentation and reporting.
- Explore additional customer behavior features.

## Author

**Vanshika**

GitHub: https://github.com/vanshika-2105

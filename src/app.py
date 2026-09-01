"""
E-commerce Multi-platform Dashboard
Combines Website, Shopee, and Lazada order data into one view.

Run with:
    pip install streamlit pandas plotly openpyxl
    streamlit run app.py
"""

import pathlib

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Data Engineering Assessment", layout="wide")

# ---------------------------------------------------------------------------
# CONFIG
# Paths are resolved relative to this script's location (app.py lives in
# src/, and data/processed/ is a sibling of src/ at the project root), so
# this works no matter what folder you run `streamlit run` from.
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FILE_PATHS = {
    "Website": PROJECT_ROOT / "data" / "processed" / "website_clean.xlsx",
    "Shopee": PROJECT_ROOT / "data" / "processed" / "shopee_clean.xlsx",
    "Lazada": PROJECT_ROOT / "data" / "processed" / "lazada_clean.xlsx",
}


# ---------------------------------------------------------------------------
# LOAD + STANDARDIZE
# Each platform has different column names, so we map each one into a common
# schema: platform, order_id, customer_key, date, product, revenue, status,
# reason, city
# ---------------------------------------------------------------------------
@st.cache_data
def load_website(path):
    df = pd.read_excel(path)

    # Remove rows without a valid order ID
    df["ID"] = df["ID"].replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(subset=["ID"])

    return pd.DataFrame({
        "platform": "Website",
        "order_id": df["ID"].astype("string").str.strip(),
        "customer_key": df["Customer Email"].astype("string").str.strip().str.lower(),
        "date": pd.to_datetime(df["Purchase Date"], errors="coerce"),
        "product": df["Order"].astype("string").str.strip(),
        "revenue": pd.to_numeric(df["Subtotal"], errors="coerce").fillna(0),
        "status": df["Status"].astype("string").str.strip(),
        "reason": pd.NA,
        "city": "N/A",
        "province": "N/A",
    })


@st.cache_data
def load_shopee(path):
    df = pd.read_excel(path)
    return pd.DataFrame({
        "platform": "Shopee",
        "order_id": df["Order ID"].astype("string").str.strip(),
        "customer_key": df["Username (Buyer)"].astype(str).str.strip().str.lower(),
        "date": pd.to_datetime(df["Order Creation Date"], errors="coerce"),
        "product": df["Product Name"].astype(str),
        "revenue": pd.to_numeric(df["Total Buyer Payment"], errors="coerce").fillna(0),
        "status": df["Order Status"].astype(str),
        "reason": df["Cancel reason"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA}),
        "city": df.get("City", "N/A"),
    })


@st.cache_data
def load_lazada(path):
    df = pd.read_excel(path)
    return pd.DataFrame({
        "platform": "Lazada",
        "order_id": df["orderNumber"].astype(str),
        "customer_key": df["customerEmail"].astype(str).str.strip().str.lower(),
        "date": pd.to_datetime(df["createTime"], errors="coerce"),
        "product": df["itemName"].astype(str),
        "revenue": pd.to_numeric(df["paidPrice"], errors="coerce").fillna(0),
        "status": df["status"].astype(str),
        "reason": df["buyerFailedDeliveryReason"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA}),
        "city": df.get("shippingCity", "N/A"),
    })


@st.cache_data
def load_all():
    frames = []
    loaders = {
        "Website": load_website,
        "Shopee": load_shopee,
        "Lazada": load_lazada,
    }
    for platform, path in FILE_PATHS.items():
        try:
            frames.append(loaders[platform](path))
        except FileNotFoundError:
            st.warning(f"Could not find {path} — skipping {platform}.")
        except Exception as e:
            st.warning(f"Error loading {platform} ({path}): {e}")
    if not frames:
        return pd.DataFrame(
            columns=["platform", "order_id", "customer_key", "date", "product",
                     "revenue", "status", "reason", "city"]
        )
    return pd.concat(frames, ignore_index=True)


df = load_all()

if df.empty:
    st.error("No data loaded. Check that the xlsx files are in data/processed/, "
              "or update FILE_PATHS at the top of the script.")
    st.stop()

# ---------------------------------------------------------------------------
# SENTIMENT PROXY (Shopee & Lazada only — Website has no cancel/return status)
# No open-ended review text exists in this data, so sentiment here is a
# proxy derived from order outcome, not an NLP model on templated strings.
# ---------------------------------------------------------------------------
# SENTIMENT_MAP = {
#     "cancelled": "Negative", "canceled": "Negative", "returned": "Negative",
#     "refunded": "Negative", "return/refund": "Negative", "failed": "Negative",
#     "pending": "Neutral", "processing": "Neutral", "shipped": "Neutral",
#     "to ship": "Neutral", "to receive": "Neutral", "unpaid": "Neutral",
#     "completed": "Positive", "delivered": "Positive", "complete": "Positive",
# }


# def classify_sentiment(status):
#     if pd.isna(status):
#         return "Unknown"
#     key = str(status).strip().lower()
#     for pattern, label in SENTIMENT_MAP.items():
#         if pattern in key:
#             return label
#     return "Unknown"


# df["sentiment"] = df["status"].apply(classify_sentiment)

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
platform_options = sorted(df["platform"].unique())
platform = st.sidebar.radio("Select Platform", platform_options)

filtered = df[df["platform"] == platform]

# Revenue-safe view: some platforms store the ORDER-level total repeated on
# every line-item row (e.g. Website's Subtotal, and originally Shopee's
# Grand Total). Summing "revenue" across all line-item rows would multiply
# it by the number of items per order. Deduplicating to one row per order
# before summing avoids this double-count.

orders_df = (
    filtered
    .dropna(subset=["order_id"])
    .drop_duplicates(subset=["platform", "order_id"])
)
# ---------------------------------------------------------------------------
# HEADER + KPIs
# ---------------------------------------------------------------------------
st.title("Data Engineering Assessment")
# st.caption("Website + Shopee + Lazada — unified order overview")

total_orders = orders_df.shape[0]
total_revenue = orders_df["revenue"].sum()
avg_order_value = total_revenue / total_orders if total_orders else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Total Revenue", f"₱{total_revenue:,.2f}")
col3.metric("Avg Order Value", f"₱{avg_order_value:,.2f}")

st.divider()

# ---------------------------------------------------------------------------
# ROW 1 — Sales trend + Revenue by platform
# ---------------------------------------------------------------------------
st.subheader("Sales Trend Over Time")
trend = (
    orders_df.dropna(subset=["date"])
    .assign(
        week_start=lambda d: d["date"].dt.to_period("W").dt.start_time,
        week_end=lambda d: d["date"].dt.to_period("W").dt.end_time.dt.normalize(),
    )
    .groupby(["week_start", "week_end", "platform"], as_index=False)["revenue"].sum()
)
trend["week_label"] = (
    trend["week_start"].dt.strftime("%b %d") + " - " + trend["week_end"].dt.strftime("%b %d")
)
trend = trend.sort_values("week_start")
fig = px.line(trend, x="week_label", y="revenue", color="platform", markers=True)
fig.update_xaxes(categoryorder="array", categoryarray=trend["week_label"].unique())
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ROW 2 — Order status breakdown (vertical bar chart)
# ---------------------------------------------------------------------------
st.subheader("Order Status by Platform")
status_counts = filtered.groupby(["platform", "status"], as_index=False)["order_id"].nunique()
status_counts.columns = ["platform", "status", "orders"]
fig = px.bar(status_counts, x="platform", y="orders", color="status", barmode="group")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ROW 2.5 — Sentiment (Shopee & Lazada only)
# Website has no cancellation/return status, so it carries no sentiment
# signal and is excluded here.
# ---------------------------------------------------------------------------
# sentiment_scope = filtered[filtered["platform"] != "Website"]

# if not sentiment_scope.empty and sentiment_scope["sentiment"].nunique() > 1:
#     st.divider()
#     st.header("Customer Sentiment")
#     st.caption(
#         "Shopee & Lazada only — proxy sentiment derived from order outcome "
#         "(Delivered = Positive, Cancelled/Returned = Negative, in-progress = Neutral)."
#     )

#     color_map = {"Positive": "#2ca02c", "Neutral": "#ff7f0e", "Negative": "#d62728", "Unknown": "#7f7f7f"}
#     sc1, sc2 = st.columns([1, 2])

#     with sc1:
#         st.subheader("Sentiment Breakdown")
#         sent_counts = sentiment_scope["sentiment"].value_counts().reset_index()
#         sent_counts.columns = ["sentiment", "orders"]
#         fig = px.pie(sent_counts, names="sentiment", values="orders", hole=0.4,
#                      color="sentiment", color_discrete_map=color_map)
#         st.plotly_chart(fig, use_container_width=True)

#     with sc2:
#         st.subheader("Sentiment by Platform")
#         sent_by_platform = sentiment_scope.groupby(["platform", "sentiment"], as_index=False)["order_id"].nunique()
#         sent_by_platform.columns = ["platform", "sentiment", "orders"]
#         fig = px.bar(sent_by_platform, x="platform", y="orders", color="sentiment",
#                      barmode="stack", color_discrete_map=color_map)
#         st.plotly_chart(fig, use_container_width=True)

#     neg = sentiment_scope[(sentiment_scope["sentiment"] == "Negative") & (sentiment_scope["customer_key"].notna())]
#     neg_by_customer = (
#         neg.groupby(["platform", "customer_key"], as_index=False)["order_id"]
#         .nunique()
#         .rename(columns={"order_id": "negative_orders"})
#     )
#     neg_by_customer = neg_by_customer[neg_by_customer["negative_orders"] > 1].sort_values(
#         "negative_orders", ascending=False
#     )
#     if not neg_by_customer.empty:
#         st.subheader("Repeat Customers with Negative Outcomes")
#         st.dataframe(neg_by_customer.head(15), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# ROW 3 — Top Repeat Purchasers (all platforms)
# ---------------------------------------------------------------------------
purchase_counts = (
    filtered.dropna(subset=["customer_key"])
    .groupby(["platform", "customer_key"], as_index=False)["order_id"]
    .nunique()
    .rename(columns={"order_id": "orders"})
)
repeat_purchasers = purchase_counts[purchase_counts["orders"] > 1].sort_values("orders", ascending=False)

if not repeat_purchasers.empty:
    st.subheader("Top Repeat Purchasers")
    st.caption("Customers who placed more than one order — a loyalty/retention signal.")
    rp1, rp2 = st.columns([1, 1])

    with rp1:
        st.dataframe(repeat_purchasers.head(15), use_container_width=True, hide_index=True)

    with rp2:
        top_repeat = repeat_purchasers.head(10)
        fig = px.bar(top_repeat, x="orders", y="customer_key", color="platform", orientation="h")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ROW 4 — Top Cancel / Failed-Delivery Reasons (vertical bar chart)
# ---------------------------------------------------------------------------
reasons = filtered["reason"].dropna()
reasons = reasons[reasons.astype(str).str.strip() != ""]
if not reasons.empty:
    st.subheader("Top Cancel / Failed-Delivery Reasons")
    reason_counts = reasons.value_counts().reset_index().head(10)
    reason_counts.columns = ["reason", "count"]
    fig = px.bar(reason_counts, x="count", y="reason", orientation="h")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ROW 5 — Top Products (vertical bar chart)
# ---------------------------------------------------------------------------
prod_df = filtered[filtered["product"] != "N/A"]
if not prod_df.empty:
    st.subheader("Top 10 Products (by number of orders)")
    top_products = (
        prod_df.groupby("product", as_index=False)["order_id"]
        .nunique()
        .rename(columns={"order_id": "orders"})
        .sort_values("orders", ascending=False)
        .head(10)
    )
    fig = px.bar(top_products, x="orders", y="product", orientation="h")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ROW 6 — Geographic distribution (vertical bar chart)
# ---------------------------------------------------------------------------
geo_df = filtered[filtered["city"] != "N/A"]
if not geo_df.empty:
    st.subheader(f"Orders by City ({platform})")
    top_cities = (
        geo_df.groupby("city", as_index=False)["order_id"]
        .nunique()
        .rename(columns={"order_id": "orders"})
        .sort_values("orders", ascending=False)
        .head(15)
    )
    fig = px.bar(top_cities, x="orders", y="city", orientation="h")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# RAW DATA (original columns from the processed file, no modification)
# ---------------------------------------------------------------------------
with st.expander("View filtered raw data"):
    raw_df = pd.read_excel(FILE_PATHS[platform])
    st.dataframe(raw_df, use_container_width=True)
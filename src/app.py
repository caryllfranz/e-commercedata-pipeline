import pathlib

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Data Engineering Assessment", layout="wide")

# ---------------------------------------------------------------------------
# CONFIG
# app.py lives in src/, and data/processed/ is next to src/ at the project
# root -- paths are built from this file's location so it works no matter
# what folder you run `streamlit run` from.
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FILE_PATHS = {
    "Website": PROJECT_ROOT / "data" / "processed" / "website_clean.xlsx",
    "Shopee": PROJECT_ROOT / "data" / "processed" / "shopee_clean.xlsx",
    "Lazada": PROJECT_ROOT / "data" / "processed" / "lazada_clean.xlsx",
}

# Identifiers must be read as text, or Excel turns them into numbers --
# Website's ID becomes "3150.0" and Shopee's zip code "0000" becomes "0".
TEXT_COLUMNS = {
    "Website": {"ID": str},
    "Shopee": {"Order ID": str, "Zip Code": str, "Phone Number": str},
    "Lazada": {"orderNumber": str, "orderItemId": str, "shippingPostCode": str},
}

# How each platform's price field behaves across a multi-item order:
#   "order_total"  -- the order's full total repeats on every row, so take
#                      it once.
#   "line_amount"  -- each row only holds its own line's amount, so add the
#                      rows together to get the order total.
# This logic stays in the code even though revenue isn't shown on the
# dashboard (see note below) -- it's what makes the order count correct,
# and it's ready to use the moment revenue reporting is needed.
REVENUE_GRAIN = {
    "Website": "order_total",
    "Shopee": "line_amount",
    "Lazada": "line_amount",
}

# One misspelling in the source files, corrected so the same product isn't
# split into two labels.
PRODUCT_NAME_FIXES = {
    "Nordic Spirirt Lush Tropics Nicotine Pouch": "Nordic Spirit Lush Tropics Nicotine Pouch",
}


# ---------------------------------------------------------------------------
# LOAD + STANDARDIZE
# Each platform names its columns differently, so each loader maps its own
# columns into one shared set: platform, order_id, customer_key, date,
# product, revenue, status, reason, city.
# ---------------------------------------------------------------------------
@st.cache_data
def load_website(path):
    df = pd.read_excel(path, dtype=TEXT_COLUMNS["Website"])
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
    })


@st.cache_data
def load_shopee(path):
    df = pd.read_excel(path, dtype=TEXT_COLUMNS["Shopee"])
    return pd.DataFrame({
        "platform": "Shopee",
        "order_id": df["Order ID"].astype("string").str.strip(),
        "customer_key": df["Username (Buyer)"].astype(str).str.strip().str.lower().replace({"nan": pd.NA, "": pd.NA}),
        "date": pd.to_datetime(df["Order Creation Date"], errors="coerce"),
        "product": df["Product Name"].astype(str),
        "revenue": pd.to_numeric(df["Total Buyer Payment"], errors="coerce").fillna(0),
        "status": df["Order Status"].astype(str),
        "reason": df["Cancel reason"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA}),
        "city": df.get("City", "N/A"),
    })


@st.cache_data
def load_lazada(path):
    df = pd.read_excel(path, dtype=TEXT_COLUMNS["Lazada"])
    return pd.DataFrame({
        "platform": "Lazada",
        "order_id": df["orderNumber"].astype(str),
        "customer_key": df["customerEmail"].astype(str).str.strip().str.lower().replace({"nan": pd.NA, "": pd.NA}),
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
    loaders = {"Website": load_website, "Shopee": load_shopee, "Lazada": load_lazada}
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
    unified = pd.concat(frames, ignore_index=True)
    unified["product"] = unified["product"].replace(PRODUCT_NAME_FIXES)
    return unified


def to_order_level(frame):
    """Collapse multiple rows of the same order into one row per order.

    This is what makes the order count correct: adding up every row would
    count a multi-item order's total more than once, and just picking one
    row would drop the other items' amounts. So each platform is combined
    the way it actually behaves (see REVENUE_GRAIN above).
    """
    frame = frame.dropna(subset=["order_id"])
    if frame.empty:
        return frame

    parts = []
    for platform_name, group in frame.groupby("platform", sort=False):
        grain = REVENUE_GRAIN.get(platform_name)
        if grain is None:
            st.warning(
                f"No revenue grain declared for {platform_name}; summing its "
                "line amounts. Add it to REVENUE_GRAIN to be explicit."
            )
        totals = group.groupby("order_id", as_index=False)["revenue"].agg(
            "first" if grain == "order_total" else "sum"
        )
        representative = group.drop_duplicates(subset=["order_id"]).drop(columns=["revenue"])
        parts.append(representative.merge(totals, on="order_id", how="left"))

    return pd.concat(parts, ignore_index=True)[frame.columns]


df = load_all()

if df.empty:
    st.error("No data loaded. Check that the xlsx files are in data/processed/, "
              "or update FILE_PATHS at the top of the script.")
    st.stop()

st.sidebar.header("Filters")
platform_options = sorted(df["platform"].unique())
platform = st.sidebar.radio("Select Platform", platform_options)

filtered = df[df["platform"] == platform]
orders_df = to_order_level(filtered)

# ---------------------------------------------------------------------------
# REPEAT CUSTOMER ACTIVITY
# A "record" here is one row of the file (one product line), not one order --
# this matches how repeat customers are counted in the cleaning report, so
# the two documents agree with each other.
# ---------------------------------------------------------------------------
with_customer = filtered.dropna(subset=["customer_key"])
records_per_customer = with_customer["customer_key"].value_counts()
repeat_records_per_customer = records_per_customer[records_per_customer > 1]
repeat_customers = int(repeat_records_per_customer.size)
repeat_records = int(repeat_records_per_customer.sum())

# ---------------------------------------------------------------------------
# HEADER + KPIs
# No revenue or average-order-value KPI is shown here. The three platforms
# calculate price differently and mix cancelled/pending/completed orders, so
# there isn't yet an agreed definition of a "real sale" to build one number
# on. The correct calculation already exists above (see to_order_level) --
# it's ready to switch on once that definition is confirmed.
# ---------------------------------------------------------------------------
st.title("Data Engineering Assessment")

total_orders = orders_df.shape[0]

col1, col2, col3 = st.columns(3)
col1.metric("Orders", f"{total_orders:,}", help="How many separate orders were placed.")
col2.metric("Repeat Customers", f"{repeat_customers:,}", help="Customers who ordered more than once.")
col3.metric(
    "Items From Repeat Customers",
    f"{repeat_records:,}",
    help="Total product lines bought by those repeat customers, combined.",
)

st.divider()

# ---------------------------------------------------------------------------
# ROW 1 — Orders per week
# ---------------------------------------------------------------------------
st.subheader("Orders Over Time")
st.caption("Number of orders placed each week.")
trend = (
    orders_df.dropna(subset=["date"])
    .assign(
        week_start=lambda d: d["date"].dt.to_period("W").dt.start_time,
        week_end=lambda d: d["date"].dt.to_period("W").dt.end_time.dt.normalize(),
    )
    .groupby(["week_start", "week_end", "platform"], as_index=False)["order_id"].nunique()
    .rename(columns={"order_id": "orders"})
)
trend["week_label"] = (
    trend["week_start"].dt.strftime("%b %d") + " - " + trend["week_end"].dt.strftime("%b %d")
)
trend = trend.sort_values("week_start")
fig = px.line(trend, x="week_label", y="orders", color="platform", markers=True)
fig.update_xaxes(categoryorder="array", categoryarray=trend["week_label"].unique())
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ROW 2 — Order status
# ---------------------------------------------------------------------------
st.subheader("Order Status")
status_counts = filtered.groupby(["platform", "status"], as_index=False)["order_id"].nunique()
status_counts.columns = ["platform", "status", "orders"]
fig = px.bar(status_counts, x="platform", y="orders", color="status", barmode="group")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ROW 3 — Repeat customers
# ---------------------------------------------------------------------------
repeat_activity = (
    with_customer[with_customer["customer_key"].isin(repeat_records_per_customer.index)]
    .groupby(["platform", "customer_key"], as_index=False)
    .agg(records=("customer_key", "size"), orders=("order_id", "nunique"))
    .sort_values("records", ascending=False)
)

if not repeat_activity.empty:
    st.subheader("Repeat Customers")
    st.caption("orders = separate purchases. records = product lines across those purchases.")
    rp1, rp2 = st.columns([1, 1])

    with rp1:
        st.dataframe(repeat_activity.head(15), use_container_width=True, hide_index=True)

    with rp2:
        top_repeat = repeat_activity.head(10)
        fig = px.bar(top_repeat, x="records", y="customer_key", color="platform", orientation="h")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ROW 4 — Why orders were cancelled or failed to deliver
# ---------------------------------------------------------------------------
REASON_LABELS = {
    "Shopee": "Why Orders Were Cancelled",
    "Lazada": "Why Deliveries Failed",
}

reasons = filtered["reason"].dropna()
reasons = reasons[reasons.astype(str).str.strip() != ""]
if not reasons.empty:
    st.subheader(REASON_LABELS.get(platform, "Reported Reasons"))
    # st.caption("These describe what happened to the order — not how the customer felt about it.")
    reason_counts = reasons.value_counts().reset_index().head(10)
    reason_counts.columns = ["reason", "orders"]
    fig = px.bar(reason_counts, x="orders", y="reason", orientation="h")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ROW 5 — Top products
# ---------------------------------------------------------------------------
prod_df = filtered[filtered["product"] != "N/A"]
if not prod_df.empty:
    st.subheader("Top 10 Products")
    st.caption("Ranked by number of orders — includes free/promotional items.")
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
# ROW 6 — Where orders are coming from
# ---------------------------------------------------------------------------
geo_df = filtered[filtered["city"] != "N/A"]
if not geo_df.empty:
    st.subheader(f"Orders by Location ({platform})")
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
# RAW DATA — the cleaned file, unmodified, for anyone who wants to check a
# number directly against the source.
# ---------------------------------------------------------------------------
with st.expander("View cleaned data"):
    raw_df = pd.read_excel(FILE_PATHS[platform], dtype=TEXT_COLUMNS.get(platform))
    st.dataframe(raw_df, use_container_width=True)
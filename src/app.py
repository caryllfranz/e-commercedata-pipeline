

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

# Identifiers must be read as text or Excel's numeric storage corrupts them:
# Website's ID comes back as float64 and renders as "3150.0", and Shopee's
# Zip Code loses its leading zeros ("0000" -> 0). Keys naming a column that
# does not exist in a file are ignored by read_excel, so this is safe to reuse.
TEXT_COLUMNS = {
    "Website": {"ID": str},
    "Shopee": {"Order ID": str, "Zip Code": str, "Phone Number": str},
    "Lazada": {"orderNumber": str, "orderItemId": str, "shippingPostCode": str},
}

# How each platform's revenue figure relates to a single line-item row:
#   "order_total" — the order's total is repeated on every row of that order,
#                   so one row already carries the full value (take it once).
#   "line_amount" — each row holds only its own line's amount, so the order
#                   total is the sum across the order's rows.
# Measured against the processed files: Website's Subtotal is identical across
# every row of a multi-line order, but Shopee's Total Buyer Payment is
# identical in only 27 of 108 multi-line orders, and Lazada carries a distinct
# orderItemId per row at ~3.3 items per order.
REVENUE_GRAIN = {
    "Website": "order_total",
    "Shopee": "line_amount",
    "Lazada": "line_amount",
}

# Misspellings in the source exports, mapped to the correct product name, so
# one SKU is not labelled two ways. "Nordic Spirirt Lush Tropics" appears in
# 26 Shopee and 60 Lazada rows and is the only spelling of that base product.
PRODUCT_NAME_FIXES = {
    "Nordic Spirirt Lush Tropics Nicotine Pouch": "Nordic Spirit Lush Tropics Nicotine Pouch",
}


# ---------------------------------------------------------------------------
# LOAD + STANDARDIZE
# Each platform has different column names, so we map each one into a common
# schema: platform, order_id, customer_key, date, product, revenue, status,
# reason, city
# ---------------------------------------------------------------------------
@st.cache_data
def load_website(path):
    df = pd.read_excel(path, dtype=TEXT_COLUMNS["Website"])

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
    unified = pd.concat(frames, ignore_index=True)
    unified["product"] = unified["product"].replace(PRODUCT_NAME_FIXES)
    return unified


def to_order_level(frame):
    """Collapse line-item rows to one row per order, with the true order total.

    The collapse is what makes the transaction count correct at the order
    grain. It also keeps `revenue` correct, which the dashboard does not
    currently present (see the note where this is called) but which must stay
    right for whenever it is: neither naive approach works across all three
    platforms. Summing "revenue" over every row multiplies an "order_total"
    platform by its item count, while deduplicating to one row per order
    throws away the remaining lines of a "line_amount" platform. So aggregate
    per platform according to its declared grain.
    """
    frame = frame.dropna(subset=["order_id"])
    if frame.empty:
        return frame

    parts = []
    for platform_name, group in frame.groupby("platform", sort=False):
        grain = REVENUE_GRAIN.get(platform_name)
        if grain is None:
            # An undeclared platform would otherwise be silently mis-summed.
            st.warning(
                f"No revenue grain declared for {platform_name}; summing its "
                "line amounts. Add it to REVENUE_GRAIN to be explicit."
            )
        totals = group.groupby("order_id", as_index=False)["revenue"].agg(
            "first" if grain == "order_total" else "sum"
        )
        # One representative row per order carries the non-revenue attributes.
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
# Row-grain on purpose, matching repeat_user_stats() in src/clean.py and the
# figures published in reports/cleaning_report.md: a "record" is one row of the
# export (a line item), not one order, and a repeat customer is one appearing
# in more than one record. Rows with no customer identifier are excluded rather
# than grouped together.
# ---------------------------------------------------------------------------
with_customer = filtered.dropna(subset=["customer_key"])
records_per_customer = with_customer["customer_key"].value_counts()
repeat_records_per_customer = records_per_customer[records_per_customer > 1]
repeat_customers = int(repeat_records_per_customer.size)
repeat_records = int(repeat_records_per_customer.sum())
repeat_share = 100 * repeat_records / len(with_customer) if len(with_customer) else 0

# ---------------------------------------------------------------------------
# HEADER + KPIs
# ---------------------------------------------------------------------------
st.title("Data Engineering Assessment")


total_orders = orders_df.shape[0]

col1, col2, col3 = st.columns(3)
col1.metric("Transactions", f"{total_orders:,}", help="Distinct orders, counted at each platform's order grain.")
col2.metric("Repeat Customers", f"{repeat_customers:,}", help="Customers appearing in more than one record.")
col3.metric(
    "Records from Repeat Customers",
    f"{repeat_records:,}",
    help="Line-item records belonging to those customers.",
)
# st.caption(
#     f"{repeat_share:.1f}% of {platform} records with a customer identifier come from repeat "
#     "customers. Records are line items, not orders — the same grain as the cleaning report."
# )

st.divider()

# ---------------------------------------------------------------------------
# ROW 1 — Transaction volume trend
# ---------------------------------------------------------------------------
st.subheader("Transaction Volume Trend")
st.caption("Distinct orders per week.")
trend = (
    orders_df.dropna(subset=["date"])
    .assign(
        week_start=lambda d: d["date"].dt.to_period("W").dt.start_time,
        week_end=lambda d: d["date"].dt.to_period("W").dt.end_time.dt.normalize(),
    )
    .groupby(["week_start", "week_end", "platform"], as_index=False)["order_id"].nunique()
    .rename(columns={"order_id": "transactions"})
)
trend["week_label"] = (
    trend["week_start"].dt.strftime("%b %d") + " - " + trend["week_end"].dt.strftime("%b %d")
)
trend = trend.sort_values("week_start")
fig = px.line(trend, x="week_label", y="transactions", color="platform", markers=True)
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
# ROW 3 — Repeat customer activity
# Same row-grain definition as the KPI above, so the table reconciles to it.
# Orders are shown next to records so the two grains stay visible and neither
# column can be mistaken for the other.
# ---------------------------------------------------------------------------
repeat_activity = (
    with_customer[with_customer["customer_key"].isin(repeat_records_per_customer.index)]
    .groupby(["platform", "customer_key"], as_index=False)
    .agg(records=("customer_key", "size"), orders=("order_id", "nunique"))
    .sort_values("records", ascending=False)
)

if not repeat_activity.empty:
    st.subheader("Repeat Customer Activity")
    # st.caption(
    #     "Customers appearing in more than one record — a loyalty and retention signal. "
    #     "`records` counts line items; `orders` counts distinct transactions."
    # )
    rp1, rp2 = st.columns([1, 1])

    with rp1:
        st.dataframe(repeat_activity.head(15), use_container_width=True, hide_index=True)

    with rp2:
        top_repeat = repeat_activity.head(10)
        fig = px.bar(top_repeat, x="records", y="customer_key", color="platform", orientation="h")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ROW 4 — Operational issues
# Shopee contributes cancellation reasons and Lazada failed-delivery reasons.
# These are operational indicators only: they describe what happened to an
# order, and are deliberately NOT presented as customer sentiment, which would
# require review text this data does not contain.
# ---------------------------------------------------------------------------
REASON_LABELS = {
    "Shopee": "Cancellation Reasons",
    "Lazada": "Failed-Delivery Reasons",
}

reasons = filtered["reason"].dropna()
reasons = reasons[reasons.astype(str).str.strip() != ""]
if not reasons.empty:
    st.subheader(f"Operational Issues — {REASON_LABELS.get(platform, 'Reported Reasons')}")
    # st.caption(
    #     "Operational indicators describing what happened to an order. "
    #     "Not a sentiment measure — this data contains no customer review text."
    # )
    reason_counts = reasons.value_counts().reset_index().head(10)
    reason_counts.columns = ["reason", "records"]
    fig = px.bar(reason_counts, x="records", y="reason", orientation="h")
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
    raw_df = pd.read_excel(FILE_PATHS[platform], dtype=TEXT_COLUMNS.get(platform))
    st.dataframe(raw_df, use_container_width=True)
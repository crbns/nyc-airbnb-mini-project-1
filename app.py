"""NYC Airbnb dashboard. Run locally with: uv run streamlit run app.py"""

from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

DATASET = "dgomonov/new-york-city-airbnb-open-data"
LOCAL_CSV = Path(__file__).parent / "data" / "AB_NYC_2019.csv"
RANDOM_STATE = 42
TARGET = "price"
CATEGORICAL = ["neighbourhood_group", "room_type"]
NUMERIC = ["minimum_nights", "number_of_reviews", "availability_365", "calculated_host_listings_count"]

st.set_page_config(page_title="NYC Airbnb Prices", layout="wide")
sns.set_theme()


def load_data() -> pd.DataFrame:
    csv = LOCAL_CSV if LOCAL_CSV.exists() else Path(kagglehub.dataset_download(DATASET)) / "AB_NYC_2019.csv"
    df = pd.read_csv(csv)

    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df[["name", "host_name"]] = df[["name", "host_name"]].fillna("Unknown")
    df = df[df["price"].between(1, 1000) & (df["minimum_nights"] <= 365)].copy()

    df["last_review"] = pd.to_datetime(df["last_review"])
    df["days_since_review"] = (df["last_review"].max() - df["last_review"]).dt.days
    df["log_price"] = np.log(df["price"])
    return df


def fit_models(df: pd.DataFrame) -> dict:
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    categories = {c: sorted(train[c].unique()) for c in CATEGORICAL}

    def make_features(data):
        cats = pd.DataFrame({c: pd.Categorical(data[c], categories=categories[c]) for c in CATEGORICAL}, index=data.index)
        return pd.concat([data[NUMERIC], pd.get_dummies(cats, drop_first=True, dtype=int)], axis=1)

    X_train, X_test = make_features(train), make_features(test)
    y_train, y_test = train[TARGET], test[TARGET]

    ols_numeric = LinearRegression().fit(X_train[NUMERIC], y_train)
    ols_full = LinearRegression().fit(X_train, y_train)
    preds = {
        "Constant (train mean)": np.full(len(y_test), y_train.mean()),
        "OLS: numeric only": ols_numeric.predict(X_test[NUMERIC]),
        "OLS: numeric + one-hot": ols_full.predict(X_test),
    }
    results = pd.DataFrame(
        [{"model": name, "RMSE": root_mean_squared_error(y_test, p), "MAE": mean_absolute_error(y_test, p)} for name, p in preds.items()]
    ).round(2)
    return {
        "results": results,
        "coefs": pd.Series(ols_full.coef_, index=X_train.columns).round(2),
        "intercept": ols_full.intercept_,
        "reference": {c: cats[0] for c, cats in categories.items()},
        "test": test.assign(predicted=preds["OLS: numeric + one-hot"]),
        "n_train": len(train),
    }


df = load_data()

st.title("NYC Airbnb Prices")
st.markdown("**Central question:** ✍️ _e.g. How much do borough and room type drive the nightly price of an NYC Airbnb?_")

st.sidebar.header("Filters")
boroughs = st.sidebar.multiselect(
    "Borough", sorted(df["neighbourhood_group"].unique()), default=sorted(df["neighbourhood_group"].unique())
)
room_type = st.sidebar.selectbox("Room type", ["All", *sorted(df["room_type"].unique())])
price_range = st.sidebar.slider("Nightly price ($)", int(df["price"].min()), int(df["price"].max()), (int(df["price"].min()), 500))

view = df[df["neighbourhood_group"].isin(boroughs) & df["price"].between(*price_range)]
if room_type != "All":
    view = view[view["room_type"] == room_type]

if view.empty:
    st.warning("No listings match these filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", f"{len(view):,}")
c2.metric("Median price", f"${view['price'].median():,.0f}")
c3.metric("Mean price", f"${view['price'].mean():,.0f}")
c4.metric("Median reviews", f"{view['number_of_reviews'].median():,.0f}")

with st.expander("About the data: Faithfulness & Granularity"):
    st.markdown(
        """
- **Each row is:** ✍️ …
- **Scope:** ✍️ …
- **Strange values / duplicates we handled:** ✍️ …
"""
    )

st.header("Price by borough and room type")
st.dataframe(view.groupby(["neighbourhood_group", "room_type"])["price"].agg(["count", "median", "mean"]).round(1))

st.header("Explore")
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=view, x="neighbourhood_group", hue="room_type", ax=ax)
    ax.set(title="Listings by borough and room type", xlabel="Borough", ylabel="Listings")
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(view["price"], bins=50, ax=ax)
    ax.set(title="Nightly price distribution", xlabel="Price ($)")
    st.pyplot(fig)

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(view["longitude"], view["latitude"], c=view["log_price"], s=2, alpha=0.5, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="log(price)")
    ax.set(title="Where are the expensive listings?", xlabel="Longitude", ylabel="Latitude")
    st.pyplot(fig)

with right:
    num_cols = ["price", "minimum_nights", "number_of_reviews", "reviews_per_month", "availability_365"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(view[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation between numeric features")
    st.pyplot(fig)

st.markdown("✍️ _One sentence per chart: what should the viewer take away?_")

st.header("Predicting price")
model = fit_models(df)

st.markdown(
    f"80/20 random train/test split ({model['n_train']:,} train rows). "
    f"Reference levels (dropped): **{model['reference']['neighbourhood_group']}** borough, "
    f"**{model['reference']['room_type']}** room type."
)

left, right = st.columns(2)
with left:
    st.subheader("Test error")
    st.dataframe(model["results"], hide_index=True)
    st.subheader("OLS coefficients")
    st.dataframe(model["coefs"].rename("coefficient"))

with right:
    test = model["test"]
    color_by = st.radio("Color predictions by", CATEGORICAL, horizontal=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=test, x=TARGET, y="predicted", hue=color_by, s=8, alpha=0.4, ax=ax)
    lims = [min(test[TARGET].min(), test["predicted"].min()), max(test[TARGET].max(), test["predicted"].max())]
    ax.plot(lims, lims, "r--", label="perfect prediction")
    ax.set(xlabel=f"Actual {TARGET}", ylabel=f"Predicted {TARGET}", title="Predicted vs. actual (test set)")
    ax.legend(fontsize="small")
    st.pyplot(fig)

st.markdown(
    """
- **Coefficient interpretation:** ✍️ "Listings in ___ are predicted to cost about $___ more than listings in ___, holding other features constant."
- **Did one-hot encoding help?** ✍️ …
- **Evaluation observation:** ✍️ …
"""
)

st.header("Takeaway")
st.markdown("✍️ _The one clear finding a first-time visitor should walk away with._")

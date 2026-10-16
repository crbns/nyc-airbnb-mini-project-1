from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLEAN_CSV = Path(__file__).parent / "data" / "airbnb_clean.csv"
RANDOM_STATE = 42
TARGET = "price"
CATEGORICAL = ["neighbourhood_group", "room_type"]
NUMERIC = ["minimum_nights", "availability_365", "calculated_host_listings_count"]
PRICER_CATEGORICAL = ["neighbourhood", "room_type"]
PRICER_NUMERIC = ["availability_365", "minimum_nights"]
MIN_LISTINGS = 20

st.set_page_config(page_title="NYC Airbnb Prices", layout="wide")
sns.set_theme()


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


def fit_pricer(df: pd.DataFrame) -> tuple[Pipeline, float]:
    X, y = df[PRICER_CATEGORICAL + PRICER_NUMERIC], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    encoder = OneHotEncoder(drop="first", min_frequency=MIN_LISTINGS, handle_unknown="infrequent_if_exist")
    pipeline = Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", StandardScaler(), PRICER_NUMERIC),
            ("cat", encoder, PRICER_CATEGORICAL),
        ])),
        ("model", LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline, root_mean_squared_error(y_test, pipeline.predict(X_test))


if not CLEAN_CSV.exists():
    st.error(f"{CLEAN_CSV.name} not found. Run Section 1 of main.ipynb to create it.")
    st.stop()

df = pd.read_csv(CLEAN_CSV)

st.title("NYC Airbnb Prices")
st.markdown("**Central question:** How much do borough and room type drive the nightly price of an NYC Airbnb?")

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
c4.metric("Median 7-night stay", f"${view['price_per_week'].median():,.0f}")

with st.expander("About the data: Faithfulness & Granularity"):
    st.markdown(
        """
- **Each row is:** one Airbnb listing (an entire home, private room, or shared room) as it appeared on the site in a single snapshot around July 2019. It is not a booking or a host: 37,457 hosts own the 48,895 raw listings, and one operator runs 327 of them.
- **Scope:** all five NYC boroughs and 221 neighbourhoods at one point in time, so the data can't show seasonal or year-over-year changes. `price` is the host's asking nightly price, not what guests actually paid (no cleaning fees or discounts). After cleaning, the dashboard covers 47,924 short-term listings (minimum stay of 30 nights or fewer) priced $1–$1,000.
- **Strange values / duplicates we handled:** we removed 11 listings priced at $0, 239 listings over $1,000/night (up to $10,000), and 747 listings requiring stays over 30 nights (up to 1,250), which are monthly rentals rather than short stays. There were no duplicate listings. We dropped the review columns: 20.6% of listings have never been reviewed, and those listings are pricier on average, so dropping the *rows* instead would have biased prices downward.
"""
    )

st.header("Price by borough and room type")
st.dataframe(view.groupby(["neighbourhood_group", "room_type"])["price"].agg(["count", "median", "mean"]).round(1))
st.caption(
    "In every borough an entire home has about twice the median price of a private room, and Manhattan is the most "
    "expensive borough for every room type. Medians are the fairer summary here: a few expensive listings pull every mean up."
)

st.header("Explore")
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=view, x="neighbourhood_group", hue="room_type", ax=ax)
    ax.set(title="Listings by borough and room type", xlabel="Borough", ylabel="Listings")
    st.pyplot(fig)
    st.caption(
        "Manhattan and Brooklyn hold about 85% of all listings. Entire homes and private rooms split the market "
        "almost evenly; shared rooms are only about 2%."
    )

with right:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(view["price"], bins=50, ax=ax)
    ax.set(title="Nightly price distribution", xlabel="Price ($)")
    st.pyplot(fig)
    st.caption(
        "Prices are strongly right-skewed: the middle half of listings costs $69–$175 a night, but a long tail "
        "stretches to $1,000, which pulls the mean above the median."
    )

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(view["longitude"], view["latitude"], c=view["log_price"], s=2, alpha=0.5, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="log(price)")
    ax.set(title="Where are the expensive listings?", xlabel="Longitude", ylabel="Latitude")
    st.pyplot(fig)
    st.caption(
        "The priciest listings cluster in Manhattan, especially Midtown, Hell's Kitchen and the Upper West Side, "
        "plus Williamsburg in Brooklyn. Location is a big part of the price story."
    )

with right:
    num_cols = ["price", "neighbourhood_median_price", "minimum_nights", "calculated_host_listings_count", "availability_365"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(view[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation between numeric features")
    st.pyplot(fig)
    st.caption(
        "Minimum nights, availability and host listing count barely relate to price (|r| ≤ 0.13). Only the "
        "neighbourhood median price does, which again points to location and room type as the real drivers."
    )

st.header("Predicting price")
model = fit_models(df)
coefs, reference = model["coefs"], model["reference"]
results = model["results"].set_index("model")

st.markdown(
    f"80/20 random train/test split ({model['n_train']:,} train rows). "
    f"Reference levels (dropped): **{reference['neighbourhood_group']}** borough, "
    f"**{reference['room_type']}** room type."
)

left, right = st.columns(2)
with left:
    st.subheader("Test error")
    st.dataframe(model["results"], hide_index=True)
    st.subheader("OLS coefficients")
    st.dataframe(coefs.rename("coefficient"))

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

manhattan = coefs["neighbourhood_group_Manhattan"]
private = coefs["room_type_Private room"]
rmse_numeric, rmse_full = results.loc["OLS: numeric only", "RMSE"], results.loc["OLS: numeric + one-hot", "RMSE"]
mae_numeric, mae_full = results.loc["OLS: numeric only", "MAE"], results.loc["OLS: numeric + one-hot", "MAE"]
top_fifth = test[test[TARGET] >= test[TARGET].quantile(0.8)]
st.markdown(
    f"""
- **Coefficient interpretation:** listings in Manhattan are predicted to cost about **${manhattan:,.0f} more** per night than listings in the {reference['neighbourhood_group']}, holding the other features constant. A private room is predicted to cost about **${abs(private):,.0f} less** than an {reference['room_type'].lower()}, so room type moves price more than any borough does.
- **Did one-hot encoding help?** Yes. Without borough and room type, the model barely beats always guessing the average (RMSE ${rmse_numeric:,.2f} vs. ${results.loc['Constant (train mean)', 'RMSE']:,.2f}). Adding them cuts RMSE to ${rmse_full:,.2f} ({(rmse_full - rmse_numeric) / rmse_numeric:.0%}) and MAE from ${mae_numeric:,.2f} to ${mae_full:,.2f} ({(mae_full - mae_numeric) / mae_numeric:.0%}).
- **Evaluation observation:** the model **underpredicts at the high end**. For the most expensive fifth of test listings, it predicts about ${(top_fifth[TARGET] - top_fifth['predicted']).mean():,.0f} below the actual price on average, and its highest prediction is only ${test['predicted'].max():,.0f}, while real prices reach ${test[TARGET].max():,.0f}. It also overpredicts the cheapest listings.
"""
)

st.header("What should I charge?")
st.markdown(
    "Describe your listing and a linear regression model suggests a nightly price based on similar NYC listings. "
    "The model's pipeline standardises the numeric inputs and one-hot encodes neighbourhood and room type."
)
pricer, pricer_rmse = fit_pricer(df)
rare_neighbourhoods = set(pricer.named_steps["preprocess"].named_transformers_["cat"].infrequent_categories_[0])
neighbourhoods = df.groupby("neighbourhood")["neighbourhood_group"].first().sort_index()

inputs, output = st.columns(2)
with inputs:
    my_neighbourhood = st.selectbox(
        "Neighbourhood",
        neighbourhoods.index,
        index=neighbourhoods.index.get_loc(df["neighbourhood"].mode()[0]),
        format_func=lambda n: f"{n} ({neighbourhoods[n]})",
    )
    my_room_type = st.selectbox("Room type", sorted(df["room_type"].unique()), key="pricer_room_type")
    my_availability = st.slider("Nights available per year", 0, 365, 180)
    my_min_nights = st.slider("Minimum nights", 1, 30, 2)

with output:
    listing = pd.DataFrame([{
        "neighbourhood": my_neighbourhood,
        "room_type": my_room_type,
        "availability_365": my_availability,
        "minimum_nights": my_min_nights,
    }])
    suggested = pricer.predict(listing)[0]
    m1, m2 = st.columns(2)
    m1.metric("Suggested nightly price", f"${suggested:,.0f}")
    m2.metric("7-night stay", f"${suggested * 7:,.0f}")
    if suggested <= 0:
        st.warning("The model predicts no positive price for this combination; there's too little similar data to trust it.")
    if my_neighbourhood in rare_neighbourhoods:
        st.info(
            f"{my_neighbourhood} has fewer than {MIN_LISTINGS} listings in the training data, so the model groups it "
            "with other small neighbourhoods instead of estimating it on its own."
        )

    similar = df.loc[(df["neighbourhood"] == my_neighbourhood) & (df["room_type"] == my_room_type), "price"]
    if similar.empty:
        st.caption(f"There are no {my_room_type.lower()} listings in {my_neighbourhood} in the data, so treat this estimate with caution.")
    else:
        st.caption(
            f"For comparison: {len(similar):,} {my_room_type.lower()} listing(s) in {my_neighbourhood}, "
            f"median ${similar.median():,.0f} a night."
        )
    st.caption(
        f"Typical error is about ±${pricer_rmse:,.0f} (test RMSE). Like the model above, it underprices high-end "
        "listings, and neighbourhoods with only a handful of listings give noisy estimates."
    )

st.header("Takeaway")
st.markdown(
    f"**Room type is the biggest lever on an NYC Airbnb's price, and location is the second.** An entire home costs "
    f"roughly twice as much as a private room in every borough. For the same room type, Manhattan adds about "
    f"${manhattan:,.0f} a night over the Bronx. Details like minimum stay and availability barely move the price."
)

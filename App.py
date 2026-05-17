import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Munich Rental Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Load and train model ──────────────────────────────────────
@st.cache_resource
def train_model():
    DATA_PROC = Path("data/processed")
    listings  = pd.read_csv(DATA_PROC / "listings_clean.csv", low_memory=False)

    listings["room_type_enc"]       = listings["room_type"].astype("category").cat.codes
    listings["neighbourhood_enc"]   = listings["neighbourhood_cleansed"].astype("category").cat.codes

    FEATURES = [
        "accommodates", "bedrooms", "beds", "bathrooms",
        "minimum_nights", "number_of_reviews", "review_scores_rating",
        "review_scores_cleanliness", "review_scores_location",
        "review_scores_value", "calculated_host_listings_count",
        "availability_365", "room_type_enc", "neighbourhood_enc"
    ]

    model_df = listings[FEATURES + ["price"]].dropna()
    X = model_df[FEATURES]
    y = np.log1p(model_df["price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    room_types     = listings["room_type"].dropna().unique().tolist()
    neighbourhoods = listings["neighbourhood_cleansed"].dropna().unique().tolist()
    room_cat       = listings["room_type"].astype("category").cat.categories.tolist()
    nb_cat         = listings["neighbourhood_cleansed"].astype("category").cat.categories.tolist()

    return model, sorted(room_types), sorted(neighbourhoods), room_cat, nb_cat

model, room_types, neighbourhoods, room_cat, nb_cat = train_model()

# ── UI ────────────────────────────────────────────────────────
st.title("🏠 Munich Vacation Rental Price Predictor")
st.markdown("Enter your listing details below to get an estimated nightly price.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    room_type     = st.selectbox("Room Type", room_types)
    neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)
    accommodates  = st.slider("Guests", 1, 16, 2)
    bedrooms      = st.slider("Bedrooms", 0, 10, 1)

with col2:
    beds          = st.slider("Beds", 1, 16, 1)
    bathrooms     = st.slider("Bathrooms", 0.0, 5.0, 1.0, step=0.5)
    minimum_nights= st.slider("Minimum Nights", 1, 30, 2)
    availability  = st.slider("Availability (days/year)", 0, 365, 180)

st.divider()
col3, col4 = st.columns(2)

with col3:
    review_rating       = st.slider("Review Score", 1.0, 5.0, 4.8, step=0.1)
    review_cleanliness  = st.slider("Cleanliness Score", 1.0, 5.0, 4.8, step=0.1)

with col4:
    review_location     = st.slider("Location Score", 1.0, 5.0, 4.8, step=0.1)
    review_value        = st.slider("Value Score", 1.0, 5.0, 4.7, step=0.1)

number_of_reviews   = st.number_input("Number of Reviews", 0, 1000, 10)
host_listings       = st.number_input("Host Total Listings", 1, 100, 1)

st.divider()

# ── Predict ───────────────────────────────────────────────────
if st.button("💶 Predict Price", use_container_width=True):
    room_enc = room_cat.index(room_type) if room_type in room_cat else 0
    nb_enc   = nb_cat.index(neighbourhood) if neighbourhood in nb_cat else 0

    features = pd.DataFrame([{
        "accommodates"                  : accommodates,
        "bedrooms"                      : bedrooms,
        "beds"                          : beds,
        "bathrooms"                     : bathrooms,
        "minimum_nights"                : minimum_nights,
        "number_of_reviews"             : number_of_reviews,
        "review_scores_rating"          : review_rating,
        "review_scores_cleanliness"     : review_cleanliness,
        "review_scores_location"        : review_location,
        "review_scores_value"           : review_value,
        "calculated_host_listings_count": host_listings,
        "availability_365"              : availability,
        "room_type_enc"                 : room_enc,
        "neighbourhood_enc"             : nb_enc,
    }])

    prediction = np.expm1(model.predict(features)[0])

    st.success(f"### Estimated Nightly Price: €{prediction:.0f}")
    st.caption("Based on Random Forest model trained on 5,487 Munich Airbnb listings.")

    # Show feature breakdown
    st.markdown("**Key factors in this prediction:**")
    col5, col6, col7 = st.columns(3)
    col5.metric("Room Type",      room_type)
    col6.metric("Neighbourhood",  neighbourhood)
    col7.metric("Accommodates",   f"{accommodates} guests")
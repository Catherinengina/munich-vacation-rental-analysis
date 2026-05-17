from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# ── Paths ─────────────────────────────────────────────────────
from pathlib import Path
DATA_RAW  = Path("/opt/airflow/data/raw")
DATA_PROC = Path("/opt/airflow/data/processed")


# ══════════════════════════════════════════════════════════════
# TASK 1 — EXTRACT
# ══════════════════════════════════════════════════════════════
def extract():
    import os
    from pathlib import Path

    DATA_RAW  = Path("/opt/airflow/data/raw")
    DATA_PROC = Path("/opt/airflow/data/processed")

    folders = ["data/raw", "data/processed", "outputs"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("EXTRACTION COMPLETE!")
    print("-" * 40)
    for f in sorted(DATA_RAW.iterdir()):
        print(f.name)
    print(f"Total files: {len(list(DATA_RAW.iterdir()))}")
    print("✅ Extract done!")


# ══════════════════════════════════════════════════════════════
# TASK 2 — TRANSFORM
# ══════════════════════════════════════════════════════════════
def transform():
    import pandas as pd
    import numpy as np
    from pathlib import Path

    DATA_RAW  = Path("/opt/airflow/data/raw")
    DATA_PROC = Path("/opt/airflow/data/processed")
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    # Load listings
    listings = pd.read_csv(DATA_RAW / "listings.csv", low_memory=False)
    print(f"Loaded: {listings.shape}")

    # Drop useless columns
    cols_to_drop = [
        "neighbourhood_group_cleansed",
        "calendar_updated",
        "license",
        "host_neighbourhood",
        "neighborhood_overview",
        "neighbourhood",
        "host_about",
        "estimated_revenue_l365d",
        "host_location",
        "host_response_time",
        "host_acceptance_rate",
        "description",
        "has_availability",
        "bathrooms_text",
        "maximum_maximum_nights",
        "minimum_maximum_nights",
        "maximum_minimum_nights",
        "minimum_minimum_nights",
    ]
    cols_to_drop = [c for c in cols_to_drop if c in listings.columns]
    listings.drop(columns=cols_to_drop, inplace=True)

    # Clean price
    listings["price"] = (listings["price"]
                         .astype(str)
                         .str.replace("$", "", regex=False)
                         .str.replace(",", "", regex=False)
                         .pipe(pd.to_numeric, errors="coerce"))

    # Drop missing prices
    listings = listings.dropna(subset=["price"])

    # Remove outliers
    Q1 = listings["price"].quantile(0.01)
    Q3 = listings["price"].quantile(0.99)
    listings = listings[listings["price"].between(Q1, Q3)]

    # Fill review scores with median
    review_cols = [c for c in listings.columns if c.startswith("review_scores")]
    for col in review_cols:
        listings[col] = listings[col].fillna(listings[col].median())

    # Fill other columns
    listings["bedrooms"]          = listings["bedrooms"].fillna(listings["bedrooms"].median())
    listings["beds"]              = listings["beds"].fillna(listings["beds"].median())
    listings["bathrooms"]         = listings["bathrooms"].fillna(listings["bathrooms"].median())
    listings["host_is_superhost"] = listings["host_is_superhost"].fillna("f")
    listings["host_response_rate"]= listings["host_response_rate"].fillna("unknown")
    listings["reviews_per_month"] = listings["reviews_per_month"].fillna(0)
    listings["first_review"]      = listings["first_review"].fillna("unknown")
    listings["last_review"]       = listings["last_review"].fillna("unknown")

    # Drop rows where host_since is missing
    listings = listings.dropna(subset=["host_since"])

    # Save cleaned data
    listings.to_csv(DATA_PROC / "listings_clean.csv", index=False)
    print(f"✅ Transform complete! {listings.shape[0]:,} clean rows saved.")


# ══════════════════════════════════════════════════════════════
# TASK 3 — LOAD
# ══════════════════════════════════════════════════════════════
def load():
    import pandas as pd
    import sqlite3
    import mysql.connector
    from pathlib import Path

    DATA_PROC = Path("/opt/airflow/data/processed")
    DB_PATH   = DATA_PROC / "munich_airbnb.db"

    listings = pd.read_csv(DATA_PROC / "listings_clean.csv", low_memory=False)

    # ── Load into SQLite ──────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    listings.to_sql("listings", conn, if_exists="replace", index=False)
    count = pd.read_sql("SELECT COUNT(*) AS total FROM listings", conn)
    print(f"✅ SQLite: {count['total'][0]:,} rows loaded!")
    conn.close()

    # ── Load into MySQL ───────────────────────────────────────
    conn_mysql = mysql.connector.connect(
        host="host.docker.internal",
        port=3306,
        user="root",
        password="Catherine_08051999",
        database="munich_airbnb"
    )

    cursor = conn_mysql.cursor()
    cursor.execute("DROP TABLE IF EXISTS listings")
    cursor.execute("""
        CREATE TABLE listings (
            id BIGINT,
            name TEXT,
            neighbourhood_cleansed VARCHAR(255),
            room_type VARCHAR(100),
            price FLOAT,
            minimum_nights INT,
            number_of_reviews INT,
            review_scores_rating FLOAT,
            bedrooms FLOAT,
            beds FLOAT,
            accommodates INT
        )
    """)

    for _, row in listings[["id","name","neighbourhood_cleansed",
                              "room_type","price","minimum_nights",
                              "number_of_reviews","review_scores_rating",
                              "bedrooms","beds","accommodates"]].iterrows():
        cursor.execute("""
            INSERT INTO listings VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, tuple(row))

    conn_mysql.commit()
    cursor.close()
    conn_mysql.close()
    print(f"✅ MySQL: {listings.shape[0]:,} rows loaded!")


# ══════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════
default_args = {
    "owner"      : "catherine",
    "retries"    : 1,
    "retry_delay": __import__("datetime").timedelta(minutes=5),
}

with DAG(
    dag_id="munich_airbnb_etl",
    default_args=default_args,
    description="Automated ETL pipeline for Munich Airbnb data",
    schedule_interval="@daily",
    start_date=__import__("datetime").datetime(2026, 1, 1),
    catchup=False,
    tags=["etl", "airbnb", "munich"],
) as dag:

    t1 = PythonOperator(task_id="extract",   python_callable=extract)
    t2 = PythonOperator(task_id="transform", python_callable=transform)
    t3 = PythonOperator(task_id="load",      python_callable=load)

    t1 >> t2 >> t3

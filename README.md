# Munich Vacation Rental Market Analysis- AIRBNB

A full end-to-end data analytics project analysing the Munich 
short-term rental market using real Airbnb data from Inside Airbnb.

## Project Structure

| Phase | File | Description |
|-------|------|-------------|
| Extract | `Extraction.ipynb` | Download & decompress raw data |
| Transform | `Transform.ipynb` | Clean & process listings data |
| Load | `Load.ipynb` | Load into SQLite & MySQL |
| Analysis | `Analysis.ipynb` | SQL queries & KPI visualisations |
| Model | `Prediction_Model.ipynb` | Random Forest pricing model |
| Automation | `airflow/dags/ETL_PIPELINE.py` | Automated daily pipeline |

## Key Findings

- **5,487 listings** across **25 neighbourhoods**
- **Average price:** €239/night
- **Most expensive neighbourhood:** Altstadt-Lehel (€391/night)
- **Cheapest neighbourhood:** Aubing-Lochhausen-Langwied (€166/night)
- **Average review score:** 4.81/5
- **Dominant listing type:** Entire home/apt (3,663 listings, avg €280/night)
- **Casual hosts** (1 listing) get the best reviews (4.84) vs commercial (4.61)
- **Top price driver:** Number of guests accommodated

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, matplotlib, seaborn
- **SQL** — SQLite, MySQL
- **Machine Learning** — Random Forest Regressor
- **Automation** — Apache Airflow + Docker
- **Version Control** — Git & GitHub

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## Data Source

[Inside Airbnb](http://insideairbnb.com/get-the-data/) — 
Munich dataset, snapshot 2024-12-25


# SudEnergy Luxembourg — Weather-Driven Energy Demand Forecasting

![Databricks](https://img.shields.io/badge/Databricks-Community%20Edition-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.0-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-Facebook%2FMeta-0866FF?style=for-the-badge&logo=meta&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-Delta%20Lake-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)

> **Phase 1 — Proof of Concept**  
> Built entirely on **Databricks** · Tracked with **MLflow 3** · Forecasted with **AutoML**  
> Author: Elham Khorasani · aeonic-intelligence.de · May 2026

---

## Project Overview

This project demonstrates how **Databricks** — the leading unified data and AI platform — can be used to build an end-to-end weather-driven energy demand forecasting pipeline for **SudEnergy Luxembourg**.

The entire project runs on **Databricks**, leveraging:
- **Apache Spark** for distributed data processing
- **Delta Lake** for reliable, versioned data storage
- **Databricks AutoML** for automated model selection and training
- **MLflow 3** for full experiment tracking, model registry, and deployment
- **Databricks Serverless Compute** — zero infrastructure management

Since SudEnergy has not yet provided actual energy consumption (kWh) data, this phase uses **weather variables as energy demand proxies** — particularly **Heating Degree Days (HDD)** and **Cooling Degree Days (CDD)**, which are industry-standard metrics used by European utility companies to model energy demand.

---

## Objectives

- Ingest 5 years of historical hourly weather data for Luxembourg City using the Open-Meteo API
- Build a production-grade data pipeline on **Databricks** using the **Medallion Architecture** (Bronze → Silver → Gold)
- Perform exploratory data analysis and validate data quality
- Use **Databricks AutoML** to automatically train and compare multiple forecasting models
- Track all experiments with **MLflow 3** — parameters, metrics, models, and artifacts
- Register the best model in the **MLflow Model Registry**
- Produce a 1-year temperature forecast as an energy demand proxy
- Demonstrate the full power of the **Databricks platform** to SudEnergy

---

## Databricks Architecture

```
                        ┌─────────────────────────────────────────┐
                        │         DATABRICKS PLATFORM             │
                        │                                         │
  Open-Meteo API ──────▶│  MEDALLION ARCHITECTURE (Delta Lake)    │
                        │                                         │
                        │  BRONZE          SILVER        GOLD     │
                        │  ─────────   ──────────────   ──────    │
                        │  Raw hourly  Daily features   Forecast  │
                        │  43,824 rows  1,827 rows      365 rows  │
                        │                                         │
                        │         Apache Spark (Serverless)       │
                        └──────────────────┬──────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────┐
                        │           DATABRICKS AutoML             │
                        │                                         │
                        │  Tested models:                         │
                        │  ✓ Prophet      ✓ DeepAR                │
                        │  ✓ ARIMA        ✓ AutoARIMA             │
                        │                                         │
                        │  Winner: Prophet (MAPE: 0.4728)         │
                        └──────────────────┬──────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────┐
                        │              MLflow 3                   │
                        │                                         │
                        │  • Experiment tracking                  │
                        │  • Model registry (v1)                  │
                        │  • Artifact logging                     │
                        │  • Batch inference                      │
                        │                                         │
                        │  Model: sudenergy_lux_temp_forecast     │
                        └─────────────────────────────────────────┘
```

---

## Why Databricks?

| Feature | How We Used It |
|---------|---------------|
| **Serverless Compute** | Zero cluster management — notebooks run instantly |
| **Delta Lake** | Reliable, versioned storage for Bronze/Silver/Gold tables |
| **Unity Catalog** | Centralized data governance and table management |
| **AutoML** | Automated model training — tested 10+ model configurations automatically |
| **MLflow 3** | Full experiment tracking, model versioning, and registry |
| **Apache Spark** | Distributed processing of 43,824 hourly records |
| **Databricks Notebooks** | Interactive, collaborative development environment |
| **Git Integration** | Native GitHub connection for version control |

---

## 🤖 Databricks AutoML — Model Comparison

Databricks AutoML automatically tested multiple forecasting models and tracked every run in **MLflow**. Here is the full leaderboard:

| Rank | Model | MAPE ↓ | Training Time | Parameters |
|------|-------|--------|---------------|------------|
| 🥇 1 | **Prophet** | **0.4728** | 50s | US holidays, CI=0.8 |
| 🥈 2 | Prophet | 0.4754 | 46s | No holidays, CI=0.95 |
| 🥉 3 | Prophet | 0.4756 | 48s | No holidays, CI=0.8 |
| 4 | Prophet | 0.4815 | 47s | US holidays, CI=0.95 |
| 5 | DeepAR | 0.4797 | 5.6 min | batch=64, context=365 |
| 6 | DeepAR | 0.5531 | 5.6 min | batch=32, context=730 |
| 7 | DeepAR | 0.6136 | 7.4 min | batch=32, context=730 |
| 8 | ARIMA | 0.6884 | 1.2 min | Auto-configured |

> **MAPE** = Mean Absolute Percentage Error. Lower is better.  
> All runs automatically logged to **MLflow** with full parameter and metric tracking.

### Why Prophet Won
- Luxembourg temperature has strong, consistent **yearly seasonality** — Prophet's core strength
- Dataset size (1,827 rows) is ideal for Prophet — too small for DeepAR to outperform
- ARIMA struggles with **multiple seasonality** (weekly + yearly) — Prophet handles both natively
- Prophet is **interpretable** — trend, seasonality, and holiday components can be visualized separately

---

## MLflow Experiment Tracking

Every model run was automatically tracked in **MLflow 3** with:

```
MLflow Experiment: sudenergy_lux_temp_forecast
│
├── Run: shivering-cow-449  ← BEST MODEL (registered as v1)
│   ├── Parameters
│   │   ├── model: prophet
│   │   ├── holiday_country: US
│   │   ├── interval_width: 0.8
│   │   └── random_state: 607099633
│   ├── Metrics
│   │   ├── val_smape: 0.4728
│   │   └── test_smape: 0.4728
│   └── Artifacts
│       └── model/ (pyfunc format — deployable)
│
├── Run: peaceful-toad-98
├── Run: placid-jay-296
├── Run: vaunted-squirrel-409
├── Run: dashing-shrike-576
├── Run: bemused-cow-862  ← DeepAR
├── Run: sassy-jay-939   ← DeepAR
└── Run: burly-auk-99    ← ARIMA
```

### MLflow Model Registry
The best model is registered as:
```
Model:   workspace.default.sudenergy_lux_temp_forecast
Version: v1
Stage:   Production-ready
Format:  pyfunc (deployable via REST API)
```

---

## Data Pipeline (Databricks Delta Lake)

### Bronze Layer — Raw Hourly Data
```sql
SELECT * FROM workspace.default.bronze_weather_hourly
```
- **43,824 rows** · Hourly granularity · 21 columns
- Raw data as-is from Open-Meteo API
- Stored as **Delta table** in Databricks Unity Catalog

### Silver Layer — Daily Aggregated Features
```sql
SELECT * FROM workspace.default.silver_weather_daily
```
- **1,827 rows** · Daily granularity · 30 columns
- Aggregated from Bronze using **Apache Spark**
- Engineered features: HDD, CDD, season, is_weekend

### Gold Layer — Forecast Output
```sql
SELECT * FROM workspace.default.forecast_predictions_1780046331583
```
- **365 rows** · Daily forecast · 5 columns
- Generated by best AutoML Prophet model
- Includes confidence intervals (lower/upper bounds)

---

## Data Quality Results

| Check | Result |
|-------|--------|
| Total rows (Silver) | 1,827 daily rows ✓ |
| Date range | 2021-05-29 → 2026-05-29 ✓ |
| Missing values | None ✓ |
| Duplicate dates | None ✓ |
| Date gaps | None ✓ |
| Temperature range | -5.44°C to 27.57°C ✓ |
| Outliers | Real weather events — kept intentionally ✓ |
| Snow depth | Cross-validated vs Open-Meteo API ✓ |

---

## Forecast Results

### 1-Year Temperature Forecast (Prophet via Databricks AutoML)

| Period | Predicted Avg Temp | Confidence Band |
|--------|-------------------|-----------------|
| Summer 2026 (Jun–Aug) | ~17–18°C | 10°C – 22°C |
| Autumn 2026 (Sep–Oct) | ~11–16°C | 8°C – 20°C |
| Winter 2026–2027 (Nov–Jan) | ~3–5°C | -2°C – 10°C |
| Spring 2027 (Feb–May) | ~5–14°C | 2°C – 19°C |

> Forecast validated against Open-Meteo historical charts for Luxembourg City (49.60°N, 6.06°E, 311m asl)

---

## Notebook Structure

```
sudenergy-luxembourg/
│
├── README.md                    ← You are here
│
├── 01_weather_ingest.ipynb
│   ├── STEP 1 — Install libraries
│   ├── STEP 2 — Imports
│   ├── STEP 3 — Fetch 5 years weather data (Open-Meteo API)
│   ├── STEP 4 — Save Bronze layer (Delta table)
│   ├── STEP 5 — Create Silver layer (Spark aggregation + features)
│   └── STEP 6 — Data quality check
│
└── 02_prophet_forecast.ipynb
    ├── STEP 1 — Install Prophet
    ├── STEP 2 — Imports
    ├── STEP 3 — Load Silver data & prepare for Prophet
    ├── STEP 4 — Visualize raw data (4 plots)
    ├── STEP 5 — Train / Test split
    ├── STEP 6 — Visualize AutoML forecast
    └── STEP 7 — Historical + Forecast combined chart
```

---

## How to Run

### Prerequisites
- [Databricks Community Edition](https://community.cloud.databricks.com) account (free)
- No API keys required
- No Azure subscription required

### Steps
```bash
# 1. Clone this repo into Databricks Workspace
#    Workspace → Users → Your folder → Git folder → paste repo URL

# 2. Open 01_weather_ingest notebook
#    Run all cells top to bottom

# 3. Open 02_prophet_forecast notebook
#    Run all cells top to bottom

# 4. View results in Databricks Catalog
#    Catalog → workspace → default → Tables
```

---

## Phase 2 Roadmap

| Step | Technology | Description |
|------|-----------|-------------|
| Real consumption data | Delta Live Tables | Replace HDD proxy with actual kWh |
| Advanced models | XGBoost + LightGBM | More accurate with real consumption data |
| Multi-location | DeepAR | Forecast across all Luxembourg substations simultaneously |
| Production platform | Azure Databricks Premium | Move from Community Edition |
| Data governance | Unity Catalog | Full data lineage and access control |
| Automated pipeline | Databricks Workflows | Weekly forecast refresh |
| Live dashboard | Databricks Dashboard | Client-facing energy forecast visualization |
| Model monitoring | MLflow + Lakehouse Monitoring | Detect model drift over time |

---

## Tech Stack

| Category | Technology | Version |
|----------|-----------|---------|
| **Platform** | Databricks Community Edition | Latest |
| **Compute** | Databricks Serverless | — |
| **Storage** | Delta Lake | — |
| **Processing** | Apache Spark | 3.x |
| **ML Tracking** | MLflow | 3.0 |
| **Forecasting** | Prophet (Facebook/Meta) | Latest |
| **Deep Learning** | DeepAR | AutoML |
| **Data Source** | Open-Meteo Historical API | Free |
| **Language** | Python | 3.10 |
| **Libraries** | Pandas, NumPy, Matplotlib | Latest |
| **Version Control** | GitHub + Databricks Git | — |

---

## Author

**Elham Khorasani**  
Data Engineer · aeonic-intelligence.de  
Project: SudEnergy Luxembourg — Phase 1 PoC  
Platform: Databricks Community Edition  
Date: May 2026

---

*This is a Phase 1 proof of concept built entirely on Databricks. All forecasts are based on weather data only and should be validated against actual energy consumption data before use in production.*
# SudEnergy Luxembourg — Weather-Driven Energy Demand Forecasting

> **Phase 1 — Proof of Concept**  
> Databricks Community Edition · Open-Meteo API · Prophet Forecasting  
> Author: Elham Khorasani · aeonic-intelligence.de

---

## Project Overview

This project is a **Phase 1 proof of concept** for SudEnergy Luxembourg, demonstrating how historical weather data can be used to understand and forecast energy demand patterns using **Databricks** and **machine learning**.

Since SudEnergy has not yet provided actual energy consumption (kWh) data, this phase uses **weather variables as energy demand proxies** — particularly Heating Degree Days (HDD) and Cooling Degree Days (CDD), which are industry-standard metrics used by European utility companies to model energy demand.

---

## Objectives

- Ingest 5 years of historical hourly weather data for Luxembourg City
- Build a clean, structured data pipeline using the Medallion Architecture (Bronze → Silver → Gold)
- Perform exploratory data analysis and validate data quality
- Train a forecasting model using Databricks AutoML
- Produce a 1-year temperature forecast as a proxy for energy demand
- Demonstrate the capabilities of Databricks to the client

---

## Architecture

```
Open-Meteo API
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                  MEDALLION ARCHITECTURE              │
│                                                     │
│  BRONZE                SILVER               GOLD    │
│  Raw hourly   →→→   Daily aggregated  →→→  Forecast │
│  43,824 rows         1,827 rows             365 rows │
│                                                     │
└─────────────────────────────────────────────────────┘
      │
      ▼
  MLflow / AutoML
      │
      ▼
  Prophet Forecast Model
```

---

## Data Source

| Property | Value |
|----------|-------|
| Source | [Open-Meteo Historical Weather API](https://open-meteo.com) |
| Location | Luxembourg City (49.61°N, 6.13°E, 311m asl) |
| Period | 2021-05-29 → 2026-05-29 (5 years) |
| Granularity | Hourly |
| Total rows | 43,824 hourly records |
| Cost | Free — no API key required |

---

## Weather Variables

| Variable | Unit | Energy Relevance |
|----------|------|-----------------|
| `temperature_2m` | °C | Primary demand driver |
| `apparent_temp` | °C | Comfort index |
| `wind_speed_10m` | km/h | Standard meteorological level |
| `wind_speed_100m` | km/h | Wind turbine estimation |
| `wind_direction_10m` | degrees | Wind pattern analysis |
| `wind_gusts_10m` | km/h | Grid stability |
| `shortwave_radiation` | W/m² | Solar energy potential |
| `direct_radiation` | W/m² | Solar panel estimation |
| `diffuse_radiation` | W/m² | Scattered light |
| `sunshine_duration` | seconds | Sunshine hours per day |
| `precipitation` | mm | Total rain + snow |
| `rain` | mm | Liquid rain |
| `snowfall` | cm | Cold weather indicator |
| `snow_depth` | cm | Ground snow accumulation |
| `cloud_cover` | % | Solar blocking |
| `relative_humidity` | % | Comfort & cooling demand |
| `dew_point` | °C | Condensation point |
| `pressure_msl` | hPa | Weather system indicator |
| `soil_temp_0_7cm` | °C | Ground heat / building heating |

---

## Data Pipeline

### Bronze Layer — Raw Hourly Data
**Table:** `workspace.default.bronze_weather_hourly`  
**Rows:** 43,824  
**Description:** Raw hourly weather data as downloaded from Open-Meteo API. No transformations applied. Date and time separated into individual columns for clarity.

### Silver Layer — Daily Aggregated Features
**Table:** `workspace.default.silver_weather_daily`  
**Rows:** 1,827  
**Description:** Daily aggregations of all weather variables. Includes energy-relevant engineered features:

| Feature | Description |
|---------|-------------|
| `HDD` | Heating Degree Days (base 15.5°C) — EU standard |
| `CDD` | Cooling Degree Days (base 15.5°C) |
| `season` | Winter / Spring / Summer / Autumn |
| `is_weekend` | 1 if Saturday or Sunday |
| `avg_temp` | Daily average temperature |
| `max_temp` | Daily maximum temperature |
| `min_temp` | Daily minimum temperature |
| `max_wind_gusts` | Peak wind gust of the day |
| `total_radiation` | Total daily solar radiation |
| `total_precipitation` | Total daily precipitation |

### Gold Layer — Forecast Output
**Table:** `workspace.default.forecast_predictions_1780046331583`  
**Rows:** 365  
**Description:** 1-year daily temperature forecast produced by the best AutoML model (Prophet).

| Column | Description |
|--------|-------------|
| `date` | Forecast date |
| `predicted_avg_temp` | Predicted daily average temperature |
| `predicted_avg_temp_lower` | Lower bound (80% confidence) |
| `predicted_avg_temp_upper` | Upper bound (80% confidence) |

---

## Data Quality Results

| Check | Result |
|-------|--------|
| Total rows (Silver) | 1,827 daily rows |
| Date range | 2021-05-29 → 2026-05-29 |
| Missing values | None ✓ |
| Duplicate dates | None ✓ |
| Date gaps | None ✓ |
| Temperature range | -5.44°C to 27.57°C ✓ |
| Outliers | Real weather events — kept intentionally ✓ |
| Snow depth validation | Cross-checked vs Open-Meteo chart ✓ |

---

## Forecasting Model

### AutoML Results

Databricks AutoML automatically tested multiple models and selected the best one:

| Model | MAPE | Training Time |
|-------|------|---------------|
| **Prophet (US holidays, 0.8 CI)** | **0.4728 🏆** | 50s |
| Prophet (no holidays, 0.95 CI) | 0.4754 | 46s |
| Prophet (no holidays, 0.8 CI) | 0.4756 | 48s |
| DeepAR (64, 365) | 0.4797 | 5.6 min |
| DeepAR (32, 730) | 0.5531 | 5.6 min |
| ARIMA | 0.6884 | 1.2 min |

**Winner: Prophet** with multiplicative seasonality, US holidays, 80% confidence interval.

### Why Prophet?
- Strong yearly seasonality in Luxembourg climate
- Only 1,827 rows — too small for deep learning (DeepAR)
- Interpretable results — easy to explain to client
- Automatic changepoint detection
- Beautiful confidence interval visualization

### Forecast Summary (May 2026 → May 2027)

| Period | Predicted Avg Temp | Notes |
|--------|-------------------|-------|
| Summer 2026 (Jun–Aug) | ~17–18°C | Realistic for Luxembourg |
| Autumn 2026 (Sep–Oct) | ~11–16°C | Gradual cooling |
| Winter 2026–2027 (Nov–Jan) | ~3–5°C | Heating demand peak |
| Spring 2027 (Feb–May) | ~5–14°C | Gradual warming |

---

## Notebook Structure

```
sudenergy-luxembourg/
│
├── 01_weather_ingest.ipynb
│   ├── STEP 1 — Install libraries
│   ├── STEP 2 — Imports
│   ├── STEP 3 — Fetch 5 years weather data (Open-Meteo)
│   ├── STEP 4 — Save Bronze layer
│   ├── STEP 5 — Create Silver layer (daily aggregation + features)
│   └── STEP 6 — Data quality check
│
└── 02_prophet_forecast.ipynb
    ├── STEP 1 — Install Prophet
    ├── STEP 2 — Imports
    ├── STEP 3 — Load Silver data & prepare for Prophet
    ├── STEP 4 — Visualize raw data (4 plots)
    ├── STEP 5 — Train / Test split
    └── STEP 6 — Visualize AutoML forecast
```

---

## How to Run

1. Open [Databricks Community Edition](https://community.cloud.databricks.com)
2. Clone this repo into your Workspace
3. Run `01_weather_ingest` — all cells top to bottom
4. Run `02_prophet_forecast` — all cells top to bottom
5. View results in **Catalog → workspace → default**

**Requirements:** Databricks Community Edition (free) · No API keys needed

---

## Phase 2 Roadmap

Once SudEnergy provides actual kWh consumption data:

| Step | Description |
|------|-------------|
| Real consumption data | Replace HDD proxy with actual kWh readings |
| XGBoost model | More accurate with real consumption data |
| Multiple locations | DeepAR across all Luxembourg substations |
| Azure Databricks | Move to production environment |
| Unity Catalog | Proper data governance |
| Automated refresh | Weekly forecast updates via Databricks Workflows |
| Dashboard | Live client-facing energy forecast dashboard |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Databricks Community Edition | Data platform |
| Apache Spark | Distributed data processing |
| Delta Lake | Reliable data storage |
| MLflow | Experiment tracking & model registry |
| Prophet (Facebook/Meta) | Time series forecasting |
| Open-Meteo API | Free historical weather data |
| Python / Pandas | Data manipulation |
| Matplotlib | Visualization |

---

## Author

**Elham Khorasani**  
Data & AI consultant
Project: SudEnergy Luxembourg — Phase 1 PoC  
Date: May 2026

---

*This is a Phase 1 proof of concept. All forecasts are based on weather data only and should be validated against actual energy consumption data before use in production.*
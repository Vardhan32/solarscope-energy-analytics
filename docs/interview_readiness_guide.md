# SolarScope Energy Analytics: Technical Learning & Interview Readiness Guide

---

## 1. Project Overview & Problem Statement

### What problem does SolarScope solve?
Utility-scale photovoltaic (PV) power plants generate massive streams of time-series telemetry from hundreds of inverters, pyranometers, and ambient weather stations. Plant operators and performance engineers face three daily challenges:
1. **Performance Benchmarking:** Determining whether lower plant output is caused by weather (low solar irradiance / cloud cover) or equipment degradation (soiling, inverter derating, wiring losses).
2. **Supervisory Anomaly Detection:** Detecting inverter faults, localized string losses, and abnormal clipping in real time before they result in significant revenue losses.
3. **Power Generation Estimation:** Accurately mapping weather variables and diurnal patterns to electrical output to support grid integration, scheduling, and dispatch planning.

SolarScope ingests **136,000+ 15-minute telemetry intervals** from two operational solar PV power plants (combined 61.5 MW AC capacity across 44 inverters) in Gandhinagar, Gujarat, India. It standardizes telemetry, computes IEC 61724 international solar metrics, detects anomalies, evaluates tabular ML generation estimation models on a chronological holdout, and provides an interactive analytics dashboard with an embedded SQL exploration engine.

---

## 2. Architecture & Design Decisions

### System Architecture
```
                                 [ Raw Kaggle Datasets ]
                        (Plant 1 & 2 Generation & Weather CSVs)
                                           │
                                           ▼
                                [ src/data/loader.py ]
                           (Schema & Type Validation)
                                           │
                                           ▼
                             [ src/data/preprocessor.py ]
               (Timestamp parsing, DC/10 correction, Outlier clipping,
                 Forward-fill weather gaps, Causal lag engineering)
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 ▼                                                   ▼
       [ data/processed/*.parquet ]                        [ data/processed/solarscope.db ]
        (Fast columnar cached tables)                       (Embedded analytical SQLite DB)
                 │                                                   │
                 ├───────────────────────────────┐                   │
                 ▼                               ▼                   ▼
    [ src/analytics/metrics.py ]     [ src/analytics/anomaly.py ]    │
      (Daily yields, Diurnal curves,   (Daytime outages, Underperf,  │
       Capacity Factor, PR)             Clipping, Conversion loss)   │
                 │                               │                   │
                 └───────────────────────┬───────┘                   │
                                         ▼                           │
                           [ src/models/forecasting.py ]             │
                             (Chronological holdout split,           │
                              HistGradientBoosting, RF, Ridge,       │
                              Persistence & Climatology baselines)   │
                                         │                           │
                                         └───────────┬───────────────┘
                                                     ▼
                                            [ Streamlit App ]
                            (Overview, Inverter Rankings, Anomaly Studio,
                              ML Generation Studio, SQL Explorer)
```

### Why this architecture?
* **Columnar Parquet for Analytics:** High-frequency 15-minute time-series data with dozens of columns is up to 10x faster to slice and aggregate in Parquet than reading CSVs or hitting an unindexed database.
* **Embedded SQLite for Ad-Hoc SQL:** Provides relational integrity and allows analysts to run raw analytical SQL queries (`GROUP BY`, window functions) directly in the dashboard without requiring an external MySQL/PostgreSQL server.
* **Modular Decoupling:** Decoupling `data`, `analytics`, `models`, and `app` allows all processing and ML pipelines to run headlessly in CI (`scripts/process_data.py`) or via automated unit tests without launching the Streamlit server.

---

## 3. End-to-End Data Flow

1. **Ingestion (`src/data/loader.py`):** Loads raw generation CSVs (`DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`) and weather sensor CSVs (`AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`).
2. **Sanitization (`src/data/preprocessor.py`):**
   * Multi-format datetime parsing handles `%d-%m-%Y %H:%M` and `%Y-%m-%d %H:%M:%S`.
   * Inverter DC power normalization: Plant 1 DC readings are scaled by 10 (`DC_POWER / 10.0`) to match physical inverter inverter ratios (DC-to-AC ratio ~1.1 to 1.3).
   * Physical boundaries enforced: Negative generation clipped to 0; ambient temp constrained to [0, 60] °C; module temp constrained to [0, 85] °C; irradiance constrained to [0, 1.5] kW/m².
   * Interval energy: $E_{\text{interval}} = P_{\text{AC}} \times 0.25\text{ h}$ (since intervals are 15 minutes = 0.25 hours).
3. **Persistence:** Saves cleaned tables to Apache Parquet format and creates indexed tables in `data/processed/solarscope.db`.
4. **Analytics & Anomaly Detection:** Computes plant-level daily totals, capacity factors, performance ratios, and applies supervisory anomaly rules.
5. **Model Estimation:** Splits data chronologically (last 7 days held out). Fits baseline and ML models. Evaluates MAE, RMSE, $R^2$, nMAE, nRMSE, and WAPE.
6. **Visualization:** Streamlit renders Plotly charts and tables.

---

## 4. Key Files & Responsibilities

| File | Primary Responsibility |
|---|---|
| `src/config.py` | Configuration constants, plant metadata (Plant 1: 31.02 MW, Plant 2: 30.47 MW), filepaths, and inverter mapping. |
| `src/data/loader.py` | Validates file presence, columns, and data types from raw CSVs. |
| `src/data/preprocessor.py` | Timestamp alignment, boundary validation, DC scaling, weather merging, causal lag creation. |
| `src/data/database.py` | SQLite schema creation, batch inserts, and pre-built analytical SQL queries. |
| `src/analytics/metrics.py` | IEC 61724 metrics: Capacity Factor (CF %), Performance Ratio (PR), diurnal envelopes. |
| `src/analytics/anomaly.py` | Rule-based asset health detection: daytime inverter trips, localized string underperformance, inverter clipping. |
| `src/models/baseline.py` | Naive benchmark models: 24-hour persistence and diurnal climatology. |
| `src/models/forecasting.py` | Feature engineering (lags, solar geometry, cyclic time), chronological splitting, and model training. |
| `src/models/evaluation.py` | Standardized metrics: MAE, RMSE, $R^2$, normalized MAE/RMSE (% capacity), WAPE. |
| `app/app.py` | Streamlit entrypoint and sidebar routing. |

---

## 5. Core Algorithms & Physics Principles

### 1. Solar Elevation Proxy (Geometric Bell Curve)
Solar irradiance follows a diurnal curve driven by solar zenith angle. Rather than requiring external ephemeris packages, SolarScope computes an empirical solar elevation proxy:
$$\text{Solar Proxy}(t) = \begin{cases} \sin\left(\pi \cdot \frac{t_{\text{hour}} - 6}{12}\right), & 6 \le t_{\text{hour}} \le 18 \\ 0, & \text{otherwise} \end{cases}$$

### 2. IEC 61724 Standard Metrics
* **Capacity Factor (CF %):**
  $$\text{CF} = \frac{\text{Total Actual Energy Produced (kWh)}}{\text{Installed AC Capacity (kW)} \times 24\text{ h}} \times 100$$
* **Performance Ratio (PR):**
  $$\text{PR} = \frac{Y_f}{Y_r} = \frac{\text{Actual Specific Yield (kWh/kWp)}}{\text{Reference Yield based on Irradiation (kWh/kWp)}}$$
  A healthy utility-scale plant typically operates between 0.75 and 0.85 (75%–85%).

### 3. Thermal Derating Model
PV module efficiency drops as module temperature rises above Standard Test Conditions (STC = 25 °C):
$$P_{\text{expected}} = P_{\text{STC}} \times [1 + \gamma \cdot (T_{\text{module}} - 25^\circ\text{C})]$$
where $\gamma \approx -0.38\% / ^\circ\text{C}$ (negative temperature power coefficient).

---

## 6. Bugs Discovered & Fixed During Audit

1. **Future-Data Leakage in Lag Features (`src/models/forecasting.py`):**
   * *Problem:* Lag features (`LAG_96`, `LAG_192`) used `.bfill().fillna(0.0)`. Backward filling copied future generation measurements backward into the initial rows of the dataset, creating lookahead leakage.
   * *Fix:* Removed `.bfill()`, replacing it with strictly causal `.fillna(0.0)` for the initial cold-start window. Added regression test `test_engineered_lags_never_backfill_from_future_rows()`.
2. **Weather Gap Backfilling (`src/data/preprocessor.py`):**
   * *Problem:* Weather interpolation previously allowed backward filling.
   * *Fix:* Enforced forward-fill only from past timestamps; explicitly raises `ValueError` if leading timestamps have missing weather data. Added regression tests `test_clean_weather_data_forward_fills_only_from_past()` and `test_clean_weather_data_rejects_leading_missing_values()`.
3. **Misleading "Day-Ahead Forecasting" Claims:**
   * *Problem:* The original README and docstrings claimed "Zero-leakage Day-Ahead Forecasts", but the models used actual observed pyranometer readings and actual measured 15-minute lags.
   * *Fix:* Re-scoped all documentation, docstrings, and UI headers to **"Retrospective Generation Estimation Benchmark"**.
4. **Missing Tuple Import in Database Type Hints (`src/data/database.py`):**
   * *Problem:* Type annotation `Tuple[pd.DataFrame, pd.DataFrame]` failed when `Tuple` was not imported from `typing`.
   * *Fix:* Added `from typing import Tuple`.

---

## 7. 40 Technical Interview Questions & Answers

### Part A: Basic Fundamentals (15 Questions)

#### 1. What does SolarScope do?
SolarScope is an analytics and machine learning estimation platform for utility-scale solar PV power plants. It ingests 15-minute generation and weather telemetry from 44 inverters across two plants in India, calculates standard solar performance metrics (Capacity Factor, Performance Ratio), detects equipment anomalies, and benchmarks regression models for power generation.

#### 2. What is the difference between AC power and DC power in a solar plant?
Solar panels generate Direct Current (DC) electricity through the photovoltaic effect. Inverters convert DC into Alternating Current (AC), which is synchronized with the electrical grid frequency and voltage.

#### 3. Why are intervals 15 minutes, and how do you calculate energy (kWh) from power (kW)?
Utility telemetry and grid meters record generation in 15-minute blocks (96 intervals per day). Energy is the integral of power over time:
$$\text{Energy (kWh)} = \text{Power (kW)} \times 0.25\text{ hours}$$

#### 4. What is the Capacity Factor (CF)?
Capacity Factor measures how much energy a plant produced compared to the theoretical maximum it could have produced running at 100% rated capacity 24 hours a day. For solar, CF is typically between 18% and 26% because solar plants do not generate at night.

#### 5. What is the Performance Ratio (PR)?
PR is an IEC 61724 standard metric defined as the ratio of actual energy produced to the energy theoretically available given the incident solar irradiance. It accounts for temperature, inverter efficiency, soiling, and cabling losses. A well-performing plant has a PR of 0.75 to 0.85.

#### 6. Why did Plant 1 DC power need to be divided by 10?
In Plant 1's raw dataset, the DC power values were recorded in deca-watts or scaled by a factor of 10. Comparing DC power directly with AC power revealed an unrealistic DC-to-AC ratio (~12:1). Dividing DC power by 10 restored the expected physical ratio (~1.15:1 to 1.25:1).

#### 7. What technologies and libraries are used in SolarScope?
Python 3.11+, Pandas and NumPy for data manipulation, Scikit-learn for ML regression models, Plotly and Streamlit for interactive dashboard visualization, SQLite for relational querying, and Pytest for automated testing.

#### 8. What is the difference between Ambient Temperature and Module Temperature?
Ambient temperature is the surrounding air temperature measured in the shade. Module temperature is the actual surface temperature of the solar PV cells, measured by a thermocouple behind the panel. Under full sun, module temperature is often 20°C to 30°C hotter than ambient temperature.

#### 9. Why does module temperature matter for power generation?
Solar cells exhibit a negative temperature coefficient (typically around -0.38% to -0.45% per °C above 25°C). As panels get hotter, their semiconductor bandgap changes, causing voltage and efficiency to drop significantly.

#### 10. How does SolarScope detect an inverter outage?
It flags an interval as an outage when the solar pyranometer detects strong daylight (irradiation > 0.2 kW/m²), but an individual inverter outputs zero AC power (`AC_POWER_KW == 0`).

#### 11. What is inverter clipping?
Inverter clipping occurs when the DC power generated by the panels exceeds the maximum AC power rating of the inverter. The inverter saturates and limits the AC output to its nameplate capacity, creating a flat "table-top" profile on daytime power curves.

#### 12. What baseline models are implemented in SolarScope?
Two baseline models are implemented:
- **24-Hour Persistence Model:** Predicts that power today at 12:00 PM equals power yesterday at 12:00 PM (`LAG_96`).
- **Diurnal Climatology Model:** Predicts power based on the historical mean power for that exact time-of-day interval (hour and minute) across all prior training days.

#### 13. What machine learning models are evaluated?
HistGradientBoostingRegressor, RandomForestRegressor, and Ridge Regression (L2-regularized linear regression).

#### 14. Why is data stored in both Parquet and SQLite?
Parquet provides columnar compression and high-speed loading for whole-table analytical queries in Pandas. SQLite provides structured SQL capabilities so users can execute ad-hoc SQL queries with `WHERE`, `GROUP BY`, and window functions in the UI.

#### 15. How do you run the tests for this repository?
Run `pytest tests/ -v` from the project root. All 27 tests execute deterministically in ~6 seconds without external network dependencies.

---

### Part B: Intermediate Engineering & ML (15 Questions)

#### 16. What was the future-data leakage bug in the original code, and how did you fix it?
In `src/models/forecasting.py`, lag features were constructed using `df["LAG_96"].bfill()`. Backward filling takes future values and fills them into previous rows. In the first 24 hours of data, this copied future generation measurements backward into earlier timestamps. The fix was removing `.bfill()` and using `.fillna(0.0)` so that cold-start rows only contain zero or known historical values.

#### 17. Why is random train/test splitting (`train_test_split`) invalid for time-series data?
Random shuffling samples data points from future timestamps into the training set and past timestamps into the test set. Models learn to interpolate between adjacent 15-minute time steps rather than predicting forward in time, artificially inflating performance scores.

#### 18. How does `chronological_split()` work in SolarScope?
It determines the test horizon (last 7 days = 672 intervals). The training set consists of all data prior to that cutoff timestamp ($t < t_{\text{cutoff}}$), and the test set consists strictly of all data after ($t \ge t_{\text{cutoff}}$), ensuring a realistic out-of-time evaluation.

#### 19. Why was "Day-Ahead Forecasting" corrected to "Retrospective Generation Estimation"?
A true operational day-ahead forecast cannot use actual ground-truth weather sensor data from tomorrow because it hasn't happened yet—it must use Numerical Weather Predictions (NWP) from meteorological services. Because SolarScope evaluates models using observed weather readings recorded during the test window, it evaluates regression estimation capacity on historical telemetry, not a true day-ahead forecast.

#### 20. What is WAPE, and why is it preferred over MAPE in solar forecasting?
MAPE (Mean Absolute Percentage Error) divides error by actual power: $\frac{|y - \hat{y}|}{y}$. During early morning, late afternoon, or nighttime, when power is close to zero, MAPE explodes toward infinity or produces division-by-zero errors. WAPE (Weighted Absolute Percentage Error) sums absolute errors across all intervals and divides by total actual generation:
$$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100$$
It provides a robust, scale-sensitive percentage error that does not distort at near-zero values.

#### 21. What is nMAE (% Cap), and why is it useful?
Normalized MAE divides the Mean Absolute Error in kW by the total rated AC capacity of the plant:
$$\text{nMAE} = \frac{\text{MAE (kW)}}{\text{Installed Capacity (kW)}} \times 100$$
This allows engineers to compare forecasting accuracy across plants of different sizes (e.g., comparing a 10 MW plant to a 100 MW plant).

#### 22. What features are engineered in `engineer_features()`?
- `SOLAR_ELEVATION_PROXY`: Trigonometric diurnal bell curve between 6:00 AM and 6:00 PM.
- Cyclic time features: $\sin$ and $\cos$ transforms of `HOUR` and `DAY_OF_YEAR`.
- Backward lags: `LAG_96` (24 hours prior), `LAG_192` (48 hours prior), `LAG_672` (7 days prior).
- `ROLLING_MEAN_PREV_DAY`: Mean power of the preceding 24 hours, shifted by 96 intervals to prevent intra-day lookahead leakage.
- Weather observations: `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`.

#### 23. How does the anomaly detector identify localized string underperformance?
It calculates the median power generated across all 22 inverters at each daytime interval ($P_{\text{fleet\_median}}$). If a specific inverter generates less than 70% of the fleet median during high irradiance conditions ($>0.3\text{ kW/m}^2$) while other inverters are operating normally, it flags an underperformance anomaly (likely caused by blown string fuses, inverter derating, or severe localized soiling).

#### 24. What SQLite schema is used in `data/processed/solarscope.db`?
- `plant_timeseries`: Aggregated plant-level 15-minute generation and weather.
- `inverter_generation`: Inverter-level telemetry with `SOURCE_KEY`, `AC_POWER_KW`, `DC_POWER_KW`, and calculated specific yields.
- `daily_metrics`: Daily aggregated energy, Capacity Factor, and Performance Ratio.
- `anomalies`: Log of detected anomaly events with timestamp, inverter key, anomaly type, and severity.

#### 25. How are database queries protected against SQL injection in the SQL Explorer?
The UI provides pre-built parameter-driven queries. When custom SQL execution is enabled, queries are executed via read-only cursor connections without write/DDL privileges, and standard input sanitization checks prevent destructive commands (`DROP`, `DELETE`, `UPDATE`).

#### 26. Why is HistGradientBoostingRegressor used instead of standard GradientBoostingRegressor?
HistGradientBoosting bins continuous numerical features into integer bins (histograms), reducing split-finding complexity from $O(n \cdot d)$ to $O(\text{bins} \cdot d)$. It is substantially faster on large datasets (136,000 rows) and handles non-linear interactions natively.

#### 27. What is the cold-start problem in time-series lag features?
Because `LAG_96` looks back 24 hours (96 intervals), the first 96 rows of the dataset have no past observation. If filled with mean or backfilled, it leaks information. SolarScope fills them with 0.0, representing the unobserved initial boundary condition, and verifies this behavior via automated tests.

#### 28. How does SolarScope verify data integrity during ingestion?
`src/data/loader.py` enforces schema contracts: verifies that expected columns exist, datetimes are parseable, plant IDs match configuration (4135001 or 4136001), and power values are numeric. If invalid, it raises descriptive `ValueError` exceptions before downstream processing.

#### 29. How is the Performance Ratio calculated when module temperature is not available?
The standard PR formula relies on incident plane-of-array irradiance:
$$\text{PR} = \frac{\sum E_{\text{AC}}}{\sum (G \cdot A \cdot \eta_{\text{STC}})}$$
When temperature correction is added (Temperature-Corrected PR), expected output is scaled down by $(1 + \gamma \cdot (T_{\text{module}} - 25))$. SolarScope provides both standard and temperature-derated comparisons.

#### 30. How does SolarScope avoid lookahead leakage in the rolling mean feature?
`df["ROLLING_MEAN_PREV_DAY"] = df["AC_POWER_KW"].shift(96).rolling(96).mean()`
By shifting the series by 96 intervals (24 hours) *before* applying the 24-hour rolling window, the rolling mean is computed strictly over yesterday's generation, ensuring zero intra-day data from today can leak into today's predictions.

---

### Part C: Advanced Architectural & Domain Questions (10 Questions)

#### 31. Why does Plant 1 have significantly lower WAPE (3.5%) than Plant 2 (15.2%) on the holdout set?
Plant 1 operated under stable clear-sky conditions with consistent inverter performance throughout the test window. Plant 2 experienced severe intermittent inverter trips and localized cloud cover transients. Inverter trips in Plant 2 decoupled electrical output from irradiance, leading to higher prediction errors when using weather-driven models.

#### 32. In a production environment, how would you evolve this project into a real-time operational forecasting system?
1. **NWP Integration:** Ingest GFS, ECMWF, or satellite-derived cloud motion vectors (e.g., from Solcast or NOAA) for day-ahead irradiance forecasts.
2. **Recursive / Direct Multi-Step Forecasting:** Instead of 1-step estimation with observed weather, train 96 separate models or a sequence-to-sequence model (Temporal Fusion Transformer or LightGBM multi-output) for horizons $t+1$ to $t+96$.
3. **Streaming Pipeline:** Ingest SCADA telemetry via Kafka or MQTT, updating rolling features in Redis.
4. **Model Retraining:** Scheduled weekly retraining with automated drift checks.

#### 33. How does inverter clipping affect linear regression vs tree-based models?
Inverter clipping creates a sharp non-linear ceiling where power plateaus even as irradiance continues rising. Linear regression (Ridge) struggles with this non-linearity, overpredicting peak generation and underpredicting mid-day generation. Decision trees and gradient boosting split on feature thresholds (e.g., `if IRRADIATION > 0.8: return 1200 kW`), modeling the saturation plateau accurately.

#### 34. What are the limitations of using a single weather station for a 30 MW plant?
A 30 MW solar plant spans hundreds of acres. A cloud passing over one section of the plant can cause a 50% drop in generation across 10 inverters, while the single pyranometer located at the central substation remains under direct sunlight. This spatial mismatch creates apparent sensor disagreement that point models cannot resolve without spatially distributed sensors.

#### 35. Explain the mathematical formulation of L2 Regularization in Ridge Regression.
$$\min_{w} \left( \sum_{i=1}^n (y_i - x_i^T w)^2 + \alpha \sum_{j=1}^p w_j^2 \right)$$
Ridge adds an L2 penalty on weight magnitudes. In solar feature engineering, lagged features (`LAG_96`, `LAG_192`) and solar proxies exhibit strong multicollinearity (high Pearson correlation). Ordinary Least Squares produces unstable, high-variance coefficients; the L2 penalty shrinks correlated coefficients, stabilizing estimation.

#### 36. How does the CI pipeline ensure reproducibility and code quality?
GitHub Actions (`.github/workflows/ci.yml`) executes on every push and PR:
1. Provisions an isolated Ubuntu container with Python 3.11.
2. Installs pinned dependencies from `requirements.txt`.
3. Runs `scripts/process_data.py` to verify data pipelines, SQLite generation, and model training from scratch.
4. Runs `pytest tests/ -v` to enforce all 27 unit, integration, and leakage-regression tests.

#### 37. What is the difference between specific yield and performance ratio?
* **Specific Yield ($Y_f$):** $\frac{\text{Total Generated Energy (kWh)}}{\text{Installed DC Capacity (kWp)}}$, expressed in $\text{kWh/kWp}$ (or equivalent peak sun hours). It measures total generation per installed kilowatt.
* **Performance Ratio (PR):** Normalizes specific yield by the actual solar irradiation received ($Y_f / Y_r$). Specific yield varies heavily by season; PR measures how efficiently the plant converts whatever sunlight it received, remaining relatively stable across seasons.

#### 38. How does SolarScope handle daylight masking during feature engineering?
Nighttime solar power is physically 0 kW. SolarScope sets `IS_DAYLIGHT = IRRADIATION > 0.05`. Nighttime intervals are bounded to prevent models from predicting small non-zero negative or positive values when the sun is down.

#### 39. What is the impact of soiling on solar PV performance and how could it be detected?
Soiling (dust, dirt, bird droppings) attenuates light reaching solar cells, causing a gradual decline in power output (0.1% to 1% per week depending on geography). Soiling can be detected by tracking the trend in daily Performance Ratio over 14–30 rain-free days. A sudden jump in PR following rainfall confirms soiling recovery.

#### 40. What are the key architectural lessons learned from this audit?
1. **Never use backward-fill on time-series lag features:** It introduces subtle lookahead leakage that invalidates benchmark claims.
2. **Be precise with terminology:** Using observed weather in model evaluation is retrospective estimation, not day-ahead forecasting.
3. **Decouple business logic from UI:** Keeping analytics and ML in pure Python modules (`src/`) enabled 100% automated test coverage in CI without needing browser automation or Streamlit mock servers.

---

## 8. Summary Table: Verified Model Results

### Plant 1 (Gandhinagar Solar Park - 31.02 MW AC Fleet)
*Evaluated on 7-Day Chronological Holdout (672 consecutive 15-minute intervals):*

| Model Architecture | MAE (kW) | RMSE (kW) | $R^2$ Score | nMAE (% Cap) | nRMSE (% Cap) | WAPE (%) |
|---|---|---|---|---|---|---|
| **Gradient Boosting** *(Best)* | **208.36** | **435.15** | **0.9969** | **0.67%** | **1.40%** | **3.48%** |
| **Random Forest** | 215.95 | 447.37 | 0.9967 | 0.70% | 1.44% | 3.60% |
| **Linear Regression (Ridge)** | 274.23 | 522.96 | 0.9955 | 0.88% | 1.69% | 4.58% |
| **Baseline (Diurnal Climatology)** | 1,554.17 | 2,932.03 | 0.8600 | 5.01% | 9.45% | 25.93% |
| **Baseline (24h Persistence)** | 1,900.36 | 3,651.32 | 0.7829 | 6.13% | 11.77% | 31.71% |

### Plant 2 (Gandhinagar Solar Park - 30.47 MW AC Fleet)
| Model Architecture | MAE (kW) | RMSE (kW) | $R^2$ Score | nMAE (% Cap) | nRMSE (% Cap) | WAPE (%) |
|---|---|---|---|---|---|---|
| **Gradient Boosting** *(Best)* | **660.64** | **1,696.89** | **0.9043** | **2.17%** | **5.57%** | **15.18%** |
| **Random Forest** | 687.74 | 1,749.01 | 0.8984 | 2.26% | 5.74% | 15.80% |
| **Linear Regression (Ridge)** | 1,076.24 | 2,051.22 | 0.8602 | 3.53% | 6.73% | 24.73% |
| **Baseline (Diurnal Climatology)** | 1,379.75 | 2,668.15 | 0.7635 | 4.53% | 8.76% | 31.70% |
| **Baseline (24h Persistence)** | 1,679.64 | 3,253.80 | 0.6483 | 5.51% | 10.68% | 38.59% |

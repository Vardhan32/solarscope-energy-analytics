"""End-to-End Orchestration Pipeline for SolarScope.

Processes raw data for both plants, initializes the SQLite database,
runs anomaly detection, and trains & serializes forecasting models.
"""

from typing import Dict, Any, Tuple
import joblib
import pandas as pd

from src.config import (
    PLANTS,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    DB_PATH,
    PlantMetadata,
)
from src.data.preprocessor import process_plant
from src.data.database import (
    get_db_connection,
    init_database_schema,
    persist_plant_data,
    compute_and_save_daily_summary,
)
from src.analytics.metrics import calculate_daily_metrics, calculate_kpi_summary
from src.analytics.anomaly import detect_operational_anomalies
from src.models.forecasting import SolarForecastEngine
from src.utils.logger import get_logger

logger = get_logger("pipeline")


def run_full_pipeline() -> Dict[str, Any]:
    """Executes the complete SolarScope data & ML pipeline."""
    logger.info("Starting SolarScope full end-to-end processing pipeline...")

    # 1. Initialize SQLite Database
    if DB_PATH.exists():
        DB_PATH.unlink()  # Fresh build
    conn = get_db_connection()
    init_database_schema(conn)

    results = {}

    for plant_id, meta in PLANTS.items():
        logger.info(f"Processing {meta.name} (ID: {plant_id})...")

        # Load & Clean
        inverter_df, plant_ts_df = process_plant(plant_id)

        # Save to Parquet for fast cache
        inv_parquet = DATA_PROCESSED_DIR / f"plant_{plant_id}_inverter.parquet"
        ts_parquet = DATA_PROCESSED_DIR / f"plant_{plant_id}_timeseries.parquet"
        inverter_df.to_parquet(inv_parquet, index=False)
        plant_ts_df.to_parquet(ts_parquet, index=False)

        # Persist to SQLite
        persist_plant_data(conn, inverter_df, plant_id)
        compute_and_save_daily_summary(conn, plant_id)

        # Anomaly Detection
        anomalies_df = detect_operational_anomalies(inverter_df, meta)
        anom_parquet = DATA_PROCESSED_DIR / f"plant_{plant_id}_anomalies.parquet"
        anomalies_df.to_parquet(anom_parquet, index=False)

        # Insert anomalies into SQLite
        if not anomalies_df.empty:
            anom_db = anomalies_df[[
                "DATE_TIME", "PLANT_ID", "SOURCE_KEY", "ANOMALY_TYPE",
                "SEVERITY", "AC_POWER_KW", "EXPECTED_AC_POWER_KW",
                "IRRADIATION", "DESCRIPTION"
            ]].copy()
            anom_db["DATE_TIME"] = anom_db["DATE_TIME"].astype(str)
            anom_db.columns = [c.lower() for c in anom_db.columns]
            anom_db.to_sql("anomalies", conn, if_exists="append", index=False)

        # Daily Metrics & KPIs
        daily_df = calculate_daily_metrics(plant_ts_df, meta)
        kpis = calculate_kpi_summary(plant_ts_df, daily_df, meta)

        # ML Forecasting Pipeline
        logger.info(f"Training ML forecasting models for plant {plant_id}...")
        engine = SolarForecastEngine(meta)
        metrics_table, predictions_df = engine.train_and_evaluate(plant_ts_df)

        pred_parquet = DATA_PROCESSED_DIR / f"plant_{plant_id}_forecast_predictions.parquet"
        predictions_df.to_parquet(pred_parquet, index=False)

        # Save trained engine
        model_file = MODELS_DIR / f"plant_{plant_id}_engine.joblib"
        joblib.dump(engine, model_file)

        results[plant_id] = {
            "meta": meta,
            "inverter_count": len(inverter_df["SOURCE_KEY"].unique()),
            "total_intervals": len(plant_ts_df),
            "anomalies_count": len(anomalies_df),
            "kpis": kpis,
            "metrics_table": metrics_table,
        }

    conn.close()
    logger.info("Pipeline execution completed successfully.")
    return results


if __name__ == "__main__":
    run_full_pipeline()

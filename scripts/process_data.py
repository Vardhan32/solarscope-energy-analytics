"""Script to run data preprocessing, database initialization, and model training."""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pipeline import run_full_pipeline

if __name__ == "__main__":
    print("Running SolarScope full data pipeline...")
    results = run_full_pipeline()
    for plant_id, res in results.items():
        print(f"\n================ Plant {plant_id} Summary ================")
        print(f"Name: {res['meta'].name}")
        print(f"Capacity: {res['meta'].total_ac_capacity_mw:.2f} MW")
        print(f"Total Energy: {res['kpis']['total_energy_mwh']} MWh")
        print(f"Avg Capacity Factor: {res['kpis']['avg_capacity_factor_pct']}%")
        print(f"Anomalies Detected: {res['anomalies_count']}")
        print("\nModel Evaluation Metrics (7-day Holdout Test):")
        print(res["metrics_table"].to_string(index=False))

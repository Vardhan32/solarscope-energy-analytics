"""SolarScope SQLite Analytics Database Layer.

Provides schema creation, indexing, batch persistence, and analytical SQL queries
for fast interactive dashboard operations.
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from src.config import DB_PATH, PLANTS
from src.utils.logger import get_logger

logger = get_logger("database")


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Creates a connection to the SQLite analytics database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_database_schema(conn: sqlite3.Connection) -> None:
    """Creates tables and indexes if they do not exist."""
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS plants (
        plant_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        location TEXT NOT NULL,
        inverter_count INTEGER NOT NULL,
        nameplate_ac_kw_per_inverter REAL NOT NULL,
        total_ac_capacity_kw REAL NOT NULL,
        total_ac_capacity_mw REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS inverters (
        source_key TEXT NOT NULL,
        plant_id INTEGER NOT NULL,
        PRIMARY KEY (source_key, plant_id),
        FOREIGN KEY (plant_id) REFERENCES plants(plant_id)
    );

    CREATE TABLE IF NOT EXISTS generation_15min (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_time TEXT NOT NULL,
        date TEXT NOT NULL,
        hour INTEGER NOT NULL,
        minute INTEGER NOT NULL,
        plant_id INTEGER NOT NULL,
        source_key TEXT NOT NULL,
        dc_power_kw REAL NOT NULL,
        ac_power_kw REAL NOT NULL,
        interval_energy_kwh REAL NOT NULL,
        daily_yield REAL,
        total_yield REAL,
        ambient_temperature REAL,
        module_temperature REAL,
        irradiation REAL,
        is_daylight INTEGER,
        inverter_efficiency_pct REAL,
        expected_ac_power_kw REAL,
        performance_ratio REAL,
        FOREIGN KEY (plant_id) REFERENCES plants(plant_id)
    );

    CREATE INDEX IF NOT EXISTS idx_gen_datetime ON generation_15min (date_time);
    CREATE INDEX IF NOT EXISTS idx_gen_plant_date ON generation_15min (plant_id, date);
    CREATE INDEX IF NOT EXISTS idx_gen_inverter ON generation_15min (source_key);

    CREATE TABLE IF NOT EXISTS plant_daily_summary (
        plant_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        total_ac_energy_kwh REAL NOT NULL,
        total_dc_energy_kwh REAL NOT NULL,
        peak_ac_power_kw REAL NOT NULL,
        avg_irradiation REAL NOT NULL,
        peak_module_temp REAL NOT NULL,
        avg_module_temp REAL NOT NULL,
        fleet_avg_efficiency_pct REAL,
        capacity_factor_pct REAL NOT NULL,
        performance_ratio REAL,
        PRIMARY KEY (plant_id, date),
        FOREIGN KEY (plant_id) REFERENCES plants(plant_id)
    );

    CREATE TABLE IF NOT EXISTS anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_time TEXT NOT NULL,
        plant_id INTEGER NOT NULL,
        source_key TEXT NOT NULL,
        anomaly_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        ac_power_kw REAL,
        expected_ac_power_kw REAL,
        irradiation REAL,
        description TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_anom_plant_dt ON anomalies (plant_id, date_time);
    """)

    # Populate plants table metadata
    for p in PLANTS.values():
        cursor.execute("""
            INSERT OR REPLACE INTO plants 
            (plant_id, name, location, inverter_count, nameplate_ac_kw_per_inverter, total_ac_capacity_kw, total_ac_capacity_mw)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            p.plant_id,
            p.name,
            p.location,
            p.inverter_count,
            p.nameplate_ac_kw_per_inverter,
            p.total_ac_capacity_kw,
            p.total_ac_capacity_mw
        ))

    conn.commit()
    logger.info("Database schema initialized successfully.")


def persist_plant_data(
    conn: sqlite3.Connection,
    inverter_df: pd.DataFrame,
    plant_id: int
) -> None:
    """Inserts processed inverter data into the SQLite database."""
    logger.info(f"Persisting data for plant {plant_id} ({len(inverter_df)} records)")

    # Insert unique inverters
    inverter_keys = inverter_df["SOURCE_KEY"].unique()
    cursor = conn.cursor()
    for key in inverter_keys:
        cursor.execute(
            "INSERT OR IGNORE INTO inverters (source_key, plant_id) VALUES (?, ?)",
            (key, plant_id)
        )

    # Format for DB insertion
    df_to_save = inverter_df[[
        "DATE_TIME", "DATE", "HOUR", "MINUTE", "PLANT_ID", "SOURCE_KEY",
        "DC_POWER_KW", "AC_POWER_KW", "INTERVAL_ENERGY_KWH", "DAILY_YIELD",
        "TOTAL_YIELD", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION",
        "IS_DAYLIGHT", "INVERTER_EFFICIENCY_PCT", "EXPECTED_AC_POWER_KW", "PERFORMANCE_RATIO"
    ]].copy()

    df_to_save["DATE_TIME"] = df_to_save["DATE_TIME"].astype(str)
    df_to_save["DATE"] = df_to_save["DATE"].astype(str)
    df_to_save["IS_DAYLIGHT"] = df_to_save["IS_DAYLIGHT"].astype(int)

    df_to_save.columns = [c.lower() for c in df_to_save.columns]

    # Batch insert via pandas
    df_to_save.to_sql("generation_15min", conn, if_exists="append", index=False)
    conn.commit()


def compute_and_save_daily_summary(conn: sqlite3.Connection, plant_id: int) -> None:
    """Executes a SQL aggregation query to populate the plant_daily_summary table."""
    meta = PLANTS[plant_id]
    cursor = conn.cursor()

    query = f"""
    INSERT OR REPLACE INTO plant_daily_summary (
        plant_id, date, total_ac_energy_kwh, total_dc_energy_kwh,
        peak_ac_power_kw, avg_irradiation, peak_module_temp,
        avg_module_temp, fleet_avg_efficiency_pct, capacity_factor_pct,
        performance_ratio
    )
    SELECT 
        plant_id,
        date,
        SUM(interval_energy_kwh) as total_ac_energy_kwh,
        SUM(plant_dc_kw * 0.25) as total_dc_energy_kwh,
        MAX(plant_ac_kw) as peak_ac_power_kw,
        AVG(irradiation) as avg_irradiation,
        MAX(module_temperature) as peak_module_temp,
        AVG(module_temperature) as avg_module_temp,
        AVG(inverter_efficiency_pct) as fleet_avg_efficiency_pct,
        (SUM(interval_energy_kwh) / ({meta.total_ac_capacity_kw} * 24.0)) * 100.0 as capacity_factor_pct,
        (SUM(plant_ac_kw) / NULLIF(SUM(expected_ac_power_kw), 0)) as performance_ratio
    FROM (
        SELECT 
            plant_id,
            date,
            date_time,
            SUM(ac_power_kw) as plant_ac_kw,
            SUM(dc_power_kw) as plant_dc_kw,
            SUM(interval_energy_kwh) as interval_energy_kwh,
            SUM(expected_ac_power_kw) as expected_ac_power_kw,
            AVG(inverter_efficiency_pct) as inverter_efficiency_pct,
            AVG(irradiation) as irradiation,
            AVG(module_temperature) as module_temperature
        FROM generation_15min
        WHERE plant_id = ?
        GROUP BY plant_id, date, date_time
    )
    GROUP BY plant_id, date;
    """

    cursor.execute(query, (plant_id,))
    conn.commit()
    logger.info(f"Daily summary populated for plant {plant_id}")


def query_daily_generation(plant_id: Optional[int] = None) -> pd.DataFrame:
    """Executes SQL to return daily generation metrics."""
    conn = get_db_connection()
    try:
        if plant_id:
            sql = "SELECT * FROM plant_daily_summary WHERE plant_id = ? ORDER BY date ASC"
            return pd.read_sql_query(sql, conn, params=(plant_id,))
        else:
            sql = "SELECT * FROM plant_daily_summary ORDER BY plant_id, date ASC"
            return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def query_inverter_rankings(plant_id: int) -> pd.DataFrame:
    """Calculates inverter performance rankings via SQL."""
    conn = get_db_connection()
    try:
        sql = """
        SELECT 
            source_key,
            SUM(interval_energy_kwh) as total_energy_kwh,
            MAX(ac_power_kw) as peak_ac_kw,
            AVG(inverter_efficiency_pct) as avg_efficiency_pct,
            AVG(performance_ratio) as avg_performance_ratio,
            COUNT(CASE WHEN irradiation > 0.2 AND ac_power_kw < 5 THEN 1 END) as zero_sun_intervals
        FROM generation_15min
        WHERE plant_id = ?
        GROUP BY source_key
        ORDER BY total_energy_kwh DESC;
        """
        return pd.read_sql_query(sql, conn, params=(plant_id,))
    finally:
        conn.close()


def query_custom_sql(sql: str, params: Optional[Tuple] = None) -> pd.DataFrame:
    """Safe analytical query execution for custom SQL views."""
    conn = get_db_connection()
    try:
        if params:
            return pd.read_sql_query(sql, conn, params=params)
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()

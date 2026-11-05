"""Unit tests for SolarScope SQLite database."""

import sqlite3
import pytest
import pandas as pd
from src.config import PLANTS
from src.data.database import (
    get_db_connection,
    init_database_schema,
    query_daily_generation,
    query_inverter_rankings,
    query_custom_sql,
)


def test_database_tables_exist():
    conn = get_db_connection()
    try:
        init_database_schema(conn)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = ["plants", "inverters", "generation_15min", "plant_daily_summary", "anomalies"]
        for t in expected_tables:
            assert t in tables, f"Expected table {t} missing from SQLite database"
    finally:
        conn.close()


def test_query_daily_generation():
    df = query_daily_generation(4135001)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "capacity_factor_pct" in df.columns
    assert "total_ac_energy_kwh" in df.columns


def test_query_inverter_rankings():
    df = query_inverter_rankings(4135001)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == PLANTS[4135001].inverter_count
    assert "total_energy_kwh" in df.columns


def test_query_custom_sql():
    sql = "SELECT COUNT(*) as cnt FROM plants"
    res = query_custom_sql(sql)
    assert res["cnt"].iloc[0] == 2

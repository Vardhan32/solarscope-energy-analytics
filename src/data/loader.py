"""Data Loading and Schema Validation Module for SolarScope."""

from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd

from src.config import DATA_RAW_DIR, PLANTS, PlantMetadata
from src.utils.logger import get_logger

logger = get_logger("data_loader")

# Expected raw schemas
EXPECTED_GEN_COLS = [
    "DATE_TIME",
    "PLANT_ID",
    "SOURCE_KEY",
    "DC_POWER",
    "AC_POWER",
    "DAILY_YIELD",
    "TOTAL_YIELD",
]

EXPECTED_WEATHER_COLS = [
    "DATE_TIME",
    "PLANT_ID",
    "SOURCE_KEY",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
]


class RawDataLoadError(Exception):
    """Raised when raw dataset file is missing or corrupted."""
    pass


def get_raw_filepaths(plant_id: int) -> Tuple[Path, Path]:
    """Returns (gen_filepath, weather_filepath) for a given plant."""
    if plant_id == 4135001:
        gen_file = DATA_RAW_DIR / "Plant_1_Generation_Data.csv"
        weather_file = DATA_RAW_DIR / "Plant_1_Weather_Sensor_Data.csv"
    elif plant_id == 4136001:
        gen_file = DATA_RAW_DIR / "Plant_2_Generation_Data.csv"
        weather_file = DATA_RAW_DIR / "Plant_2_Weather_Sensor_Data.csv"
    else:
        raise ValueError(f"Unknown plant_id: {plant_id}. Supported: {list(PLANTS.keys())}")

    return gen_file, weather_file


def verify_raw_datasets() -> Dict[str, bool]:
    """Checks whether all four real-world dataset files are present and non-empty."""
    results = {}
    for plant_id in PLANTS.keys():
        gen_path, w_path = get_raw_filepaths(plant_id)
        results[f"Plant_{plant_id}_Gen"] = gen_path.exists() and gen_path.stat().st_size > 1000
        results[f"Plant_{plant_id}_Weather"] = w_path.exists() and w_path.stat().st_size > 1000
    return results


def load_raw_generation(plant_id: int) -> pd.DataFrame:
    """Loads and validates raw inverter generation records."""
    gen_file, _ = get_raw_filepaths(plant_id)
    if not gen_file.exists():
        raise RawDataLoadError(f"Missing generation file: {gen_file}")

    logger.info(f"Loading raw generation data for plant {plant_id} from {gen_file.name}")
    df = pd.read_csv(gen_file)

    missing = [c for c in EXPECTED_GEN_COLS if c not in df.columns]
    if missing:
        raise RawDataLoadError(f"Generation dataset missing columns: {missing}")

    return df


def load_raw_weather(plant_id: int) -> pd.DataFrame:
    """Loads and validates raw weather sensor records."""
    _, weather_file = get_raw_filepaths(plant_id)
    if not weather_file.exists():
        raise RawDataLoadError(f"Missing weather file: {weather_file}")

    logger.info(f"Loading raw weather data for plant {plant_id} from {weather_file.name}")
    df = pd.read_csv(weather_file)

    missing = [c for c in EXPECTED_WEATHER_COLS if c not in df.columns]
    if missing:
        raise RawDataLoadError(f"Weather dataset missing columns: {missing}")

    return df

"""SolarScope Platform Configuration.

Centralized configuration parameters, plant specifications, paths,
and solar engineering constants.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_PROCESSED_DIR / "solarscope.db"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure runtime directories exist
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PlantMetadata:
    """Plant configuration and technical specifications."""
    plant_id: int
    name: str
    location: str
    inverter_count: int
    nameplate_ac_kw_per_inverter: float
    nameplate_dc_kw_per_inverter: float
    dc_scale_factor: float  # Plant 1 raw DC is scaled by 10 relative to kW
    date_format_gen: str
    date_format_weather: str

    @property
    def total_ac_capacity_kw(self) -> float:
        return self.inverter_count * self.nameplate_ac_kw_per_inverter

    @property
    def total_ac_capacity_mw(self) -> float:
        return self.total_ac_capacity_kw / 1000.0


# Technical specifications based on Plant 1 and Plant 2 PV array field parameters
PLANTS: Dict[int, PlantMetadata] = {
    4135001: PlantMetadata(
        plant_id=4135001,
        name="Gandhinagar Solar Park - Plant 1",
        location="Gujarat, India (23.2156° N, 72.6369° E)",
        inverter_count=22,
        nameplate_ac_kw_per_inverter=1410.0,
        nameplate_dc_kw_per_inverter=1447.0,
        dc_scale_factor=10.0,  # Plant 1 raw DC values are 10x kW
        date_format_gen="%d-%m-%Y %H:%M",
        date_format_weather="%Y-%m-%d %H:%M:%S",
    ),
    4136001: PlantMetadata(
        plant_id=4136001,
        name="Gandhinagar Solar Park - Plant 2",
        location="Gujarat, India (23.2156° N, 72.6369° E)",
        inverter_count=22,
        nameplate_ac_kw_per_inverter=1385.0,
        nameplate_dc_kw_per_inverter=1420.0,
        dc_scale_factor=1.0,  # Plant 2 raw DC values are in kW
        date_format_gen="%Y-%m-%d %H:%M:%S",
        date_format_weather="%Y-%m-%d %H:%M:%S",
    ),
}

# Standard Solar Engineering Constants
STC_IRRADIANCE = 1.0  # kW/m^2 standard test condition
STC_MODULE_TEMP = 25.0  # °C standard test cell temp
TEMP_COEFFICIENT_PMAX = -0.004  # -0.4% per °C temperature derating

# Anomaly Detection Thresholds
DAYLIGHT_MIN_IRRADIANCE = 0.05  # kW/m^2 threshold to define daylight hours
OUTAGE_IRRADIANCE_THRESHOLD = 0.20  # kW/m^2 threshold to declare day sun
OUTAGE_MAX_AC_POWER = 5.0  # kW under which an active inverter is declared down
UNDERPERFORMANCE_RATIO = 0.70  # producing < 70% of fleet median AC under high sun
CLIPPING_THRESHOLD_RATIO = 0.985  # AC at > 98.5% of max inverter capacity

# Forecasting Configuration
FORECAST_HORIZON_DAYS = 7
INTERVAL_MINUTES = 15
INTERVALS_PER_DAY = 96
RANDOM_STATE = 42

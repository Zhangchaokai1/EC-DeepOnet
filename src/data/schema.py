from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "Data_10Params.csv"
DEFAULT_TIME_SERIES_DIR = PROJECT_ROOT / "time_series_data"

INPUT_COLUMNS = [
    "Input_rho_s",
    "Input_rho_l",
    "Input_cp_s",
    "Input_cl_l",
    "Input_k_s",
    "Input_k_l",
    "Input_Lf",
    "Input_wt",
    "Input_Tm",
    "Input_dT",
]

TARGET_TOUT_COLUMN = "Output_Tout"
TARGET_LIQUID_FRAC_COLUMN = "Output_LiquidFrac"
TARGET_ENERGY_COLUMN = "Output_TotalEnergy_MJ"
TIME_SERIES_FILE_COLUMN = "TimeSeries_File"
STATUS_COLUMN = "Status"
SUCCESS_STATUS = "Success"

TIME_SERIES_TIME_KEY = "times"
TIME_SERIES_TEMPERATURE_KEY = "temperatures"

PIPE_DIAMETER_MM = 20.0
PIPE_RADIUS_M = (PIPE_DIAMETER_MM / 1000.0) / 2.0
PIPE_AREA_M2 = np.pi * (PIPE_RADIUS_M**2)
RHO_WATER_KG_PER_M3 = 1000.0
CP_WATER_J_PER_KG_K = 4182.0
FIXED_VELOCITY_M_PER_S = 0.6

DEFAULT_INLET_TEMPERATURE_C = 35.0
DEFAULT_T_START_HOURS = 1.0
DEFAULT_T_END_HOURS = 100.0
DEFAULT_INTEGRATION_POINTS = 500


def compute_mass_flow_rate_kg_per_s(
    density_kg_per_m3: float = RHO_WATER_KG_PER_M3,
    area_m2: float = PIPE_AREA_M2,
    velocity_m_per_s: float = FIXED_VELOCITY_M_PER_S,
) -> float:
    return density_kg_per_m3 * area_m2 * velocity_m_per_s


DEFAULT_MASS_FLOW_RATE_KG_PER_S = compute_mass_flow_rate_kg_per_s()

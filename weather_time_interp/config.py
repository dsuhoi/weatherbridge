"""
Конфигурация проекта: имена погодных переменных и индексы каналов для RMSE.
Соответствует ERA5: 5 переменных × 4 уровня давления = 20 каналов.
Порядок: temperature(0-3), u_wind(4-7), v_wind(8-11), humidity(12-15), geopotential(16-19).
"""
from typing import List, Tuple

# Имена переменных для логов и метрик RMSE по каждому параметру
WEATHER_VARIABLE_NAMES: List[str] = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]

# Defaults for dataset channel selection. Keep this in sync with
# ERA5MemmapDataset.PL_LEVELS and the paper/eval channel order.
# Q = specific_humidity (доступно в json_stats.nc)
DEFAULT_VARIABLES: List[str] = ["T", "U", "V", "Q", "Z"]
DEFAULT_PRESSURE_LEVELS: List[int] = [1000, 925, 850, 700]

# Уровни давления (соответствуют порядку в данных)
PRESSURE_LEVELS: List[str] = ["1000", "925", "850", "700"]

# Map short variable codes to full names for metric labels
VAR_SHORT_TO_NAME: dict[str, str] = {
    "T": "temperature",
    "U": "u_component_of_wind",
    "V": "v_component_of_wind",
    "Q": "specific_humidity",
    "Z": "geopotential",
}

# Map short variable codes to ERA5 variable names in Zarr files
VAR_TO_ERA5: dict[str, str] = {
    "T": "t",
    "U": "u",
    "V": "v",
    "Q": "q",
    "Z": "z",
}

# Число каналов на одну переменную
CHANNELS_PER_VARIABLE: int = 4

# Число переменных
NUM_WEATHER_VARIABLES: int = len(WEATHER_VARIABLE_NAMES)

# Общее число каналов входа/выхода
TOTAL_CHANNELS: int = NUM_WEATHER_VARIABLES * CHANNELS_PER_VARIABLE  # 20

# Срезы каналов по переменным: (start, end) для каждой переменной
# Используется для расчёта RMSE по каждому погодному параметру отдельно
CHANNEL_SLICES_PER_VARIABLE: List[Tuple[int, int]] = [
    (i * CHANNELS_PER_VARIABLE, (i + 1) * CHANNELS_PER_VARIABLE)
    for i in range(NUM_WEATHER_VARIABLES)
]

# Соответствие латентного времени tau и часов: tau=0 -> 0ч (x0), tau=1 -> 6ч (x1).
# Интерполяция между x0 и x1: учим на часах 1–5 (tau = 1/6, 2/6, ..., 5/6).
HOURS_PER_TAU_UNIT: float = 6.0

# Шаги tau для обучения: 0 (warmup, предсказание x0), затем часы 1–5 между x0 и x1
TAU_TRAIN_STEPS: List[float] = [0.0, 1/6, 2/6, 3/6, 4/6, 5/6]

# Шаги tau для валидации (промежуточные моменты)
TAU_VAL_STEPS: List[float] = [0.1667, 0.5]

# Маппинг: час 0 -> tau=0 (x0), час 6 -> tau=1 (x1), часы 1–5 -> tau = hour/6
def hour_to_tau(hour: int) -> float:
    """Час (0..6) в латентное время [0, 1]."""
    return hour / HOURS_PER_TAU_UNIT


def tau_to_hour(tau: float) -> float:
    """Латентное время в часы."""
    return tau * HOURS_PER_TAU_UNIT

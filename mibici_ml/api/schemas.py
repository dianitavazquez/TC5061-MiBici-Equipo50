
"""Esquemas de entrada y salida para la API del modelo MiBici P2."""

from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    """Valida las 23 variables requeridas por el modelo P2."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # Capacidad y actividad reciente de la estación.
    capacity: int = Field(gt=0)
    departures_lag_1h: float = Field(ge=0)
    arrivals_lag_1h: float = Field(ge=0)
    net_departures_lag_1h: float

    # Promedios históricos de salidas.
    departures_mean_3h: float = Field(ge=0)
    departures_mean_24h: float = Field(ge=0)
    departures_mean_168h: float = Field(ge=0)

    # Promedios históricos de llegadas.
    arrivals_mean_3h: float = Field(ge=0)
    arrivals_mean_24h: float = Field(ge=0)
    arrivals_mean_168h: float = Field(ge=0)

    # Balance de movimientos.
    net_departures_mean_3h: float
    net_departures_mean_24h: float
    net_departures_mean_168h: float

    # Indicadores de riesgo de la hora anterior.
    risk_no_bikes_lag_1h: float = Field(ge=0, le=1)
    risk_no_docks_lag_1h: float = Field(ge=0, le=1)

    # Variables temporales cíclicas.
    hour_sin: float = Field(ge=-1, le=1)
    hour_cos: float = Field(ge=-1, le=1)
    weekday_sin: float = Field(ge=-1, le=1)
    weekday_cos: float = Field(ge=-1, le=1)
    month_sin: float = Field(ge=-1, le=1)
    month_cos: float = Field(ge=-1, le=1)

    # Fin de semana e identificador de estación.
    is_weekend: int = Field(ge=0, le=1)
    station_id: int = Field(gt=0)


class PredictionOutput(BaseModel):
    """Valida las dos predicciones binarias del modelo P2."""

    risk_no_bikes: int = Field(ge=0, le=1)
    risk_no_docks: int = Field(ge=0, le=1)
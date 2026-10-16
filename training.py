"""Training, on Ray, and the shared solar and hourly arithmetic.

The target is the hourly mean of the inverter's total AC power, labelled by the
start of the hour. The operator issues a forecast at the start of hour h for hour
h + 24 h. The model uses four inputs, all known at issue time:

  - lag:   the mean of the last completed hour, which is 25 h before the target
  - clim:  the mean daytime output for the target's month and hour, from training
  - sun:   the sine of the solar elevation at the target (clear-sky shape)
  - a constant

Hours with the sun below the horizon are forecast as zero. No weather is used in
this version; cloud cover is the largest term still missing.

provide_historic_data() is Operator Lib's reader over the platform's timeseries
store. It returns Ray datasets with a "time" column (naive UTC) and a "value"
column.
"""

import datetime
import typing

import numpy as np
import pandas as pd
import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


# How much history one training pass reads: the last year before the training end.
TRAINING_WINDOW = datetime.timedelta(days=365)

# Plant location, approximate (51 deg 30 min N, 10 deg 00 min E).
LATITUDE = 51.5
LONGITUDE = 10.0

# The newest completed hour relative to the target hour, for a forecast issued at
# the start of the hour 24 h before the target.
LAG_HOURS = 25

_J2000 = pd.Timestamp("2000-01-01 12:00:00")


def sun_sin_elevation(times) -> np.ndarray:
    """Sine of the solar elevation at each UTC time, at the plant's location.

    Uses the standard low-precision solar position approximation. Values above
    zero mean the sun is above the horizon.
    """
    t = pd.DatetimeIndex(times)
    n = np.asarray((t - _J2000).total_seconds(), dtype=float) / 86400.0
    mean_long = np.radians((280.460 + 0.9856474 * n) % 360.0)
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = mean_long + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 0.0000004 * n)
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    gmst = np.radians((280.46061837 + 360.98564736629 * n) % 360.0)
    hour_angle = gmst + np.radians(LONGITUDE) - ra
    lat = np.radians(LATITUDE)
    return np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(hour_angle)


def _naive_utc(ts: datetime.datetime) -> datetime.datetime:
    if ts.tzinfo is not None:
        return ts.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return ts


def hourly_table(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly mean and sample count of the value column, indexed by hour start."""
    s = df.set_index(pd.to_datetime(df["time"])).sort_index()["value"]
    s = pd.to_numeric(s, errors="coerce").clip(lower=0)
    g = s.resample("h")
    return pd.DataFrame({"mean": g.mean(), "n": g.count()})


def _clim_array(times, cell: dict, hour: dict) -> np.ndarray:
    t = pd.DatetimeIndex(times)
    return np.array(
        [cell.get((int(m), int(h)), hour.get(int(h), np.nan)) for m, h in zip(t.month, t.hour)],
        dtype=float,
    )


def _climatology(times, y: np.ndarray, known: np.ndarray) -> typing.Tuple[dict, dict]:
    t = pd.DatetimeIndex(times)
    frame = pd.DataFrame({"m": t.month[known], "h": t.hour[known], "y": y[known]})
    cell = {(int(m), int(h)): float(v) for (m, h), v in frame.groupby(["m", "h"])["y"].mean().items()}
    hour = {int(h): float(v) for h, v in frame.groupby("h")["y"].mean().items()}
    return cell, hour


class PvForecastModel(PythonModel):
    """The model MLflow registers and op.py later loads.

    predict() receives one payload per issued forecast: the target hour, the lag
    hour and its observed mean (None when that hour is unknown). It returns the
    forecast mean power in W for the target hour.
    """

    def __init__(self, coef, clim_cell: dict, clim_hour: dict) -> None:
        self.coef = [float(c) for c in coef]
        self.clim_cell = {(int(m), int(h)): float(v) for (m, h), v in clim_cell.items()}
        self.clim_hour = {int(h): float(v) for h, v in clim_hour.items()}

    def _clim(self, t: datetime.datetime) -> typing.Optional[float]:
        v = self.clim_cell.get((t.month, t.hour))
        if v is None:
            v = self.clim_hour.get(t.hour)
        return v

    def predict(self, context, model_input=None, params=None):
        payload = model_input if model_input is not None else context
        target = _naive_utc(payload["target"])
        lag_time = _naive_utc(payload["lag_time"])
        sun = float(sun_sin_elevation([target])[0])
        if sun <= 0.0:
            return 0.0
        clim = self._clim(target)
        lag = payload.get("lag_value")
        if lag is None:
            lag = self._clim(lag_time)
        if clim is None or lag is None:
            return None
        b0, b1, b2, b3 = self.coef
        return float(max(b0 + b1 * float(lag) + b2 * clim + b3 * sun, 0.0))


def fit_model(df: pd.DataFrame):
    """Fit the four-term linear model on daylight hours. Returns (model, mae, rows)."""
    table = hourly_table(df)
    if table.empty:
        return None, None, 0
    grid = pd.date_range(table.index.min().floor("h"), table.index.max().floor("h"), freq="h")
    y = table["mean"].reindex(grid).to_numpy(dtype=float)
    sun = sun_sin_elevation(grid)
    day = sun > 0.0
    # Night hours with no reading are zero output, not missing.
    y = np.where(~day & np.isnan(y), 0.0, y)
    known = day & ~np.isnan(y)

    clim_cell, clim_hour = _climatology(grid, y, known)
    clim = _clim_array(grid, clim_cell, clim_hour)

    lag = np.full(len(y), np.nan)
    lag[LAG_HOURS:] = y[:-LAG_HOURS]
    lag_clim = np.full(len(y), np.nan)
    lag_clim[LAG_HOURS:] = clim[:-LAG_HOURS]
    lag = np.where(np.isnan(lag), lag_clim, lag)

    mask = known & ~np.isnan(clim) & ~np.isnan(lag)
    rows = int(mask.sum())
    if rows < 100:
        return None, None, rows
    X = np.column_stack([np.ones(rows), lag[mask], clim[mask], sun[mask]])
    coef, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    mae = float(np.mean(np.abs(X @ coef - y[mask])))
    return PvForecastModel(coef, clim_cell, clim_hour), mae, rows


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, fit, and hand back a model for MLflow to register."""
    with logger.trace("read history"):
        datasets = provide_historic_data(TRAINING_WINDOW)
    if not datasets:
        # Explicitly nothing rather than a model fitted on no data: returning None
        # leaves the previously registered model in place.
        return None

    with logger.trace("fit"):
        frames = [ray.get(d) if isinstance(d, ray.ObjectRef) else d for d in datasets]
        df = frames[0].to_pandas()
        if df.empty:
            return None
        model, mae, rows = fit_model(df)
        if model is None:
            return None

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "horizon_hours": 24,
        "lag_hours": LAG_HOURS,
        "features": "intercept,lag,climatology,sun",
    })
    logger.log_metrics({"train_mae_w": mae, "train_daylight_hours": rows})
    return model

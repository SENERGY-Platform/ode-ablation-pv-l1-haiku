"""Training, on Ray, and the shared solar and hourly arithmetic.

The target is the hourly mean of the inverter's total AC power, labelled by the
start of the hour. The operator issues a forecast at the start of hour h for hour
h + 24 h. The model uses these inputs, all known at issue time:

  - lag:   the mean of the last completed hour, which is 25 h before the target
  - clim:  the mean daytime output for the target's month and hour, from training
  - sun:   the sine of the solar elevation at the target (clear-sky shape)
  - cloud: the cloud cover fraction at the target hour (five-term model only)
  - a constant

Two fits are kept. The five-term fit uses cloud cover; the four-term fit does not
and is the fallback when no cloud forecast is available for the target hour.

Training uses observed cloud cover from the open-meteo history export. Inference
uses the yr.no forecast for the target hour. These are different sources, so the
cloud coefficient describes observed cloud and is applied to forecast cloud.

Hours with the sun below the horizon are forecast as zero.

provide_historic_data() is Operator Lib's reader over the platform's timeseries
store. It returns one Ray dataset per input topic, each with a "time" column
(naive UTC) and one column per mapped destination.
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


def hourly_table(df: pd.DataFrame, column: str = "value") -> pd.DataFrame:
    """Hourly mean and sample count of one column, indexed by hour start."""
    s = df.set_index(pd.to_datetime(df["time"])).sort_index()[column]
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
    hour and its observed mean (None when that hour is unknown), and the cloud
    cover fraction forecast for the target hour (None when unavailable). It
    returns the forecast mean power in W for the target hour.
    """

    def __init__(self, coef4, coef5, clim_cell: dict, clim_hour: dict) -> None:
        self.coef4 = [float(c) for c in coef4]
        self.coef5 = None if coef5 is None else [float(c) for c in coef5]
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
        cloud = payload.get("cloud")
        if cloud is not None and self.coef5 is not None:
            b = self.coef5
            x = [1.0, float(lag), clim, sun, float(cloud)]
        else:
            b = self.coef4
            x = [1.0, float(lag), clim, sun]
        return float(max(sum(bi * xi for bi, xi in zip(b, x)), 0.0))


def _lstsq_fit(X: np.ndarray, y: np.ndarray):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    mae = float(np.mean(np.abs(X @ coef - y)))
    return coef, mae


def fit_model(df_pv: pd.DataFrame, df_cloud: typing.Optional[pd.DataFrame] = None):
    """Fit the four-term and five-term models on daylight hours.

    Returns (model, mae4, mae5, rows4, rows5). mae5 and rows5 are None when no
    observed cloud history was read or too few hours had both values.
    """
    table = hourly_table(df_pv, "value")
    if table.empty:
        return None, None, None, 0, 0
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
    rows4 = int(mask.sum())
    if rows4 < 100:
        return None, None, None, rows4, 0
    ones = np.ones(rows4)
    X4 = np.column_stack([ones, lag[mask], clim[mask], sun[mask]])
    coef4, mae4 = _lstsq_fit(X4, y[mask])

    coef5, mae5, rows5 = None, None, 0
    if df_cloud is not None and not df_cloud.empty and "cloud_obs" in df_cloud.columns:
        cloud = hourly_table(df_cloud, "cloud_obs")["mean"].reindex(grid).to_numpy(dtype=float) / 100.0
        cloud = np.clip(cloud, 0.0, 1.0)
        mask5 = mask & ~np.isnan(cloud)
        rows5 = int(mask5.sum())
        if rows5 >= 100:
            X5 = np.column_stack([np.ones(rows5), lag[mask5], clim[mask5], sun[mask5], cloud[mask5]])
            coef5, mae5 = _lstsq_fit(X5, y[mask5])

    model = PvForecastModel(coef4, coef5, clim_cell, clim_hour)
    return model, mae4, mae5, rows4, rows5


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
        dfs = [f.to_pandas() for f in frames]
        # Pick each input by the column it carries, not by its position: the
        # topics are listed in the launch and the order is not a contract.
        pv = next((d for d in dfs if "value" in d.columns and not d.empty), None)
        cloud = next((d for d in dfs if "cloud_obs" in d.columns and not d.empty), None)
        if pv is None:
            return None
        model, mae4, mae5, rows4, rows5 = fit_model(pv, cloud)
        if model is None:
            return None

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "horizon_hours": 24,
        "lag_hours": LAG_HOURS,
        "features_4term": "intercept,lag,climatology,sun",
        "features_5term": "intercept,lag,climatology,sun,cloud_obs",
        "cloud_at_inference": "yr.no forecast instant_cloud_area_fraction",
    })
    metrics = {"train_daylight_rows_4term": rows4}
    if mae4 is not None:
        metrics["train_mae_w_4term"] = mae4
    if mae5 is not None:
        metrics["train_mae_w_5term"] = mae5
        metrics["train_daylight_rows_5term"] = rows5
    logger.log_metrics(metrics)
    return model

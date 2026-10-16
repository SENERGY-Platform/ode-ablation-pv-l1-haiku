"""The operator: what it infers per message, and when it retrains.

MLOperator is the machine-learning half of Operator Lib. It loads the model
registered under this pipeline and operator from MLflow, calls infer() for every
message that matches a selector, and calls train() when there is no model yet or
when need_retraining() says so. Training runs on Ray.

Three inputs arrive, each under its own selector:

  - "value":    the inverter's total AC power, one reading per message.
  - "forecast": the yr.no forecast, one cloud-cover value per forecasted hour.
                Read live only; kept as the cloud term for the hour it is about.
  - "history":  open-meteo observed cloud cover. Used by training only; infer
                ignores it.

At the start of each hour infer() closes the previous hour and issues one forecast:
the mean power for the hour 24 h ahead. The result carries that hour's timestamp,
so the forecast is scored against the hour it is about.

Three methods are yours. Everything else — Kafka, the model registry, the
lifecycle — belongs to the library.
"""

import datetime
import typing

import pandas as pd
from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger
from operator_lib.util.helpers.data import provide_historic_data_local

from training import train_model, hourly_table, sun_sin_elevation, LAG_HOURS


# How much history to read when the operator starts, so the lag hour and the
# current hour are known before the first forecast.
SEED_HOURS = 49


def _utc_naive(ts: datetime.datetime) -> datetime.datetime:
    if ts.tzinfo is not None:
        return ts.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return ts


def _hour_of(value) -> datetime.datetime:
    """The start of the hour a timestamp (string or datetime) falls in, naive UTC."""
    t = pd.Timestamp(value)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.floor("h").to_pydatetime()


class CustomConfig(Config):
    """Deployment configuration, typed.

    The base Config already carries mlflow_url, ray_url and ts_conn. Anything
    added here arrives from the operator's deployment config under the same name,
    with this value as the default.
    """

    # Retrain at most this often, in seconds. A day, so a deployment does not
    # spend its life training.
    retrain_after_s = 86400


class Operator(MLOperator):
    configType = CustomConfig

    selectors = [
        Selector({"name": "value", "args": ["value"]}),
        Selector({"name": "forecast", "args": ["cloud_fc", "forecasted_for"]}),
        Selector({"name": "history", "args": ["cloud_obs"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns, so anything set after it is missing
        # during the replay and overwrites what train() just recorded.
        self.trained_at: typing.Optional[datetime.datetime] = None
        self._hour: typing.Optional[datetime.datetime] = None  # start of the hour being filled
        self._sum = 0.0  # sum of readings in that hour
        self._n = 0  # number of readings in that hour
        self._hourly: typing.Dict[datetime.datetime, typing.Optional[float]] = {}
        self._cloud: typing.Dict[datetime.datetime, float] = {}  # forecast hour -> cloud %
        super().init(*args, **kwargs)

    def _seed(self, hour: datetime.datetime) -> None:
        """Fill the hourly state from the history before the first message."""
        self._hour = hour
        self._sum, self._n = 0.0, 0
        datasets = provide_historic_data_local(datetime.timedelta(hours=SEED_HOURS))
        if not datasets:
            return
        frames = [d.to_pandas() for d in datasets]
        pv = next((f for f in frames if "value" in f.columns and not f.empty), None)
        if pv is None:
            return
        table = hourly_table(pv, "value")
        for h, row in table.iterrows():
            h = h.to_pydatetime()
            if h > hour:
                continue
            n = int(row["n"])
            if h == hour:
                if n > 0:
                    self._sum = float(row["mean"]) * n
                    self._n = n
                continue
            if n > 0:
                self._hourly[h] = float(row["mean"])
            else:
                # No reading in an hour with the sun down is zero output.
                self._hourly[h] = 0.0 if float(sun_sin_elevation([h])[0]) <= 0.0 else None
        self._trim(hour)

    def _trim(self, hour: datetime.datetime) -> None:
        cutoff = hour - datetime.timedelta(hours=SEED_HOURS)
        for k in [k for k in self._hourly if k < cutoff]:
            del self._hourly[k]

    def _close_hour(self) -> None:
        """Record the mean of the hour that has just ended."""
        if self._n > 0:
            self._hourly[self._hour] = self._sum / self._n
        else:
            self._hourly[self._hour] = 0.0 if float(sun_sin_elevation([self._hour])[0]) <= 0.0 else None
        self._trim(self._hour)

    def _note_forecast(self, data: typing.Dict[str, typing.Any]) -> None:
        """Keep the latest yr.no cloud forecast for each forecast hour."""
        cloud = data.get("cloud_fc")
        when = data.get("forecasted_for")
        if cloud is None or when is None:
            return
        try:
            target = _hour_of(when)
        except (ValueError, TypeError):
            return
        self._cloud[target] = float(cloud)
        if self._hour is not None:
            for k in [k for k in self._cloud if k < self._hour]:
                del self._cloud[k]

    def _issue(self, model: typing.Optional[PyFuncModel], hour: datetime.datetime):
        """Forecast the hour 24 h after the start of this hour."""
        if model is None:
            return None, None, None
        target = hour + datetime.timedelta(hours=24)
        lag_time = target - datetime.timedelta(hours=LAG_HOURS)
        cloud_pct = self._cloud.get(target)
        payload = {
            "target": target,
            "lag_time": lag_time,
            "lag_value": self._hourly.get(lag_time),
            "cloud": None if cloud_pct is None else cloud_pct / 100.0,
        }
        prediction = model.predict(payload)
        if prediction is None:
            return None, None, None
        return target.replace(tzinfo=datetime.timezone.utc), {"prediction": float(prediction)}, None

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        """Called for every message. Returns (result timestamp, result, new model).

        Forecast messages and historic cloud messages only update state. Inverter
        readings add to the current hour; the first reading of each hour closes the
        previous hour and issues one forecast for the hour 24 h ahead.
        """
        if selector == "history":
            return None, None, None
        if selector == "forecast":
            self._note_forecast(data)
            return None, None, None

        value = data.get("value")
        if value is None:
            return None, None, None

        ts = _utc_naive(timestamp)
        hour = ts.replace(minute=0, second=0, microsecond=0)
        if self._hour is None:
            self._seed(hour)

        result = (None, None, None)
        if hour != self._hour:
            self._close_hour()
            self._hour = hour
            self._sum, self._n = 0.0, 0
            result = self._issue(model, hour)

        self._sum += max(float(value), 0.0)
        self._n += 1
        return result

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        """Called when there is no model, or when need_retraining() said so.

        Runs inside a Ray session the library opened, with an MLflow run already
        started — so params and metrics logged through "logger" land on the run
        the resulting model is registered from.
        """
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        """Called after every inference, so it has to be cheap.

        Time-based to start with. A better policy watches the data — a drift in the
        input distribution, or an error that stops falling — and that is a decision
        to make with the profile in front of you rather than a default to inherit.
        """
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s

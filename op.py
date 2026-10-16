"""The operator: what it infers per message, and when it retrains.

MLOperator is the machine-learning half of Operator Lib. It loads the model
registered under this pipeline and operator from MLflow, calls infer() for every
message that matches a selector, and calls train() when there is no model yet or
when need_retraining() says so. Training runs on Ray.

infer() receives one inverter reading per message. It keeps the hourly means it
has seen, and at the start of each hour it issues one forecast: the mean power for
the hour 24 h ahead. The result carries that hour's timestamp, so the forecast is
scored against the hour it is about.

Three methods are yours. Everything else — Kafka, the model registry, the
lifecycle — belongs to the library.
"""

import datetime
import typing

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

    # The inverter's total AC power, in W, arrives as "value".
    selectors = [
        Selector({"name": "value", "args": ["value"]}),
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
        super().init(*args, **kwargs)

    def _seed(self, hour: datetime.datetime) -> None:
        """Fill the hourly state from the history before the first message."""
        self._hour = hour
        self._sum, self._n = 0.0, 0
        datasets = provide_historic_data_local(datetime.timedelta(hours=SEED_HOURS))
        if not datasets:
            return
        df = datasets[0].to_pandas()
        if df.empty:
            return
        table = hourly_table(df)
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

    def _issue(self, model: typing.Optional[PyFuncModel], hour: datetime.datetime):
        """Forecast the hour 24 h after the start of this hour."""
        if model is None:
            return None, None, None
        target = hour + datetime.timedelta(hours=24)
        lag_time = target - datetime.timedelta(hours=LAG_HOURS)
        payload = {
            "target": target,
            "lag_time": lag_time,
            "lag_value": self._hourly.get(lag_time),
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

        Most messages only add to the current hour and return nothing. The first
        message of each hour closes the previous hour and issues one forecast for
        the hour 24 h ahead.
        """
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

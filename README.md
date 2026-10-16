# ode-ablation-pv-l1-haiku

## Approach: 24-hour-ahead PV forecast

**Problem.** Forecast the plant's hourly mean power, in W, 24 hours ahead. The
criterion is in `evaluation.yaml`: mean absolute error in W, against the withheld
target at 1 h resolution, over 2026-09-01 to 2026-10-01. The threshold is 30 W,
and lower is better.

**Status.** The operator is not complete. The best evaluation MAE so far is
65.0 W, which does not meet the threshold. The runs and their results are listed
below.

### Data

- **Target.** `root.powerTotal` from the inverter "Wechselrichter" (APSystems
  DS3-S), device `urn:infai:ses:device:89169393-068d-4fb1-a032-2f238ad7ac9d`,
  service `9ddd93b2-0988-407f-931d-a2ebcc8f6d94`. It is an instantaneous W series
  sampled every 60 s, with history from 2022-10. The developer confirmed this
  series as the target. Nightly gaps are read as the inverter sleeping, so they are
  zero output. Daytime outages are treated as missing, not as zero.
- **Weather at inference.** The yr.no 48-hour forecast import
  (`urn:infai:ses:import:a50aa583-282e-56c2-b101-7aaf68ebd2b9`, export
  `e31775ba-4bf5-46a5-ba8f-81adb3977104`), read as `instant_cloud_area_fraction`
  per forecast hour. The export holds only a few days of history, so its coverage
  of the test window has not been confirmed.
- **Weather at training.** The open-meteo history import
  (`urn:infai:ses:import:57ebf2a0-f019-6426-bdf4-b12a1e7363f0`, export
  `808761f2-43ea-40c3-bf3d-c51b44aa1b53`), read as observed `cloudcover`. This is
  realised weather, so it is used only to fit the cloud term and is never used to
  forecast future hours.
- **Location.** The plant is at approximately 51°30′ N, 10°00′ E, and that is used
  for solar geometry. Both weather exports are labelled 51.7 / 10, about 22 km
  north of the plant.
- **Training window.** The 365 days before the training end, 2026-09-01.

### Model

- The target is the hourly mean, labelled by the start of the hour. At the start of
  hour *h* the operator issues one forecast for hour *h* + 24 h.
- **Four-term fit,** by least squares on daylight hours: an intercept, the lag (the
  most recent completed hour, 25 h before the target), the month-and-hour
  climatology of daytime output, and the sine of the solar elevation at the target
  hour. Night hours are forecast as zero.
- **Five-term fit,** which adds cloud cover at the target hour. Training uses the
  observed cloud, and inference uses the yr.no forecast for that hour. When no
  forecast is available for the target hour, the operator falls back to the
  four-term fit.
- Each forecast reports `prediction` in W and `cloud_used`, which is 1.0 when a
  forecast cloud value was present for that hour.

### Runs

All three are evaluations with the data split confirmed.

| Run | Inputs | Evaluation MAE (W) |
|---|---|---|
| `e7d856e7` baseline | inverter only | 67.06 |
| `826a008b` cloud term | inverter, yr.no forecast, open-meteo history | 64.996 |
| `d8b56a54` cloud_used output | as above, plus `cloud_used` in the output | 64.996 |

In each run, 422 forecasts were issued and 390 were scored. The 32 unscored
forecasts are not explained by the run summary.

### Open items

- The MAE split by `cloud_used` has not been computed. It is in
  `evaluation/predictions.csv` of run `d8b56a54`. The count of forecasts that used
  the cloud term, and the MAE for each group, would show whether the cloud term
  contributes at all.
- The yr.no export's coverage of September 2026 has not been confirmed.
- The 32 unscored forecasts have not been explained.
- Whether 30 W is reachable with these inputs has not been decided. `evaluation.yaml`
  belongs to the developer.

### Limitations

- The cloud term is fitted on observed cloud and applied to forecast cloud. The
  two sources differ, so the coefficient may be biased at inference.
- The model is linear. Its in-sample MAE on daylight hours is about 77 to 78 W, so it
  underfits. Irradiance, temperature and soiling are not included.
- There is one plant and one year of training data. Seasonal robustness has not
  been tested.
- The run summary labels the direction of elapsed-time metrics the wrong way round.
  Read those deltas as noise.
- Retraining is time-based, once a day. There is no drift detection.

### Code

- `training.py`: solar geometry, hourly arithmetic, the two fits and the training
  pass.
- `op.py`: the three selectors (inverter, yr.no forecast, open-meteo history), the
  hourly state, and the forecast issued at each hour start.
- `operator.yaml`: the inputs and outputs the code reads and writes.

---

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.8.1". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-pv-l1-haiku". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-pv-l1-haiku:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.8.1", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.

# Training Guide

The active uncertainty workflow is policy-agnostic. The Evidential Neural
Network (ENN) is trained by behavior cloning on Grid2Op observation/action
pairs produced by the configured policy.

Any policy object exposing `act(...)` can be used. For external policies,
configure a factory with:

```dotenv
AGENT_NAME=my_agent
AGENT_FACTORY=my_package.my_agent:make_agent
```

The factory receives the Grid2Op environment and returns the policy instance:

```python
def make_agent(env):
    return MyGrid2OpAgent(env.action_space)
```

## Recommended End-To-End Path

Collect agent rollouts:

```bash
python training/collect_rollouts.py
```

Train and export the ENN bundle:

```bash
python training/train_enn.py
```

Run the local example:

```bash
python run_example.py
```

The original `run_pipeline.py` workflow is archived under
`archive/legacy/old_version/` and is not part of the active training path.

## Rollout Collection

For the configured default CurriculumAgent:

```bash
python training/collect_rollouts.py --agent-name curriculum --episodes 50
```

For a custom policy:

```bash
python training/collect_rollouts.py \
  --agent-name my_agent \
  --agent-factory my_package.my_agent:make_agent \
  --episodes 50
```

By default, rollout bundles are written to:

```text
artifacts/<ENV_NAME>/<agent>/rollouts/
|-- observations.npy
|-- labels.npy
`-- actions.npy
```

`actions.npy` contains distinct action vectors in stable first-observed order.
`labels.npy` contains the corresponding action-class index for each
observation.

At least two distinct policy actions are required for meaningful ENN training.

When copying artifacts between machines, copy the contents into the existing
`artifacts/` directory. Do not copy the directory itself into `artifacts/`,
which creates an unintended `artifacts/artifacts/` tree.

## ENN Training

Train from the collected rollout bundle:

```bash
python training/train_enn.py --agent-name curriculum
```

For explicit paths:

```bash
python training/train_enn.py \
  --agent-name my_agent \
  --data-dir artifacts/<ENV_NAME>/my_agent/rollouts \
  --out-dir artifacts/<ENV_NAME>/my_agent/model
```

The trainer exports:

```text
artifacts/<ENV_NAME>/<agent>/model/
|-- enn_<agent>.pth
|-- scaler_params.json
|-- enn_meta.json
`-- enn_pctile_calib.npz
```

## ENN Objective

The ENN uses an evidential classification objective based on a Dirichlet
distribution. The training loss combines Bayes-risk cross-entropy with an
annealed KL penalty toward the uniform Dirichlet prior.

The resulting vacuity signal is used as the direct epistemic uncertainty
measure:

```text
u = K / S
```

where `K` is the number of retained action classes and `S` is the total
Dirichlet strength.

## Calibration

ENN percentile calibration is generated during `training/train_enn.py` and
stored beside the trained model as:

```text
enn_pctile_calib.npz
```

The calibration reference maps raw ENN vacuity to relative percentiles used by
`recommendation_uncertainty.py`. For a different reference set, use
`recommendation_uncertainty.build_calibration()` and
`recommendation_uncertainty.save_calibration()` with the chosen scaled input
vectors.

## Validation

Run:

```bash
python -m unittest discover -s tests -v
```

The fast suite covers ENN inference and calibration, the API contract, and the
failure-forecast classifier without requiring trained production artifacts or a
live Grid2Op environment.

A live Grid2Op validation requires the target environment dataset, active agent
assets, trained ENN artifacts, and the pinned Grid2Op/LightSim dependency stack.

For a small end-to-end pipeline check that writes only to `artifacts_smoke/`,
run:

```bash
python smoke_test.py
```

The smoke test uses short runs and adds clearly marked synthetic positive rows
only when the tiny sample does not contain both classifier labels.

## Failure Forecasting

The failure forecast predicts whether the configured RL agent will fail one
hour ahead after a candidate line disconnection. Forecast inputs contain the
current, 1-hour, 1-day, and 1-week grid values plus time features. Collection
stores one origin every 12 steps, rotates the five-minute offset by episode,
and pairs each origin with its actual `t+12` injections. The resulting
`forecast_X.npy`, `forecast_y.npy`, and `forecast_episode_ids.npy` files support
an episode-separated tuning split for the mean forecaster.

Failure-row collection builds one common forecasted `t+12` state and evaluates
each currently connected selected line in an isolated contingency branch. The
agent action used for the label is requested from the post-contingency state;
runtime prediction does not execute the agent. During row collection, forecast
or contingency states whose power flow does not converge are omitted because
they do not provide a valid classifier feature row. Runtime raises for an
invalid forecast base state and skips invalid contingency branches.

Run the full pipeline:

```bash
python training/run_failure_forecast_pipeline.py \
  --agent-artifact-dir artifacts/<ENV_NAME>/<AGENT_NAME> \
  --policy-agent-dir assets/<ENV_NAME> \
  --episodes 50 \
  --lines line_a,line_b \
  --smoke-line line_a
```

Train the mean and aleatoric forecast models:

```bash
python training/train_failure_forecasters.py \
  --agent-artifact-dir artifacts/<ENV_NAME>/<AGENT_NAME> \
  --episodes 50
```

Collect rows:

```bash
python training/collect_failure_forecast.py \
  --mean-model artifacts/<ENV_NAME>/<AGENT_NAME>/failure_forecast/mean_forecaster.pkl \
  --aleatoric-model artifacts/<ENV_NAME>/<AGENT_NAME>/failure_forecast/aleatoric_forecaster.pkl \
  --episodes 50 \
  --lines line_a,line_b
```

Train the classifier:

```bash
python training/train_failure_forecast.py
```

Smoke-test one prediction:

```bash
python training/predict_failure_forecast.py \
  --mean-model artifacts/<ENV_NAME>/<AGENT_NAME>/failure_forecast/mean_forecaster.pkl \
  --aleatoric-model artifacts/<ENV_NAME>/<AGENT_NAME>/failure_forecast/aleatoric_forecaster.pkl \
  --line line_a
```

`--line` accepts a comma-separated list. Omit it to predict all lines stored in
the classifier metadata. Each returned item contains the line name, failure
probability, and thresholded binary prediction.

For the production failure-forecast bundle, configure `.env` and run the
in-process trainer:

```bash
python forecaster_full_train.py
```

This script collects and validates genuine `t -> t+12` pairs, trains both
forecasters and the classifier, reopens the saved bundle, and checks per-line
inference.

Artifacts are written under:

```text
artifacts/<ENV_NAME>/<AGENT_NAME>/failure_forecast/
|-- forecast_X.npy
|-- forecast_y.npy
|-- forecast_episode_ids.npy
|-- mean_forecaster.pkl
|-- aleatoric_forecaster.pkl
|-- failure_forecast_rows.csv
|-- failure_classifier.pkl
`-- failure_classifier_metadata.json
```

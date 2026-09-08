# Grid2Op Agent Failure Forecasting and Evidential Uncertainty

This repository provides uncertainty estimation and contingency-risk prediction for Grid2Op control policies.

It exposes two complementary signals:

1. **Epistemic uncertainty for an agent recommendation**, estimated with an Evidential Neural Network (ENN) trained by behavior cloning.
2. **Probability of failure for a requested line disconnection**, estimated with a supervised classifier using the current grid state, forecast-derived variables, aleatoric uncertainty, and ENN epistemic uncertainty.

The ENN training pipeline supports both pre-generated tutor datasets and direct policy rollouts. If tutor data is unavailable, the configured Grid2Op agent is executed to collect the observation/action pairs required for ENN training.

## Repository structure

```text
app/
  main.py                     FastAPI recommendation service
  API.md                      API documentation

src/
  agent_runtime.py            Generic Grid2Op policy loading and invocation
  collect_data.py             Failure-label dataset generation
  config.py                   Model/training configuration
  enn_data.py                 ENN tutor/rollout dataset resolution
  enn_models.py               Evidential neural network and losses
  failure_probability.py      Observation + line -> P(failure)
  pipeline_artifacts.py       Artifact validation and metadata helpers
  rule_predictor.py           Symbolic-rule inference utilities
  train_classifier.py         Failure classifier training
  train_forecast.py           Forecast and aleatoric-uncertainty training
  training_enn.py             ENN training and artifact export
  utils.py                    Shared utilities

training/
  collect_rollouts.py         Standalone policy rollout collector
  train_curriculumagent.py    CurriculumAgent training entry point
  train_enn.py                ENN training entry point
  TRAINING.md                 Additional training notes

tests/
  test_api.py
  test_enn_training_fallback.py
  test_new_components.py
  validate_module.py

project_config.py             Project-level paths and runtime configuration
recommendation_uncertainty.py Public ENN uncertainty KPI API
run_pipeline.py               End-to-end training/artifact pipeline
run_example.py                Live recommendation example
```

## Requirements

The pinned dependencies target Python 3.9 or 3.10 because several training dependencies do not provide compatible wheels for newer Python versions.

Install the environment with:

```bash
python -m pip install -r requirements.txt
```

The project uses Grid2Op `1.9.8` and LightSim2Grid `0.10.3` to keep the environment/action representation consistent with the expected simulator stack.

## Configuration

Copy the environment template:

```bash
cp .env.example .env
```

Set the Grid2Op environment, agent, artifact paths, and training parameters in `.env`.

A generic external agent can be configured with:

```dotenv
ENV_NAME=ai4realnet_small
ENV_LOCATION=/path/to/grid2op/data
AGENT_NAME=my_agent
AGENT_FACTORY=my_package.my_agent:make_agent
```

`AGENT_FACTORY` must use the form:

```text
package.module:function
```

The factory receives the already-created Grid2Op environment and returns an object exposing `act(...)`:

```python
def make_agent(env):
    return MyGrid2OpAgent(env.action_space)
```

`src.agent_runtime.call_agent` supports common Grid2Op agent signatures including:

```python
act(obs)
act(obs, reward, done)
act(obs, reward=reward, done=done)
```

If `AGENT_FACTORY` is empty, the bundled CurriculumAgent-compatible asset loader is used.

## ENN training data

The ENN models the configured policy by learning the policy's action distribution from observation/action pairs.

`src.enn_data.load_or_collect_enn_data` supports three data-source modes:

- `auto`: use tutor splits when all required files are available; otherwise collect policy rollouts.
- `tutor`: require the configured tutor train/validation/test files.
- `rollout`: always execute the configured policy and construct the ENN dataset from rollouts.

### Automatic rollout path

When policy rollouts are used, the collector:

1. creates the configured Grid2Op environment;
2. creates the configured policy;
3. executes the policy for the configured number of episodes;
4. stores `obs.to_vect()` and the corresponding `action.to_vect()`;
5. deduplicates action vectors in stable first-observed order;
6. converts action vectors to class labels;
7. creates train, validation, and test splits.

Typical configuration:

```dotenv
ROLLOUT_EPISODES=50
ENN_ROLLOUT_MAX_STEPS=0
ENN_EPOCHS=100
ENN_BATCH_SIZE=512
ENN_LR=1e-3
ENN_VAL_FRAC=0.1
```

`ENN_ROLLOUT_MAX_STEPS=0` means that no explicit per-episode rollout cap is applied.

Programmatic training:

```python
from src.training_enn import train_enn

# Tutor data when available, otherwise direct agent rollouts.
model = train_enn(data_source="auto")

# Always collect supervision from the configured policy.
model = train_enn(data_source="rollout")

# Require tutor files.
model = train_enn(data_source="tutor")
```

The rollout dataset contains:

```text
observations.npy
actions.npy
labels.npy
train.npz
validation.npz
test.npz
```

At least two distinct policy actions are required for meaningful ENN training. The collector raises an error if the collected policy behavior contains only one action class.

## Epistemic uncertainty KPI

The public uncertainty API is implemented in `recommendation_uncertainty.py`.

For a Dirichlet ENN with `K` retained action classes and total Dirichlet strength `S`, evidential vacuity is:

```text
u = K / S
```

The public percentage is:

```text
epistemic_uncertainty_pct = 100 * clip(u, 0, 1)
```

The result also contains a percentile calibrated against the reference ENN uncertainty distribution:

```text
epistemic_uncertainty_total_pctile in [0, 100]
```

When the recommended action can be mapped to an ENN class, an action-conditional percentile is also returned:

```text
epistemic_uncertainty_action_pctile
```

### Uncertainty and confidence bands

The calibrated total percentile is mapped to three bands:

| Calibrated percentile | Uncertainty | Confidence |
|---:|---|---|
| `< 33.33` | `low` | `high` |
| `33.33 - < 66.67` | `medium` | `medium` |
| `>= 66.67` | `high` | `low` |

Example output:

```python
{
    "epistemic_uncertainty_pct": 57.9,
    "epistemic_uncertainty_total_pctile": 4.6,
    "epistemic_uncertainty_action_pctile": 12.3,
    "epistemic_uncertainty_level": "low",
    "epistemic_confidence_level": "high",
}
```

The `high` / `medium` / `low` value is a categorical confidence band. It is not a frequentist confidence interval with a specified coverage probability.

### Scoring a recommendation

Use the action already selected by the policy:

```python
from recommendation_uncertainty import assess_recommendation

action = agent.act(obs, reward, done)
info = assess_recommendation(
    obs,
    agent,
    enn,
    calibration,
    action=action,
)
```

Passing the selected action is important for stochastic policies because the uncertainty score must correspond to the exact recommendation returned to the caller.

## Failure probability for a line disconnection

The public component is `src.failure_probability.FailureProbabilityPredictor`.

It accepts a Grid2Op observation and a line identifier and returns the classifier estimate of:

```text
P(failure = 1 | observation, requested line disconnection)
```

Load the trained pipeline artifacts with:

```python
from src.failure_probability import FailureProbabilityPredictor

predictor = FailureProbabilityPredictor.from_pipeline()

probability = predictor.predict_proba(obs, "62_58_180")
```

`predict_proba` returns a float in `[0, 1]`.

For a structured result:

```python
result = predictor.predict(obs, "62_58_180")
```

Example:

```python
{
    "line": "62_58_180",
    "failure_probability": 0.27,
    "failure_probability_pct": 27.0,
    "threshold": 0.41,
    "predicted_failure": False,
}
```

An integer line id is accepted when the observation exposes `name_line`.

### Failure target

The classifier target is generated by `src/collect_data.py`.

For each sample, the environment is advanced according to the configured policy, the requested line contingency is applied, and `failure=1` is assigned when the contingency terminates the Grid2Op episode according to the collection procedure.

### Supported line domain

Line identity is treated as a categorical feature. The predictor rejects lines that were absent from classifier training instead of extrapolating an arbitrary encoded category.

To support an additional contingency line:

1. include the line during data collection;
2. regenerate the classifier dataset;
3. retrain the classifier.

### Feature contract

The classifier uses the feature order stored in `classifier_metadata.json`. The feature set includes current-state grid statistics, forecast statistics, aleatoric uncertainty, and ENN epistemic uncertainty.

`FailureProbabilityPredictor.from_pipeline()` loads the trained dependencies and constructs the required feature vector from the Grid2Op observation/history.

For dependency injection or unit testing:

```python
predictor = FailureProbabilityPredictor.from_artifacts(...)
```

The returned value is a model-estimated probability. For safety-critical probability interpretation, evaluate calibration on an independent held-out environment distribution using metrics such as Brier score and reliability diagrams, and apply an explicit probability calibrator if required.

## End-to-end training pipeline

Run:

```bash
python run_pipeline.py
```

The runner validates existing artifacts before reuse. ENN artifacts are checked for internally consistent model dimensions, scaler parameters, calibration arrays, action mappings, and metadata. Classifier metadata is checked for feature ordering, line mappings, and the selected decision threshold.

When ENN training is required and tutor data is unavailable, the pipeline collects the required policy rollouts automatically.

## Model artifacts

The primary artifact directory is:

```text
artifacts/<environment>/<agent>/model/
```

The ENN/classifier bundle includes files such as:

```text
enn_36.pth
enn_<agent>.pth
enn_best_<environment>.pth
scaler_<environment>_enn.pkl
scaler_params.json
enn_meta_<environment>.json
enn_meta.json
actions.npy
enn_pctile_calib.npz
final_classifier_36.pkl
classifier_metadata.json
```

Some filenames retain `_36` to preserve compatibility with existing scripts. Model dimensions are validated against the configured environment and metadata rather than inferred from the filename.

Treat the following ENN files as one model contract:

- ENN checkpoint;
- scaler/scaler parameters;
- ENN metadata;
- `actions.npy`;
- ENN percentile calibration.

Treat `final_classifier_36.pkl` and `classifier_metadata.json` as one classifier contract.

## Standalone rollout collection

Policy behavior can be collected without running the full training pipeline:

```bash
python training/collect_rollouts.py \
  --agent-factory my_package.my_agent:make_agent \
  --episodes 50 \
  --out-dir artifacts/my_rollouts
```

Use:

```bash
python training/collect_rollouts.py --help
```

for the complete command-line interface.

## Inference example

Run a live recommendation example with:

```bash
python run_example.py
```

The output includes the policy recommendation and ENN uncertainty KPI.

## FastAPI service

Start the API with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The recommendation response exposes the uncertainty values under `kpis`, including:

```text
epistemic_uncertainty_pct
epistemic_uncertainty_total_pctile
epistemic_uncertainty_action_pctile
epistemic_uncertainty_level
epistemic_confidence_level
```

See `app/API.md` for endpoint details.

## Testing

Run the repository regression suite with:

```bash
python -m compileall -q .
pytest -q
```

The synthetic tests cover:

- ENN training when tutor files are absent;
- policy-rollout dataset generation;
- uncertainty percentage and confidence/uncertainty bands;
- failure-probability inference and line-domain validation;
- FastAPI recommendation output.

A live Grid2Op integration test additionally requires the target Grid2Op dataset and the pinned Grid2Op/LightSim dependency stack.

## Main Python interfaces

### Build a policy

```python
from src.agent_runtime import build_agent

agent = build_agent(env, factory_spec="my_package.agent:make_agent")
```

### Resolve ENN data

```python
from src.enn_data import load_or_collect_enn_data

bundle = load_or_collect_enn_data(CFG, source="auto")
```

### Compute epistemic uncertainty

```python
from recommendation_uncertainty import assess_recommendation

info = assess_recommendation(obs, agent, enn, calibration, action=action)
```

### Compute line-disconnection failure probability

```python
from src.failure_probability import FailureProbabilityPredictor

predictor = FailureProbabilityPredictor.from_pipeline()
p_failure = predictor.predict_proba(obs, line)
```

## Artifact integrity

When moving or deploying trained models:

- keep ENN weights, scaler, metadata, action set, and calibration files together;
- keep classifier metadata beside the classifier model;
- do not replace `actions.npy` independently of the ENN checkpoint and class mapping;
- regenerate model artifacts when switching to an incompatible Grid2Op environment;
- retrain the classifier when adding previously unseen contingency lines.

## License

See `LICENSE`.

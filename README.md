# RL Agent Recommendation Uncertainty

This repository now keeps the active workflow focused on agent-agnostic
epistemic uncertainty scoring for Grid2Op recommendations.

The current flow observes any agent that exposes:

```python
agent.act(obs, reward, done)
```

It collects the agent's own rollout behavior, trains an Evidential Neural
Network (ENN) on those observation/action pairs, and adds uncertainty
percentiles to each recommendation.

The original `run_pipeline.py` failure-forecast workflow has been archived in
`archive/legacy/old_version/`.

## Quick Links

- [Installation](#installation)
- [Active Workflow](#active-workflow)
- [Supported Grid2Op Environment](#supported-grid2op-environment)
- [Curriculum agent](#curriculum-agent)
- [Configuration](#configuration)
- [ENN Training](#enn-training)
- [Project Structure](#project-structure)
- [API](#api)
- [Tests](#tests)
- [Legacy Workflow](#legacy-workflow)


## Installation

Use Python 3.9 or 3.10. Several pinned ML dependencies, especially Grid2Op,
TensorFlow, Ray, and Torch, should not be silently upgraded.

```bash
conda create -n enn_uq python=3.10 -y
conda activate enn_uq
pip install -r requirements.txt
```


## Active Workflow

```bash
python training/collect_rollouts.py
python training/train_enn.py
python run_example.py
```

The active workflow uses:

- `.env` and `project_config.py` for shared configuration.
- `training/collect_rollouts.py` to collect `(observation, action)` pairs from
  a live agent.
- `training/train_enn.py` to train and export the ENN bundle.
- `recommendation_uncertainty.py` to score recommendations.
- `run_example.py` to run an end-to-end local example.
- `app/main.py` to expose the recommendation API.

Generated rollout and ENN artifacts are written under:

```text
artifacts/<ENV_NAME>/<agent>/
|-- rollouts/
|   |-- observations.npy
|   |-- labels.npy
|   `-- actions.npy
`-- model/
    |-- enn_<agent>.pth
    |-- scaler_params.json
    |-- enn_meta.json
    `-- enn_pctile_calib.npz
```

## Supported Grid2Op Environment

The active configuration defaults to:

```text
ENV_NAME=ai4realnet_small # from https://github.com/ainetus/grid2op-scenario
```

The local Grid2Op scenario files are expected under:

```text
<ENV_LOCATION>/<ENV_NAME>
```

With the default `.env.example`, the committed local scenario is:

```text
environment/ai4realnet_small/
```

### Curriculum Agent

The active workflow expects a pre-trained CurriculumAgent package for the
`ai4realnet_small` Grid2Op environment. This package is distributed through the
project's GitHub Releases and should be placed under:

```text
assets/ai4realnet_small/
|-- model/
`-- actions/
```

The `model/` directory contains the trained agent model, and `actions/`
contains the discrete action set used by the agent. The rollout, ENN training,
example, and API scripts all expect this package to be available before they
can collect agent behavior or produce recommendations.

If the released agent artifact is not available, or if the agent needs to be
trained again for a changed environment or action space, retrain it with:

```bash
python training/train_curriculumagent.py
```

That script builds the Grid2Op environment from `.env` / `project_config.py`,
initializes `CurriculumAgent`, and runs its full Teacher -> Tutor -> Junior ->
Senior training pipeline. The trained package is saved back to:

```text
assets/<ENV_NAME>/
```

With the default configuration, this resolves to `assets/ai4realnet_small/`.


## Configuration
Create a local `.env` file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

The active scripts read configuration from environment variables first, then
`.env`, then defaults in `project_config.py`.

Main settings:

```text
ENV_NAME=ai4realnet_small
ENV_LOCATION=environment
AGENT_NAME=curriculum
AGENT_FACTORY=
ASSETS_DIR=assets
ARTIFACTS_DIR=artifacts
ROLLOUT_EPISODES=50
ENN_ROLLOUT_MAX_STEPS=0
ENN_EPOCHS=100
ENN_ANNEAL_EPOCHS=10
ENN_BATCH_SIZE=512
ENN_LR=1e-3
ENN_VAL_FRAC=0.1
EXAMPLE_N_STEPS=5
SEED=0
CURRICULUM_ITERATIONS=50
CURRICULUM_JOBS=1
CURRICULUM_TUTOR_DO_NOTHING_THRESHOLD=0.85
CURRICULUM_TUTOR_BEST_ACTION_THRESHOLD=0.999
CURRICULUM_TUTOR_MIN_UNIQUE_ROWS=100
```

The active CurriculumAgent package should be available under:

```text
assets/<ENV_NAME>/model/
assets/<ENV_NAME>/actions/
```

With the default environment name:

```text
assets/ai4realnet_small/model/
assets/ai4realnet_small/actions/
```

## ENN Training

Collect rollouts:

```bash
python training/collect_rollouts.py --agent curriculum --episodes 50
```

Train and export the ENN bundle:

```bash
python training/train_enn.py --agent-name curriculum
```

For a different agent, configure `AGENT_FACTORY=module:function`. The factory
receives the Grid2Op environment and must return an object exposing
`agent.act(obs, reward, done)`.

More training details are in `training/TRAINING.md`.

## Project Structure

```text
.
|-- .env.example
|-- project_config.py
|-- recommendation_uncertainty.py
|-- run_example.py
|-- app/
|   |-- main.py
|   `-- API.md
|-- training/
|   |-- collect_rollouts.py
|   |-- train_enn.py
|   |-- train_curriculumagent.py
|   `-- TRAINING.md
|-- src/
|   |-- agent_runtime.py
|   |-- enn_data.py
|   `-- enn_models.py
|-- tests/
|   |-- validate_module.py
|   |-- test_api.py
|   `-- synthetic fixtures
|-- assets/
|-- artifacts/
|-- environment/
|-- curriculumagent/
`-- archive/legacy/old_version/
```



## API

Run locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker:

```bash
docker build -t curriculum-agent-api .
docker run --env-file .env -p 8000:8000 curriculum-agent-api
```

Endpoint:

```text
POST /api/v1/recommendation
GET  /health
```

The API returns main-project recommendation dictionaries. ENN uncertainty KPIs
are included under `kpis`:

- `uncertainty`
- `epistemic_uncertainty_total_pctile`
- `epistemic_uncertainty_action_pctile`

See `app/API.md` for the request/response contract and deployment notes.

## Tests

```bash
python tests/validate_module.py
python tests/test_api.py
```

`tests/validate_module.py` checks the uncertainty module with a synthetic ENN.
`tests/test_api.py` validates the FastAPI response shape without requiring real
Grid2Op assets.

## Legacy Workflow

The original failure-forecast pipeline, including `run_pipeline.py`,
forecaster/classifier training, tutor-data ENN training, LLM rule generation,
and old model/data paths, lives in:

```text
archive/legacy/old_version/
```

Its preserved documentation is:

```text
archive/legacy/old_version/README.md
```

# RL Agent Recommendation Uncertainty

This repository provides agent-agnostic epistemic uncertainty scoring for
Grid2Op recommendations. It collects an agent's rollout behavior, trains an
Evidential Neural Network (ENN) on observation/action pairs, and adds calibrated
uncertainty KPIs to the agent's recommendations.

Compatible agents expose:

```python
agent.act(obs, reward, done)
```

The active workflow is implemented by `run_pipeline.py`, `run_example.py`,
`recommendation_uncertainty.py`, `training/`, and `app/`. The original workflow
is preserved under `archive/legacy/old_version/`.

## Quick Links

- [Installation](#installation)
- [Active Workflow](#active-workflow)
- [Supported Grid2Op Environment](#supported-grid2op-environment)
- [Curriculum agent](#curriculum-agent)
- [Configuration](#configuration)
- [Pipeline](#pipeline)
- [ENN Training](#enn-training)
- [Project Structure](#project-structure)
- [API](#api)
- [Docker Instructions](DOCKER.md)
- [Tests](#tests)
- [Legacy Workflow](#legacy-workflow)

## Installation

Use Python 3.9 or 3.10. The pinned Grid2Op, TensorFlow, Ray, and Torch versions
do not support newer Python versions and should not be upgraded independently.

```bash
conda create -n enn_uq python=3.10 -y
conda activate enn_uq
pip install -r requirements.txt
```

## Active Workflow

Pre-trained agent assets and ENN artifacts are distributed separately in the
project release archives. Extract them into the repository root so the
following directories exist:

```text
assets/<ENV_NAME>/
artifacts/<ENV_NAME>/<AGENT_NAME>/
environment/<ENV_NAME>
```

Create `.env` as described in [Configuration](#configuration), then run the
end-to-end example:

```bash
python run_example.py
```

The workflow uses:

- `run_pipeline.py` to run data collection and model training.
- `training/collect_rollouts.py` and `training/train_enn.py` for ENN training.
- `recommendation_uncertainty.py` to score an agent's selected action.
- `app/main.py` to serve recommendations and KPIs through FastAPI.

Generated ENN data is stored under:

```text
artifacts/<ENV_NAME>/<AGENT_NAME>/
|-- rollouts/
|   |-- observations.npy
|   |-- labels.npy
|   `-- actions.npy
`-- model/
    |-- enn_<AGENT_NAME>.pth
    |-- scaler_params.json
    |-- enn_meta.json
    `-- enn_pctile_calib.npz
```

## Supported Grid2Op Environment

The default environment is `ai4realnet_small`, sourced from the
[Grid2Op scenario repository](https://github.com/ainetus/grid2op-scenario). The
scenario directory must resolve to:

```text
<ENV_LOCATION>/<ENV_NAME>
```

With the default configuration, this is
`environment/ai4realnet_small/`.

### Curriculum Agent

The default policy is a pre-trained CurriculumAgent for
`ai4realnet_small`. Its release archive must provide:

```text
assets/ai4realnet_small/
|-- model/
`-- actions/
```

To retrain the policy for a changed environment or action space, run:

```bash
python training/train_curriculumagent.py
```

The trained package is written to `assets/<ENV_NAME>/`.

## Configuration

Create a local configuration file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Configuration precedence is: environment variables, `.env`, then defaults in
`project_config.py`. The main path and identity settings are:

```dotenv
ENV_NAME=ai4realnet_small
ENV_LOCATION=environment
AGENT_NAME=curriculum
AGENT_FACTORY=
ASSETS_DIR=assets
ARTIFACTS_DIR=artifacts
```

Training parameters, episode limits, thresholds, and seeds are documented in
`.env.example`. Relative paths are resolved from the repository root. To use a
different policy, set `AGENT_FACTORY=module:function`; the factory receives the
Grid2Op environment and returns an agent with an `act` method.

## Pipeline

Run the full training pipeline with the values configured in `.env`:

```bash
python run_pipeline.py
```

Valid artifacts are reused automatically. Force individual stages when needed:

```bash
python run_pipeline.py --force-stage enn
python run_pipeline.py --force-stage forecast --force-stage classifier
python run_pipeline.py --force-stage all
```

Available stages are `enn-data`, `enn`, `forecast`, `failure-rows`, and
`classifier`. Run `python run_pipeline.py --help` for all options.

## ENN Training

To train only the uncertainty model:

```bash
python training/collect_rollouts.py --agent-name curriculum --episodes 50
python training/train_enn.py --agent-name curriculum
```

See [training/TRAINING.md](training/TRAINING.md) for artifact formats, training
options, and failure-forecast stages.


## API

Run the API locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or build and run it with Docker:

```bash
docker build -t curriculum-agent-api .
docker run --env-file .env -p 8000:8000 curriculum-agent-api
```

Available endpoints:

```text
GET  /health
GET  /diagnostics
GET  /docs
POST /api/v1/recommendation
```

Before requesting a recommendation, confirm that `/diagnostics` reports
`artifact_validation.ok: true` and `services.can_load: true`. In Swagger at
`http://localhost:8000/docs`, execute `POST /api/v1/recommendation` with:

```json
{
  "event": {},
  "context": {}
}
```

An empty context uses `env.reset()` for a smoke test. A successful response is
a list of recommendation dictionaries containing `actions` and `kpis`. The KPI
object includes the uncertainty percentage, total and action percentiles, and
uncertainty/confidence levels.

See [Docker Instructions](DOCKER.md) and [app/API.md](app/API.md) for operational
details and the complete request/response contract.

## Tests

Run the synthetic test suite:

```bash
python -m unittest discover -s tests -v
```

These tests cover ENN inference, uncertainty KPIs, the FastAPI contract, and
failure-forecast behavior without requiring trained weights or a live Grid2Op
environment.

## Project Structure

```text
.
|-- app/                         FastAPI service and recommendation formatter
|-- artifacts/                   Generated rollouts and trained models
|-- assets/                      Trained policy model and action set
|-- curriculumagent/             CurriculumAgent implementation
|-- environment/                 Local Grid2Op scenarios
|-- src/                         Shared agent, data, and ENN modules
|-- tests/                       Synthetic unit and API tests
|-- training/                    Data collection and training scripts
|-- project_config.py            Shared environment configuration
|-- recommendation_uncertainty.py
|-- run_example.py
`-- run_pipeline.py
```

## Legacy Workflow

The original tutor-data ENN pipeline, LLM rule generation, and its documentation
are preserved under `archive/legacy/old_version/`. They are not used by the
active API or training pipeline.

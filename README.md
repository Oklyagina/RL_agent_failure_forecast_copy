# ENN Uncertainty-Quantification Module

Epistemic-uncertainty KPI for RL grid-operation recommendations, based on an
Evidential Neural Network (ENN) trained by behavior cloning on the agent's
own decisions. For each recommendation it returns two percentiles (0–100):

- `epistemic_uncertainty_total_pctile` — total epistemic uncertainty of the
  situation (the ENN vacuity `u = K/S`); high when the grid state is
  out-of-distribution for the agent.
- `epistemic_uncertainty_action_pctile` — epistemic uncertainty of the action
  the agent selected (variance of that action's predicted probability);
  `None` for a do-nothing action (not in the curated set).

Percentiles are reported (rather than raw values) because the raw measures
sit in narrow bands; a percentile spreads them onto a readable 0–100 scale
relative to a pre-computed reference distribution. **No calibration step is
needed on the consumer side** — the reference is shipped
(`enn_pctile_calib.npz`).

## Configuration

Copy `.env.example` to `.env` before running the pipeline, then adjust the
values for your machine. The main settings are `ENV_NAME`, `ENV_LOCATION`,
`AGENT_NAME`, `ASSETS_DIR`, and `ARTIFACTS_DIR`; all training and example
scripts read these through `project_config.py`.

## Environment

**Python 3.9 or 3.10 is required.** The bundled CurriculumAgent SavedModel was
exported with Keras 2.12, so keep `tensorflow==2.12.1` and use a
`python:3.10-slim` base image for containers (see `Dockerfile`). **Grid2Op is
pinned to 1.9.8**, the same version used by the InteractiveAI simulator, so the
action format stays consistent.

```bash
conda create -n enn_uq python=3.10 -y
conda activate enn_uq
pip install -r requirements.txt
```

## Quick start — full reproduction

The scaler used by the module is **created at ENN training time**, so the
complete reproduction of the experiment is: collect rollouts → train (the
training exports the scaler, the metadata and the percentile calibration
automatically) → run the example. For the CurriculumAgent nothing needs to
be edited:

```bash
python training/train_curriculumagent.py
python training/collect_rollouts.py
python training/train_enn.py
python run_example.py
```

Shared defaults are read from `.env` (see `.env.example`). By default, the
trained CurriculumAgent policy package is stored in `assets/ai4realnet_small/`,
rollout data is written to `artifacts/ai4realnet_small/curriculum/rollouts/`,
and trained ENN artifacts to `artifacts/ai4realnet_small/curriculum/model/`.
For another agent, pass `--agent expert` to collection and `--agent-name expert`
to training; the default directories become
`artifacts/ai4realnet_small/expert/rollouts/` and
`artifacts/ai4realnet_small/expert/model/`.

On Windows, the first Grid2Op dataset download/cache setup may require running
PowerShell as Administrator. If Grid2Op reports missing files such as
`config.py` or `grid_layout.json`, rerun the rollout command from an Administrator shell.

`run_example.py` is self-configuring: it prefers trained artifacts under
`artifacts/`, then falls back to legacy `models_*` folders and other compatible
locations. It locates `scaler_params.json`, `enn_meta.json`, calibration `.npz`,
the curated action set, the ENN architecture module and the agent binaries,
printing every resolved path (anything can be overridden in its CONFIG block).
It then creates the
configured environment (LightSim backend), loads the CurriculumAgent policy
from `assets/<ENV_NAME>/`, rebuilds the training scaler from
the exported JSON (no pickled scaler, so it reloads under any scikit-learn
version), loads the calibration, and calls `assess_recommendation` on live
observations, printing the two percentiles per step and the recommendations
list in the InteractiveAI format. If no trained artifacts exist yet, it
prints the exact training commands to run first.

### Artifacts

| File | What it is |
|---|---|
| `recommendation_uncertainty.py` | the module: `load_calibration`, `assess_recommendation` |
| `src/enn_models.py` | ENN architecture (`EvidentialNetwork`) |
| `assets/<ENV_NAME>/` (produced by CurriculumAgent training) | trained RL policy package with `model/` and `actions/` |
| `artifacts/<ENV_NAME>/<agent>/rollouts/actions.npy` (produced by collection) | curated action set — rows are `action.to_vect()`, deduplicated from the rollouts |
| `artifacts/<ENV_NAME>/<agent>/model/` (produced by ENN training) | `enn_<agent>.pth`, `scaler_params.json` (scaler mean/std as JSON — the scaler is created at training time), `enn_meta.json`, `enn_pctile_calib.npz` |
| `curriculumagent/` | agent source/submission code |

## Usage in three lines

```python
from recommendation_uncertainty import load_calibration, assess_recommendation

calibration = load_calibration("artifacts/ai4realnet_small/curriculum/model/enn_pctile_calib.npz",
                               scaler=scaler,
                               action_set="artifacts/ai4realnet_small/curriculum/rollouts/actions.npy",
                               class_mapping="artifacts/ai4realnet_small/curriculum/model/enn_meta.json")
info = assess_recommendation(obs, agent, enn, calibration)
# {"chosen_action_id": ..., "epistemic_uncertainty_total_pctile": ...,
#  "epistemic_uncertainty_action_pctile": ...}
```

The agent returns a Grid2Op **action object** (not an index): the module
locates it in the curated action set to obtain its index, then maps it to the
ENN label via `class_mapping`. A do-nothing action is not in the set, so its
per-action value is `None` (the situation-level value is still produced).

## InteractiveAI output format

Each recommendation is a dictionary in the recommendations list; the two
percentiles go into the `kpis` field alongside `efficiency_of_the_reco`:

```json
{
  "title": "Topological recommendation (CurriculumAgent)",
  "description": "...",
  "use_case": "PowerGrid",
  "agent_type": 2,
  "actions": [ { "...": "grid2op serializable action dict" } ],
  "kpis": {
    "efficiency_of_the_reco": null,
    "epistemic_uncertainty_total_pctile": 47.8,
    "epistemic_uncertainty_action_pctile": 3.1
  }
}
```

See `to_interactiveai()` in `run_example.py` for the exact mapping.

## Tests (no trained weights needed)

```bash
python tests/validate_module.py
```

Validates the module end-to-end with a synthetic ENN exposing the same
forward interface: action location in the curated set, do-nothing handling,
percentile-mapping monotonicity and bounds, and InteractiveAI serialisation.
Useful to confirm an environment is correctly set up before touching the
real binaries.

## Training the ENN for a new agent

The module is agent-agnostic — any object exposing
`agent.act(obs, reward, done) -> grid2op action` can be wrapped. The same
two scripts used above handle any agent: `--agent expert` in
`training/collect_rollouts.py` (after plugging in the ExpertAgent
constructor) collects rollouts and curates the action set automatically by
deduplication; `training/train_enn.py` then trains, exports the scaler and
metadata, and calibrates. Full step-by-step guide: **[TRAINING.md](training/TRAINING.md)**.

## Agent API (InteractiveAI integration)

`app/main.py` exposes the CurriculumAgent as an InteractiveAI agent API
(FastAPI, `POST /api/v1/recommendation`), built to the AI4REALNET AI-agent
template. Each recommendation is returned in the InteractiveAI dictionary
format, with the two ENN epistemic-uncertainty percentiles added into the
`kpis` field alongside `efficiency_of_the_reco`. The container built from the
`Dockerfile` runs this API (uvicorn on port 8000).

```bash
docker build -t curriculum-agent-api .
docker run -p 8000:8000 curriculum-agent-api
```

The `Dockerfile` mirrors the AI4REALNET ExpertAgent reference (it sets up the
`ai4realnet_small` scenario from `grid2op-scenario`, the same the simulator
uses) on `python:3.10-slim`. Full endpoint contract, the `curl` test and the
two deployment caveats (the `get_parade_info` helper from ExpertOp4Grid and the
environment/Grid2Op alignment) are in **[API.md](API.md)**. `python tests/test_api.py` validates the API with a
synthetic ENN, no environment needed.

For a pure reproducibility check (no API), run `python run_example.py`
directly, inside or outside the container.

## Maintainers

`export_artifacts.py` regenerates the JSON artifacts from an existing fitted
scaler (`.pkl`) and checkpoint, for the case where a training was done
outside `training/train_enn.py`. It is fully self-configuring (it infers
`input_dim`/`num_classes` from the checkpoint's state_dict and cross-checks
everything). With the standard training flow it is not needed —
`train_enn.py` already exports everything.

## License

Mozilla Public License 2.0 — see [LICENSE](LICENSE).

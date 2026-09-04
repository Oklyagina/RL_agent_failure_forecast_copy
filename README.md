# Forecast RL Agent Failure

This repository implements a framework to **quantify and predict the realiability of pre-trained Reinforcement Learning (RL) agents** used fot real-time congestion management in power grids.

Assessing the reliability of AI-assisted decision support systems under unseen operating conditions is critical. This project anticipates unreliable AI recommendations and provides early warnings to human operators.

The pipeline integrates **Uncertainty Quantification (UQ)** to support risk-aware decision making by separating uncertainty into two components:
  - **Aleatoric Uncertainty:** The uncertainty of the forecasted values (predictive variance). It captures the inherent stochastic variability and forecast errors of load and generation, estimated by modeling the residuals of the primary Forecaster (HistGradientBoosting).
  - **Epistemic Uncertainty:** The uncertainty associated with the RL agent's decisions when facing out-of-distribution or unseen grid states, computed using an **Evidential Neural Network (ENN)**.



These indicators are integrated into a **failure prediction model** that estimates the probability of RL agent failure under future contigencies (disconnection of lines).

Finally, a **Dual LLM Architecture** takes the outputs of the predictive classifiers and synthesizes robust, symbolic Python rules (`best_rule.py`). This translates complex, black-box uncertainty metrics into intepretable, human-readable operational guidelines, ensuring the AI assistant's boundaries are transparent and safe.

---

## Supported Environments

- **Network 36** (`l2rpn_icaps_2021_small`)
- **AI4REALNET small** (`ai4realnet_small`)

---

## Project Structure

```text
grid_security_project/
|
├── .env.example                         # Example settings for refactored scripts
├── .env                                 # Local settings, not committed
├── project_config.py                    # Shared .env/environment configuration
├── recommendation_uncertainty.py        # ENN uncertainty module
|
├── assets/
│   ├── ai4realnet_small/                # Refactored CurriculumAgent policy package
│   │   ├── model/
│   │   └── actions/
|
├── artifacts/
│   └── <ENV_NAME>/<agent>/              # Generated rollout and ENN artifacts
│       ├── rollouts/
│       └── model/
|
├── environment/
│   └── ai4realnet_small/                # Local Grid2Op scenario files
|
├── app/
│   ├── main.py                          # FastAPI InteractiveAI recommendation endpoint
│   └── API.md                           # API contract and deployment notes
|
├── training/
│   ├── collect_rollouts.py              # Refactored rollout collection
│   ├── train_enn.py                     # Refactored ENN training/export pipeline
│   ├── train_curriculumagent.py         # CurriculumAgent training entrypoint
│   └── TRAINING.md                      # ENN training guide
|
├── src/
│   ├── collect_data.py                  # Simulation and dataset generation
│   ├── config.py                        # Pipeline configuration and execution flags
│   ├── dual_llm.py                      # Dual LLM: generator and critic
│   ├── enn_models.py                    # ENN architectures
│   ├── rule_predictor.py                # Rule inference and natural-language translation
│   ├── test_rule_predictor.py           # Live rule inference and natural-language translation
│   ├── train_classifier.py              # Classifier training and inference
│   ├── train_forecast.py                # Forecaster training
│   ├── training_enn.py                  # Original ENN training pipeline
│   └── utils.py                         # Feature extraction and grid statistics
|
├── run_pipeline.py                      # Main entry point for the failure-forecast pipeline
├── run_example.py                       # Refactored ENN uncertainty example
├── Dockerfile                           # InteractiveAI API container
├── requirements.txt                     # Python dependencies
```

## Installation

### 1. Clone the Repository
```
git clone <repository_url>
```

### 2. Set up Conda environment and install dependencies
This project requires **Python 3.9 or 3.10**. The bundled CurriculumAgent SavedModel was exported with Keras 2.12, so keep the pinned `tensorflow==2.12.1`. Grid2Op is pinned to `1.9.8`.

```bash
conda create -n enn_uq python=3.10 -y
conda activate enn_uq
pip install -r requirements.txt
```

### 3. Set up .env file

Copy the example .env file before running CurriculumAgent/ENN scripts.
Then adjust the values for your desired experiment and machine.

```bash
cp .env.example .env
```
Or, on Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

### 4. Agent Setup
The pre-trained agent downloadable archive is available at project github page under the last release.
For the CurriculumAgent/ENN workflow, ensure the pre-trained agent package is available under:

```text
assets/<ENV_NAME>/model/
assets/<ENV_NAME>/actions/
```

With the default `.env`, this is:

```text
assets/ai4realnet_small/model/
assets/ai4realnet_small/actions/
```

The rollout loader also checks the legacy-compatible locations `assets/network36/` and `src/models/network36/`.

### Configuration
There are two configuration entrypoints in the current repository.

The original failure-forecast pipeline is controlled via `src/config.py`. **You do not need to modify the logic scripts directly**.

The refactored CurriculumAgent/ENN scripts (`training/collect_rollouts.py`, `training/train_enn.py`, `training/train_curriculumagent.py`, and `run_example.py`) read shared defaults from `.env` through `project_config.py`. Environment variables override `.env`, and `.env` overrides the defaults in `project_config.py`.

The main `.env` settings are:

```text
ENV_NAME=ai4realnet_small
ENV_LOCATION=C:\Users\
AGENT_NAME=curriculum
ASSETS_DIR=assets
ARTIFACTS_DIR=artifacts
CURRICULUM_ITERATIONS=3
CURRICULUM_JOBS=1
ROLLOUT_EPISODES=50
ENN_EPOCHS=100
ENN_ANNEAL_EPOCHS=10
ENN_BATCH_SIZE=512
ENN_LR=1e-3
ENN_VAL_FRAC=0.1
EXAMPLE_N_STEPS=5
SEED=0
```

#### Select Environment
For the CurriculumAgent/ENN scripts, set `ENV_NAME` and `ENV_LOCATION` in `.env`. `ENV_LOCATION` should point to the directory containing the Grid2Op environment folder, so `project_config.py` resolves the environment as:

```text
<ENV_LOCATION>/<ENV_NAME>
```

On Windows, the first Grid2Op dataset download/cache setup may require running PowerShell as Administrator. 
If Grid2Op reports missing files such as `config.py` or `grid_layout.json`, rerun the command from an Administrator shell.

### 5. Pre-trained Models

The forecaster models are too large to be stored directly in this repository.
These files are generated by the training pipeline: 
- `src/train_forecast.py` creates the main forecaster, `src/train_classifier.py` 
- trains/uses the failure classifier inputs, and `src/training_enn.py` trains the original ENN model. 

If the release artifacts are not available, launch the training pipeline with `TRAIN_MODE = True` in `src/config.py` and run `python run_pipeline.py`.


The pre-trained models are strored as attachments in the GitHub Releases section.

1. Go to the https://github.com/AI4REALNET/RL_agent_failure_forecast/releases/tag/v1.0-models
   of this repository.
2. Download the following files from release `v1.0-models`:
   - `HBGB_36.pkl` -> place in `forecasts/`
   - `HBGB_36_aleatoric.pkl` -> place in `models/`
   - `enn_36.pth` -> place in `models/`


For the refactored ENN workflow, generated rollout data is written by default to:

```text
artifacts/<ENV_NAME>/<agent>/rollouts/
```

and trained ENN artifacts are written to:

```text
artifacts/<ENV_NAME>/<agent>/model/
```

The ENN training exports `enn_<agent>.pth`, `scaler_params.json`, `enn_meta.json`, and `enn_pctile_calib.npz`.

#### Usage & Execution Modes
Use the main script to run the pipeline. The behaviour depends on the flags set in ```src/config.py```.

##### 1. Training Pipeline
Use this mode to train the Forecasters, collect simulation data, and train the final Classifier.
1. **Config**: Set `TRAIN_MODE = True` in `src/config.py`.
2. **Run**:
    ```
   python run_pipeline.py
   ```
3. **Outcome**: All models wil be trained and saved in the `models/` directory.

##### 2. Testing & Inference
If you have trained models, you can use the following modes to test the system.

A. **Single Episode Simulation**
Use this to analyse a specific episode (seed) from start to finish. It simulates the agent interacting with the grid and records how the uncertainty metrics behave over time.
1. *Config*:
```python
TRAIN_MODE = False
TEST_SINGLE_EPISODE = True
EPISODE_ID_TO_TEST = 50 # The seed of the episode
```
2. *Run*: python run_pipeline.py
3. *Outcome*: Generates a CSV trace of that specific episode in `data/`.

B. **Single Observation Inference (Probabilities)**
Use this to predict the failure probability for **one specific grid state** (Observation). This mode **does not** run a physical simulation (no disconnection). It purely calculates risk based on the model's knowledge.
1. *Config*:
```python
TRAIN_MODE = False
TEST_SINGLE_EPISODE = FALSE
PREDICT_PROBA_MODE = True

# Define which state to fetch from the environment
PROBA_TEST_EPISODE_ID = 50
PROBA_TEST_STEP = 50
```
2. *Run*: python run_pipeline.py
3. *Outcome*: Generates a CSV trace of that specific episode in `data/`.

C. **LLM Rule Inference (Natural-Language Explanations)**
Use this mode to apply the symbolic rules generated by the Dual LLM to a live simulation episode. For each monitored line, at every analysis step the system runs the forecast pipeline internally, evaluates the corresponding rule, and prints a human-readable explanation of the prediction.

1. *Config*:
```python
TRAIN_MODE          = False
TEST_SINGLE_EPISODE = False
PREDICT_PROBA_MODE  = False
LLM_RULE_MODE       = True

# Path to the folder containing the generated rules
LLM_RULES_DIR     = "llm_rules_results/temp_0.5"

# Episode seed to simulate
LLM_RULES_EPISODE = 50
```
2. *Run*: `python run_pipeline.py`
3. *Outcome*: For each monitored line and analysis step, prints the binary prediction (`OK` or `FAILURE PREDICTED`) together with a plain-English explanation sentence. At startup, all available rule sentences are also printed for use as operational guidelines.

Example output:
```
  [41_48_131]
  Following a contingency on line 41_48_131, the RL agent is predicted to fail
  to provide a recommendation that solves a problem if the maximum
  line loading (rho) at t is >= 0.82, or if the forecasted maximum line loading
  (rho) at t+12 is >= 0.66 while the epistemic uncertainty at t is >= 0.77 and
  the forecasted total active power load at t+12 is <= 643 MW.

  --- Step 40 ---
    Line 41_48_131    -> FAILURE PREDICTED
      Following a contingency on line 41_48_131, ...
    Line 34_35_110    -> OK
```

#### Critical Lines
The system automatically monitors specific critical lines defined in CFG.

D. **Live Episode Rule Test**

Use `src/test_rule_predictor.py` to run a live episode with the CurriculumAgent and verify the symbolic rules in real time. For each monitored line at every step, the system:

1. Applies the rule to predict failure 1 hour ahead.
2. If failure is predicted, simulates the actual line disconnection to confirm whether the grid would really fail.
3. Only prints an alert if both the rule **and** the simulation agree on failure.

At the end of the episode it reports the step at which the agent failed and whether any rule issued a confirmed warning in the 12 steps (1 hour) before the actual failure.

**Run (normal episode):**
```bash
python src/test_rule_predictor.py
```

**Run (with adversarial HeavyAttack_1 line attacks):**
```bash
python src/test_rule_predictor.py --attack
```

Example output:
```
═════════════════════════════════════════════════════════════════
  Scenario: Normal  |  seed=50
═════════════════════════════════════════════════════════════════

[INFO] Lines monitored: ['34_35_110', '41_48_131', ...]
[INFO] Episode started. Running...

  [step   100] running...
  [step   200] running...

  Step  247 | [41_48_131] FAILURE PREDICTED
  Following a contingency on line 41_48_131, the RL agent is predicted
  to fail if the maximum line loading (rho) at t is >= 0.74 while the
  epistemic uncertainty at t is >= 0.79.

─────────────────────────────────────────────────────────────────
  Agent failed at step 259.

  Did the rule warn in the 12 steps before failure?

  Line                  Warned?   Warning steps
  ────────────────────  ────────  ─────────────────────────
  34_35_110             NO        —
  41_48_131             YES       [247, 251]
  43_44_125             NO        —
```

The episode seed and rules directory can be configured at the top of `src/test_rule_predictor.py` via `EPISODE_SEED` and `RESULTS_DIR`.

##### 3. Refactored ENN Uncertainty Workflow

The standard refactored workflow is: start from a trained CurriculumAgent, collect rollouts, train the ENN, then run the example.

```bash
python training/collect_rollouts.py
python training/train_enn.py
python run_example.py
```

The scripts read their defaults from `.env` through `project_config.py`. For another agent, pass `--agent expert` to collection and `--agent-name expert` to training after plugging in the ExpertAgent constructor in `training/collect_rollouts.py`.

`run_example.py` prefers trained artifacts under `artifacts/`, then falls back to legacy compatible locations such as `models_*` folders, `assets/network36/`, and `src/models/network36/`. It prints the resolved paths before running the live example.

##### 4. Agent API (InteractiveAI integration)

`app/main.py` exposes the CurriculumAgent as an InteractiveAI agent API with FastAPI:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The endpoint is `POST /api/v1/recommendation`, and `/health` reports the active Grid2Op environment. The API reads `GRID2OP_ENV` from the environment, defaulting to `l2rpn_icaps_2021_small` in `app/main.py`.

The Docker container uses `python:3.10-slim`, installs the pinned requirements, copies the repository, and starts uvicorn on port 8000:

```bash
docker build -t curriculum-agent-api .
docker run -p 8000:8000 curriculum-agent-api
```

The `Dockerfile` sets `GRID2OP_ENV=ai4realnet_small` and copies the `ai4realnet_small` scenario from `grid2op-scenario`. It currently runs `pip install .`, so add a packaging file or remove that line before relying on the image build. See `app/API.md` for the endpoint contract and deployment notes.

## Methodology

This work proposes a **failure probability forecasting framework** that combines
**power grid forecasting**, **uncertainty quantification**, and **risk classification**
to anticipate cascading failures caused by line disconnections.

The methodology is structured into four main stages.

---

### 1. Uncertainty Decomposition

The framework explicitly separates uncertainty into **aleatoric** and **epistemic**
components, each capturing different sources of risk.

#### Aleatoric Uncertainty (Data Uncertainty)

Aleatoric uncertainty captures the **stochastic variability** inherent to load and
generation dynamics.

- A **multi-output time-series forecaster** (HistGradientBoosting) predicts active and
  reactive power injections for all loads and generators.
- Forecasts are generated for a **1-hour horizon** (12 timesteps ahead).
- Squared residuals between ground-truth values and mean forecasts are computed.
- A secondary regression model is trained on these squared residuals to estimate
  the **forecast variance**, which is used as a proxy for aleatoric uncertainty.

This process allows the framework to quantify how unpredictable future operating
conditions are, independently of the agent's knowledge.

---

#### Epistemic Uncertainty (Model Uncertainty)

Epistemic uncertainty reflects the **lack of knowledge of the agent** about the current
grid state and is used as an indicator of **out-of-distribution (OOD)** situations.

- An **Evidential Neural Network (ENN)** is trained via **knowledge distillation** to
  replicate the policy of a Senior (expert) agent.
- Instead of producing softmax probabilities, the ENN outputs the parameters
  of a **Dirichlet distribution** over the action space.
- Model ignorance is computed analytically as:

u = K / sum(alpha_i)

where `K` is the number of actions and `alpha_i` are the Dirichlet parameters.

High epistemic uncertainty indicates that the agent is operating in rarely observed or
unknown grid conditions.

---

### 2. Forecasting Future Grid States

To anticipate failures before they occur, the framework predicts **future grid states**.

- Load and generation forecasts are injected into the power grid model.
- A power flow simulation is executed to obtain the **forecasted grid state**
  one hour ahead.
- These future states are combined with aleatoric uncertainty estimates, capturing
  intrinsic forecast variability.

---

### 3. Contingency Analysis

For each candidate critical line:

- A **what-if disconnection** is simulated on the forecasted grid state.
- The system evaluates whether the grid remains stable or reaches a failure condition
  one hour after the contingency.
- This process generates labeled data linking grid conditions, uncertainties, and
  line disconnections to observed failures.

---

### 4. Risk Classification

A final **binary classifier** is trained to predict cascading failures **before action
execution**.

**Inputs:**
- Current grid state indicators (e.g., load-generation balance, thermal stress).
- Epistemic uncertainty (confidence in the current state).
- Aleatoric uncertainty (forecast variability).
- Identifier of the disconnected transmission line.

**Output:**
- `0` - Stable operation expected.
- `1` - Failure predicted (alarm triggered).

---

### 5. LLM-Guided Symbolic Rule Generation (Dual LLM)

To convert the black-box classifier outputs into interpretable operational guidelines, a **Dual LLM Architecture** (Generator-Evaluator) processes the data. The system automatically iterates over multiple critical lines and explores various LLM temperature settings (hyperparameter search) to find the optimal balance between logical strictness and creative problem-solving.

- **Dynamic Rule Synthesis:** The generator LLM writes explicit, symbolic Python rules (`best_rule.py`) for each targeted transmission line based on thresholds of grid statistics and uncertainty metrics.
- **Evaluation & Refinement:** An evaluator LLM critiques the generated rules against false-alarm and oversight metrics. Changes and logical justifications are systematically logged (`best_feedback.txt`, `best_justification.txt`).
- **Iterative Tracking:** The framework iteratively tests seeds and records performance metrics across different temperature folders, ensuring convergence on the safest and most accurate operational rule for every monitored line.

---

### 6. Rule Translation (Natural-Language Explanations)

To make the symbolic rules accessible to human operators, `src/rule_predictor.py` automatically translates each `best_rule.py` into a plain-English sentence that describes the conditions under which the RL agent is predicted to fail.

The translation is performed by parsing the Python rule as an Abstract Syntax Tree (AST) and mapping each condition to a human-readable description of the corresponding grid feature. Each distinct failure path within the rule becomes an "or if" clause, and multiple AND conditions within the same path are joined with "while ... and".

For example, the following rule:

```python
def rule(x):
    if x["max_line_rho"] >= 0.65:
        if x["epistemic_before"] >= 0.7891:
            if x["aleatoric_gen_p_mean"] <= 0.3434:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        if x["fcast_sum_load_q"] >= 155.5429:
            if x["aleatoric_gen_p_mean"] <= 0.3434:
                return 1
            else:
                return 0
        else:
            return 0
```

is automatically translated to:

> *Following a contingency on line 34_35_110, the RL agent is predicted to fail if the maximum line loading at t is >= 0.65 while the epistemic uncertainty is <= 0.79 and the mean aleatoric generation uncertainty is <= 0.34, or if the forecasted reactive load at t+12 is >= 155.54 MVAR and the mean aleatoric generation uncertainty is <= 0.34*.

In LLM Rule Inference mode, the system also evaluates each rule against the current grid state in real time: it runs the forecast pipeline internally (computing t+12 features from the live observation), applies the rule, and reports the prediction alongside the explanation sentence.

---

### Final Objective

The ultimate goal of this framework is to provide **real-time confidence levels**
that allow:

- Validation of autonomous agent decisions.
- Prevention of unsafe operations in critical power grid environments through transparent, human-readable guidelines.

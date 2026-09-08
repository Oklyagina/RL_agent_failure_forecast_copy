# Legacy Failure-Forecast Pipeline

This folder preserves the original `run_pipeline.py` workflow that existed
before the repository was narrowed to the refactored agent-agnostic ENN
uncertainty path.

The active repository README now documents the current workflow:

```bash
python training/collect_rollouts.py
python training/train_enn.py
python run_example.py
```

## What Is Archived Here

The legacy workflow combines:

- forecaster training for future load/generation values;
- original tutor-data ENN training from `src/training_enn.py`;
- disconnection simulation and feature-table generation;
- failure classifier training;
- optional Dual-LLM symbolic rule generation and rule inference.

The main entrypoint is:

```bash
python run_pipeline.py
```

The original pipeline is configured through:

```text
src/config.py
```

## Archived Structure

```text
archive/legacy/old_version/
|-- run_pipeline.py
|-- calibrate_uncertainty.py
|-- compute_shap_rankings.py
|-- dual_llm.py
|-- llm_prompts.md
|-- call_real_simulator_api.py
|-- enn_pctile_calib.npz
|-- data/
|-- forecasts/
|-- misc/
|-- models/
|-- src/
|   |-- config.py
|   |-- collect_data.py
|   |-- train_forecast.py
|   |-- training_enn.py
|   |-- train_classifier.py
|   |-- utils.py
|   |-- failure_probability.py
|   |-- pipeline_artifacts.py
|   |-- rule_predictor.py
|   |-- test_rule_predictor.py
|   `-- models/network36/
`-- tests/
    |-- test_training_enn_data.py
    |-- test_enn_training_fallback.py
    `-- test_new_components.py
```

## Legacy Training Pipeline

Use this mode to train the forecasters, original ENN, collect simulation data,
and train the final failure classifier.

1. Set the flags in `src/config.py`:

```python
TRAIN_MODE = True
TEST_SINGLE_EPISODE = False
PREDICT_PROBA_MODE = False
LLM_RULE_MODE = False
```

2. Run:

```bash
python run_pipeline.py
```

3. Outputs are written to the legacy `data/`, `forecasts/`, `models/`, and
   `src/models/network36/` paths in this folder.

## Original ENN Training

The original ENN is trained by `src/training_enn.py` from tutor/junior-style
data configured in `src/config.py`:

```text
CFG.TUTOR_DIR
CFG.TRAIN_FILE
CFG.VAL_FILE
CFG.TEST_FILE
```

At runtime it prints:

```text
[ENN] Loading Tutor Data from: ...
```

This is the main difference from the active refactored workflow, which trains
from actual agent rollouts collected by `training/collect_rollouts.py`.

## Legacy Execution Modes

Single episode simulation:

```python
TRAIN_MODE = False
TEST_SINGLE_EPISODE = True
PREDICT_PROBA_MODE = False
LLM_RULE_MODE = False
EPISODE_ID_TO_TEST = 50
```

```bash
python run_pipeline.py
```

Single-observation probability inference:

```python
TRAIN_MODE = False
TEST_SINGLE_EPISODE = False
PREDICT_PROBA_MODE = True
LLM_RULE_MODE = False
PROBA_TEST_EPISODE_ID = 50
PROBA_TEST_STEP = 50
```

```bash
python run_pipeline.py
```

LLM rule inference:

```python
TRAIN_MODE = False
TEST_SINGLE_EPISODE = False
PREDICT_PROBA_MODE = False
LLM_RULE_MODE = True
LLM_RULES_DIR = "llm_rules_results/temp_0.5"
LLM_RULES_EPISODE = 50
```

```bash
python run_pipeline.py
```

## Dual-LLM Rule Generation

The archived symbolic rule workflow is:

```bash
python compute_shap_rankings.py
python dual_llm.py
```

`compute_shap_rankings.py` ranks per-line features from the HGB classifier.
`dual_llm.py` generates and evaluates Python `best_rule.py` files using the
generator/critic/repair prompt templates described in `llm_prompts.md`.

## Live Rule Test

The archived live rule predictor is:

```bash
python src/test_rule_predictor.py
```

With adversarial line attacks:

```bash
python src/test_rule_predictor.py --attack
```

It applies symbolic rules during a live episode and checks whether warnings
appear before the agent fails.

## Notes

- This folder is kept for reproducibility and comparison with the original
  failure-forecast implementation.
- The active code path does not depend on these files.
- Paths in the archived scripts may still assume execution from this legacy
  folder or from the original repository layout.

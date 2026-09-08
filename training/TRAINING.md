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
|-- actions.npy
|-- train.npz
|-- validation.npz
`-- test.npz
```

`actions.npy` contains distinct action vectors in stable first-observed order.
`labels.npy` contains the corresponding action-class index for each
observation.

At least two distinct policy actions are required for meaningful ENN training.

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
python tests/validate_module.py
python tests/test_api.py
```

`tests/validate_module.py` checks the uncertainty module with a synthetic ENN.
`tests/test_api.py` validates the FastAPI response shape without requiring real
Grid2Op assets.

A live Grid2Op validation requires the target environment dataset, active agent
assets, trained ENN artifacts, and the pinned Grid2Op/LightSim dependency stack.

# Training the ENN for a new agent (e.g. the ExpertAgent)

The uncertainty module is agent-agnostic: the ENN is trained by **behavior
cloning** on (observation, action) pairs collected from the trained agent, so
its epistemic uncertainty acts as a familiarity signal — low in states the
agent has effectively encountered, high out-of-distribution. Nothing from the
agent's internals is needed: any object exposing
`agent.act(obs, reward, done) -> grid2op action` works.

**Requirement:** the ENN classifies over a **discrete action set**. If the
agent selects from a fixed list, that list is the class set; if it builds
actions dynamically (rule/simulation-based), the collection step below curates
a set automatically by deduplicating the actions seen in the rollouts.

Environment: Python 3.9/3.10, `pip install -r requirements.txt`,
Grid2Op pinned to **1.9.8** (same as the InteractiveAI simulator).

## Step 1 — Collect rollouts

For the **CurriculumAgent** (binaries already in the repository) nothing
needs to be edited:

```bash
python training/collect_rollouts.py
```

For the **ExpertAgent**, plug its constructor into `make_expert_agent()` in
`training/collect_rollouts.py`, then:

```bash
python training/collect_rollouts.py --agent expert
```

Produces `observations.npy` (obs vectors), `labels.npy` (index of each chosen
action in the curated set) and `actions.npy` (the curated set, deduplicated
action vectors). Guidance: aim for at least a few tens of thousands of pairs,
covering both calm and stressed grid conditions (the same chronics mix you
will face at inference time).

## Step 2 — Train the ENN

```bash
python training/train_enn.py --agent-name expert
```

Both scripts derive their default directories from the agent name:
`artifacts/<ENV_NAME>/<agent>/rollouts/` for collected rollout arrays and
`artifacts/<ENV_NAME>/<agent>/model/` for trained ENN artifacts. The trained
RL policy package used for rollouts lives separately under `assets/<ENV_NAME>/`.

**The scaler is created here, at training time** — fitted on the training
split — and exported together with everything else. The script trains the
`EvidentialNetwork` with the evidential classification objective (Bayes-risk
cross-entropy + annealed KL to the uniform Dirichlet), keeps the best
checkpoint by validation loss, and exports:

- `enn_expert.pth` — trained weights;
- `scaler_params.json` — scaler mean/std as JSON (version-proof);
- `enn_meta.json` — `input_dim`, `num_classes`, identity `class_mapping`;
- `enn_pctile_calib.npz` — the percentile calibration (see Step 4).

`run_example.py` picks artifacts under `artifacts/` first, then falls back to
legacy `models_*` folders.

Sanity checks: validation top-1 accuracy well above chance (1/num_classes);
if one action dominates the rollouts (e.g. do-nothing), consider collecting
more stressed scenarios so minority actions get enough evidence.

## Step 3 — Do-nothing actions

Do-nothing is typically excluded from the curated set: the module returns
`epistemic_uncertainty_action_pctile = None` for it (the total, state-level
percentile is still produced). If your rollouts are dominated by do-nothing
steps, you may drop those pairs before training — keep this choice recorded
in `enn_meta.json`.

## Step 4 — Percentile calibration (automatic)

The two outputs are **percentiles (0–100)** relative to a reference
distribution. `train_enn.py` builds this calibration automatically at the
end of training (on a sample of the scaled training states) and writes
`enn_pctile_calib.npz` next to the other artifacts. To calibrate on a
different reference set (e.g. specific normal-operation episodes), use the
repository's `calibrate_uncertainty.py` instead.

## Step 5 — Use it

Exactly as in `run_example.py`, pointing the CONFIG block at the new
artifacts (`artifacts/ai4realnet_small/expert/model/enn_expert.pth`,
`artifacts/ai4realnet_small/expert/rollouts/actions.npy`,
`artifacts/ai4realnet_small/expert/model/enn_pctile_calib.npz`,
`artifacts/ai4realnet_small/expert/model/scaler_params.json`,
`artifacts/ai4realnet_small/expert/model/enn_meta.json`) and
loading the ExpertAgent instead of the CurriculumAgent:

```python
calibration = load_calibration(npz_path, scaler=scaler,
                               action_set=actions_path,
                               class_mapping=meta_path)
info = assess_recommendation(obs, agent, enn, calibration)
```

The returned percentiles go into the `kpis` field of each InteractiveAI
recommendation, alongside `efficiency_of_the_reco`.

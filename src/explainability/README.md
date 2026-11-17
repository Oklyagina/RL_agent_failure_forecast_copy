# Failure-forecast explainability

This package is an optional advisory layer around the active HGB failure
forecast. The HGB probability and binary prediction remain authoritative.
Distilled rules provide a short explanation and an explicitly labelled
surrogate prediction.

Rule targets use the HGB failure probability compared with `>= decision_threshold`
from classifier metadata, matching runtime decisions. If the threshold is absent,
the runtime default of 0.5 applies. Seeds, LLM feedback, candidate selection, and
reported rule metrics all use these targets rather than the simulation labels.
When the classifier stage saved `failure_classifier_oof.npz` (metadata
`teacher_oof`), rules are fitted only on the train+val episodes, whose targets come
from episode-grouped out-of-fold HGB probabilities instead of in-sample ones.
The test episodes are held out and the rules are scored on them against the
runtime HGB: see the `holdout_*` fields. Without that file, every row is used
with in-sample targets and a warning is logged. The other metrics measure fidelity
on the data used to select the rules. Newly generated per-line `metrics.json` files
and `summary.json` entries record `teacher_decision_threshold` and `teacher_source`.
Existing saved rules retain their original targets until they are regenerated.
Generator and critic guidance maximize full-population F2, explicitly accounting
for false positives and false negatives. Both receive population/confusion counts
and examples of both error types when available, with missed failures shown first.
The example budget is split evenly (an odd extra slot goes to missed failures),
with unused slots filled from the other type. The critic also receives the best
rule's metrics to distinguish rejected candidates from the retained baseline.
Each run starts from a deterministic seed, not the previous run's saved rules.
`prompts/llm_prompts.md` preserves the legacy prompts for reference; the active
templates are `generator_prompt.md`, `critic_prompt.md`, and `repair_prompt.md`.

## Rule selection (summary of experiments E1–E10)

How one rule per contingency line is selected:

1. **HGB operating point.** The classifier stage picks the decision threshold
   with the best F2 on episode-grouped out-of-fold probabilities for train+val.
   For the current model this is 0.7056; it was fixed at 0.5 before.
2. **Teacher.** Rules imitate the HGB decision, not the simulation label. On
   train+val, that decision comes from the out-of-fold probabilities, because
   in-sample HGB predictions are near-perfect and give the rules a target they
   cannot reach. The 38 test episodes are held out.
3. **Start rule.** For each line, the start rule is the best by F2 of three
   candidates, all on the line's SHAP top-8 features: one split, a balanced
   depth-2 tree, and a depth-3 tree.
4. **LLM.** The generator/critic loop runs 5 rounds with the current prompts and
   replaces the rule only when F2 improves.

Changes that were tried and dropped:

| Change | Experiment | Outcome |
|---|---|---|
| Removing the forecast features | E1 | worse |
| Grouping sparse lines into one rule | E3 | worse |
| The legacy evaluation protocol | E4 | worse |
| Legacy and missed-failure prompts | E6 | about 5 F2 lower |
| 20 LLM rounds | E10 | +0.2 F2; not worth 4× the LLM time |

Results on the test episodes, server HGB at 0.7056:

| Model | Scored vs | F2 | Acc | OVR | FA |
|---|---|---|---|---|---|
| Legacy reference (LLM rule) | teacher | 73.8 | 89.4 | 13.8 | 11.3 |
| Start rules | teacher | 75.3 | 89.4 | 8.7 | 10.8 |
| LLM rules, 5 rounds | teacher | 75.3 | 89.4 | 8.7 | 10.8 |
| LLM rules, 5 rounds | label | 51.9 | 85.1 | 20.5 | 14.6 |
| HGB | label | 64.4 | 93.7 | 25.5 | 5.3 |

Most of the gain comes from the out-of-fold teacher, the threshold and the tree
start rule. In this run the LLM improved one line (`48_53_141`: held-out F2 0.649
→ 0.672). The experiment scripts are in `misc/llm_rules_experiments/`, and all
results are logged in `misc/metrics_log_1111.csv`.

## Workflow

Offline (run occasionally):

```text
trained HGB + saved training rows + LLM
-> distil one surrogate rule per contingency
-> save rules/line_<line>/best_rule.py
```

Online (run for each new operating scenario):

```text
new operating scenario
-> compute the authoritative HGB failure probability and binary prediction
-> use the saved rules loaded when the orchestrator starts
-> evaluate the matching contingency rule and return its explanation
```

The modules implement this workflow as follows:

1. `compute_shap_rankings.py` ranks classifier features per contingency and
   stores a provenance-checked cache beside the failure-forecast artifacts.
2. `dual_llm.py` selects a deterministic seed rule: the best by F2 among one
   split and balanced depth-2/3 decision trees written as `if/else`. It then optionally
   asks one OpenAI-compatible model to generate, repair, and critique compact
   rule candidates.
3. `rule_predictor.py` validates and loads `line_*/best_rule.py` files.
4. `orchestrator.py` attaches explanations to unchanged HGB results. Missing or
   invalid rules produce `explanation_status="unavailable"`; they do not replace
   or suppress the classifier result.

Train with explanations from one combined profile, from the repository root:

```powershell
Copy-Item src/explainability/llm_config.example.json src/llm_config.json
python main.py train --profile                      # uses src/llm_config.json
python main.py train --profile path/to/profile.json
```

Prefer `api_key_env` over storing a key in JSON. `project_config.py` already
loads the repository `.env`, so the named variable can be defined there or in
the process environment. The populated `llm_config.json` path is ignored by
Git.

For Qwen models, set `"enable_thinking": false`. The client sends the vLLM
hard switch and falls back to `/no_think` when a compatible proxy rejects or
ignores that request option.

Rules and the SHAP cache are written under:

```text
artifacts/<environment>/<agent>/failure_forecast/explainability/
|-- shap_rankings.json
`-- rules/
    |-- summary.json
    `-- line_<line>/
        |-- best_rule.py
        `-- metrics.json
```

Run the focused checks and manual performance comparison with:

```powershell
python tests/test_explainability.py
python tests/benchmark_explainability_seed.py --rows <failure_forecast_rows.csv>
```

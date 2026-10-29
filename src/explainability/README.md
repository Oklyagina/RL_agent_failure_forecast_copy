# Failure-forecast explainability

This package is an optional advisory layer around the active HGB failure
forecast. The HGB probability and binary prediction remain authoritative.
Distilled rules provide a short explanation and an explicitly labelled
surrogate prediction.

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
2. `dual_llm.py` selects a vectorized deterministic seed rule, then optionally
   asks one OpenAI-compatible model to generate, repair, and critique compact
   rule candidates.
3. `rule_predictor.py` validates and loads `line_*/best_rule.py` files.
4. `orchestrator.py` attaches explanations to unchanged HGB results. Missing or
   invalid rules produce `explanation_status="unavailable"`; they do not replace
   or suppress the classifier result.

Use the combined profile runner from the repository root:

```powershell
Copy-Item src/explainability/llm_config.example.json `
  src/explainability/llm_config.json
python run_llm_pipeline.py --config src/explainability/llm_config.json
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

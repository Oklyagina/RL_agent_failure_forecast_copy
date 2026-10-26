# Failure-forecast explainability

This package is an optional layer around the active HGB failure forecast. It
does not change forecast construction, classifier features, probabilities, or
binary predictions.

The offline workflow follows the legacy implementation:

1. Rank the HGB features for each contingency.
2. For each line, use one OpenAI-compatible model in three stages: generate a
   compact `if/else` rule from the HGB decisions, critique the rule's fidelity,
   and repair it when the output is invalid or incomplete.
3. Load those rules at runtime and translate them into text.

## Module map

| Module | Main purpose | Inputs and outputs |
| --- | --- | --- |
| `__init__.py` | Defines the package's public API. | Re-exports the LLM client/configuration, runtime orchestrator, rule loader, compiler, and translators. |
| `compute_shap_rankings.py` | Ranks the most influential classifier features for each line. It uses mean absolute SHAP values and falls back to permutation importance if SHAP fails. | Reads failure rows, the HGB classifier, and classifier metadata; writes the provenance-checked SHAP ranking cache. This is an executable module. |
| `shap_cache.py` | Stores and validates reusable feature rankings. | Hashes the rows, classifier, and metadata; records the feature contract, line map, and SHAP settings; writes the cache atomically and rejects stale or malformed caches. |
| `dual_llm.py` | Distils the HGB teacher into one compact surrogate rule per line. It starts from a deterministic rule, iterates generator and critic prompts, and repairs invalid LLM output. | Reads the teacher artifacts, compatible SHAP cache, prompt templates, and LLM configuration; writes each `best_rule.py`, per-line metrics, and `summary.json`. This is an executable module. |
| `llm_client.py` | Calls an OpenAI-compatible chat-completions endpoint. | Loads JSON connection settings, resolves the API key directly or from an environment variable, and applies bounded retries and timeouts. |
| `rule_predictor.py` | Validates, loads, evaluates, and explains generated rules. | Restricts rules to feature comparisons, `if/else` branches, and binary returns; compiles without Python built-ins and translates failure branches into plain English. |
| `orchestrator.py` | Connects current forecast feature construction and authoritative HGB predictions to optional surrogate explanations. | Returns the unchanged HGB probability/prediction plus explanation text, surrogate prediction, and agreement status when a rule is available. |

## Supporting files

| File | Purpose |
| --- | --- |
| `llm_prompts.md` | Documents the generator, critic, and repair roles and the restricted rule format. |
| `prompts/generator_prompt.md` | Requests a compact rule using feature summaries, errors, feedback, and the best rule so far. |
| `prompts/critic_prompt.md` | Requests targeted feedback to improve fidelity, especially failure detection. |
| `prompts/repair_prompt.md` | Repairs malformed output into the permitted rule syntax. |
| `llm_config.example.json` | Safe template for endpoint, credential source, model, timeout, retry, and TLS settings. |

Run the optional offline steps from the repository root:

```bash
python -m src.explainability.compute_shap_rankings
python -m src.explainability.dual_llm --llm-config path/to/llm_config.json
```

Alternatively, include both steps and explained inference in the active
end-to-end pipeline:

```powershell
python run_pipeline.py --with-explainability --llm-config path\to\llm_config.json
```

For the small disposable smoke run, use one LLM iteration per line:

```powershell
python smoke_test.py --llm-config path\to\llm_config.json
```

Copy `llm_config.example.json` outside version control and provide the path to
`--llm-config`. The JSON configuration uses the legacy keys:

```json
{
  "api_url": "https://example/v1/chat/completions",
  "api_key_env": "OPENAI_API_KEY",
  "model": "model-name"
}
```

The legacy setup appears to have used an INESC TEC-hosted compatible endpoint.
For that service, request the endpoint URL, bearer API key, and enabled model ID
from its administrator. For the official OpenAI API, create a project key at
`https://platform.openai.com/api-keys`, use
`https://api.openai.com/v1/chat/completions` as `api_url`, and use a model ID
enabled for that project. Never commit the populated configuration file.
Set the key for the current PowerShell session before starting the pipeline:

```powershell
$env:OPENAI_API_KEY = "your-secret-key"
```

For a compatible internal service, the legacy `"api_key": "..."` JSON field
is also accepted, although an environment variable is preferable.

Rules are written under the configured output directory as
`line_<line>/best_rule.py`. Pass that directory when enabling runtime
explanations:

```bash
python training/predict_failure_forecast.py \
  --mean-model artifacts/.../mean_forecaster.pkl \
  --aleatoric-model artifacts/.../aleatoric_forecaster.pkl \
  --explain-rules-dir artifacts/.../explainability/rules
```

The offline workflow creates:

```text
artifacts/<ENV_NAME>/<AGENT_NAME>/failure_forecast/explainability/
|-- shap_rankings.json
`-- rules/
    |-- summary.json
    `-- line_<line>/
        |-- best_rule.py
        `-- metrics.json
```

`ExplainabilityOrchestrator` reuses `forecast_injections`,
`simulate_forecast_state`, `build_feature_row`, and
`FailureForecastPredictor.predict_from_features`. The HGB result stays
authoritative; `surrogate_prediction` and `surrogate_agrees` make clear that
the human-readable rule is only an approximation.

# Failure-forecast explainability

This package is an optional layer around the active HGB failure forecast. It
does not change forecast construction, classifier features, probabilities, or
binary predictions.

The offline workflow follows the legacy implementation:

1. Rank the HGB features for each contingency.
2. Ask one OpenAI-compatible model, with generator, critic, and repair prompts,
   to distil the HGB decisions into a small `if/else` rule per line.
3. Load those rules at runtime and translate them into text.

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

`ExplainabilityOrchestrator` reuses `forecast_injections`,
`simulate_forecast_state`, `build_feature_row`, and
`FailureForecastPredictor.predict_from_features`. The HGB result stays
authoritative; `surrogate_prediction` and `surrogate_agrees` make clear that
the human-readable rule is only an approximation.

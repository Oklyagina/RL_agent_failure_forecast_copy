# Agent API (InteractiveAI integration)

FastAPI wrapper exposing the CurriculumAgent as an InteractiveAI agent, built
to the AI4REALNET AI-agent template. It accepts the template recommendation
request and returns the main-project recommendation list directly. The ENN
epistemic-uncertainty KPIs are added to each recommendation's `kpis` field.

Files: `app/main.py` (the API), `app/__init__.py`, `project_config.py` (shared
`.env`/environment configuration), `Dockerfile`.

## Endpoint

```
POST /api/v1/recommendation
  body: {"event": ..., "context": {..., "observation": <grid2op observation>}}
  ->   [ {"title", "description", "use_case", "agent_type",
          "actions": [...], "kpis": {...}} ]
GET  /health
```

## Build and run

```bash
docker build -t curriculum-agent-api .
docker run --env-file .env -p 8000:8000 curriculum-agent-api
```

Locally (without Docker):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Test

```bash
curl -X POST http://localhost:8000/api/v1/recommendation \
     -H "Content-Type: application/json" \
     --data @rte_recommendation.json
```

Output — a list of recommendation dictionaries. ENN uncertainty is exposed as
`kpis.uncertainty` and the detailed ENN KPI fields are kept for traceability:

```json
[
  {
    "title": "Topological recommendation (CurriculumAgent)",
    "description": "...",
    "use_case": "PowerGrid",
    "agent_type": 2,
    "actions": [ { "_set_topo_vect": [ ... ], "...": "..." } ],
    "kpis": {
      "type_of_the_reco": "Topological",
      "efficiency_of_the_reco": 0.8976841568946838,
      "uncertainty": 50.0,
      "epistemic_uncertainty_pct": 50.0,
      "epistemic_uncertainty_total_pctile": 47.8,
      "epistemic_uncertainty_action_pctile": 3.1,
      "epistemic_uncertainty_level": "medium",
      "epistemic_confidence_level": "medium"
    }
  }
]
```

## Two things to confirm before deployment

1. **`efficiency_of_the_reco` / description.** These come from the helper
   `get_parade_info(action, obs)`, provided on the ExpertAgent side by
   **ExpertOp4Grid** (installed in the ExpertAgent container). `app/main.py`
   tries to import it (`app.parade`, `parade`, `expertop4grid`); wire the exact
   import there, or drop the helper into the repo as `app/parade.py`. Until
   then, `efficiency_of_the_reco` is `null` for the platform to fill and the
   description is a plain serialisation — the uncertainty percentiles are added
   on top either way.

2. **Environment / Grid2Op alignment.** The API uses `project_config.py`:
   environment variables override `.env`, and `.env` overrides the defaults.
   Configure `ENV_NAME` and `ENV_LOCATION` so they resolve to the Grid2Op scenario directory:
   `<ENV_LOCATION>/<ENV_NAME>`. The container uses the same project-relative
   layout as `run_example.py`; pass the existing `.env` at runtime with
   `--env-file .env`, or provide equivalent environment variables another way.

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

## Deployment notes

1. **`efficiency_of_the_reco` / description.** The vendored `app/parade.py`
   formatter provides the InteractiveAI title, description, action payload, and
   simulated maximum line loading in `efficiency_of_the_reco`. If that formatter
   cannot produce a recommendation, `app/main.py` falls back to plain action
   serialization and leaves `efficiency_of_the_reco` as `null`. ENN uncertainty
   fields are merged into either result.

2. **Environment / Grid2Op alignment.** The API uses `project_config.py`:
   environment variables override `.env`, and `.env` overrides the defaults.
   Configure `ENV_NAME` and `ENV_LOCATION` so they resolve to the Grid2Op scenario directory:
   `<ENV_LOCATION>/<ENV_NAME>`. The container uses the same project-relative
   layout as `run_example.py`; pass the existing `.env` at runtime with
   `--env-file .env`, or provide equivalent environment variables another way.

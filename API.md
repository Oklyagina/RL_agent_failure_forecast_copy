# Agent API (InteractiveAI integration)

FastAPI wrapper exposing the CurriculumAgent as an InteractiveAI agent, built
to the AI4REALNET AI-agent template. It returns recommendations in the
InteractiveAI dictionary format and adds the two ENN epistemic-uncertainty
percentiles into the `kpis` field, alongside `efficiency_of_the_reco`.

Files: `app/main.py` (the API), `app/__init__.py`, `setup.py` (root, so the
Docker build's `pip install .` works), `Dockerfile`.

## Endpoint

```
POST /api/v1/recommendation
  body: {"event": ..., "context": {..., "observation": <grid2op observation>}}
  ->   [ {"title", "description", "use_case", "agent_type",
          "actions": [...], "kpis": {...}}, ... ]
GET  /health
```

## Build and run

```bash
docker build -t curriculum-agent-api .
docker run -p 8000:8000 curriculum-agent-api
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

Output — a list of recommendation dictionaries; note the two uncertainty
percentiles inside `kpis`:

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
      "epistemic_uncertainty_total_pctile": 47.8,
      "epistemic_uncertainty_action_pctile": 3.1
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

2. **Environment / Grid2Op alignment.** The `Dockerfile` sets up
   `ai4realnet_small` (the same scenario the simulator and the ExpertAgent
   container use) and defaults `GRID2OP_ENV` to it. But the CurriculumAgent was
   trained on `l2rpn_icaps_2021_small` — if it does not run natively on
   `ai4realnet_small` (this is the observation shape mismatch that was fixed on
   first runs), set `GRID2OP_ENV=l2rpn_icaps_2021_small`, or keep whatever that
   fix established as the correct environment. Local runs (`run_example.py`,
   tests) fall back to `l2rpn_icaps_2021_small`, which Grid2Op downloads
   automatically.

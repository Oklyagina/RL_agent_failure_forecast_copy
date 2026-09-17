# Docker Instructions

This API container runs the FastAPI recommendation service from `app/main.py`.
It exposes the CurriculumAgent recommendation endpoint with ENN uncertainty
KPIs.

## Prerequisites

- Docker is installed and running.
- Run commands from the repository root.
- A local `.env` file exists, or create one from the example:

```powershell
Copy-Item .env.example .env
```

On Linux or macOS:

```bash
cp .env.example .env
```

The `.env` values are read by `project_config.py`. Runtime environment
variables override `.env`, and `.env` overrides the project defaults.

## Build

```powershell
docker build -t curriculum-agent-api .
```

## Run

```powershell
docker run --env-file .env -p 8000:8000 curriculum-agent-api
```

The API will be available at:

```text
http://localhost:8000
```

## Grid2Op Environment

With the default repository layout, the Docker image includes the following
Grid2Op environment:

```text
/app/environment/ai4realnet_small
```

The default `.env` values point to that in-container path:

```text
ENV_LOCATION=environment
ENV_NAME=ai4realnet_small
```

## Available URLs

```text
GET  http://localhost:8000/health
GET  http://localhost:8000/diagnostics
GET  http://localhost:8000/docs
POST http://localhost:8000/api/v1/recommendation
```

## Check The Container

Check that the API is running:

```powershell
curl http://localhost:8000/health
```

Check whether the Grid2Op environment, agent assets, and ENN artifacts can be
loaded:

```powershell
curl http://localhost:8000/diagnostics
```

## Get A Recommendation In Swagger

1. Open `http://localhost:8000/docs`.
2. Expand `POST /api/v1/recommendation`.
3. Select `Try it out`.
4. Use the built-in example:

```json
{
  "event": {},
  "context": {}
}
```

5. Select `Execute`.

When `context.observation` is omitted, the API uses `env.reset()` internally.
To score a simulator state, send the InteractiveAI/Grid2Op observation payload
under `context.observation`.

## Equivalent Curl Smoke Test

```powershell
curl.exe -X POST http://localhost:8000/api/v1/recommendation `
  -H "Content-Type: application/json" `
  --data-raw '{"event":{},"context":{}}'
```

Expected response shape:

```json
[
  {
    "title": "Topological recommendation (CurriculumAgent)",
    "description": "...",
    "use_case": "PowerGrid",
    "agent_type": 2,
    "actions": [],
    "kpis": {
      "type_of_the_reco": "Topological",
      "efficiency_of_the_reco": null,
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

If `/diagnostics` reports `artifact_validation.ok: false` or
`services.can_load: false`, the recommendation endpoint may return `503` until
the Grid2Op environment, CurriculumAgent assets, and ENN artifacts are available
inside the image or mounted at the configured path. Check `/diagnostics` first;
only mount the environment if the image does not include it or `ENV_LOCATION`
intentionally points to external scenario data.

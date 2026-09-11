# InteractiveAI agent API container: CurriculumAgent + ENN uncertainty.
# Mirrors the AI4REALNET ExpertAgent Dockerfile, adapted for the CurriculumAgent.
#
# python:3.10-slim (the ExpertAgent uses 3.12): the bundled CurriculumAgent
# SavedModel was exported with Keras 2.12, and each agent has its own container,
# so this stack can keep its Python/TensorFlow compatibility target.
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

# NOTE: the API uses project_config.py for shared settings.
# Runtime environment variables override .env, and .env overrides defaults.

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# API + agent binaries + module + trained artifacts (whole repo)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# InteractiveAI agent API container: CurriculumAgent + ENN uncertainty.
# Mirrors the AI4REALNET ExpertAgent Dockerfile, adapted for the CurriculumAgent.
#
# python:3.10-slim (the ExpertAgent uses 3.12): tensorflow 2.12.1, ray 2.5.1 and
# torch 2.1.2 have no wheels for Python >= 3.11. Each agent has its own
# container, so the Python version is free (confirmed with IRT-SystemX).
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

# InteractiveAI environment data: same scenario as the simulator and the
# ExpertAgent container (ai4realnet_small, from the grid2op-scenario repo).
RUN git clone https://github.com/AI4REALNET/grid2op-scenario.git /tmp/grid2op-scenario \
    && mkdir -p /root/data_grid2op \
    && cp -r /tmp/grid2op-scenario/ai4realnet_small /root/data_grid2op/ai4realnet_small \
    && rm -rf /tmp/grid2op-scenario

# NOTE: the CurriculumAgent was trained on l2rpn_icaps_2021_small. If it does
# not run natively on ai4realnet_small (cf. the observation shape mismatch you
# fixed), set GRID2OP_ENV=l2rpn_icaps_2021_small below, or keep whatever your
# fix established as the correct environment.

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# API + agent binaries + module + trained artifacts (whole repo)
COPY . .
RUN pip install .

EXPOSE 8000

ENV GRID2OP_ENV=ai4realnet_small
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

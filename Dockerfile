# FunCAD-Harness: CAD + PyChrono environments, Codex CLI, Claude Code CLI and the web GUI.
#   docker compose up            # then open the link printed in the log
FROM mambaorg/micromamba:2.3.2

ARG CODEX_VERSION=0.162.1
ARG CLAUDE_CODE_VERSION=2.1.296

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git libgl1 libglu1-mesa libxrender1 libxi6 libxkbcommon0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
COPY environment-cad.yml environment-chrono.yml /tmp/env/
RUN micromamba create -y -n funcad-cad -f /tmp/env/environment-cad.yml \
 && micromamba create -y -n funcad-chrono -f /tmp/env/environment-chrono.yml \
 && micromamba create -y -n node -c conda-forge nodejs=22 \
 && micromamba clean -a -y && rm -rf /tmp/env
RUN PATH=/opt/conda/envs/node/bin:$PATH npm install -g --prefix /opt/conda/envs/node \
        @openai/codex@${CODEX_VERSION} @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} \
 && rm -rf /root/.npm

# The image is the isolation boundary for agent-written programs. Agents run
# unprivileged: the harness code and environments stay root-owned (read-only);
# only .runtime_runs (mounted from ./runs) and HOME are writable. HOME accepts
# any uid because docker-compose runs as the host user.
RUN useradd -m -u 1000 -s /bin/bash funcad && chmod 1777 /home/funcad \
 && echo 'export PATH=/opt/conda/envs/funcad-cad/bin:/opt/conda/envs/node/bin:$PATH' > /etc/profile.d/funcad.sh
WORKDIR /opt/funcad-harness
COPY . /opt/funcad-harness
RUN mkdir -p /opt/funcad-harness/.runtime_runs && chown funcad:funcad /opt/funcad-harness/.runtime_runs

ENV SIMREADY_CAD_PYTHON=/opt/conda/envs/funcad-cad/bin/python \
    SIMREADY_PYCHRONO_EXECUTABLE=/opt/conda/envs/funcad-chrono/bin/python \
    LD_LIBRARY_PATH=/opt/conda/envs/funcad-cad/lib \
    PATH=/opt/conda/envs/funcad-cad/bin:/opt/conda/envs/node/bin:$PATH \
    SIMREADY_CONTAINER_ISOLATION=1 \
    SIMREADY_RENDER_LANG=en \
    HOME=/home/funcad

USER funcad
EXPOSE 8765
ENTRYPOINT []
CMD ["/opt/conda/envs/funcad-cad/bin/python", "-X", "utf8", "-m", "simulation_ready_cad_workflow.validation_pipeline.dashboard", \
     "serve", "--enable-launch", "--allow-remote-launch", "--host", "0.0.0.0", "--port", "8765"]

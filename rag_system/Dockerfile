# ===============================
# Stage: Base Builder
# ===============================
FROM ubuntu:22.04 AS base-builder

RUN apt-get update -qqy && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        ca-certificates \
        libpoppler-dev \
        unzip \
        curl \
        cargo \
        git \
        ssh \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

WORKDIR /app

# Download pdfjs
COPY kotaemon/scripts/download_pdfjs.sh /app/scripts/download_pdfjs.sh
RUN chmod +x /app/scripts/download_pdfjs.sh
ENV PDFJS_PREBUILT_DIR="/app/libs/ktem/ktem/assets/prebuilt/pdfjs-dist"
RUN bash scripts/download_pdfjs.sh $PDFJS_PREBUILT_DIR

# Copy project files
COPY kotaemon/libs /app/libs
COPY kotaemon/launch.sh /app/launch.sh
COPY kotaemon/app.py /app/app.py
COPY kotaemon_pipeline_scripts/.env /app/.env
COPY kotaemon_pipeline_scripts/flowsettings.py /app/flowsettings.py
COPY kotaemon_pipeline_scripts /app/pipeline_scripts
COPY taxonomy /app/taxonomy


RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

RUN uv venv --python=3.10
RUN . .venv/bin/activate
RUN chmod +x .venv/bin/activate



# ===============================
# Stage: dev (editable installs)
# ===============================
FROM base-builder AS dev

RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install -e "libs/kotaemon[adv]" && \
    uv pip install -e "libs/ktem" && \
    uv pip install -e "libs/pipelineblocks" && \
    uv pip install -e "pipeline_scripts" && \
    uv pip install -e "taxonomy" && \
    uv pip install boto3 && \
    uv pip install "pdfservices-sdk@git+https://github.com/niallcm/pdfservices-python-sdk.git@bump-and-unfreeze-requirements"


# ===============================
# Stage: prod (non-editable installs)
# ===============================
FROM base-builder AS prod

RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install "libs/kotaemon[adv]" && \
    uv pip install "libs/ktem" && \
    uv pip install "libs/pipelineblocks" && \
    uv pip install "pipeline_scripts" && \
    pip install "taxonomy" && \
    pip installboto3 && \
    uv pip install psycopg2-binary && \
    uv pip install "pdfservices-sdk@git+https://github.com/niallcm/pdfservices-python-sdk.git@bump-and-unfreeze-requirements"

RUN uv pip install llama-index==0.10.68 && \
    uv pip install llama-index-vector-stores-qdrant==0.2.17 && \
    uv pip install requests-toolbelt==0.10.1 urllib3==1.26.20 voyageai

ENTRYPOINT ["bash", "/app/launch.sh"]

# ===============================
# Shared minimal runtime base
# ===============================
# FROM ubuntu:22.04 AS runtime-base
#
# RUN apt-get update -qqy && \
#     apt-get install -y --no-install-recommends \
#         ssh \
#         git \
#         poppler-utils \
#         curl \
#         ca-certificates \
#     && rm -rf /var/lib/apt/lists/*
#
# # Install uv
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
#
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PYTHONIOENCODING=UTF-8 \
#     PATH="/root/.local/bin:/app/.venv/bin:$PATH"
#
# WORKDIR /app
#
#
# # ===============================
# # Final: dev-runtime
# # ===============================
# FROM runtime-base AS dev-runtime
#
# COPY --from=dev /app /app
#
# ENTRYPOINT ["/bin/sh", "-c", "tail -f /dev/null"]
#
#
# # ===============================
# # Final: prod-runtime
# # ===============================
# FROM runtime-base AS prod-runtime
#
# COPY --from=prod /app /app
#
# ENTRYPOINT ["sh", "/app/launch.sh"]
# ===============================
# Stage: Base Builder
# ===============================
FROM python:3.10-slim AS base-builder

RUN apt-get update -qqy && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
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


# ===============================
# Stage: dev (editable installs)
# ===============================
FROM base-builder AS dev

RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache/pip \
    pip install -e "libs/kotaemon[adv]" && \
    pip install -e "libs/ktem" && \
    pip install -e "libs/pipelineblocks" && \
    pip install -e "pipeline_scripts" && \
    pip install -e "taxonomy" && \
    pip install boto3 && \
    pip install "pdfservices-sdk@git+https://github.com/niallcm/pdfservices-python-sdk.git@bump-and-unfreeze-requirements"


# ===============================
# Stage: prod (non-editable installs)
# ===============================
FROM base-builder AS prod

RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache/pip \
    pip install "libs/kotaemon[adv]" && \
    pip install "libs/ktem" && \
    pip install "libs/pipelineblocks" && \
    pip install "pipeline_scripts" && \
    pip install "taxonomy" && \
    pip install boto3 && \
    pip install "pdfservices-sdk@git+https://github.com/niallcm/pdfservices-python-sdk.git@bump-and-unfreeze-requirements"


# ===============================
# Shared minimal runtime base
# ===============================
FROM python:3.10-slim AS runtime-base

RUN apt-get update -qqy && \
    apt-get install -y --no-install-recommends \
        ssh \
        git \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

WORKDIR /app


# ===============================
# Final: dev-runtime
# ===============================
FROM runtime-base AS dev-runtime

COPY --from=dev /app /app
COPY --from=dev /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dev /usr/local/bin /usr/local/bin

ENTRYPOINT ["/bin/sh", "-c", "tail -f /dev/null"]


# ===============================
# Final: prod-runtime
# ===============================
FROM runtime-base AS prod-runtime

COPY --from=prod /app /app
COPY --from=prod /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=prod /usr/local/bin /usr/local/bin

ENTRYPOINT ["sh", "/app/launch.sh"]
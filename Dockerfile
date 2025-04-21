# Lite version
FROM python:3.10-slim AS lite

# Common dependencies
RUN apt-get update -qqy && \
    apt-get install -y --no-install-recommends \
        ssh \
        git \
        gcc \
        g++ \
        poppler-utils \
        libpoppler-dev \
        unzip \
        curl \
        cargo

# Setup args
ARG TARGETPLATFORM
ARG TARGETARCH

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV TARGETARCH=${TARGETARCH}

# Create working directory
WORKDIR /app

# Download pdfjs
RUN ls
ADD rag_system/kotaemon/scripts/download_pdfjs.sh /app/scripts/download_pdfjs.sh
RUN chmod +x /app/scripts/download_pdfjs.sh
ENV PDFJS_PREBUILT_DIR="/app/libs/ktem/ktem/assets/prebuilt/pdfjs-dist"
RUN bash /app/scripts/download_pdfjs.sh $PDFJS_PREBUILT_DIR

# Copy contents
COPY rag_system /app
COPY rag_system/kotaemon/launch.sh /app/launch.sh
COPY rag_system/kotaemon/.env.example /app/.env

WORKDIR /app/kotaemon

# Install pip packages
RUN pip install -e "libs/kotaemon" \
    && pip install -e "libs/ktem" \
    && pip install -e "libs/pipelineblocks" \
    && pip install "pdfservices-sdk@git+https://github.com/niallcm/pdfservices-python-sdk.git@bump-and-unfreeze-requirements"

# Clean up
RUN apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache

ENTRYPOINT ["sh", "/app/launch.sh"]

# Full version
FROM lite AS full

# Additional dependencies for full version
RUN apt-get update -qqy && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-jpn \
        libsm6 \
        libxext6 \
        libreoffice \
        ffmpeg \
        libmagic-dev

# Install torch and torchvision for unstructured
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install psycopg2-binary logfire

# Install additional pip packages
RUN pip install -e "libs/kotaemon[adv]" \
    && pip install unstructured[all-docs]

# Install lightRAG
ENV USE_LIGHTRAG=false

RUN pip install "docling<=2.5.2"

# Clean up
RUN apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache

CMD ["sh", "/app/launch.sh"]
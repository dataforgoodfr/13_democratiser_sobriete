# to avoid OOM issues and give it a max amount of memory / CPUs, run this with
# create image : docker build -t <image_name> -f scraping/Dockerfile .
# run it docker run --memory="Ng" --memory-swap="Ng" --cpus="K" <image_name>

FROM python:3.13-slim

# Install wget for downloading files
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY .env ./

RUN uv sync --frozen --no-dev --group pdfscraping

COPY src/library ./library
COPY scraping/extract_text_from_pdfs.py .

CMD ["uv", "run", "extract_text_from_pdfs.py", "--md"]
# Installation Guide

> [!Warning]
> Needs updating after refactoring. Possibly, delete this document and split its content between general and sub-projects READMEs.

## Prerequisites

Before you begin, ensure you have:
- Python 3.12 or higher
- git
- 4GB+ RAM recommended

## 🚀 Quick Start

### 1. Install `uv` Package Manager

`uv` is a fast Python package installer and resolver that we use for dependency management.

**macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative - via pip:**
```bash
pip install uv
```

For more information, see the [official uv documentation](https://astral.sh/uv).

### 2. Clone the Repository

```bash
git clone https://github.com/dataforgoodfr/13_democratiser_sobriete.git
cd 13_democratiser_sobriete
```

### 3. Install Project Dependencies

```bash
# Install all dependencies using uv
uv sync

# Or if using poetry
poetry install
```

### 4. Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 5. Verify Installation

```bash
# Run tests to ensure everything is set up correctly
uv run pytest tests/

# Or with poetry
poetry run pytest tests/
```

## Component-Specific Setup

### RAG System (Kotaemon)

```bash
cd rag_system/kotaemon

# Copy settings template
cp settings.yaml.example settings.yaml

# Edit settings with your API keys and configuration
nano settings.yaml

# Run the RAG system
./scripts/run_linux.sh    # Linux
./scripts/run_macos.sh    # macOS
./scripts/run_windows.bat # Windows
```

### PDF Extraction Module

```bash
# Install additional dependencies for PDF processing
uv pip install pymupdf pypdf2

# Test PDF extraction
uv run python tests/pdfextraction/pdf/test_pymu.py
```

### Taxonomy System

```bash
cd rag_system/taxonomy

# The taxonomy module is automatically available after main installation
# Test it with:
uv run python -c "from taxonomy import geographical_taxonomy; print('Taxonomy loaded')"
```

## Docker Setup (Optional)

If you prefer to run the project in Docker:

```bash
# Build the Docker image
docker build -t kotaemon .

# Run with Docker Compose
docker-compose up -d

# For the RAG system specifically
cd rag_system
docker-compose up -d
```

## Troubleshooting

### Common Issues

**Issue: `uv` command not found**
```bash
# Add uv to your PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or restart your terminal
```


**Issue: Dependencies conflict**
```bash
# Clear cache and reinstall
uv cache clean
rm -rf .venv
uv sync
```

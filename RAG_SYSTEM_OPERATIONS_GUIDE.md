# RAG System Operations Guide
**Document Processing Pipeline for Academic Papers**

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Paper Processing Pipeline](#2-paper-processing-pipeline)
3. [Models & Components](#3-models--components)
4. [Document Ingestion Process](#4-document-ingestion-process)
5. [Production Deployment](#5-production-deployment)
6. [Data Transformation Details](#6-data-transformation-details)

---

## 1. Architecture Overview

### System Components

The RAG system operates in two environments:

#### Local Development Environment
```
┌─────────────────────────────────────────────────────────────────────┐
│                        Docker Compose Local Setup                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │   Kotaemon   │   │    Ollama    │   │        Qdrant            │ │
│  │    App       │   │   (LLM Host) │   │   (Vector Store)         │ │
│  │  Port: 7860  │   │ Port: 11434  │   │   Ports: 6333-6335       │ │
│  └──────────────┘   └──────────────┘   └──────────────────────────┘ │
│         │                    │                       │              │
│         └────────────────────┼───────────────────────┘              │
│                              │                                      │
│  ┌──────────────────────────────────────────────────────────────────┤
│  │               Pipeline Scripts                                   │
│  │  • Paper scraping & processing                                   │
│  │  • Metadata extraction                                           │
│  │  • 12-folder parallel processing                                 │
│  └──────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

#### Production Environment
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Production Docker Swarm                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────┐                                                         │
│  │  Kotaemon App  │                                                         │
│  │   Container    │                                                         │
│  │   (Swarm)      │                                                         │
│  └────────────────┘                                                         │
│           │                                                                 │
│           ├─────────── External Dependencies ──────────────┐                │
│           │                                                │                │
│  ┌─────────────────┐                              ┌─────────────────────┐   │
│  │   PostgreSQL    │                              │   Qdrant Cloud      │   │
│  │ (Clever Cloud)  │                              │   Vector Database   │   │
│  │                 │                              │                     │   │
│  │ • User data     │                              │ • Document vectors  │   │
│  │ • Processing    │                              │ • Embeddings        │   │
│  │   queue         │                              │ • Collections       │   │
│  └─────────────────┘                              └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### External API Dependencies
- **OpenAlex API**: Paper metadata and discovery
- **DeepSeek API**: LLM inference for metadata extraction  
- **Ollama Local**: Embedding models (Snowflake Arctic Embed 2)

---

## 2. Paper Processing Pipeline

### Complete Workflow Overview

```mermaid
graph TD
    A[Paper Discovery] --> B[Queue Population]
    B --> C[Targeted Scraping]
    C --> D[PDF Distribution]
    D --> E[Parallel Processing]
    E --> F[Metadata Extraction]
    F --> G[Data Reconciliation]
    G --> H[Vector Embedding]
    H --> I[Storage & Indexing]

    subgraph "Phase 1: Discovery & Acquisition"
        A --> A1[OpenAlex API Query]
        A1 --> A2[Filter Papers by Criteria]
        A2 --> B
        B --> B1[PostgreSQL Queue]
    end

    subgraph "Phase 2: Document Retrieval"
        C --> C1[Selenium Web Scraper]
        C1 --> C2[DOI Fallback Logic]
        C2 --> C3[PDF Download]
        C3 --> D
        D --> D1[12-Folder Distribution]
    end

    subgraph "Phase 3: Processing & Extraction"
        E --> E1[Parallel Workers x12]
        E1 --> F
        F --> F1[PDF → Markdown]
        F1 --> F2[LLM Metadata Inference]
        F2 --> G
        G --> G1[OpenAlex + LLM Merge]
    end

    subgraph "Phase 4: Storage & Indexing"
        H --> H1[Text Embedding]
        H1 --> H2[Vector Generation]
        H2 --> I
        I --> I1[Qdrant Collection]
        I1 --> I2[Document Store]
    end
```

### Detailed Task Sequence & Dependencies

#### Phase 1: Discovery & Queue Management
```
┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 1: DISCOVERY                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Paper Discovery                                                 │
│     └── Source: OpenAlex API                                        │
│     └── Criteria: Academic papers matching research topics          │
│     └── Output: Paper IDs + basic metadata                          │
│                                                                     │
│  2. Queue Population                                                │
│     └── Input: OpenAlex paper IDs                                   │
│     └── Process: Insert into PostgreSQL queue                       │
│     └── Fields: paper_id, doi, url, status, retry_count             │
│     └── Command: python cli.py queue --populate --limit 250000      │
│                                                                     │
│  3. Queue Management                                                │
│     └── Monitor: python cli.py queue --stats                        │
│     └── Reset failed: python cli.py queue --reset-failed            │
│     └── Clear queue: python cli.py queue --clear                    │
└─────────────────────────────────────────────────────────────────────┘
```

#### Phase 2: Document Acquisition
```
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: DOCUMENT ACQUISITION                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Targeted Scraping                                               │
│     └── Command: python cli.py scrape --batch-size 50               │
│     └── Technology: Selenium WebDriver + Chrome                     │
│     └── Logic:                                                      │
│         ┌── Try OpenAlex ID → PDF URL                               │
│         └── If fails → Try DOI → Publisher site → PDF               │
│                                                                     │
│  2. Download Process                                                │
│     └── Input: Paper queue (pending status)                         │
│     └── Process:                                                    │
│         ┌── Batch processing (configurable size)                    │
│         ├── Error handling & retry logic                            │
│         ├── Progress tracking in database                           │
│         └── PDF validation                                          │
│                                                                     │
│  3. File Distribution                                               │
│     └── Output: 12 folders (folder_00 to folder_11)                 │
│     └── Logic: Round-robin distribution                             │
│     └── Location: ./scraping_output/folder_XX/                      │
│     └── Format: Papers named by OpenAlex ID (W1234567890.pdf)       │
└─────────────────────────────────────────────────────────────────────┘
```

#### Phase 3: Parallel Processing
```
┌─────────────────────────────────────────────────────────────────────┐
│                   PHASE 3: PARALLEL PROCESSING                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Parallel Worker Launch                                          │
│     └── Command: ./run_metadata_extraction.sh                       │
│     └── Workers: 12 parallel processes                              │
│     └── Each worker processes one folder independently              │
│                                                                     │
│  2. Per-Folder Processing                                           │
│     └── Worker Command:                                             │
│         python historized_ingestion_pipeline.py --folder-path       │
│           ./scraping_output/folder_XX                               │
│                                                                     │
│  3. Individual Paper Processing (per worker)                        │
│     └── For each PDF in folder:                                     │
│         ┌── PDF → Markdown extraction                               │
│         ├── OpenAlex metadata retrieval                             │
│         ├── LLM metadata inference (DeepSeek)                       │
│         ├── Metadata reconciliation                                 │
│         ├── Text embedding generation                               │
│         ├── Vector store ingestion (Qdrant)                         │
│         └── Document store ingestion                                │
│                                                                     │
│  4. Success/Failure Tracking                                        │
│     └── Success: Move to ./scraping_output/done/folder_XX/          │
│     └── Failure: Log to failed_extractions.txt                      │
│     └── Progress: Real-time console output                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Models & Components

### Language Models & APIs

| Component | Model/Service | Purpose | Configuration |
|-----------|---------------|---------|---------------|
| **LLM Inference** | DeepSeek API | Metadata extraction from PDFs | `base_url: https://api.deepseek.com` |
| **Text Embeddings** | Snowflake Arctic Embed 2 | Vector generation for retrieval | Via Ollama: `localhost:11434/v1/` |
| **Paper Metadata** | OpenAlex API | Academic paper discovery & metadata | `api.openalex.org` |

### Storage Components

| Component | Technology | Purpose | Location |
|-----------|------------|---------|----------|
| **Vector Database** | Qdrant Cloud | Document embeddings & similarity search | `qdrant_host: europe-west3-0.gcp.cloud.qdrant.io` |
| **Relational Database** | PostgreSQL (Clever Cloud) | Queue management, user data | Environment: `PG_DATABASE_URL` |
| **Document Store** | SimpleFileDocumentStore | Raw document storage | `/app/ktem_app_data/user_data/docstore` |
| **File Storage** | Local/Docker volumes | PDF files, processed data | `./scraping_output/`, `./ktem_app_data/` |

### Processing Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROCESSING COMPONENT STACK                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ PDF Extraction  │  │ Metadata LLM    │  │   Embedding         │  │
│  │                 │  │ Inference       │  │   Generation        │  │
│  │ • GroupAll      │  │                 │  │                     │  │
│  │   method        │  │ • DeepSeek API  │  │ • Arctic Embed 2    │  │
│  │ • Markdown      │  │ • Scientific    │  │ • Ollama local      │  │
│  │   output        │  │   taxonomy      │  │ • 1024 dimensions   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Vector          │  │ Document Store  │  │   Queue             │  │
│  │ Indexing        │  │                 │  │   Management        │  │
│  │                 │  │ • File paths    │  │                     │  │
│  │ • Qdrant        │  │ • Metadata      │  │ • PostgreSQL        │  │
│  │ • Collection:   │  │ • Chunks        │  │ • Status tracking   │  │
│  │   index_1       │  │ • Taxonomy      │  │ • Retry logic       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Document Ingestion Process

### Prerequisites & Setup

#### Local Development Setup
```bash
# 1. Environment Setup
cd rag_system/
docker-compose up -d

# 2. Install Chrome & Dependencies  
brew install --cask google-chrome  # macOS
# OR sudo apt install google-chrome-stable  # Ubuntu
pip install selenium webdriver-manager

# 3. Install Pipeline Dependencies
cd pipeline_scripts/
pip install -r requirements.txt

# 4. Environment Variables
export PG_DATABASE_URL="postgresql://user:pass@host/db"
export VECTOR_STORE_API_KEY="your-qdrant-key"
export DS_SECRET_KEY="your-deepseek-key"
export VECTORSTORE_URL="https://your-cluster.gcp.cloud.qdrant.io:6333"
```

#### Production Deployment Check
```bash
# Verify external connections
python cli.py queue --stats  # Test PostgreSQL
# Check Qdrant connectivity in app settings
# Verify DeepSeek API key
```

### Step-by-Step Operations Guide

#### Step 1: System Testing
```bash
# Test with small batch first
python cli.py test
# Expected output: 10 papers downloaded & processed
# Expected folders: ./test_scraping_output/folder_00/
# Expected database entries: 10 test papers in queue
```

#### Step 2: Queue Population  
```bash
# Production queue setup
python cli.py queue --populate --limit 50000
# This queries OpenAlex API and populates PostgreSQL queue
# Monitor: python cli.py queue --stats
```

#### Step 3: Paper Acquisition
```bash
# Start scraping in batches
python cli.py scrape --batch-size 100

# Monitor progress
python cli.py queue --stats
python cli.py queue --show-failed --limit 10

# Expected output structure:
# ./scraping_output/folder_00/W1234567890.pdf
# ./scraping_output/folder_01/W9876543210.pdf
# ... (distributed across 12 folders)
```

#### Step 4: Parallel Metadata Extraction
```bash
# Launch 12 parallel processing workers
chmod +x run_metadata_extraction.sh
./run_metadata_extraction.sh

# Monitor progress
tail -f failed_extractions.txt

# Expected behavior:
# - 12 parallel processes start
# - Each processes one folder independently  
# - Success: PDFs moved to ./scraping_output/done/
# - Failures: Logged to failed_extractions.txt
```

#### Step 5: Validation & Monitoring
```bash
# Check processing results
ls -la scraping_output/done/  # Successfully processed papers
wc -l failed_extractions.txt  # Count failed papers

# Validate vector store ingestion
# (Check via Kotaemon app interface)
# - Navigate to Documents section
# - Verify new papers appear
# - Test search functionality
```

### Error Recovery Procedures

#### Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Chrome/Selenium Error** | Scraping fails immediately | `brew install google-chrome` + restart |
| **API Rate Limiting** | OpenAlex/DeepSeek timeouts | Reduce batch size, add delays |
| **Database Connection** | Queue commands fail | Check `PG_DATABASE_URL` environment variable |
| **Qdrant Connection** | Vector storage fails | Verify `VECTOR_STORE_API_KEY` and URL |
| **Parallel Processing Stuck** | Workers hang indefinitely | Kill processes, check folder permissions |

#### Recovery Commands
```bash
# Reset failed papers for retry
python cli.py queue --reset-failed

# Clear entire queue (testing)
python cli.py queue --clear

# Restart specific folder processing
python historized_ingestion_pipeline.py --folder-path ./scraping_output/folder_05

# Check worker status
ps aux | grep "historized_ingestion_pipeline"
```

---

## 5. Production Deployment

### Docker Swarm Configuration

The production system runs on Docker Swarm with external database dependencies:

```yaml
# Production stack (conceptual)
services:
  kotaemon:
    image: custom-kotaemon:latest
    environment:
      - PG_DATABASE_URL=${PG_DATABASE_URL}
      - VECTOR_STORE_API_KEY=${VECTOR_STORE_API_KEY}  
      - VECTORSTORE_URL=${VECTORSTORE_URL}
      - DS_SECRET_KEY=${DS_SECRET_KEY}
    volumes:
      - pipeline_data:/app/pipeline_scripts
      - kotaemon_data:/app/ktem_app_data
```

### External Dependencies Configuration

#### PostgreSQL (Clever Cloud)
- **Purpose**: User data, processing queue, application state
- **Connection**: Environment variable `PG_DATABASE_URL`
- **Tables**: Users, documents, processing queue, metadata cache

#### Qdrant Cloud  
- **Purpose**: Vector embeddings, similarity search
- **Connection**: `VECTORSTORE_URL` + `VECTOR_STORE_API_KEY`
- **Collections**: `index_1` (primary document vectors)

### Performance Considerations

| Metric | Local Development | Production |
|--------|------------------|------------|
| **Scraping Throughput** | ~100-200 papers/hour | ~1,000-1,250 papers/hour |
| **Processing Time** | Sequential | 12x parallel (200-250 hours for 250k) |
| **Storage Requirements** | ~10GB (testing) | ~500GB-1TB (250k papers) |
| **Memory Usage** | 4-8GB | 16-32GB (parallel processing) |

---

## 6. Data Transformation Details

### Source Data & Filters

#### OpenAlex API Input
**Source**: `api.openalex.org/works`
**Filters Applied**:
- Academic papers only (type: `article`)
- Open access when possible
- English language preference  
- Publication date filters (configurable)
- Subject area filters (research-specific)

**Sample API Response**:
```json
{
  "id": "W1234567890",
  "doi": "10.1000/example.doi", 
  "title": "Example Academic Paper",
  "abstract": "Research abstract...",
  "publication_year": 2023,
  "concepts": [...],
  "authors": [...],
  "locations": [{"pdf_url": "https://..."}]
}
```

### Data Extraction Pipeline

#### PDF → Markdown Transformation
```
┌─────────────────────────────────────────────────────────────────────┐
│                    PDF EXTRACTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input:   W1234567890.pdf                                           │
│           └── Raw PDF binary                                        │
│                                                                     │
│  Process: PdfExtractionToMarkdownBlock                              │
│           └── Method: "group_all"                                   │
│           └── Engine: Advanced PDF parser                           │
│           └── Layout: Preserve structure                            │
│                                                                     │
│  Output:  Structured Markdown                                       │
│           ├── # Title                                               │
│           ├── ## Abstract                                           │
│           ├── ## Introduction                                       │
│           ├── ## Methods                                            │
│           ├── ## Results                                            │
│           ├── ## Discussion                                         │
│           └── ## References                                         │
│                                                                     │
│  Quality: Text cleaning, structure preservation                     │
│           └── Remove headers/footers                                │
│           └── Preserve section hierarchy                            │
│           └── Handle tables and figures                             │
└─────────────────────────────────────────────────────────────────────┘
```

#### LLM Metadata Inference
```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM METADATA EXTRACTION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input:   Markdown text + OpenAlex metadata                         │
│           └── Full paper content                                    │
│           └── Existing metadata (title, authors, abstract)          │
│                                                                     │
│  Process: DeepSeek LLM Inference                                    │
│           └── Model: DeepSeek API                                   │
│           └── Prompt: Scientific paper analysis                     │
│           └── Type: "scientific_advanced"                           │
│           └── Taxonomy: PaperTaxonomy schema                        │
│                                                                     │
│  Extracted Fields:                                                  │
│  ├── Research Domain Classification                                 │
│  ├── Methodology Tags                                               │
│  ├── Key Concepts & Terms                                           │
│  ├── Research Questions                                             │
│  ├── Main Findings                                                  │
│  ├── Data Sources & Types                                           │
│  ├── Geographic Scope                                               │
│  ├── Time Period Covered                                            │
│  ├── Policy Relevance                                               │
│  └── Citation Relationships                                         │
│                                                                     │
│  Output:  Structured JSON metadata                                  │
│           └── Taxonomy-compliant format                             │
└─────────────────────────────────────────────────────────────────────┘
```

#### Metadata Reconciliation Process
```
┌─────────────────────────────────────────────────────────────────────┐
│                    METADATA RECONCILIATION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Inputs:  ┌── OpenAlex metadata (authoritative)                     │
│           └── LLM-extracted metadata (enriched)                     │
│                                                                     │
│  Process: reconcile_metadata() function                             │
│                                                                     │
│  Rules:   ┌── OpenAlex takes precedence for:                        │
│           │   ├── Title, authors, publication year                  │
│           │   ├── DOI, journal information                          │
│           │   └── Citation counts                                   │
│           │                                                         │
│           └── LLM enriches with:                                    │
│               ├── Subject classification                            │
│               ├── Methodology analysis                              │
│               ├── Research findings summary                         │
│               └── Domain-specific tags                              │
│                                                                     │
│  Output:  Unified metadata object                                   │
│           ├── Authoritative bibliographic data                      │
│           ├── Enhanced subject classification                       │
│           ├── Research methodology tags                             │
│           └── Content-derived insights                              │
└─────────────────────────────────────────────────────────────────────┘
```

#### Vector Embedding Generation
```
┌─────────────────────────────────────────────────────────────────────┐
│                    VECTOR EMBEDDING PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input:   Reconciled text + metadata                                │
│           └── Full markdown content                                 │
│           └── Structured metadata                                   │
│                                                                     │
│  Process: Text Chunking & Embedding                                 │
│           ├── Chunk size: 1024 tokens (configurable)                │
│           ├── Overlap: 200 tokens                                   │
│           ├── Model: Snowflake Arctic Embed 2                       │
│           ├── Dimensions: 1024                                      │
│           └── API: Ollama localhost:11434/v1/                       │
│                                                                     │
│  Output:  Vector embeddings + metadata                              │
│           ├── Text chunks with embeddings                           │
│           ├── Document-level metadata                               │
│           ├── Chunk-level metadata                                  │
│           └── Relationships preserved                               │
│                                                                     │
│  Storage: Qdrant Collection                                         │
│           ├── Collection: "index_1"                                 │
│           ├── Payload: Full metadata                                │
│           ├── Vector: 1024-dim embedding                            │
│           └── ID: Chunk identifier                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Quality & Validation

#### Processing Success Metrics
- **PDF Extraction**: Text length > 1000 chars, structure detected
- **Metadata Quality**: Required fields populated, taxonomy compliance
- **Embedding Quality**: Vector dimensions correct, similarity scores reasonable
- **Storage Integrity**: Qdrant insertion confirmed, document store updated

#### Failure Handling
- **PDF Corrupt**: Skip file, log error, continue processing
- **API Failures**: Retry with exponential backoff
- **LLM Timeouts**: Queue for later retry
- **Vector Store Errors**: Log failure, continue with next document

This comprehensive data transformation ensures high-quality, searchable content optimized for RAG retrieval tasks.

---

## Summary

This operations guide provides the framework for processing 250,000+ academic papers through the RAG system. The pipeline emphasizes:

- **Scalable Architecture**: 12x parallel processing for optimal throughput
- **Robust Error Handling**: Comprehensive retry logic and failure recovery
- **Quality Assurance**: Multi-stage validation and metadata reconciliation  
- **Operational Monitoring**: Real-time progress tracking and diagnostics

The system transforms raw academic papers into a searchable knowledge base through systematic document acquisition, content extraction, metadata enrichment, and vector indexing.
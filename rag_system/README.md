# RAG System

The RAG System is a collection of tools for RAG-based document QA.

## RAG System Structure

The RAG System is structured as follows:

```bash
rag_system
├── docker-compose.yml
├── Dockerfile
├── flowsettings.py
├── kotaemon
├── kotaemon_install_guide
├── pipeline_scripts  
├── README.md
└── taxonomy
```

## Pipeline Scripts Instructions

The pipeline scripts folder contains script to make extraction and analysis of documents.

To setup the pipeline scripts, run the following command:


```bash
cd rag_system/pipeline_scripts
uv sync
```

You can find a detailed guide here: [📄](../rag_system/pipeline_scripts/agentic_data_policies_extraction/policies_transformation_to_matrices/README.md)


## Kotaemon Subtree Setup

The Kotaemon folder is a shared Data4Good subtree, synchronized with the common project:

🔗 https://github.com/dataforgoodfr/kotaemon

For setup, synchronization, and contribution instructions, please see the detailed guide here:
[📄](../docs/development/setup-kotaemon.md)
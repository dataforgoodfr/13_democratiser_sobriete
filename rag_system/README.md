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
├── kotaemon_pipeline_scripts
├── new_pipeline_scripts
├── README.md
└── taxonomy
```

There are 2 pipeline ingestion projects here...
... that share the same taxonomy.

The first one (with Kotaemon) use these folders :

```bash
├── docker-compose.yml
├── Dockerfile
├── kotaemon
├── kotaemon_install_guide
├── kotaemon_pipeline_scripts
└── taxonomy  
```

The second one (currently without Kotaemon ?) use these folders:

```bash
├── new_pipeline_scripts
├── README.md
└── taxonomy
```

## NEW Pipeline Scripts Instructions

The pipeline scripts folder contains script to make extraction and analysis of documents.

To setup the pipeline scripts, run the following command:


```bash
cd rag_system/new_pipeline_scripts
uv sync
```

You can find a detailed guide here: [📄](../rag_system/pipeline_scripts/agentic_data_policies_extraction/policies_transformation_to_matrices/README.md)


### Running the RAG System

We recommend running as a Python module, or using the Docker Compose file:

```bash
cd rag_system/new_pipeline_scripts
uv run python -m  agentic_data_policies_extraction.main
```


## KOTAEMON Pipeline Scripts Instructions

This framework is build according to Kotaemon to allow a new custom built 'fast' ingestion script (multi-threading ingestion for hundred and hundred document with one batch), side-to-side with the standard 'drag-and-drop' Kotaemon ingestion from the UI.


### DEV set-up deployment

You have two config files to check:


#### - the official Kotaemon file 'flowsettings.py" :

This file is at the root of 'rag_system'. (It will overwrite the official 'flowsettings.py' during the docker build.)

where are declared (among other things but the main declared components...):

- ```KH_OLLAMA_URL``` : the uri used to connect to the Ollama service inference (LLM models inference service)
- ```KH_APP_DATA_DIR``` : The main app data root directory where Kotaemon store all the internal data
- ```KH_DOCSTORE``` : The Kotaemon Docstore used and the path for it. Local Lancedb by default, but you could choose a remote LanceDB database
- ```KH_VECTORSTORE``` : The Kotaemon VectorStore used and the url for it. Qdrant by default for the dev team.
- ```KH_DATABASE``` : The Kotaemon internal SQL database. Could be sqllite (defaut) or any sql backend.
- ```KH_FILESTORAGE_PATH``` : The Kotaemon path storage fo all the raw documents (pdf images for each page, etc.)
- ...

You should not touch all these config for now... (during your dev setup)


#### - an additionnal .env to set inside the 'kotaemon_pipeline_scripts' folder :

This file lives inside 'kotaemon_pipeline_scripts'.

First : you have to generate your own .env from the .env.example template :

```bash
cd kotaemon_pipeline_scripts
cp .env.example .env 
```

And now check all the .env values :

- ```PG_DATABASE_URL```  = The URL of the Data4Good database that maintains the OpenAlex articles metadata (ask to the team)
- ```LLM_INFERENCE_URL```  = The URL for the LLM inference stack (your Ollama service for local dev)
- ```LLM_INFERENCE_MODEL```  = The model used for the chunk inference on metadatas
- ```LLM_INFERENCE_API_KEY```  = The API Key for the LLM inference stack
- ```EMBEDDING_MODEL_URL```  = The URL for the LLM embedding model stack (Ollama for local dev)
- ```EMBEDDING_MODEL```  = The model used for the embedding
- ```EMBEDDING_MODEL_API_KEY```  = The API Key for the LLM embedding model
- ```COLLECTION_ID```  = The id of the collection within Kotaemon App (BE CAREFULL TO CHOOSE THE RIGHT ID)
- ```USER_ID```  = The User ID taken from the Kotaemon App (BE CAREFULL TO CHOOSE THE RIGHT ID)

For now, do not touch the 'USER_ID' before launching the Kotaemon app for the first time. (see further)


### Running the RAG System

1) The 'dev' deployment is used to launch, work and debug with the python package in editable mode.
Moreover, all the 'kotaemon_pipeline_scripts' folder is mapped (as a volume) inside the container, to allow working on it during this dev stage.

First, launch the different services with the docker compose provided in this folder.

Nothing to do — everything’s already set up: the Docker Compose file was created to save you the hassle.

You only need to pay attention, if necessary, to the volume mappings.

And if you don't have anny GPU on your local device and you don't have set-up cuda with docker, remove these line for the Ollama service ;

```yaml
deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

```bash
docker compose up
```

Additionally, the command that normally launches the Kotaemon app (./launch.sh) has been deliberately disabled so you can develop on the app — coding the different libraries (kotaemon, ktem, and our custom ones) — without having to stop/restart the Kotaemon container.

Indeed, to run the Kotaemon app for testing, you need to enter the container:

From the rag_system folder where the Docker Compose file is located:

```bash
docker compose exec -it kotaemon bash
```

And launch the Kotaemon app :
```bash
./launch.sh
```

IMPORTANT: After launching the Kotaemon app, open any page and check the logs to retrieve the USER ID.
Then, shut down the Kotaemon app from inside the container (or stop the container if you prefer).
Update your .env file with the correct USER ID.
Finally, restart the Kotaemon app — your fast ingestion pipeline scripts should now be consistent with the correct user collection.


2) You also need to pull the different models with the Ollama service.
Read and follow the point 2 of the README inside the 'kotaemon_install_guide' (FR) relative to this.

3) And now, for your first steps on the Kotaemon app, read and follow the point 3 of of the README inside the 'kotaemon_install_guide' (FR) relative to this.


### Running the 'Fast' ingestion pipeline scripts

The 'fast_ingestion_good_version.py' script calls the shortcut_indexing_pipeline.py that describes all the ingestion steps, build througth the Kotaemon API.
This script launchs an ingestion with the documents that are not ingested on the Data4Good metadata database.
To force a re-index, you could add a '-fr' argument do an incrementation of the version.

This pipeline uses also the 'pipelineblocks' modules (inside the folder kotaemon) which is a 'plugin' package build side-to-side with kotaemon.

To run the pipeline for a new ingestion, launch the script inside the container :
```bash
python3 pipeline_scripts/fast_ingestion_pipeline_good_version.py
```
You can update the ingestion version (for example : 2) from two ways:
- by changing the ```INGESTION_VERSION=version_2``` env variable within your .env
- by providing an '--ingestion_version=version_2' argument directly to the fast ingestion python script. 

The second choice will override the environment variable.

If you provide a new version for the first time, all the documents, without any exception, will be ingested again for the first time.


## Kotaemon Subtree Setup

The Kotaemon folder is a shared Data4Good subtree, synchronized with the common project:

🔗 https://github.com/dataforgoodfr/kotaemon

For setup, synchronization, and contribution instructions, please see the detailed guide here:
[📄](../docs/development/setup-kotaemon.md)
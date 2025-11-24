# ChatSufficiency

Ce README est composé de deux parties :
1. Quick start technique en anglais
2. Explications du travail effectué et roadmap en français.

Il est prévu de bouger le sous-dossier `policy_analysis` dans un dossier frère car il s'agit d'un axe de travail à part entière, en bonne partie indépendant du RAG.

# Quick start

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
├── policy_analysis
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
├── policy_analysis
├── README.md
└── taxonomy
```

[README policy analysis](policy_analysis/README.md)


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


# Etat du projet et roadmap

## Choix d'utiliser puis de sortir de Kotaemon
L'option retenue initialement pour le chatbot était [Kotaemon](https://github.com/Cinnamon/kotaemon), un projet open source implémentant une interface de chatbot RAG en local, censément facilement customisable.
Il apportait notamment les fonctionnalités suivantes déjà codées :
- interface graphique avec affichage des sources PDF ;
- nombreux algorithmes de RAG disponibles ;
- pipeline d'ingestion de documents.

Il "suffisait" donc de connecter Kotaemon à la library pour avoir une v1 du chatbot.

Mais Kotaemon imposait des contraintes fortes sur l'ingestion de documents : il est conçu pour ajouter des documents en local mais pas pour utiliser une base séparée déjà traitée avec un pipeline d'ingestion en propre, ce que nous voulions pour la library. Il fallait donc pour le faire marcher passer par le code de Kotaemon pour certaines étapes de la création de la library. Cela a imposé un couplage entre les deux sous-projets qui a compliqué le développement et la coordination, et obligé à multiplier les bases de données. Bref, cela a ajouté une complexité énorme qui a finalement nuit au projet.

Les autres facteurs plaidant pour une sortie de Kotaemon sont les suivants :
- le projet n'est plus maintenu ;
- son interface est celle d'un outil personnel ou interne, pas d'un site web grand public ;
- il contient une gestion des utilisateurs inutile pour le projet mais qui impose des contraintes qui n'ont pas été clarifiées ;
- les possibilités d'améliorations et de customisation futures sont restreintes.

## Travail effectué
En conséquence du choix d'utiliser Kotaemon, le travail s'est concentré sur l'intégration de Kotaemon au reste du projet via :
- une customisation du code internet de Kotaemon (`rag_system/kotaemon/libs/pipelineblocks`) ;
- de nombreux fichiers de config (`rag_system/kotaemon_pipeline_scripts/fast_ingestion/`) ;
- un pipeline d'ingestion de documents adapté au projet (`rag_system/kotaemon_pipeline_scripts/fast_ingestion/`), incluant notamment l'extraction de la taxonomie.

Malheureusement, une grosse partie de ce travail est propre à Kotaemon et n'est pas réutilisable si nous en sortons. Seul le dernier point peut être réutilisé pour l'enrichissement des métadonnées de la library.

En revanche, un travail d'évaluation a été fait sur la branche `retrieval_evaluation` que nous pourrons réemployer pour l'optimisation du retrieval et de la génération.

## Roadmap
En remplacement de Kotaemon, la solution proposée est de simplement recoder les fonctionnalités dont nous avons besoins. Des projets alternatifs comme OpenWebUI ont été considérés mais exposent aux mêmes écueils que Kotaemon.

- [ ] Retrieval = moteur de recherche sur la library (abstract puis full text, recherche par mot clé puis par similarité sémantique)
- [ ] Interface web pour ce moteur de recherche (API FastAPI, app SvelteKit)
- [ ] Ingestion de la library en base vectorielle : chunking et embedding (nécessaire à la recherche sémantique)
- [ ] Génération avec citation des sources
- [ ] Interface web pour le chatbot = extension de celle du moteur de recherche utilisant [Svelte AI Elements](https://svelte-ai-elements.vercel.app/)
- [ ] Complexification progressive (affichage des PDF, des graphiques...)
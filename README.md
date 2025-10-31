# Democratiser la sobriété


Un projet visant à démocratiser les approches de sobriété numérique et énergétique grâce à un système RAG (Retrieval-Augmented Generation) et des outils d'analyse automatisée.


## Structure du projet

Le projet est structuré en plusieurs sous-projets :


- `rag_system` : Système RAG (Retrieval-Augmented Generation) pour l'extraction et l'analyse de politiques de sobriété
- `src` : Scripts pour la librairie `WSL` (World Sufficiency Lab)


Le dossier principal contient les fichiers suivants :

```
.
├── Dockerfile
├── docs  
├── env_cluster.txt
├── failed_files.txt
├── LICENSE
├── notebooks
├── poetry.lock
├── pyproject.toml
├── rag_system
├── README.md
├── src
├── tests
├── tox.ini
└── uv.lock
```


## 🚀 Quick Start


### 1. Installer les dépendances `uv` et `pip`


```bash
# macOS et Linux
curl -LsSf https://astral.sh/uv/install.sh | sh


# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"


# Alternative : via pip
pip install uv
```


Plus d'informations : [documentation officielle de uv](https://astral.sh/uv)


### 2. Lancer les precommit-hooks localement

## Lancer les precommit-hook localement

[Installer les precommit](https://pre-commit.com/)

    pre-commit run --all-files

## Utiliser Tox pour tester votre code

    tox -vv


> [!IMPORTANT]
> Projet en développement actif, pas de garantie de fonctionnement, notamment pour les tests.

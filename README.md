# Democratiser la sobriété (refonte du README en cours)

Un projet visant à démocratiser les approches de sobriété numérique et énergétique grâce à un système RAG (Retrieval-Augmented Generation) et des outils d'analyse automatisée.


## Présentation du projet

Le projet est structuré en plusieurs sous-projets :

- **Visualisation** (branche `visualizations-combined` non encore mergée) :
    - [Carbon budget](https://app-d066b92e-20ba-4dbf-af25-73c7e5657091.cleverapps.io/) : budget carbone restant par pays pour rester sour les 2°C de réchauffement
    - Décomposition [monde](https://app-e1c3f118-5441-449a-99f3-fa4036bb2ad4.cleverapps.io/) et [UE](https://app-ac31ad44-d32f-4998-87c6-b9b699c29c63.cleverapps.io/) de scénarios de décarbonation en Population - Décarbonation - Efficacité énergétique - Sobriété
    - [Indicateurs de bien-être](https://app-aa62786e-21f6-42ab-b0ff-ddca6575e4f8.cleverapps.io/) (EWBI)
- **Library** : base d'articles sur la sobriété dans `src`
- **Policy analysis** (dans `rag_system/policy_analysis`) : pipeline d'extraction et d'analyse de politiques de sobriété et de leurs impacts dont les résultats ont vocation à être inclus dans le RAG
- **ChatSufficiency** (dans `rag_system`) : chatbot destiné aux experts des politiques publiques, branchés en RAG sur la library et les résultats de la policy analysis

De plus, une taxonomie a été développée visant à enrichir les articles de la library de métadonnées "métier" (en plus des métadonnées issues d'OpenAlex) via un traitement par LLM. Celle-ci est présente en double dans le code, dans `rag_system/taxonomy` et `src/wsl_library_domain`. Un refactoring du code est prévu pour éliminer ce doublon, mieux séparer les sous-projet (notamment policy analysis et RAG) et remplacer la librairie de RAG Kotaemon par du code custom.


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

[Installer les precommit](https://pre-commit.com/)

    pre-commit run --all-files

### 3. Utiliser Tox pour tester votre code

    tox -vv


## Roadmap

- [ ] Réduire les requirements dans `rag_system`
- [ ] Fusionner `rag_system` et `src` dans un seul dossier
- [ ] Ajouter des tests unitaires
- [ ] Ajouter des tests d'intégration
- [ ] Améliorer la documentation
- [ ] Améliorer l'extraction de politiques de sobriété


> [!IMPORTANT]
> Projet en développement actif, pas de garantie de fonctionnement, notamment pour les tests.

# Democratiser la sobriété (refonte du README en cours)

Un projet visant à démocratiser les approches de sobriété grâce à un système RAG (Retrieval-Augmented Generation) et des outils d'analyse automatisée.


## Présentation du projet

Le projet est structuré en plusieurs sous-projets :

- **Visualisation** (branche `visualizations-combined` non encore mergée) :
    - [Carbon budget](https://app-d066b92e-20ba-4dbf-af25-73c7e5657091.cleverapps.io/) : budget carbone restant par pays pour rester sour les 2°C de réchauffement
    - Décomposition [monde](https://app-e1c3f118-5441-449a-99f3-fa4036bb2ad4.cleverapps.io/) et [UE](https://app-ac31ad44-d32f-4998-87c6-b9b699c29c63.cleverapps.io/) de scénarios de décarbonation en Population - Décarbonation - Efficacité énergétique - Sobriété
    - [Indicateurs de bien-être](https://app-aa62786e-21f6-42ab-b0ff-ddca6575e4f8.cleverapps.io/) (EWBI)
- **Library** : base d'articles sur la sobriété dans `src`
- **Policy analysis** (dans `rag_system/policy_analysis`) : pipeline d'extraction et d'analyse de politiques de sobriété et de leurs impacts dont les résultats ont vocation à être inclus dans le RAG
- **ChatSufficiency** (dans `rag_system`) : chatbot destiné aux experts des politiques publiques, branchés en RAG sur la library et les résultats de la policy analysis

De plus, une **taxonomie** a été développée visant à enrichir les articles de la library de métadonnées "métier" (en plus des métadonnées issues d'OpenAlex) via un traitement par LLM. Celle-ci est présente en double dans le code, dans `rag_system/taxonomy` et `src/wsl_library_domain`.

Un refactoring est prévu pour éliminer ce doublon, mieux séparer les sous-projet et remplacer la librairie de RAG Kotaemon par du code custom.


## Library
### 1. Pré-screening
La source de départ est OpenAlex.
L'intégration de sources alternatives est laissée à de futurs travaux.
Le pré-screening à partir d'OpenAlex es fait en deux étapes :
1. Ensemble de requêtes par mots-clés (choisis par des experts) à l'API OpenAlex -> 1.6M d'articles
2. Filtrage des articles par une classification `about sufficiency / not about sufficiency` fondée sur leur abstract.

L'étape 2 a été effectuée et [documentée](https://theolvs.notion.site/Documentation-et-m-thodo-Pr-screening-1f8819109fa4807b842ecd568785004c) par Théo Alves avec un modèle BERT entraîné avec SetFit sur un dataset annoté à la main, ce qui a conduit à garder 250k articles.
Ce code n'est pas (encore) sur le repo GitHub mais il est disponible [sur Collab](https://colab.research.google.com/drive/1onirKPHdBxHTqcQKGOTgupNVpbgCJQgz?usp=sharing), le modèle entraîné [sur HuggingFace](https://huggingface.co/TheoLvs/wsl-prescreening-multi-v0.0/tree/main) et les jeu de données [sur Drive](https://drive.google.com/drive/folders/1EQkQQaUN11jvZAeP8Uf5YFC9yjLCs2Kx).
Théo rapporte une accuracy sur le dataset de test (20% du dataset annoté) de 100%, mais le recall réel (métrique la plus importante) est inconnu.
Les articles sélectionnés sont stockés dans la table `policies_abstracts_all` de la base postgres (ID OpenAlex, DOI et abstract, étrangement sans leur titre).

Des tentatives d'amélioration de l'étape 2 ont été effectuées sans être utilisées :
- dans la branche `feature/pre-screening` avec des API d'IA générative (Mistral, OpenAI) ;
- dans la branche `prescreening-experimentation` avec un entraînement de modèles BERT (dont SciBERT) par pytorch-lightning.

La fonction `search_openalex` de `src/wsl_library/scraping/extract_openalex.py` permet quant à elle de reproduire l'étape 1.
L'ensemble de mots-clés à utiliser ne semble toutefois pas documenté.

### 2. Extraction full-text
Cette étape regroupe à nouveau deux sous-étapes :
1. Obtention quand disponible (open access) d'un lien pour le texte complet, généralement en PDF.
2. Téléchargement et lecture du PDF pour obtenir le texte converti en format markdown.

Les PDF téléchargés doivent être stockés pour affichage aux utilisateurs finaux quand ils sont cités.

Le code pour l'étape 1 (à perfectionner car il ne gère pas les cas où il faut cliquer sur une popup avant d'accéder au PDF) est dans `src/wsl_library/scraping/extract_openalex.py` et celui de l'étape 2 dans `src/wsl_library/pdfextraction/pdf/`.

### 3. Extraction des métadonnées
En plus des métadonnées d'OpenAlex, des métadonnées "métier" correspondant à la taxonomie sont ajoutées par un traitement LLM (API DeepSeek) aux articles.
Les petits modèles n'ayant pas des performances satisfaisantes sur cette tâche, il est recommandé d'utiliser des modèles d'au moins 50-60 Md de paramètres (totaux pour les MoE).
Les contraintes de Kotaemon ont imposé d'effectuer cette étape en même temps que l'ingestion dans la base lancedb de Kotaemon (dans `rag_system/kotaemon_pipeline_scripts/fast_ingestion/`) mais elle est en principe distincte.
Du code est d'ailleurs disponible pour ce faire dans `src/wsl_library/pdfextraction/llm/`.

Le traitement des chunks pour cette étape reste à clarifier (métadonnées en propre ou copie de celles du document original).

### Policy analysis
A COMPLETER (EDOUARD)


### ChatSufficiency
A COMPLETER (François)


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

# Extraction des policies et de la taxonomie avec DSPy

L’objectif de ce dossier est de mettre en place une pipeline complète permettant :

* le nettoyage des chunks de texte en amont
* l’entraînement de modèles d’extraction de policies et de taxonomie
* l’extraction automatique des policies et de la taxonomie à partir de documents
* la génération d’une sortie structurée au format JSON, réutilisable dans un pipeline global

L’ensemble de la pipeline repose sur DSPy, utilisé pour optimiser les prompts et entraîner des modèles robustes pour les tâches d’extraction.

# Initial Chunk Cleaning

Cette partie contient le code chargé de nettoyer les chunks de texte avant toute extraction.

Objectifs :

* supprimer le bruit inutile
* réduire la taille des chunks et le nombre de tokens
* améliorer la qualité de l’extraction des policies et de la taxonomie

Le script principal est :

* `clean_chunks.py`

Les résultats nettoyés sont sauvegardés sous forme de fichiers parquet et jsonl pour inspection et réutilisation.

Cette étape pourrait être déplacée plus en amont dans un pipeline global, mais elle est actuellement intégrée ici.

# Extraction des policies

Cette partie est dédiée à l’entraînement du modèle d’extraction de policies.

L’objectif est d’extraire les policies présentes dans les chunks de texte, en particulier dans les conclusions des papiers de recherche.

DSPy est utilisé afin d’optimiser le prompt servant à l’extraction des policies.
L’optimisation vise à maximiser la similarité entre une policy prédite et une policy de référence, en utilisant un cross-encoder comme métrique.

Il existe 32 données labellisées qui peuvent être utilisées comme dataset de validation.

Des données synthétiques ont été générées afin d’augmenter le volume de données d’entraînement et d’améliorer la robustesse du modèle.

Une fois l’optimisation terminée, le modèle DSPy généré est sauvegardé et peut être rechargé pour les phases d’inférence.


# Données d’entraînement

Le dossier `model_training_data` contient :

* des données gold pour les policies et la taxonomie,
* des données synthétiques générées pour l’entraînement des modèles.

Ces données sont utilisées à la fois pour :

* l’optimisation des prompts DSPy,
* la validation des performances,
* l’entraînement des modèles d’extraction.


# Génération de données synthétiques (taxonomie)

La génération de données synthétiques est principalement utilisée pour la taxonomie.

Le script `taxonomy_data_generator.py` permet de :

* créer des exemples structurés de taxonomie,
* améliorer la couverture des catégories,
* standardiser les formats attendus par le modèle d’extraction.

# Pipeline policy et taxonomie

Le script `pipeline_policy_and_taxonomy_extraction.py` est le point d’entrée principal.

Il permet de :

* charger les modèles DSPy de policy et de taxonomie,
* exécuter l’extraction sur l’ensemble des chunks disponibles,
* consolider les résultats.

La sortie finale est un fichier JSONL contenant l’ensemble des policies et de la taxonomie extraites.


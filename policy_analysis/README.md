# Policy Analysis

Ce sous-projet vise à extraire les politiques à partir de documents, à les analyser, à les regrouper en clusters, puis à extraire les impacts de ces clusters selon les différentes dimensions de la taxonomie avant d'en déduire une classification sobriété ou non.

La policy analysis se divise en trois grandes étapes :
1. Extraction des politiques dans `policy_extraction`
2. Clustering dans `clustering`
3. Extraction des impacts et classification en `impacts_analysis`.

On peut ajouter l'étape d'extraction de la taxonomie, qui devrait techniquement faire partie de la partie `library` mais qui a été traitée conjointement avec l'extraction des politiques par un bénévole (Edouard) dans le dossier `dspy_policies_and_taxonomy_extraction`.

Tuos les résultats intermédiaires et finaux sont sur [notre dataset HF](https://huggingface.co/datasets/sufficiencylab/sufficiency-library/tree/main).

Nous avons bien sûr tenté de tout extraire d'un coup (`all_at_once_extraction attempt.ipynb`) mais c'est trop demander au LLM. 

## Extraction de la taxonomie
Dans `dspy_policies_and_taxonomy_extraction`. L'extraction finale de politiques n'utilise pas ce code (voir ci-dessous), donc ce dossier ne sert qu'à extraire la taxonomie.

### Extraction des taxons géographiques
Fait par LLM avec un modèle DSPy "entraîné" (DSPy ne fait qu'optimiser le prompt) via `geography_only_dspy_model_creation.py` sur un dataset annoté dans `model_training_data`.

Afin de pouvoir utiliser Jean-Zay et VLLM, DSPY n'étant pas compatible avec VLLM, on extrait ensuite le prompt optimisé du modèle DSPy, qu'on utilise dans `vllm_geo_extraction.py`.

Ce script est invoqué sur Jean-Zay via slurm via `sbatch jz_geo_extraction.sh`. Ces scripts (python et shell) sont faits pour lancer 10 jobs de 1 GPU en parallèle, sur 10 ensemble de données différents. Cela va 10x plus vite qu'un seul GPU et il est beaucoup plus facile d'obtenir 10 allocations de 1 GPU qu'une seule de 10 GPU sur Jean-Zay. La même astuce est utilisée plusieurs fois i-dessous.

### Extraction des impacts et satisfiers
Ici nous n'avons pas réussi à obtenir des résultats satisfaisants par LLM alors nous avons opté pour une approche de classification multilabel avec des logreg 1-vs-rest sur les embeddings déjà calculés de chaque chunk. Les classifieurs sont entraînés sur des données annotées à la main et par LLM (moins bonne qualité) dans `model_training_data`. Les performances sont acceptables quoique largement améliorables, notamment en annotant plus de données à la main. Cette approche a l'avantage d'être très frugale et très rapide.

## Extraction des politiques
Dans `policy_extraction`, vous trouverez un notebook, le prompt final ainsi qu'un script python et un script shell à utiliser sur Jean-Zay via la commande `sbatch`comme décrit ci-dessus. Avec 10 jobs en parallèle, cela prend environ 7 heures (x10) sur des A100 sur la v1 de la library contenant 557k conclusions chunkées.

Le travail original avait été fait avec DSPy dans `geography_only_dspy_model_creation.py` mais les résultats n'étaient pas satisfaisants. L'amélioration du prompt et du schéma de structured outputs ont permis d'améliorer les résultats en se passant de DSPy.

## Clustering des politiques
Dans `clustering`. Après avoir extrait les politiques et calculé leur embeddings (par exemple avec `compute_embeddings.py`), faire tourner (en comprenant ce qui se passe car tout n'est pas utile) `leiden_clustering.ipynb` puis `postprocess_clusters.ipynb`. A l'échelle du dataset entier, il faut une machine avec 64-128 GB de RAM pour le faire sans problème.

Après avoir expérimenté sans succès avec K-Means et HDBSCAN (clusters de mauvaise qualité à l'inspection manuelle), l'algorithme de clustering retenu est l'algorithme de Leiden, un algo de détection de communautés dans des graphes. Il faut donc construire un graphe suffisamment connecté à partir des embeddings, où la matrice de connectivité du graphe est en gros la matrice des similarités cosinus filtrés pour ne garder que les valeurs au-dessus d'un certain seuil (0,5 est une bonne valeur sans quoi le graphe n'est pas suffisamment connecté et on a plus de composantes connexes qu'on ne voudrait de clusters). Il tourne vite sur CPU, le principal facteur limitant sur cette étape est la RAM (j'ai pris une machine Scaleway).

On désigne ensuite un représentant à chaque cluster en calculant son médoïde, le point le plus proche de la moyenne renormalisée du cluster.

## Calcul des impacts et classification
Dans `impacts_analysis`.

### Extraction des directions d'impact
Il s'agit, pour chaque chunk duquel au moins une politique a été extraite, de :
- prendre le cluster (représentant, cf ci-dessus) associé à chaque politique du chunk
- déterminer la direction d'impact (positif, négatif, neutre, inconnue) de ce cluster pour chaque taxon impact et satisfier détecté sur ce chunk dans la phase d'extraction de la taxonomie.

Cela fait beaucoup de prédictions donc le script `vllm_impact_direction_extractioN.py` est optimisé pour exploiter le caching des inputs (automatic prefix caching) au maximum et générer aussi peu de tokens de sortie que possible. Cela permet d'avoir une prédiction en 10 x 3h sur des A100, utilisant le parallélisme décrit plus haut. On retraite ensuite les résultats dans `postprocess_impacts.ipynb`.

### Classification S/PS/NS
L'étape précédente permet de faire remonter pour chaque cluster des chunks dans lequel le cluster apparaît comme ayant un impact positif/négatif/neutre sur une dimension d'impact donnée, ce dans toute la library. On est donc en capacité de sortir facilement par exemple des impacts négatifs même si les impacts positifs sont ultra-dominants dans la littérature.

Pour chaque cluster, on  construit donc un résumé de ses impacts avec une partie quantitative correspondant, pour chaque dimension d'impact, au nombre de référence trouvées rapportant chaque direction d'impact, ainsi qu'une partie qualitative contenant des chunks exemples (2 ou moins en pratique, sélectionnés au hasard s'il y en a plus) pour chaque dimension et direction d'impact. Ce résumé est donné à un LLM, auquel on demande deux étapes de classification dans `classifiy_sufficiency.ipynb`:
1. DECARBONATION / EFFICIENCY / SUFFICIENCY-COMPATIBLE / NOT-COMPATIBLE
2. Si SUFFICIENCY-COMPATIBLE, SUFFICIENCY / POTENTIAL SUFFICIENCY / NOT SUFFICIENCY.

Les clusters classifiés SUFFICIENCY-COMPATIBLE et SUFFICIENCY sont ensuite ingérés en base Qdrant dans `cluster_qdrant_ingestion.ipynb` et mis à disposition du chat.

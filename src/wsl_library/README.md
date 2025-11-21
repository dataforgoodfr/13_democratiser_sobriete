# Library

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

La fonction `search_openalex` de `scraping/extract_openalex.py` permet quant à elle de reproduire l'étape 1.
L'ensemble de mots-clés à utiliser ne semble toutefois pas documenté.


### 2. Extraction full-text
Cette étape regroupe à nouveau deux sous-étapes :
1. Obtention quand disponible (open access) d'un lien pour le texte complet, généralement en PDF.
2. Téléchargement et lecture du PDF pour obtenir le texte converti en format markdown.

Les PDF téléchargés doivent être stockés pour affichage aux utilisateurs finaux quand ils sont cités.

Le code pour l'étape 1 (à perfectionner car il ne gère pas les cas où il faut cliquer sur une popup avant d'accéder au PDF, la branche `scraping` contient de légères améliorations) est dans `scraping/extract_openalex.py` et celui de l'étape 2 dans `pdfextraction/pdf/`. 


### 3. Extraction de la taxonomie
En plus des métadonnées d'OpenAlex, des métadonnées "métier" correspondant à la taxonomie sont ajoutées par un traitement LLM (API DeepSeek) aux articles.
Les petits modèles n'ayant pas des performances satisfaisantes sur cette tâche, il est recommandé d'utiliser des modèles d'au moins 50-60 Md de paramètres (totaux pour les MoE).
Les contraintes de Kotaemon ont imposé d'effectuer cette étape en même temps que l'ingestion dans la base lancedb de Kotaemon (dans `rag_system/kotaemon_pipeline_scripts/fast_ingestion/`) mais elle est en principe distincte.
Du code est d'ailleurs disponible pour ce faire dans `pdfextraction/llm/`.

Le traitement des chunks pour cette étape reste à clarifier (métadonnées en propre ou copie de celles du document original).


### Roadmap
- [ ] Nettoyer la base de données Postgres et repartir d'une table propre de 250k articles avec a minima OpenAlex ID, DOI, titre et abstract
- [ ] Récupérer le texte complet d'autant de ces articles que possible, le stocker en format texte dans Postgres et stocker les PDF dans un object storage sur CleverCloud
- [ ] Traiter les textes complets par NLP pour extraire la taxonomie, la stocker en métadonnées sur Postgres
- [ ] Mettre en place un pipeline pour mettre à jour automatiquement la library de façon régulière
- [ ] Intégrer d'autres sources qu'OpenAlex 
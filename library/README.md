# Library

### 1. Pré-screening
[Voir la doc dédiée](prescreening/README.md).


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
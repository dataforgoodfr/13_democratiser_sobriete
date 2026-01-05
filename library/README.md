# Library

### 1. Pré-screening
[Voir la doc dédiée](prescreening/README.md).


### 2. Obtention des PDF et extraction des textes complets
[Voir la doc dédiée](scraping/README.md).


### 3. Extraction de la taxonomie
En plus des métadonnées d'OpenAlex, des métadonnées "métier" correspondant à la taxonomie sont ajoutées par un traitement LLM (API DeepSeek) aux articles.
Les petits modèles n'ayant pas des performances satisfaisantes sur cette tâche, il est recommandé d'utiliser des modèles d'au moins 50-60 Md de paramètres (totaux pour les MoE).
Les contraintes de Kotaemon ont imposé d'effectuer cette étape en même temps que l'ingestion dans la base lancedb de Kotaemon (dans `rag_system/kotaemon_pipeline_scripts/fast_ingestion/`) mais elle est en principe distincte.
Du code est d'ailleurs disponible pour ce faire dans `pdfextraction/llm/`.

Le traitement des chunks pour cette étape reste à clarifier (métadonnées en propre ou copie de celles du document original).


### Roadmap
- [x] Mettre au propre le jeu de mots-clés
- [x] Etape 1 du pré-screening : obtenir les références des articles candidats en par des recherches par mot-clé sur l'API OpenAlex
- [x] Etape 2 du pré-screning : filtrer les résultats de l'étape 1 en faisant classifier l'abstract à un modèle BERT fine-tuné
- [x] Récupérer quand c'est possible les PDF des articles et en extraire les textes complets -> textes bruts et non markdown, md serait mieux
- [x] Extraire les sections Résultats et Conclusion
- [ ] Extraire la taxonomie
- [ ] Mettre en place un pipeline pour mettre à jour automatiquement la library de façon régulière
- [ ] Intégrer d'autres sources qu'OpenAlex
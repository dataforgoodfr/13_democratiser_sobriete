# Pré-screening

La prescreening consiste à identifier tous les travaux relevant de la sobriété. Dans un premier temps, nous prenons pour seule source OpenAlex (270M de références). L'intégration d'autres sources est prévue à l'avenir.

Le pré-screening se fait en deux étapes :
1. Ensemble de requêtes par [mots-clés](stage1/sufficiency_keywords.csv) à l'API OpenAlex -> 24M de références.
2. Filtrage des articles par une classification `about sufficiency / not about sufficiency` fondée sur leur abstract -> 2,5M de références.

## Etape 1
Les mots-clés sont définis par des experts dans ce [ce fichier](stage1/sufficiency_keywords.csv) (initialement sur Google Drive). Il y en a un peu plus de 2000. Afin de limiter le nombre de requêtes à l'API OpenAlex, ils sont regroupés en 149 thèmes avec un opérateur OR pour former [ce second fichier](stage1/sufficiency_keywords_regrouped_count.csv).

Même avec ce système, qui permet de transférer à OpenAlex une grande partie de la déduplication, les thèmes restent larges et se recoupent beaucoup, ce qui oblige à récupérer quasiment 100M de références par pages de 200, pour les dédupliquer localement et obtenir les 24M de références uniques. On pourrait encore regrouper certains thèmes pour raccourcir ce travail, qui prend plusieurs jours avec le rate limiting de l'API OpenAlex qui empêche de lancer plus de 2 processus en parallèle.

Comme la quantité de donnée récupérée influe sur la vitesse des requêtes, cette étape a été découpée en 2 sous-étapes :
1. Récupération des ID seulement et déduplication.
2. Récupération des métadonnées d'intérêt pour les ID collectés.

Il est toutefois possible que fusionner ces deux étapes eut été plus court.

Les scripts pour cette étape sont dans le dossier `stage1` et utilsent la librairie [pyalex](https://github.com/J535D165/pyalex) via le connecteur [OpenAlexConnector](../src/library/connectors/openalex/openalex_connector.py) défini dans le package local `library`. Ils ne sont pas complètement résilients aux erreurs, ils peuvent s'arrêter, mais peuvent être relancés sans (trop) perdre de travail.

L'ancien code de `scraping/extract_openalex.py` n'a pas été utilisé car il ne reflétait pas bien la succession d'étapes (on ne va pas directement chercher le full text après avoir requêté OpenAlex) ou n'apportait pas de valeur ajoutée par rapport à pyalex.


## Etape 2
L'étape 2 avait initialement été effectuée et [documentée](https://theolvs.notion.site/Documentation-et-m-thodo-Pr-screening-1f8819109fa4807b842ecd568785004c) par Théo Alves avec SetFit sur un petit modèle d'embedding (33M) sur un dataset annoté à la main. Théo rapporte une accuracy sur le dataset de test (20% du dataset annoté) de 100%, mais le recall réel (métrique la plus importante) est inconnu. Cela avait mené à partir d'un premier dataset de 1,6M d'articles à en retenir 250k.

Le modèle retourne en réalité 5 probabilités : 4 correspondent à des classes ayant trait à la sobriété et 1 à "autre". Plusieurs méthodes de sélection sont définies, la plus sélective consistant à ne garder que les articles dont la classe la plus haute est dans les 4 classes sobriété et dont la probabilité correspondante dépasse 80%. C'est celle qui a été retenue et elle conduit à garder 15% des lignes.

Comme toutes les lignes n'ont pas d'abstract et que nous ne traitons pour l'instant que les articles en anglais (car le modèle de Théo ne fonctionne que sur l'anglais), c'est finalement environ 10% des lignes qui sont gardées, pour une library finale de 2,5M de travaux.

Des tentatives d'amélioration de l'étape 2 ont été effectuées par le passé sans être utilisées :
- dans la branche `feature/pre-screening` avec des API d'IA générative (Mistral, OpenAI) - à noter que le coût de cette méthode serait bien supérieur ;
- dans la branche `prescreening-experimentation` avec un entraînement de modèles BERT (dont SciBERT) par pytorch-lightning.

## Résultats
Les résultats de chaque étape sont stockés dans l'object storage Scaleway, dans le bucket `sufficiency-library` :
- [library_v1_2025-12-08.parquet](https://sufficiency-library.s3.fr-par.scw.cloud/library_v1_2025-12-08.parquet) : résultat final de 2,5M de lignes (2,5 Go)
- [stage-2/all_screened_with_preds.parquet](https://sufficiency-library.s3.fr-par.scw.cloud/stage-2/all_screened_with_preds.parquet) : fichier similaire mais comportant les résultats des prédictions au lieu des liens open access (2,4 Go)
- [stage-2/all_screened_ids.txt](https://sufficiency-library.s3.fr-par.scw.cloud/stage-2/all_screened_ids.txt) : les 2,5M d'ID uniques de la library (29 Mo)
- [openalex_ids_0812_final_24M.txt](https://sufficiency-library.s3.fr-par.scw.cloud/openalex_ids_0812_final_24M.txt) : les 24M d'ID résultant de l'étape 1 (289 Mo)
- `stage-1/chunk_i.parquet` avec i de 0 à 24 : le dataset complet résultant de l'étape 1 découpé en 25 morceaux (~900 Mo chacun)
- `stage-2/preds/chunk_i.parquet` avec i de 0 à 24 : les prédictions correspondantes du modèle issues de l'étape 2 (~700 Mo chacun).
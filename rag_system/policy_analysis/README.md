# Policy Analysis

Ce sous-projet vise à extraire les politiques à partir de documents, à les analyser, à les regrouper en clusters, puis à générer une matrice de consensus liant chaque politique aux facteurs concernés, en tenant compte de la taxonomie.
Les données produites seront ensuite mises à disposition dans le contexte du générateur.

## Extraction des données

L’extraction fonctionne actuellement sur les abstracts, mais le code et les prompts nécessaires pour traiter le full text existent déjà.

Deux approches sont disponibles :
- Approche “monolithique” : un script unique qui tente de récupérer toutes les informations en une seule passe.
- Approche “pipeline par étapes” : Une série de scripts qui extraient successivement chaque élément souhaité, puis structurent l’ensemble en JSON.


### Points ouverts

- L’analyse agentique crée une table agentic_policy_extractions mais qui n'existe pas sur le postgres — ce code a-t-il été testé ?

- Certains tests ne disposent pas de tous les fichiers nécessaires à l’exécution (ex. “List files”).

## Pipeline d’analyse

Le pipeline prend toutes les politiques extraites, les facteurs associés et leurs corrélations, et génère un fichier contenant, pour chaque cluster de politique identifié un indice de correlation moyen d'impact sur le secteur.  

Les secteurs liés au politiques sont identifiés par mesure de similarité vectorielle entre : 
facteur ↔ secteur
policy ↔ secteur

On lie ensuite les politiques avec les secteurs impactés, avec la corrélation des facteurs ±1
Clustering des politiques avec knn et on choisit comme nom de pollitique la plus proche centrale de chaque cluster 
- Une autre version du code appelé optimisé fait un kmeans et essaye d'améliorer le clustering
On sauvegarde le résultat en tant que complete_flattened_policies.csv - et - on sauvegarde la pivot table avec toutes les pollitiques 
Genère un fichier policy_clusters.csv avec les clusters trouvés et leur politques associées 
Fait ensuite un rapport d'analyse de ce qui a été fait


## À faire
Dans un premier temps : 
- [ ] Clarifier la structure du repository et gestion des dépendences (ex. absence de project.toml ?) 
- [ ] Identifier où la table agentic_policy_extractions est écrite si ca à déjà été run ? 
- [ ] Exécuter les tests en se connectant à l’API DeepSeek
- [ ] Ajouter les fichiers de test manquants
- [ ] Analyser l’extraction des politiques à partir des abstracts et identifier les problèmes
- [ ] Tester l’extraction des politiques sur le full text

Puis : 
- [ ] Comprendre les résultats de clustering entre Kmeans, Knn et HDBScan pour voir ceux qui sont utilisés et quels sont les problèmes rencontrés 
- [ ] Exporter les résultats une fois corrects dans une bdd 
- [ ] Déveloper le lien entre la question utilisateur et l'insertion de données de pollitiques dans le contexte.
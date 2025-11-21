## Policy Analysis

Ce sous-projet vise à extraire les politiques à partir de documents, à les analyser, à les regrouper en clusters, puis à générer une matrice de consensus liant chaque politique aux facteurs concernés, en tenant compte de la taxonomie.
Les données produites seront ensuite mises à disposition dans le contexte du générateur.

# Extraction des données

L’extraction fonctionne actuellement sur les abstracts, mais le code et les prompts nécessaires pour traiter le full text existent déjà.

Deux approches sont disponibles :

Approche “monolithique”
Un script unique qui tente de récupérer toutes les informations en une seule passe.

Approche “pipeline par étapes”
Une série de scripts qui extraient successivement chaque élément souhaité, puis structurent l’ensemble en JSON.

- Points ouverts

L’analyse agentique crée une table agentic_policy_extractions mais qui n'existe pas sur le postgres — ce code a-t-il été testé ?

Certains tests ne disposent pas de tous les fichiers nécessaires à l’exécution (ex. “List files”).

# Pipeline d’analyse

Le pipeline prend toutes les politiques extraites, les facteurs associés et leurs corrélations, et génère un fichier contenant, pour chaque cluster de politique identifié un indice de correlation moyen d'impact sur le secteur.  

## Identification des secteurs
Les secteurs liés au politiques sont identifiés par mesure de similarité vectorielle entre : 
facteur ↔ secteur et policy ↔ secteur

On ajoute une valeur de corrélation pour chaque secteur pour chaque politique basé sur la valeur "increasing" ou "decreasing" de chaque factor et de ses secteurs associés

## Clustering
Le clustering des politiques utilisé dans la pipeline est avec knn. Une fois les clusters identifiés, on choisit comme nom de pollitique la plus centrale de chaque cluster identifié
 
Il y a deux autre manière de réaliser le clustering qui ont été dévelopé mais ne sont pas appelés dans la pipeline du code : Un clustering avec Kmeans et un clustering avec HDBscan.

Les pollitiques clusterisés avec chaque secteur impacté sont sauvegardés en tant que complete_flattened_policies.csv 

A des fins d'analyse le code génere aussi deux fichiers : 
- Les pollitiques non clusterisés avec chaque secteur impacté sont sauvegardés en tant que complete_policy_sector_matrix.csv
- Les clusters identifiés avec leur pollitiques associées

Le code présente ensuite les résultats de la pipeline

# À faire
Dans un premier temps : 
- Clarifier la structure du repository et gestion des dépendences (ex. absence de project.toml ?) 
- Identifier où la table agentic_policy_extractions est écrite si ca à déjà été run ? 
- Exécuter les tests en se connectant à l’API DeepSeek
- Ajouter les fichiers de test manquants
- Analyser l’extraction des politiques à partir des abstracts et identifier les problèmes
- Tester l’extraction des politiques sur le full text

Puis : 
- Comprendre les résultats de clustering entre Kmeans, Knn et HDBScan pour voir ceux qui sont utilisés et quels sont les problèmes rencontrés 
- Exporter les résultats une fois corrects dans une bdd 
- Déveloper le lien entre la question utilisateur et l'insertion de données de pollitiques dans le contexte.
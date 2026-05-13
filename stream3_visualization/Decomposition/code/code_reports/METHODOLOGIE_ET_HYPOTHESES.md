# Scénarios FR - Compilation & Visualisation des Données

## Vue d'ensemble

Ce projet compile et visualise les données des scénarios de réduction des émissions français à partir d'entrées brutes éparses vers un ensemble de données interpolées complètes avec des visualisations complètes. Il traite deux scénarios:
- **SNBC-3** (Stratégie Nationale Bas-Carbone 3): 2020-2030
- **AME-2024** (Anticipation d'Émissions 2024): 2019-2050

## Méthodologie

### 1. Compilation des données

Le script compile les métriques d'activité, la consommation d'énergie et les données d'émissions du fichier d'entrée brut dans un format standardisé correspondant à:
`stream3_visualization/Decomposition/data/2025-04-28_EC scenarios data_Decomposition_compiled.xlsx`

**Entrée**: `2025-12-15_FR scenarios data_before computation.xlsx` (621 lignes éparses)
**Sortie**: `2025-01-06_FR_scenarios_compiled.xlsx` (513 lignes pleinement interpolées)

### 2. Stratégie d'interpolation linéaire

**Hypothèse**: Entre les points de données disponibles, les valeurs changent linéairement au fil du temps.

Toutes les métriques sont interpolées pour remplir chaque année dans leur plage de données:
- Pour chaque métrique (combinaison Secteur/Catégorie/Type)
- Trouver les années minimum et maximum avec des données
- Remplir toutes les années entre min et max avec une interpolation linéaire
- Pas d'extrapolation avant/après la plage de données disponible

**Résultat**: Courbes lisses et continues dans les visualisations avec toutes les années représentées

### 3. Secteurs & Métriques

#### Bâtiment - Résidentiel
- **Activité**: Surface de plancher totale (Mm²)
  - Calculée comme: nombre de maisons × (54,8% × surface individuelle + 45,2% × surface collective)
  - Sources: Nombre de maisons (millions) + Surfaces moyennes (individuelle & collective)
- **Énergie finale**: Consommation totale d'énergie (Mtoe)
  - Extraction directe des données brutes
  - Interpolée linéairement sur toutes les années
- **Émissions GES**: Équivalent CO₂ total (MtCO2e)
  - Extraction directe des données brutes

#### Bâtiment - Tertiaire
- **Activité**: Surface de plancher totale (Mm²)
  - Extraction directe de la variable "Surface plancher tertiaire" brute
  - Interpolée linéairement sur toutes les années
- **Énergie finale**: Consommation totale d'énergie (Mtoe)
  - Extraction directe des données brutes
  - Interpolée linéairement sur toutes les années
- **Émissions GES**: Équivalent CO₂ total (MtCO2e)
  - Extraction directe des données brutes

#### Transport - Voiture
- **Activité**: Kilomètres passagers (Gpkm)
  - Extraction directe de la variable "Kilomètres passagers - Voiture"
  - Interpolée linéairement sur toutes les années
- **Énergie finale**: Calcul pondéré multi-carburants (Mtoe)
  - Formule: Gpkm × 10⁹ km × [
    - %stock(Essence) × consommation(Essence)/100 × CF(Essence) +
    - %stock(Diesel) × consommation(Diesel)/100 × CF(Diesel) +
    - %stock(Électrique) × consommation(Électrique)/100 × CF(Électrique) +
    - %stock(Hybride) × consommation(Essence)/100 × CF(Essence)
  - ]
  - Données filtrées à partir de 2021 (taux de consommation indisponibles avant 2021)
  - Interpolée linéairement sur les années disponibles
- **Émissions GES**: Équivalent CO₂ total (MtCO2e)
  - Extraction directe des données brutes

#### Transport - Deux-roues motorisé (2RM)
- **Activité**: Kilomètres passagers (Gpkm)
  - Extraction directe de la variable "Kilomètres passagers - 2RM"
  - Interpolée linéairement sur toutes les années
- **Énergie finale**: Calcul essence uniquement (Mtoe)
  - Formule: Gpkm × 10⁹ km × (consommation_essence/100) × CF(Essence)
  - Données filtrées à partir de 2021
  - Interpolée linéairement sur les années disponibles
- **Émissions GES**: Équivalent CO₂ total (MtCO2e)
  - Extraction directe des données brutes

#### Démographie
- **Population**: Population totale (Millions)
  - Extraction directe des données brutes
  - Interpolée linéairement sur les années disponibles

### 4. Facteurs de conversion énergétique

| Type de carburant | Facteur de conversion | Unité |
|---|---|---|
| Essence | 0.00082 | toe/L |
| Diesel | 0.00092 | toe/L |
| Électrique | 0.000086 | toe/kWh |

**Hypothèse**: Les facteurs de conversion restent constants sur la période de prévision.

### 5. Calcul de la surface de plancher résidentiel

**Hypothèse**: Parc immobilier mixte avec proportion fixe entre habitat individuel et collectif

$$\text{Surface plancher totale} = \text{Nombre de maisons} \times (0,548 \times \text{Surface individuelle} + 0,452 \times \text{Surface collective})$$

Cela suppose une répartition stable 54,8% individuel / 45,2% collectif basée sur les données 2021.

### 6. Modèle de consommation carburant des voitures

**Hypothèse**: Chaque type de carburant a sa part de parc et son taux de consommation indépendants

$$\text{Énergie finale} = \text{VKM} \times 10^9 \times \sum (\%\text{parc}_{\text{carburant}} \times \frac{\text{consommation}_{\text{carburant}}}{100} \times \text{CF}_{\text{carburant}})$$

- % parc: Part de marché des véhicules utilisant chaque type de carburant
- consommation: L/100km ou kWh/100km spécifique à chaque type de carburant
- CF: Facteur de conversion en toe

Cela permet de modéliser la transition vers les véhicules électriques et l'adoption des hybrides.

## Format de sortie

### Structure du fichier Excel
Colonnes: `Géographie | Scénario | Secteur | Catégorie | Type | Année | Valeur | Unité`

Exemple:
```
France | SNBC-3 | Bâtiment - Résidentiel | Activité | Surface plancher | 2021 | 2791.73 | Mm²
France | SNBC-3 | Bâtiment - Résidentiel | Activité | Surface plancher | 2022 | 2809.94 | Mm²
...
```

### Fichiers de visualisation (5 fichiers PNG)
- **FR_Comprehensive_Residential.png**: 3 panneaux (Activité, Énergie finale, Émissions)
- **FR_Comprehensive_Services.png**: 3 panneaux (Activité, Énergie finale, Émissions)
- **FR_Comprehensive_Car.png**: 3 panneaux (Activité en Gpkm, Énergie finale, Émissions)
- **FR_Comprehensive_Motorcycle.png**: 3 panneaux (Activité en Gpkm, Énergie finale, Émissions)
- **FR_Comprehensive_Demography.png**: Population seulement (ligne continue AME-2024, tirets SNBC-3)

## Décomposition LMDI

### Méthode Log-Mean Divisia Index (LMDI)

La décomposition LMDI décompose les émissions de CO₂ en quatre leviers d'effet:

$$CO_2 = \frac{CO_2}{Énergie} \times \frac{Énergie}{Activité} \times \frac{Activité}{Population} \times Population$$

**Les 4 leviers:**
1. **Population** (Démographie): Effet de la variation du nombre d'habitants
2. **Sobriété** (Activité/Population): Effet du changement d'activité par habitant
3. **Efficacité énergétique** (Énergie/Activité): Effet de la réduction de consommation d'énergie par unité d'activité
4. **Décarbonation** (CO₂/Énergie): Effet du changement du contenu carbone de l'énergie

### Formule de contribution LMDI

Pour chaque levier x:
$$\text{Contribution}_x = \frac{CO_2(t) - CO_2(0)}{\ln(CO_2(t)) - \ln(CO_2(0))} \times \ln\left(\frac{x(t)}{x(0)}\right)$$

Cette pondération LMDI garantit que la somme des contributions égale exactement la variation totale de CO₂.

### Visualisations waterfall

Quatre graphiques waterfall présentent la décomposition pour:
1. **Transport - Voiture** (2021-2030)
2. **Transport - Voiture + 2RM** (combiné, 2021-2030)
3. **Bâtiment - Résidentiel** (2021-2030)
4. **Bâtiment - Tertiaire** (2021-2030)

Chaque graphique montre côte à côte les scénarios SNBC-3 et AME-2024 avec:
- Barres d'émission initiale et finale (CO₂ 2021 et 2030)
- Contributions positives (rouge) et négatives (vert) de chaque levier
- Lignes de connexion montrant la progression cumulative
- Valeurs CO₂ affichées au-dessus des barres d'émissions
- Valeurs des leviers affichées au-dessus des barres de contribution
- Échelles d'axe Y identiques pour faciliter la comparaison entre scénarios

## Hypothèses principales & Limitations

### Hypothèses
1. **Interpolation linéaire**: Les valeurs changent linéairement entre les points de données
2. **Pas d'extrapolation**: Aucune valeur projetée avant/après la plage de données disponible
3. **Ratios constants**: Répartition résidentielle (54,8/45,2) et facteurs de conversion d'énergie restent constants
4. **Flux de carburants indépendants**: Types de carburants des voitures traités indépendamment avec leur % de parc et consommation
5. **Méthodologie fixe**: Les méthodes de calcul sont cohérentes sur toute la période de prévision
6. **Population tirée de la catégorie Population**: Les valeurs réelles de population sont extraites du fichier compilé pour la décomposition LMDI

### Limitations
1. L'énergie finale pour le transport commence à partir de 2021 (données de consommation antérieures indisponibles)
2. SNBC-3 se termine en 2030; AME-2024 s'étend jusqu'en 2050
3. Les données d'émissions éparses peuvent montrer des sauts où l'interpolation comble de grands écarts
4. Aucun intervalle de confiance ou bandes d'incertitude fourni
5. Suppose que les modèles historiques de consommation de carburant s'appliquent aux scénarios futurs

## Notes sur la qualité des données

### Complétude des données
- **SNBC-3**: Données partielles (années clés: 2021, 2025, 2030)
- **AME-2024**: Plus complètes sur la plage 2019-2050
- Années intermédiaires manquantes remplies par interpolation linéaire

### Validation
- Surface de plancher résidentiel 2021 SNBC-3: 2 791,73 Mm² (raisonnable ~279M m²)
- Énergie finale voiture 2021 SNBC-3: 27,18 Mtoe (cohérent avec historique ~28 Mtoe)
- Énergie finale 2RM 2021 SNBC-3: 0,6258 Mtoe (~2% de la voiture, raisonnable)

## Exécution du script

```bash
cd stream3_visualization/Decomposition/code/code_FR
python FR_generate_compiled_and_visuals.py
python FR_lmdi_waterfall.py
```

**Emplacements de sortie:**
- Données: `../data/2025-01-06_FR_scenarios_compiled.xlsx`
- Visualisations brutes: `../reports/FR/visuals raw/`
- Graphiques LMDI: `../reports/FR/visuals decomposition/`

## Améliorations futures

Améliorations potentielles de la méthodologie:
1. Ajouter une quantification de l'incertitude (intervalles de confiance)
2. Implémenter une interpolation par spline par morceaux pour des courbes plus lisses
3. Ajouter une pondération de scénarios ou des modèles probabilistes
4. Inclure des calculs de taux de croissance annuels
5. Valider de manière croisée par rapport aux inventaires historiques d'émissions
6. Ajouter une analyse de sensibilité pour les paramètres clés

## Références

- Source des données brutes: `2025-12-15_FR scenarios data_before computation.xlsx`
- Format cible: Structure du jeu de données EC Decomposition
- Scénarios: SNBC-3 (objectifs officiels français), AME-2024 (scénario alternatif)

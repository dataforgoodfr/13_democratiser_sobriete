# Logique de Calcul - Décomposition LMDI et Secteurs

## 3. Secteurs & Métriques

### Bâtiment - Résidentiel

**Activité**: Surface de plancher totale (Mm²)
- Calculée comme : nombre de maisons × (54,8% × surface individuelle + 45,2% × surface collective)
- Sources : Nombre de maisons (millions) + Surfaces moyennes (individuelle & collective)

**Énergie finale**: Consommation totale d'énergie (Mtoe)
- Extraction directe des données brutes
- Interpolée linéairement sur toutes les années

**Émissions GES**: Équivalent CO₂ total (MtCO2e)
- Extraction directe des données brutes

---

### Bâtiment - Tertiaire

**Activité**: Surface de plancher totale (Mm²)
- Extraction directe de la variable "Surface plancher tertiaire" brute
- Interpolée linéairement sur toutes les années

**Énergie finale**: Consommation totale d'énergie (Mtoe)
- Extraction directe des données brutes
- Interpolée linéairement sur toutes les années

**Émissions GES**: Équivalent CO₂ total (MtCO2e)
- Extraction directe des données brutes

---

### Transport - Voiture

**Activité**: Kilomètres passagers (Gpkm)
- Extraction directe de la variable "Kilomètres passagers - Voiture"
- Interpolée linéairement sur toutes les années

**Énergie finale**: Calcul pondéré multi-carburants (Mtoe)
- Formule: Gpkm × 10⁹ km × [%stock(Essence) × consommation(Essence)/100 × CF(Essence) + %stock(Diesel) × consommation(Diesel)/100 × CF(Diesel) + %stock(Électrique) × consommation(Électrique)/100 × CF(Électrique) + %stock(Hybride) × consommation(Essence)/100 × CF(Essence)]
- Données filtrées à partir de 2021 (taux de consommation indisponibles avant 2021)
- Interpolée linéairement sur les années disponibles

**Émissions GES**: Équivalent CO₂ total (MtCO2e)
- Extraction directe des données brutes

---

### Agriculture - Cultures

**Activité**: Surface agricole (kha)
- Extraction directe de la variable "Activity"
- Agrégation de tous les types de cultures (céréales, cultures industrielles, fruits, légumes)
- Interpolée linéairement sur toutes les années

**Émissions GES**: Équivalent CO₂ total (MtCO2e)
- Extraction directe des données brutes
- Inclut émissions directes (carburants) et indirectes (engrais synthétiques, décomposition matière organique)

---

### Agriculture - Élevage

**Activité**: Cheptel bovin
- Extraction directe de la variable "Activity"
- Interpolée linéairement sur toutes les années

**Émissions GES**: Équivalent CO₂e total (MtCO2e)
- Extraction directe des données brutes


---

## Décomposition LMDI

### Méthode Log-Mean Divisia Index (LMDI)

**Modèle général (4 leviers) - Bâtiment et Transport**:

$$CO_2 = \frac{CO_2}{Énergie} \times \frac{Énergie}{Activité} \times \frac{Activité}{Population} \times Population$$

**Modèle simplifié (3 leviers) - Agriculture**:

$$CO_2 = \frac{CO_2}{Activité} \times \frac{Activité}{Population} \times Population$$

### Les leviers

**Secteurs Bâtiment & Transport (4 leviers)**:
- **Population (Démographie)**: Effet de la variation du nombre d'habitants
- **Sobriété (Activité/Population)**: Effet du changement d'activité par habitant
- **Efficacité énergétique (Énergie/Activité)**: Effet de la réduction de consommation d'énergie par unité d'activité
- **Décarbonation (CO₂/Énergie)**: Effet du changement du contenu carbone de l'énergie

**Secteurs Agriculture (3 leviers)**:
- **Population (Démographie)**: Effet de la variation du nombre d'habitants
- **Sobriété (Activité/Population)**: Effet du changement d'activité par habitant (production ou effectif animal)
- **Efficacité/Décarbonation (CO₂/Activité)**: Effet du changement d'intensité carbone par unité d'activité

### Formule de contribution LMDI

Pour chaque levier x:

$$Contribution = \frac{CO_2(t) - CO_2(0)}{\ln(CO_2(t)) - \ln(CO_2(0))} \times \ln\left(\frac{x(t)}{x(0)}\right)$$

Cette pondération LMDI garantit que la somme des contributions égale exactement la variation totale de CO₂.

---

## Visualisations Waterfall

### Graphiques de décomposition LMDI

5 graphiques waterfall présentent la décomposition pour:

1. **Transport - Voiture** (2021-2030)
2. **Bâtiment - Résidentiel** (2021-2030)
3. **Bâtiment - Tertiaire** (2021-2030)
4. **Agriculture - Culture** (2021-2030)
5. **Agriculture - Élevage bovin** (2021-2030)

### Structure des waterfall

Chaque graphique affiche:
- **Barre initiale**: Émissions 2021 (ligne de base)
- **Leviers positifs** (augmentent les émissions): Population, Sobriété négative
- **Leviers négatifs** (réduisent les émissions): Efficacité énergétique, Décarbonation, Sobriété positive
- **Barre finale**: Émissions 2030 (scénario)
- **Légende**: Codes couleurs standardisés par levier

### Comparaison multi-scénarios

Chaque secteur présente comparaison SNBC-3 vs AME-2024 pour:
- Impact relatif de chaque levier
- Contribution à la réduction totale d'émissions
- Sensibilité aux trajectoires de croissance économique

---

## 📈 Résultats clés

### Secteur Bâtiment - Résidentiel
- Levier dominant: **Efficacité énergétique** (rénovation thermique)
- Défi: **Sobriété** (surface par habitant en croissance)
- Opportunity: **Décarbonation** (électrification chauffage)

### Secteur Bâtiment - Tertiaire
- Levier dominant: **Efficacité énergétique** (amélioration climatisation, LED)
- Défi: **Sobriété** (croissance surface tertiaire)
- Opportunity: **Décarbonation** (électrification équipements)

### Secteur Transport - Voiture
- Levier dominant: **Décarbonation** (électrification parc)
- Défi: **Population/Activité** (augmentation km/habitant en zones périphériques)
- Opportunity: **Sobriété** (télétravail, partage véhicules)

### Secteur Agriculture - Production agricole
- Levier dominant: **Décarbonation** (réduction intensité carbone production)
- Défi: **Sobriété** (selon scénarios, pression démographique)
- Opportunity: **Population** (effet démographique selon projections)

### Secteur Agriculture - Élevage
- Levier dominant: **Décarbonation** (réduction intensité carbone troupeaux)
- Défi: **Sobriété** (effectif animal selon croissance alimentation)
- Opportunity: **Population** (effet démographique selon projections)


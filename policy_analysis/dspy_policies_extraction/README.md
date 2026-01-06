# Extraction des pollitiques des conclusions avec DSPy

L'objectif est d'extraire les pollitiques qui sont situées dans les conclusions des papiers de recherche. 
On utilise DSPy afin d'optimiser le prompt qui va etre utilisé pour extraire ces politiques. 
On optimise DSPy en maximisant la similarité d'une pollitique avec une autre en utilisant un cross encoder. 


Il existe 32 données labélisées qui peuvent etre utiliées en tant que dataset de validation. 
Des données synthétiques ont été crées afin d'entrainer le model. 

On peut charger le model generé ensuite par l'optimisation et l'utiliser pour les executions. 
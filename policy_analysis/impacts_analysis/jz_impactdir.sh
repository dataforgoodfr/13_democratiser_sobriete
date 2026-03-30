#!/bin/bash
#SBATCH --job-name=ImpactDirExtraction # nom du job
#SBATCH --output=ImpactDirExtraction%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=ImpactDirExtraction%j.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=a100 # demander des GPU A100 80 Go
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 1 tache (ou processus)
#SBATCH --gres=gpu:1 # reserver 1 GPU par noeud
#SBATCH --cpus-per-task=8 # reserver 8 CPU par tache (et memoire associee)
#SBATCH --time=10:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=utx@a100 # comptabilite A100
#SBATCH --array=0-9 # lancer 10 instances du job avec des indices de 0 à 9 (pour paralleliser OFFSET)

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load arch/a100 # selectionner les modules compiles pour les A100
#module load pytorch-gpu/py3/2.3.0 # charger les modules
module load miniforge/25.9.1
conda activate vllmenv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
set -x # activer l’echo des commandes
cd $WORK/13_democratiser_sobriete/policy_analysis
srun python vllm_impact_direction_extraction.py

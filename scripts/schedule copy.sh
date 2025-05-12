#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment='sector_scibert_W_BCE' +logger.wandb.name='sector_scibert_W_BCE lr 5e-5' trainer.optimizer.lr=0.001

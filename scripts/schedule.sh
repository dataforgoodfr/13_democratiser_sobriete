#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment=scibert_W_BCE_betterscheduler data.data_path='${paths.data_dir}/digitalisation_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 digitalisation long'
python src/train.py experiment=scibert_W_BCE_betterscheduler data.data_path='${paths.data_dir}/nutrition_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 nutrition long'
python src/train.py experiment=scibert_W_BCE_betterscheduler data.data_path='${paths.data_dir}/mobility_full_concat_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 mobility_full_concat long'
python src/train.py experiment=scibert_W_BCE_betterscheduler data.data_path='${paths.data_dir}/urban_ecology_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 urban_ecology long'
python src/train.py experiment=scibert_W_BCE_betterscheduler data.data_path='${paths.data_dir}/urban_infra_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 urban_infra long'

python src/train.py experiment=scibert_W_BCE_betterscheduler data.data_path='${paths.data_dir}/urban_governance_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 urban_governance lr++ long' model.optimizer.lr=1e-5
python src/train.py experiment=scibert_W_BCE_betterscheduler data.data_path='${paths.data_dir}/freight_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 freight lr++ long' model.optimizer.lr=1e-5


python src/train.py experiment=scibert_W_BCE data.data_path='${paths.data_dir}/trade_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 trade lr++ long'
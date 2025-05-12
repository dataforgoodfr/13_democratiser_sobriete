#!/bin/bash

python src/train.py experiment=scibert_W_BCE_prod data.data_path='${paths.data_dir}/nutrition_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 nutrition prod'
python src/train.py experiment=scibert_W_BCE_prod data.data_path='${paths.data_dir}/mobility_full_concat_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 mobility_full_concat prod'
python src/train.py experiment=scibert_W_BCE_prod data.data_path='${paths.data_dir}/urban_ecology_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 urban_ecology prod'
python src/train.py experiment=scibert_W_BCE_prod data.data_path='${paths.data_dir}/urban_infra_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 urban_infra prod'

python src/train.py experiment=scibert_W_BCE_prod data.data_path='${paths.data_dir}/freight_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 freight lr++ prod' model.optimizer.lr=1e-5
python src/train.py experiment=scibert_W_BCE_prod data.data_path='${paths.data_dir}/urban_governance_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 urban_governance lr++ prod' model.optimizer.lr=1e-5
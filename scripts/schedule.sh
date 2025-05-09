#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment=bert_base_uncased_W_BCE data.data_path='${paths.data_dir}/digitalisation_dataset.csv' +logger.wandb.name='W_BCE 2.5 digitalisation'
python src/train.py experiment=bert_base_uncased_W_BCE data.data_path='${paths.data_dir}/nutrition_dataset.csv' +logger.wandb.name='W_BCE 2.5 nutrition'
python src/train.py experiment=bert_base_uncased_W_BCE data.data_path='${paths.data_dir}/urban_governance_dataset.csv' +logger.wandb.name='W_BCE 2.5 urban_governance'
python src/train.py experiment=bert_base_uncased_W_BCE data.data_path='${paths.data_dir}/freight_dataset.csv' +logger.wandb.name='W_BCE 2.5 freight'
python src/train.py experiment=bert_base_uncased_W_BCE data.data_path='${paths.data_dir}/mobility_full_concat_dataset.csv' +logger.wandb.name='W_BCE 2.5 mobility_full_concat'
python src/train.py experiment=bert_base_uncased_W_BCE data.data_path='${paths.data_dir}/urban_ecology_dataset.csv' +logger.wandb.name='W_BCE 2.5 urban_ecology'
python src/train.py experiment=bert_base_uncased_W_BCE data.data_path='${paths.data_dir}/urban_infra_dataset.csv' +logger.wandb.name='W_BCE 2.5 urban_infra'

python src/train.py experiment=scibert_W_BCE data.data_path='${paths.data_dir}/nutrition_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 nutrition'
python src/train.py experiment=scibert_W_BCE data.data_path='${paths.data_dir}/urban_governance_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 urban_governance'
python src/train.py experiment=scibert_W_BCE data.data_path='${paths.data_dir}/freight_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 freight'
python src/train.py experiment=scibert_W_BCE data.data_path='${paths.data_dir}/mobility_full_concat_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 mobility_full_concat'
python src/train.py experiment=scibert_W_BCE data.data_path='${paths.data_dir}/urban_ecology_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 urban_ecology'
python src/train.py experiment=scibert_W_BCE data.data_path='${paths.data_dir}/urban_infra_dataset.csv' +logger.wandb.name='SciBert W_BCE 2.5 urban_infra'

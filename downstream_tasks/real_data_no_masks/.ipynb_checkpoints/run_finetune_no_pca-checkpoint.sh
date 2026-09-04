#!/usr/bin/env bash
set -e
# change dir to the repository root
cd /home/jovyan/GENLINK

source /home/jovyan/miniconda3/bin/activate
conda activate /home/jovyan/miniconda3/envs/genlink

JSON_PATH=./downstream_tasks/real_data_no_masks
python utils/pipeline.py --data_folder /home/jovyan/final_datasets_2/all_degree/labeled --hardware "0|1|2|3|4|5|6|7" --model_list ${JSON_PATH}/model_list.json --running_params ${JSON_PATH}/running_params_no_pca.json \
--models_per_gpu 2
